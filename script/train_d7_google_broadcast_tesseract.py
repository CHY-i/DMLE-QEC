#!/usr/bin/env python3
"""Train Google d7 r10->r4 broadcast-source data with Tesseract evaluation.

This is intentionally separate from ``train_d7_google_subsample.py``.  The old
script remains the MWPM historical path; this script matches the newer
single-GPU sequential sub-TN training style and evaluates by broadcasting the
short r4 DEM back to the original r10 experiment.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import stim
import torch
from torch.utils.data import DataLoader, TensorDataset
from tesseract_decoder import TesseractSinterDecoder, utils as tesseract_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from script.train_d7_google_subsample import _load_google_round_root  # noqa: E402
from src import (  # noqa: E402
    GroupTN,
    PCM,
    broadcast_dem,
    extract_broadcast_source_dem,
    get_error_rates,
    subsample_d5_pcms_from_circuit,
    update_dem,
)


DTYPE = torch.float64


class TeeLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", buffering=1)

    def write(self, text: str) -> None:
        print(text, flush=True)
        self.file.write(text + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse bool: {value}")


def current_probs(model: GroupTN) -> np.ndarray:
    return torch.sigmoid(model.priors_logits.detach().cpu()).numpy()


def to_bool_numpy(tensor: torch.Tensor, *, max_items: int | None = None) -> np.ndarray:
    if max_items is not None:
        tensor = tensor[:max_items]
    return tensor.detach().cpu().numpy().astype(bool, copy=False)


def select_eval_subset(dets: torch.Tensor, obs: torch.Tensor, eval_size: int, generator: torch.Generator):
    if eval_size <= 0 or eval_size > dets.shape[0]:
        raise ValueError(f"eval_size must be in [1, {dets.shape[0]}].")
    indices = torch.randperm(dets.shape[0], generator=generator)[:eval_size]
    return to_bool_numpy(dets[indices]), to_bool_numpy(obs[indices]).reshape(-1), indices


def make_broadcast_eval_dem(short_initial_dem, error_rates: np.ndarray, *, broadcast_time_layer: int, repeat_chunk: int):
    short_dem = update_dem(short_initial_dem, np.clip(error_rates, 1e-15, 1.0 - 1e-15))
    return broadcast_dem(
        short_dem,
        broadcast_time_layer=int(broadcast_time_layer),
        repeat_chunk=int(repeat_chunk),
    )


def error_support_key(instruction: stim.DemInstruction):
    """Canonical detector/logical support, intentionally ignoring `^`."""
    targets = []
    for target in instruction.targets_copy():
        if target.is_relative_detector_id():
            targets.append(("D", int(target.val)))
        elif target.is_logical_observable_id():
            targets.append(("L", int(target.val)))
    return tuple(sorted(targets))


def grouped_error_key(instruction: stim.DemInstruction):
    """Canonical grouped support that preserves Stim's `^` decomposition."""
    components = [[]]
    for target in instruction.targets_copy():
        if target.is_separator():
            components.append([])
        elif target.is_relative_detector_id():
            components[-1].append(("D", int(target.val)))
        elif target.is_logical_observable_id():
            components[-1].append(("L", int(target.val)))
    return tuple(sorted(tuple(sorted(component)) for component in components if component))


def error_instructions(dem):
    return [instruction for instruction in dem.flattened() if instruction.type == "error"]


def reorder_dem_to_reference_support(source_dem, reference_dem):
    """Put source errors in reference column order while retaining source targets."""
    source_errors = error_instructions(source_dem)
    reference_errors = error_instructions(reference_dem)
    source_by_support = {error_support_key(inst): inst for inst in source_errors}
    reference_supports = [error_support_key(inst) for inst in reference_errors]

    if len(source_by_support) != len(source_errors):
        raise ValueError("Source DEM has duplicate detector/logical supports; mapping is ambiguous.")
    if len(set(reference_supports)) != len(reference_supports):
        raise ValueError("Reference DEM has duplicate detector/logical supports; mapping is ambiguous.")
    if set(source_by_support) != set(reference_supports):
        missing = len(set(reference_supports) - set(source_by_support))
        extra = len(set(source_by_support) - set(reference_supports))
        raise ValueError(f"Source/reference error supports differ: missing={missing}, extra={extra}.")

    reordered = stim.DetectorErrorModel()
    for support in reference_supports:
        reordered.append(source_by_support[support])
    for instruction in source_dem.flattened():
        if instruction.type != "error":
            reordered.append(instruction)
    return reordered


def compare_grouped_error_models(left_dem, right_dem):
    """Compare order-independent grouped supports and their probabilities."""
    def grouped_probability_map(dem):
        result = {}
        for instruction in error_instructions(dem):
            key = grouped_error_key(instruction)
            if key in result:
                raise ValueError("DEM has duplicate grouped error supports; comparison is ambiguous.")
            result[key] = float(instruction.args_copy()[0])
        return result

    left = grouped_probability_map(left_dem)
    right = grouped_probability_map(right_dem)
    supports_equal = left.keys() == right.keys()
    if not supports_equal:
        return False, float("inf")
    max_abs_probability_diff = max((abs(value - right[key]) for key, value in left.items()), default=0.0)
    return True, float(max_abs_probability_diff)


def tesseract_ler(
    *,
    label: str,
    dem,
    eval_dets: np.ndarray,
    eval_obs: np.ndarray,
    batch_size: int,
    det_beam: int,
    beam_climbing: bool,
    pqlimit: int,
    num_det_orders: int,
    det_order_method_name: str,
    logger: TeeLogger | None = None,
):
    det_order_method = getattr(tesseract_utils.DetOrder, det_order_method_name)
    if logger is not None:
        logger.write(
            f"# Starting Tesseract Eval [{label}] "
            f"(det_beam={det_beam}, beam_climbing={beam_climbing}, pqlimit={pqlimit}, "
            f"num_det_orders={num_det_orders}, det_order_method={det_order_method_name}, "
            f"eval_size={len(eval_dets)})"
        )

    compile_t0 = time.perf_counter()
    compiled = TesseractSinterDecoder(
        det_beam=int(det_beam),
        beam_climbing=bool(beam_climbing),
        pqlimit=int(pqlimit),
        num_det_orders=int(num_det_orders),
        det_order_method=det_order_method,
    ).compile_decoder_for_dem(dem=dem)
    compile_sec = time.perf_counter() - compile_t0

    wrong = 0
    total = 0
    decode_t0 = time.perf_counter()
    for start in range(0, len(eval_dets), batch_size):
        stop = min(start + batch_size, len(eval_dets))
        pred = compiled.decoder.decode_batch(np.asarray(eval_dets[start:stop], dtype=bool))
        pred = np.asarray(pred, dtype=bool).reshape(-1)
        truth = np.asarray(eval_obs[start:stop], dtype=bool).reshape(-1)
        wrong += int(np.count_nonzero(pred != truth))
        total += int(len(truth))
    decode_sec = time.perf_counter() - decode_t0
    return {
        "label": label,
        "ler": float(wrong / total),
        "wrong": int(wrong),
        "total": int(total),
        "compile_sec": float(compile_sec),
        "decode_sec": float(decode_sec),
        "num_detectors": int(dem.num_detectors),
        "num_errors": int(dem.num_errors),
        "num_observables": int(dem.num_observables),
    }


def train_one_epoch(
    *,
    model: GroupTN,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    mini_batch: int,
    dev: str,
    sequential_patch_backward: bool,
    epoch: int,
    epochs: int,
    logger: TeeLogger,
    run_t0: float,
):
    losses = []
    weighted_loss_sum = 0.0
    sample_count = 0
    optimizer_steps = len(dataloader)
    for batch_idx, (syndrome_batch,) in enumerate(dataloader, start=1):
        batch_t0 = time.perf_counter()
        syndrome_batch = syndrome_batch.to(dev, non_blocking=True).to(DTYPE)
        actual_batch_size = int(syndrome_batch.size(0))
        chunks = syndrome_batch.split(mini_batch, dim=0)

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for chunk in chunks:
            loss_scale = chunk.size(0) / float(actual_batch_size)
            if sequential_patch_backward:
                loss_k = model.sequential_sub_loss_and_grad(chunk, loss_scale=loss_scale)
            else:
                loss_k = model(chunk) * loss_scale
                loss_k.backward()
            step_loss += float(loss_k.detach().cpu().item())
        optimizer.step()

        losses.append(step_loss)
        weighted_loss_sum += step_loss * actual_batch_size
        sample_count += actual_batch_size
        elapsed_hours = (time.perf_counter() - run_t0) / 3600.0
        logger.write(
            f"epoch {epoch}/{epochs} batch {batch_idx}/{optimizer_steps}: "
            f"loss={step_loss:.6f}, batch_size={actual_batch_size}, "
            f"mini_batch={mini_batch}, lr={optimizer.param_groups[0]['lr']:.6g}, "
            f"batch_sec={time.perf_counter() - batch_t0:.2f}, elapsed={elapsed_hours:.2f}h"
        )

    avg_loss = weighted_loss_sum / sample_count
    return avg_loss, float(np.mean(losses)), sample_count, optimizer_steps


def checkpoint_payload(
    *,
    epoch: int,
    model: GroupTN,
    optimizer: torch.optim.Optimizer,
    er_ref: np.ndarray,
    oer: np.ndarray,
    loss: float | None,
    tesseract_eval: dict | None,
    baseline_eval: dict | None,
    params: dict,
    checkpoint_kind: str,
):
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "er_ref": torch.from_numpy(er_ref),
        "oer": torch.from_numpy(oer),
        "loss": None if loss is None else float(loss),
        "tesseract_eval": tesseract_eval,
        "baseline_eval": baseline_eval,
        "params": params,
        "checkpoint_kind": checkpoint_kind,
    }


def save_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def run(args):
    train_round_root = Path(args.train_round_root)
    eval_round_root = Path(args.eval_round_root)
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name
    log_dir = Path(args.log_dir) / run_name
    ckpt_dir = Path(args.checkpoint_dir) / run_name / timestamp
    log_path = log_dir / f"{run_name}_{timestamp}.log"
    logger = TeeLogger(log_path)

    try:
        torch.manual_seed(int(args.seed))
        generator = torch.Generator().manual_seed(int(args.seed))

        ideal_circuit, _noisy_circuit, dets, _obs, short_reference_dem, train_round_dir = _load_google_round_root(
            train_round_root
        )
        _eval_ideal, _eval_noisy, eval_dets_all, eval_obs_all, eval_initial_dem, eval_round_dir = (
            _load_google_round_root(eval_round_root)
        )
        if dets.shape[0] != eval_dets_all.shape[0]:
            raise ValueError(f"train/eval shot counts differ: {dets.shape[0]} vs {eval_dets_all.shape[0]}")
        eval_dets, eval_obs, eval_indices = select_eval_subset(
            eval_dets_all,
            eval_obs_all,
            int(args.eval_size),
            generator,
        )

        eval_round_count = int(args.broadcast_time_layer)
        short_initial_dem = short_reference_dem
        initial_r10_dem_path = None
        initial_r04_dem_path = None
        initial_r10_roundtrip_path = None
        initial_support_reordered = False
        initial_pcm_matches_reference = True
        initial_logical_support_matches_reference = True
        initial_roundtrip_supports_equal = None
        initial_roundtrip_max_abs_probability_diff = None
        if args.initial_r10_dem_path is not None:
            initial_r10_dem_path = Path(args.initial_r10_dem_path)
            if not initial_r10_dem_path.exists():
                raise FileNotFoundError(initial_r10_dem_path)
            initial_r10_dem = stim.DetectorErrorModel.from_file(str(initial_r10_dem_path))
            if initial_r10_dem.num_detectors != eval_initial_dem.num_detectors:
                raise ValueError(
                    f"Initial r10 DEM detector count {initial_r10_dem.num_detectors} does not match "
                    f"eval DEM detector count {eval_initial_dem.num_detectors}."
                )
            extracted = extract_broadcast_source_dem(
                initial_r10_dem,
                source_rounds=eval_round_count,
                bulk_layer=int(args.initial_bulk_layer),
            )
            short_initial_dem = reorder_dem_to_reference_support(extracted["dem"], short_reference_dem)
            initial_support_reordered = True

            initial_pcm, initial_logical = PCM(short_initial_dem)
            reference_pcm, reference_logical = PCM(short_reference_dem)
            initial_pcm_matches_reference = bool(np.array_equal(initial_pcm, reference_pcm))
            initial_logical_support_matches_reference = bool(np.array_equal(initial_logical, reference_logical))
            if not initial_pcm_matches_reference or not initial_logical_support_matches_reference:
                raise ValueError("Reordered RL-r04 DEM is not PCM-compatible with the existing TN trees.")

        train_dets = dets
        if args.max_train_shots is not None:
            max_train_shots = int(args.max_train_shots)
            if max_train_shots <= 0 or max_train_shots > dets.shape[0]:
                raise ValueError(f"max_train_shots must be in [1, {dets.shape[0]}].")
            train_indices = torch.randperm(dets.shape[0], generator=generator)[:max_train_shots]
            train_dets = dets[train_indices]

        dataset = TensorDataset(train_dets)
        dataloader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            generator=generator,
            drop_last=False,
        )
        optimizer_steps_per_epoch = len(dataloader)

        pcm, _ = PCM(short_initial_dem)
        er_ref = get_error_rates(short_initial_dem)
        init_er_t = torch.from_numpy(er_ref).to(DTYPE)

        sub_pcms, sub_dets, sub_errors, sub_full_masks = subsample_d5_pcms_from_circuit(
            ideal_circuit,
            short_initial_dem,
            print_info=True,
            return_edge_masks=True,
        )
        missing_trees = [
            str(Path(args.path_dir) / f"subsample_tree_{i}.json")
            for i in range(len(sub_pcms))
            if not (Path(args.path_dir) / f"subsample_tree_{i}.json").exists()
        ]
        if missing_trees:
            raise FileNotFoundError(f"Missing contraction trees: {missing_trees}")

        model = GroupTN(
            d=7,
            r=4,
            sub_pcms=sub_pcms,
            sub_dets=sub_dets,
            sub_errors=sub_errors,
            init_priors=init_er_t,
            dev=args.dev,
            devices=[args.dev],
            dtype=DTYPE,
            path_dir=args.path_dir,
            manual_sync_grads=False,
            sub_full_masks=sub_full_masks,
            stop_grad_partial=bool(args.stop_grad_partial),
            partial_only_grad=bool(args.partial_only_grad),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

        initial_broadcast_dem = make_broadcast_eval_dem(
            short_initial_dem,
            er_ref,
            broadcast_time_layer=eval_round_count,
            repeat_chunk=int(args.broadcast_repeat_chunk),
        )
        if args.initial_r10_dem_path is not None:
            initial_roundtrip_supports_equal, initial_roundtrip_max_abs_probability_diff = (
                compare_grouped_error_models(initial_r10_dem, initial_broadcast_dem)
            )
            if (
                not initial_roundtrip_supports_equal
                or initial_roundtrip_max_abs_probability_diff != 0.0
            ):
                raise ValueError(
                    "RL r10 -> r04 -> r10 round trip failed: "
                    f"supports_equal={initial_roundtrip_supports_equal}, "
                    f"max_abs_probability_diff={initial_roundtrip_max_abs_probability_diff}."
                )

            initial_r04_dem_path = ckpt_dir / "initial_google_rl_r04.dem"
            initial_r10_roundtrip_path = ckpt_dir / "initial_google_rl_r04_broadcast_r10.dem"
            initial_r04_dem_path.parent.mkdir(parents=True, exist_ok=True)
            short_initial_dem.to_file(str(initial_r04_dem_path))
            initial_broadcast_dem.to_file(str(initial_r10_roundtrip_path))
        baseline_dem_path = Path(args.baseline_dem_path)
        baseline_eval = None

        params = {
            "run_name": run_name,
            "timestamp": timestamp,
            "experiment": "Google d7 X/r10 broadcast-source r04 NLL training, broadcast to r10 Tesseract eval",
            "train_round_root": str(train_round_root),
            "train_round_dir": train_round_dir,
            "eval_round_root": str(eval_round_root),
            "eval_round_dir": eval_round_dir,
            "d": 7,
            "r": 4,
            "eval_r": eval_round_count,
            "train_shots_total": int(dets.shape[0]),
            "max_train_shots": args.max_train_shots,
            "samples_per_epoch": int(train_dets.shape[0]),
            "batch_size": int(args.batch_size),
            "mini_batch": int(args.mini_batch),
            "optimizer_steps_per_epoch": int(optimizer_steps_per_epoch),
            "lr": float(args.lr),
            "optimizer": "Adam",
            "dev": args.dev,
            "devices": [args.dev],
            "dtype": "float64",
            "seed": int(args.seed),
            "eval_size": int(args.eval_size),
            "eval_indices_source": "torch.randperm(eval_shots, seed)[:eval_size]",
            "eval_indices_first10": [int(x) for x in eval_indices[:10].tolist()],
            "decode_interval": int(args.decode_interval),
            "periodic_checkpoint_interval": int(args.periodic_checkpoint_interval),
            "keep_top_k": int(args.keep_top_k),
            "decoder": "TesseractSinterDecoder",
            "tesseract_det_beam": int(args.tesseract_det_beam),
            "tesseract_beam_climbing": bool(args.tesseract_beam_climbing),
            "tesseract_pqlimit": int(args.tesseract_pqlimit),
            "tesseract_num_det_orders": int(args.tesseract_num_det_orders),
            "tesseract_det_order_method": args.tesseract_det_order_method,
            "tesseract_eval_batch_size": int(args.tesseract_eval_batch_size),
            "broadcast_eval_dem": True,
            "broadcast_time_layer": eval_round_count,
            "broadcast_repeat_chunk": int(args.broadcast_repeat_chunk),
            "short_dem_num_detectors": int(short_initial_dem.num_detectors),
            "short_dem_num_errors": int(short_initial_dem.num_errors),
            "short_dem_num_observables": int(short_initial_dem.num_observables),
            "eval_dem_num_detectors": int(eval_initial_dem.num_detectors),
            "eval_dem_num_errors": int(eval_initial_dem.num_errors),
            "eval_dem_num_observables": int(eval_initial_dem.num_observables),
            "pcm_shape": tuple(int(x) for x in pcm.shape),
            "path_dir": args.path_dir,
            "log_dir": str(log_dir),
            "checkpoint_dir": str(ckpt_dir),
            "baseline_dem_path": str(baseline_dem_path),
            "initial_r10_dem_path": None if initial_r10_dem_path is None else str(initial_r10_dem_path),
            "initial_bulk_layer": int(args.initial_bulk_layer),
            "initial_support_reordered": initial_support_reordered,
            "initial_pcm_matches_reference": initial_pcm_matches_reference,
            "initial_logical_support_matches_reference": initial_logical_support_matches_reference,
            "initial_roundtrip_supports_equal": initial_roundtrip_supports_equal,
            "initial_roundtrip_max_abs_probability_diff": initial_roundtrip_max_abs_probability_diff,
            "initial_r04_dem_path": None if initial_r04_dem_path is None else str(initial_r04_dem_path),
            "initial_r10_roundtrip_path": (
                None if initial_r10_roundtrip_path is None else str(initial_r10_roundtrip_path)
            ),
            "reuse_initial_eval_checkpoint": args.reuse_initial_eval_checkpoint,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "manual_sync_grads": False,
            "sequential_patch_backward": bool(args.sequential_patch_backward),
            "stop_grad_partial": bool(args.stop_grad_partial),
            "partial_only_grad": bool(args.partial_only_grad),
            "partial_gradient_stats": getattr(model, "partial_gradient_stats", None),
        }

        logger.write(f"# {run_name} training")
        logger.write(f"# timestamp: {timestamp}")
        logger.write("# parameters:")
        for key, value in params.items():
            logger.write(f"#   {key}: {value}")

        if args.reuse_initial_eval_checkpoint is not None:
            reuse_path = Path(args.reuse_initial_eval_checkpoint)
            reused = torch.load(reuse_path, map_location="cpu", weights_only=False)
            baseline_eval = reused.get("baseline_eval")
            initial_eval = reused.get("tesseract_eval")
            if baseline_eval is None or initial_eval is None:
                raise ValueError(f"Reusable checkpoint lacks initial evaluation results: {reuse_path}")
            if baseline_eval["total"] != int(args.eval_size) or initial_eval["total"] != int(args.eval_size):
                raise ValueError("Reusable initial evaluation size does not match --eval_size.")
            logger.write(f"# Reusing initial Tesseract evaluations from: {reuse_path}")
            logger.write(
                f"# Baseline Tesseract Eval (reused, eval_size={args.eval_size}): "
                f"LER={baseline_eval['ler']:.6f}, wrong={baseline_eval['wrong']}/{baseline_eval['total']}"
            )
            logger.write(
                f"# Initial Tesseract Eval (reused r04 broadcast to r10, eval_size={args.eval_size}): "
                f"LER={initial_eval['ler']:.6f}, wrong={initial_eval['wrong']}/{initial_eval['total']}"
            )
        else:
            if baseline_dem_path.exists():
                baseline_dem = stim.DetectorErrorModel.from_file(str(baseline_dem_path))
                baseline_eval = tesseract_ler(
                    label="Google_RL_optimized_prior",
                    dem=baseline_dem,
                    eval_dets=eval_dets,
                    eval_obs=eval_obs,
                    batch_size=int(args.tesseract_eval_batch_size),
                    det_beam=int(args.tesseract_det_beam),
                    beam_climbing=bool(args.tesseract_beam_climbing),
                    pqlimit=int(args.tesseract_pqlimit),
                    num_det_orders=int(args.tesseract_num_det_orders),
                    det_order_method_name=args.tesseract_det_order_method,
                    logger=logger,
                )
            if baseline_eval is not None:
                logger.write(
                    f"# Baseline Tesseract Eval ({baseline_eval['label']}, eval_size={args.eval_size}): "
                    f"LER={baseline_eval['ler']:.6f}, wrong={baseline_eval['wrong']}/{baseline_eval['total']}, "
                    f"compile_sec={baseline_eval['compile_sec']:.3f}, decode_sec={baseline_eval['decode_sec']:.3f}"
                )
            else:
                logger.write(f"# Baseline Tesseract Eval skipped; missing baseline_dem_path={baseline_dem_path}")

            initial_eval = tesseract_ler(
                label="Initial_r04_broadcast_r10",
                dem=initial_broadcast_dem,
                eval_dets=eval_dets,
                eval_obs=eval_obs,
                batch_size=int(args.tesseract_eval_batch_size),
                det_beam=int(args.tesseract_det_beam),
                beam_climbing=bool(args.tesseract_beam_climbing),
                pqlimit=int(args.tesseract_pqlimit),
                num_det_orders=int(args.tesseract_num_det_orders),
                det_order_method_name=args.tesseract_det_order_method,
                logger=logger,
            )
            logger.write(
                f"# Initial Tesseract Eval (r04 broadcast to r10, eval_size={args.eval_size}): "
                f"LER={initial_eval['ler']:.6f}, wrong={initial_eval['wrong']}/{initial_eval['total']}, "
                f"compile_sec={initial_eval['compile_sec']:.3f}, decode_sec={initial_eval['decode_sec']:.3f}"
            )

        init_oer = er_ref.copy()
        save_checkpoint(
            ckpt_dir / f"{run_name}_{timestamp}_initial.pt",
            checkpoint_payload(
                epoch=0,
                model=model,
                optimizer=optimizer,
                er_ref=er_ref,
                oer=init_oer,
                loss=None,
                tesseract_eval=initial_eval,
                baseline_eval=baseline_eval,
                params=params,
                checkpoint_kind="initial",
            ),
        )

        best_checkpoints: list[tuple[float, int, Path]] = []
        last_loss = None
        run_t0 = time.perf_counter()

        for epoch in range(1, int(args.epochs) + 1):
            epoch_t0 = time.perf_counter()
            avg_loss, batch_mean_loss, samples_seen, steps = train_one_epoch(
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
                mini_batch=int(args.mini_batch),
                dev=args.dev,
                sequential_patch_backward=bool(args.sequential_patch_backward),
                epoch=epoch,
                epochs=int(args.epochs),
                logger=logger,
                run_t0=run_t0,
            )
            last_loss = avg_loss
            oer = current_probs(model)
            elapsed_hours = (time.perf_counter() - run_t0) / 3600.0
            logger.write(
                f"epoch {epoch}/{args.epochs}: loss={avg_loss:.6f}, "
                f"batch_mean_loss={batch_mean_loss:.6f}, samples={samples_seen}, "
                f"optimizer_steps={steps}, lr={optimizer.param_groups[0]['lr']:.6g}, "
                f"epoch_sec={time.perf_counter() - epoch_t0:.2f}, elapsed={elapsed_hours:.2f}h"
            )

            eval_row = None
            if int(args.decode_interval) > 0 and epoch % int(args.decode_interval) == 0:
                eval_dem = make_broadcast_eval_dem(
                    short_initial_dem,
                    oer,
                    broadcast_time_layer=eval_round_count,
                    repeat_chunk=int(args.broadcast_repeat_chunk),
                )
                eval_row = tesseract_ler(
                    label=f"epoch_{epoch}_r04_broadcast_r10",
                    dem=eval_dem,
                    eval_dets=eval_dets,
                    eval_obs=eval_obs,
                    batch_size=int(args.tesseract_eval_batch_size),
                    det_beam=int(args.tesseract_det_beam),
                    beam_climbing=bool(args.tesseract_beam_climbing),
                    pqlimit=int(args.tesseract_pqlimit),
                    num_det_orders=int(args.tesseract_num_det_orders),
                    det_order_method_name=args.tesseract_det_order_method,
                    logger=logger,
                )
                logger.write(
                    f"epoch {epoch}/{args.epochs}: "
                    f"Tesseract(r04 broadcast to r10, det_beam={args.tesseract_det_beam}, "
                    f"beam_climbing={args.tesseract_beam_climbing}, pqlimit={args.tesseract_pqlimit}, "
                    f"num_det_orders={args.tesseract_num_det_orders}, "
                    f"det_order_method={args.tesseract_det_order_method}, eval_size={args.eval_size}) "
                    f"LER={eval_row['ler']:.6f}, wrong={eval_row['wrong']}/{eval_row['total']}, "
                    f"compile_sec={eval_row['compile_sec']:.3f}, decode_sec={eval_row['decode_sec']:.3f}"
                )
                eval_path = ckpt_dir / (
                    f"{run_name}_{timestamp}_epoch{epoch:06d}_"
                    f"tesseract_ler{eval_row['ler']:.6f}.pt"
                )
                save_checkpoint(
                    eval_path,
                    checkpoint_payload(
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        er_ref=er_ref,
                        oer=oer,
                        loss=avg_loss,
                        tesseract_eval=eval_row,
                        baseline_eval=baseline_eval,
                        params=params,
                        checkpoint_kind="best_tesseract_candidate",
                    ),
                )
                best_checkpoints.append((eval_row["ler"], epoch, eval_path))
                best_checkpoints.sort(key=lambda item: (item[0], item[1]))
                while len(best_checkpoints) > int(args.keep_top_k):
                    _ler, _epoch, stale_path = best_checkpoints.pop()
                    if stale_path.exists():
                        stale_path.unlink()

            if int(args.periodic_checkpoint_interval) > 0 and epoch % int(args.periodic_checkpoint_interval) == 0:
                periodic_path = ckpt_dir / f"{run_name}_{timestamp}_periodic_epoch{epoch:06d}.pt"
                save_checkpoint(
                    periodic_path,
                    checkpoint_payload(
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        er_ref=er_ref,
                        oer=oer,
                        loss=avg_loss,
                        tesseract_eval=eval_row,
                        baseline_eval=baseline_eval,
                        params=params,
                        checkpoint_kind="periodic",
                    ),
                )
                logger.write(f"epoch {epoch}/{args.epochs}: periodic checkpoint={periodic_path.name}")

        final_oer = current_probs(model)
        final_eval = None
        if bool(args.final_decode):
            final_dem = make_broadcast_eval_dem(
                short_initial_dem,
                final_oer,
                broadcast_time_layer=eval_round_count,
                repeat_chunk=int(args.broadcast_repeat_chunk),
            )
            final_eval = tesseract_ler(
                label="final_r04_broadcast_r10",
                dem=final_dem,
                eval_dets=eval_dets,
                eval_obs=eval_obs,
                batch_size=int(args.tesseract_eval_batch_size),
                det_beam=int(args.tesseract_det_beam),
                beam_climbing=bool(args.tesseract_beam_climbing),
                pqlimit=int(args.tesseract_pqlimit),
                num_det_orders=int(args.tesseract_num_det_orders),
                det_order_method_name=args.tesseract_det_order_method,
                logger=logger,
            )
            logger.write(
                f"# Final Tesseract Eval: LER={final_eval['ler']:.6f}, "
                f"wrong={final_eval['wrong']}/{final_eval['total']}, "
                f"compile_sec={final_eval['compile_sec']:.3f}, decode_sec={final_eval['decode_sec']:.3f}"
            )

        final_path = ckpt_dir / f"{run_name}_{timestamp}_final_epoch{args.epochs:06d}.pt"
        save_checkpoint(
            final_path,
            checkpoint_payload(
                epoch=int(args.epochs),
                model=model,
                optimizer=optimizer,
                er_ref=er_ref,
                oer=final_oer,
                loss=last_loss,
                tesseract_eval=final_eval,
                baseline_eval=baseline_eval,
                params=params,
                checkpoint_kind="final",
            ),
        )
        logger.write("# Final results:")
        if baseline_eval is not None:
            logger.write(f"#   Baseline Tesseract LER: {baseline_eval['ler']:.6f}")
        logger.write(f"#   Initial Tesseract LER: {initial_eval['ler']:.6f}")
        if final_eval is not None:
            logger.write(f"#   Final Tesseract LER: {final_eval['ler']:.6f}")
        logger.write(f"#   Final checkpoint: {final_path}")
        if best_checkpoints:
            logger.write("# Best Tesseract checkpoints:")
            for ler, epoch, path in best_checkpoints:
                logger.write(f"#   epoch {epoch}: LER={ler:.6f}, file={path.name}")
        return 0
    except BaseException:
        logger.write("# FATAL ERROR")
        for line in traceback.format_exc().rstrip().splitlines():
            logger.write(line)
        raise
    finally:
        logger.close()


def parse_args():
    willow_root = Path(
        os.environ.get(
            "PATCHDMLE_WILLOW_DATA",
            REPO_ROOT / "dataset/google_willow",
        )
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_round_root",
        default=str(REPO_ROOT / "dataset/google_broadcast_source/d7_at_q6_7_bulk2/X/r04"),
    )
    parser.add_argument(
        "--eval_round_root",
        default=str(willow_root / "d7_at_q6_7/X/r10"),
    )
    parser.add_argument(
        "--baseline_dem_path",
        default=str(
            willow_root
            / "d7_at_q6_7/X/r10/decoding_results/"
            "correlated_matching_decoder_with_rl_optimized_prior/error_model.dem"
        ),
    )
    parser.add_argument(
        "--initial_r10_dem_path",
        default=None,
        help="Optional full r10 DEM to reverse-broadcast into the trainable r04 initialization.",
    )
    parser.add_argument("--initial_bulk_layer", type=int, default=2)
    parser.add_argument(
        "--reuse_initial_eval_checkpoint",
        default=None,
        help="Reuse baseline/initial Tesseract results from a compatible epoch-0 checkpoint.",
    )
    parser.add_argument("--run_name", default="d7r4_googleX_broadcast_tesseract")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--mini_batch", type=int, default=256)
    parser.add_argument("--max_train_shots", type=int, default=None)
    parser.add_argument("--eval_size", type=int, default=10000)
    parser.add_argument("--dev", default="cuda:0")
    parser.add_argument("--seed", type=int, default=75328)
    parser.add_argument("--decode_interval", type=int, default=1)
    parser.add_argument("--periodic_checkpoint_interval", type=int, default=50)
    parser.add_argument("--keep_top_k", type=int, default=10)
    parser.add_argument("--path_dir", default="path/d7r4_googleX_subsample")
    parser.add_argument("--broadcast_time_layer", type=int, default=10)
    parser.add_argument("--broadcast_repeat_chunk", type=int, default=1)
    parser.add_argument("--sequential_patch_backward", type=parse_bool, default=True)
    parser.add_argument("--stop_grad_partial", type=parse_bool, default=True)
    parser.add_argument("--partial_only_grad", type=parse_bool, default=False)
    parser.add_argument("--tesseract_det_beam", type=int, default=10)
    parser.add_argument("--tesseract_beam_climbing", type=parse_bool, default=True)
    parser.add_argument("--tesseract_pqlimit", type=int, default=1_000_000)
    parser.add_argument("--tesseract_num_det_orders", type=int, default=11)
    parser.add_argument(
        "--tesseract_det_order_method",
        choices=["DetBFS", "DetCoordinate", "DetIndex"],
        default="DetIndex",
    )
    parser.add_argument("--tesseract_eval_batch_size", type=int, default=1000)
    parser.add_argument("--final_decode", type=parse_bool, default=True)
    parser.add_argument("--log_dir", default="log/sc_tn/google")
    parser.add_argument("--checkpoint_dir", default="data/google")
    parser.add_argument("--timestamp", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
