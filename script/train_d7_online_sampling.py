#!/usr/bin/env python3
"""Train d7 simulated DEM parameters with online Stim sampling.

This is separate from ``train_d7_subsample.py`` so the fixed finite-dataset
training path remains available.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import stim
import torch
from tesseract_decoder import TesseractSinterDecoder, utils as tesseract_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from script.train_d7_subsample import add_perturbation  # noqa: E402
from src import (  # noqa: E402
    GroupTN,
    get_error_rates,
    subsample_d5_pcms_from_circuit,
    update_dem,
)


CODE_TASK = "surface_code:rotated_memory_z"
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


def build_true_dem(d: int, r: int, error_prob: float):
    circuit = stim.Circuit.generated(
        code_task=CODE_TASK,
        distance=d,
        rounds=r,
        after_clifford_depolarization=error_prob,
        before_round_data_depolarization=error_prob,
        before_measure_flip_probability=error_prob,
        after_reset_flip_probability=error_prob,
    )
    dem = circuit.detector_error_model(
        decompose_errors=False,
        flatten_loops=True,
    )
    return circuit, dem, get_error_rates(dem)


def sample_dem_batch(dem, shots: int, seed: int) -> torch.Tensor:
    sampler = dem.compile_sampler(seed=int(seed))
    dets, _, _ = sampler.sample(shots=int(shots))
    return torch.from_numpy(dets.astype(np.float64, copy=False))


def sample_dem_eval(dem, shots: int, seed: int):
    sampler = dem.compile_sampler(seed=int(seed))
    dets, obs, _ = sampler.sample(shots=int(shots))
    return dets.astype(bool, copy=False), obs.astype(bool, copy=False).reshape(-1)


def current_probs(model: GroupTN) -> np.ndarray:
    return torch.sigmoid(model.priors_logits.detach().cpu()).numpy()


def mean_relative_error(oer: np.ndarray, er_gt: np.ndarray) -> float:
    return float(np.mean(np.abs(oer - er_gt) / er_gt))


def tesseract_ler(
    dem,
    error_rates: np.ndarray,
    eval_dets: np.ndarray,
    eval_obs: np.ndarray,
    *,
    batch_size: int,
    det_beam: int,
    beam_climbing: bool,
    pqlimit: int,
    num_det_orders: int,
    det_order_method_name: str,
):
    eval_dem = update_dem(dem, np.clip(error_rates, 1e-15, 1.0 - 1e-15))
    det_order_method = getattr(tesseract_utils.DetOrder, det_order_method_name)

    compile_t0 = time.perf_counter()
    compiled = TesseractSinterDecoder(
        det_beam=int(det_beam),
        beam_climbing=bool(beam_climbing),
        pqlimit=int(pqlimit),
        num_det_orders=int(num_det_orders),
        det_order_method=det_order_method,
    ).compile_decoder_for_dem(dem=eval_dem)
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
        "ler": float(wrong / total),
        "wrong": int(wrong),
        "total": int(total),
        "compile_sec": float(compile_sec),
        "decode_sec": float(decode_sec),
    }


def train_one_online_epoch(
    model: GroupTN,
    optimizer: torch.optim.Optimizer,
    dem,
    *,
    shots: int,
    seed: int,
    mini_batch: int,
    dev: str,
    sequential_patch_backward: bool,
) -> float:
    syndrome_batch = sample_dem_batch(dem, shots=shots, seed=seed).to(dev).to(DTYPE)
    chunks = syndrome_batch.split(mini_batch, dim=0) if mini_batch and mini_batch < shots else (syndrome_batch,)
    optimizer.zero_grad(set_to_none=True)
    epoch_loss = 0.0
    for chunk in chunks:
        loss_scale = chunk.size(0) / float(shots)
        if sequential_patch_backward:
            loss_k = model.sequential_sub_loss_and_grad(chunk, loss_scale=loss_scale)
        else:
            loss_k = model(chunk) * loss_scale
            loss_k.backward()
        epoch_loss += float(loss_k.detach().cpu().item())
    optimizer.step()
    return epoch_loss


def checkpoint_payload(
    *,
    epoch: int,
    model: GroupTN,
    optimizer: torch.optim.Optimizer,
    er_gt: np.ndarray,
    oer: np.ndarray,
    loss: float | None,
    mre: float,
    tesseract_eval: dict | None,
    params: dict,
    checkpoint_kind: str,
):
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "er_gt": torch.from_numpy(er_gt),
        "oer": torch.from_numpy(oer),
        "loss": None if loss is None else float(loss),
        "mre": float(mre),
        "tesseract_eval": tesseract_eval,
        "params": params,
        "checkpoint_kind": checkpoint_kind,
    }


def save_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def write_dem(path: Path, dem):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(dem))


def run(args):
    d = int(args.d)
    r = int(args.r)
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "mask" if bool(args.stop_grad_partial) else "nomask"
    run_name = args.run_name or f"d{d}r{r}_online_p{str(args.error_prob).replace('.', 'p')}_{mode}"
    log_dir = Path(args.log_dir) / run_name
    ckpt_dir = Path(args.checkpoint_dir) / run_name / timestamp
    log_path = log_dir / f"{run_name}_{timestamp}.log"
    logger = TeeLogger(log_path)

    try:
        torch.manual_seed(int(args.seed))
        circuit, dem, er = build_true_dem(d, r, float(args.error_prob))
        expected_num_errors = 2807 if d == 7 and r == 4 else None
        if expected_num_errors is not None and dem.num_errors != expected_num_errors:
            raise ValueError(f"Expected {expected_num_errors} errors, got {dem.num_errors}.")

        er_gt_t, init_er_t = add_perturbation(er, float(args.perturbation_strength))
        er_gt = er_gt_t.numpy()
        init_er = init_er_t.numpy()
        initial_mre = mean_relative_error(init_er, er_gt)

        true_dem_path = ckpt_dir / "true.dem"
        initial_dem_path = ckpt_dir / "initial_perturbed.dem"
        write_dem(true_dem_path, dem)
        write_dem(initial_dem_path, update_dem(dem, init_er))

        sub_pcms, sub_dets, sub_errors, sub_full_masks = subsample_d5_pcms_from_circuit(
            circuit,
            dem,
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
            d=d,
            r=r,
            sub_pcms=sub_pcms,
            sub_dets=sub_dets,
            sub_errors=sub_errors,
            init_priors=init_er_t,
            dev=args.dev,
            devices=[args.dev],
            dtype=DTYPE,
            path_dir=args.path_dir,
            sub_full_masks=sub_full_masks,
            stop_grad_partial=bool(args.stop_grad_partial),
            partial_only_grad=bool(args.partial_only_grad),
            manual_sync_grads=False,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
        eval_dets, eval_obs = sample_dem_eval(dem, int(args.eval_size), int(args.eval_seed))

        params = {
            "run_name": run_name,
            "timestamp": timestamp,
            "code_task": CODE_TASK,
            "d": d,
            "r": r,
            "error_prob": float(args.error_prob),
            "after_clifford_depolarization": float(args.error_prob),
            "before_round_data_depolarization": float(args.error_prob),
            "before_measure_flip_probability": float(args.error_prob),
            "after_reset_flip_probability": float(args.error_prob),
            "decompose_errors": False,
            "flatten_loops": True,
            "true_dem_num_errors": int(dem.num_errors),
            "true_dem_num_detectors": int(dem.num_detectors),
            "true_dem_num_observables": int(dem.num_observables),
            "epochs": int(args.epochs),
            "num_shots": int(args.samples_per_epoch),
            "samples_per_epoch": int(args.samples_per_epoch),
            "batch_size": int(args.samples_per_epoch),
            "mini_batch": int(args.mini_batch),
            "optimizer_steps_per_epoch": 1,
            "lr": float(args.lr),
            "lr_final": None,
            "lr_decay_epochs": 0,
            "optimizer": "Adam",
            "dev": args.dev,
            "devices": [args.dev],
            "dtype": "float64",
            "perturbation_strength": float(args.perturbation_strength),
            "seed": int(args.seed),
            "online_seed": int(args.online_seed),
            "eval_size": int(args.eval_size),
            "eval_seed": int(args.eval_seed),
            "eval_source": "fixed eval set sampled from true DEM",
            "decode_interval": int(args.decode_interval),
            "periodic_checkpoint_interval": int(args.periodic_checkpoint_interval),
            "keep_top_k": int(args.keep_top_k),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "decoder": "TesseractSinterDecoder",
            "tesseract_det_beam": int(args.tesseract_det_beam),
            "tesseract_beam_climbing": bool(args.tesseract_beam_climbing),
            "tesseract_pqlimit": int(args.tesseract_pqlimit),
            "tesseract_num_det_orders": int(args.tesseract_num_det_orders),
            "tesseract_det_order_method": args.tesseract_det_order_method,
            "tesseract_eval_batch_size": int(args.tesseract_eval_batch_size),
            "stop_grad_partial": bool(args.stop_grad_partial),
            "partial_only_grad": bool(args.partial_only_grad),
            "manual_sync_grads": False,
            "sequential_patch_backward": bool(args.sequential_patch_backward),
            "partial_gradient_stats": getattr(model, "partial_gradient_stats", None),
            "path_dir": args.path_dir,
            "log_dir": str(log_dir),
            "checkpoint_dir": str(ckpt_dir),
            "true_dem_path": str(true_dem_path),
            "initial_perturbed_dem_path": str(initial_dem_path),
        }

        logger.write(f"# {run_name} online sampling training")
        logger.write(f"# timestamp: {timestamp}")
        logger.write("# parameters:")
        for key, value in params.items():
            logger.write(f"#   {key}: {value}")
        logger.write(f"# Initial MRE: {initial_mre:.6f}")
        logger.write(f"# True DEM: {true_dem_path}")
        logger.write(f"# Initial perturbed DEM: {initial_dem_path}")

        initial_eval = tesseract_ler(
            dem,
            init_er,
            eval_dets,
            eval_obs,
            batch_size=int(args.tesseract_eval_batch_size),
            det_beam=int(args.tesseract_det_beam),
            beam_climbing=bool(args.tesseract_beam_climbing),
            pqlimit=int(args.tesseract_pqlimit),
            num_det_orders=int(args.tesseract_num_det_orders),
            det_order_method_name=args.tesseract_det_order_method,
        )
        true_eval = tesseract_ler(
            dem,
            er_gt,
            eval_dets,
            eval_obs,
            batch_size=int(args.tesseract_eval_batch_size),
            det_beam=int(args.tesseract_det_beam),
            beam_climbing=bool(args.tesseract_beam_climbing),
            pqlimit=int(args.tesseract_pqlimit),
            num_det_orders=int(args.tesseract_num_det_orders),
            det_order_method_name=args.tesseract_det_order_method,
        )
        logger.write(
            f"# Initial Tesseract Eval (det_beam={args.tesseract_det_beam}, eval_size={args.eval_size}): "
            f"LER={initial_eval['ler']:.6f}, wrong={initial_eval['wrong']}/{initial_eval['total']}, "
            f"compile_sec={initial_eval['compile_sec']:.3f}, decode_sec={initial_eval['decode_sec']:.3f}"
        )
        logger.write(
            f"# True Tesseract Eval (det_beam={args.tesseract_det_beam}, eval_size={args.eval_size}): "
            f"LER={true_eval['ler']:.6f}, wrong={true_eval['wrong']}/{true_eval['total']}, "
            f"compile_sec={true_eval['compile_sec']:.3f}, decode_sec={true_eval['decode_sec']:.3f}"
        )

        initial_payload = checkpoint_payload(
            epoch=0,
            model=model,
            optimizer=optimizer,
            er_gt=er_gt,
            oer=init_er,
            loss=None,
            mre=initial_mre,
            tesseract_eval=initial_eval,
            params=params,
            checkpoint_kind="initial",
        )
        save_checkpoint(ckpt_dir / f"{run_name}_{timestamp}_initial.pt", initial_payload)

        best_checkpoints: list[tuple[float, int, Path]] = []
        best_mre = initial_mre
        best_mre_epoch = 0
        last_loss = None
        run_t0 = time.perf_counter()

        for epoch in range(1, int(args.epochs) + 1):
            train_seed = int(args.online_seed) + epoch - 1
            epoch_t0 = time.perf_counter()
            loss = train_one_online_epoch(
                model,
                optimizer,
                dem,
                shots=int(args.samples_per_epoch),
                seed=train_seed,
                mini_batch=int(args.mini_batch),
                dev=args.dev,
                sequential_patch_backward=bool(args.sequential_patch_backward),
            )
            last_loss = loss
            oer = current_probs(model)
            current_mre = mean_relative_error(oer, er_gt)
            if current_mre < best_mre:
                best_mre = current_mre
                best_mre_epoch = epoch
            elapsed_hours = (time.perf_counter() - run_t0) / 3600.0
            logger.write(
                f"epoch {epoch}/{args.epochs}: online_seed={train_seed}, "
                f"samples={args.samples_per_epoch}, loss={loss:.6f}, "
                f"MRE={current_mre:.6f}, best_MRE={best_mre:.6f}, "
                f"lr={args.lr:.6g}, train_sec={time.perf_counter() - epoch_t0:.2f}, "
                f"elapsed={elapsed_hours:.2f}h"
            )

            eval_row = None
            if int(args.decode_interval) > 0 and epoch % int(args.decode_interval) == 0:
                eval_row = tesseract_ler(
                    dem,
                    oer,
                    eval_dets,
                    eval_obs,
                    batch_size=int(args.tesseract_eval_batch_size),
                    det_beam=int(args.tesseract_det_beam),
                    beam_climbing=bool(args.tesseract_beam_climbing),
                    pqlimit=int(args.tesseract_pqlimit),
                    num_det_orders=int(args.tesseract_num_det_orders),
                    det_order_method_name=args.tesseract_det_order_method,
                )
                logger.write(
                    f"epoch {epoch}/{args.epochs}: "
                    f"Tesseract(det_beam={args.tesseract_det_beam}, "
                    f"beam_climbing={args.tesseract_beam_climbing}, "
                    f"pqlimit={args.tesseract_pqlimit}, "
                    f"num_det_orders={args.tesseract_num_det_orders}, "
                    f"det_order_method={args.tesseract_det_order_method}, "
                    f"eval_size={args.eval_size}) "
                    f"LER={eval_row['ler']:.6f}, wrong={eval_row['wrong']}/{eval_row['total']}, "
                    f"compile_sec={eval_row['compile_sec']:.3f}, decode_sec={eval_row['decode_sec']:.3f}"
                )
                eval_path = ckpt_dir / (
                    f"{run_name}_{timestamp}_epoch{epoch:06d}_"
                    f"ler{eval_row['ler']:.6f}_mre{current_mre:.6f}.pt"
                )
                save_checkpoint(
                    eval_path,
                    checkpoint_payload(
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        er_gt=er_gt,
                        oer=oer,
                        loss=loss,
                        mre=current_mre,
                        tesseract_eval=eval_row,
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
                periodic_path = ckpt_dir / f"{run_name}_{timestamp}_periodic_epoch{epoch:06d}_mre{current_mre:.6f}.pt"
                save_checkpoint(
                    periodic_path,
                    checkpoint_payload(
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        er_gt=er_gt,
                        oer=oer,
                        loss=loss,
                        mre=current_mre,
                        tesseract_eval=eval_row,
                        params=params,
                        checkpoint_kind="periodic",
                    ),
                )
                logger.write(f"epoch {epoch}/{args.epochs}: periodic checkpoint={periodic_path.name}")

        final_oer = current_probs(model)
        final_mre = mean_relative_error(final_oer, er_gt)
        final_eval = None
        if bool(args.final_decode):
            final_eval = tesseract_ler(
                dem,
                final_oer,
                eval_dets,
                eval_obs,
                batch_size=int(args.tesseract_eval_batch_size),
                det_beam=int(args.tesseract_det_beam),
                beam_climbing=bool(args.tesseract_beam_climbing),
                pqlimit=int(args.tesseract_pqlimit),
                num_det_orders=int(args.tesseract_num_det_orders),
                det_order_method_name=args.tesseract_det_order_method,
            )
            logger.write(
                f"# Final Tesseract Eval: LER={final_eval['ler']:.6f}, "
                f"wrong={final_eval['wrong']}/{final_eval['total']}, "
                f"compile_sec={final_eval['compile_sec']:.3f}, decode_sec={final_eval['decode_sec']:.3f}"
            )

        final_path = ckpt_dir / f"{run_name}_{timestamp}_final_epoch{args.epochs:06d}_mre{final_mre:.6f}.pt"
        save_checkpoint(
            final_path,
            checkpoint_payload(
                epoch=int(args.epochs),
                model=model,
                optimizer=optimizer,
                er_gt=er_gt,
                oer=final_oer,
                loss=last_loss,
                mre=final_mre,
                tesseract_eval=final_eval,
                params=params,
                checkpoint_kind="final",
            ),
        )
        logger.write("# Final results:")
        logger.write(f"#   Initial MRE: {initial_mre:.6f}")
        logger.write(f"#   Final MRE: {final_mre:.6f}")
        logger.write(f"#   Best MRE: {best_mre:.6f}")
        logger.write(f"#   Best MRE epoch: {best_mre_epoch}")
        logger.write(f"#   Final checkpoint: {final_path}")
        if best_checkpoints:
            logger.write("# Best Tesseract checkpoints:")
            for ler, epoch, path in best_checkpoints:
                logger.write(f"#   epoch {epoch}: LER={ler:.6f}, file={path.name}")
        return final_mre
    finally:
        logger.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=7)
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--error_prob", type=float, default=0.005)
    parser.add_argument("--perturbation_strength", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--samples_per_epoch", type=int, default=1024)
    parser.add_argument("--mini_batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dev", default="cuda:0")
    parser.add_argument("--path_dir", default="path/d7r4")
    parser.add_argument("--seed", type=int, default=75328)
    parser.add_argument("--online_seed", type=int, default=95328)
    parser.add_argument("--eval_size", type=int, default=10000)
    parser.add_argument("--eval_seed", type=int, default=75329)
    parser.add_argument("--decode_interval", type=int, default=50)
    parser.add_argument("--periodic_checkpoint_interval", type=int, default=500)
    parser.add_argument("--keep_top_k", type=int, default=10)
    parser.add_argument("--stop_grad_partial", type=parse_bool, default=False)
    parser.add_argument("--partial_only_grad", type=parse_bool, default=True)
    parser.add_argument("--sequential_patch_backward", type=parse_bool, default=True)
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
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--log_dir", default="log/sc_tn/simulation")
    parser.add_argument("--checkpoint_dir", default="data/simulation")
    parser.add_argument("--timestamp", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
