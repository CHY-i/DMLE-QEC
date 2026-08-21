#!/usr/bin/env python3
"""Train a full d7r10 synthetic DEM with tied repeated-round parameters.

The five d5 patch TNs consume the complete r10 detector graph. Repeated bulk
mechanisms share one trainable logit, so the 8137 full-DEM mechanisms are
represented by the 2809 parameters of the exact r4 broadcast source.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock

import numpy as np
import stim
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from script.train_d7_fixed_pool_sampling import (  # noqa: E402
    TeeLogger,
    parse_bool,
    sample_dem_dets,
    sample_dem_eval,
    tesseract_ler,
    write_dem,
)
from script.train_d7_subsample import add_perturbation  # noqa: E402
from src import (  # noqa: E402
    GroupTN,
    broadcast_dem,
    extract_broadcast_source_dem,
    get_error_rates,
    subsample_d5_pcms_from_circuit,
    update_dem,
)


DTYPE = torch.float64
EXPECTED_FULL_ERRORS = 8137
EXPECTED_SHARED_PARAMETERS = 2809


def _error_rows(dem: stim.DetectorErrorModel):
    rows = []
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        support = tuple(str(target) for target in instruction.targets_copy())
        probability = float(instruction.args_copy()[0])
        rows.append((support, probability))
    return rows


def build_tied_parameterization(full_dem: stim.DetectorErrorModel):
    """Recover the exact r4 source and map every r10 error to its source group."""
    source_info = extract_broadcast_source_dem(
        full_dem,
        source_rounds=10,
        bulk_layer=2,
    )
    source_dem = source_info["dem"]
    if source_dem.num_errors != EXPECTED_SHARED_PARAMETERS:
        raise ValueError(
            f"Expected {EXPECTED_SHARED_PARAMETERS} shared parameters, "
            f"got {source_dem.num_errors}."
        )

    full_rows = _error_rows(full_dem)
    full_supports = [row[0] for row in full_rows]
    if len(set(full_supports)) != len(full_supports):
        raise ValueError("Full r10 DEM has duplicate error supports; mapping is ambiguous.")

    # Unique valid probabilities let the normal broadcast routine carry each
    # source group ID to every repeated full-r10 mechanism.
    markers = np.linspace(0.01, 0.49, source_dem.num_errors, dtype=np.float64)
    marked_source = update_dem(source_dem, markers)
    marked_full = broadcast_dem(marked_source, broadcast_time_layer=10, repeat_chunk=1)
    marked_rows = _error_rows(marked_full)
    marker_to_group = {float(value).hex(): index for index, value in enumerate(markers)}

    support_to_group = {}
    for support, marker in marked_rows:
        marker_key = float(marker).hex()
        if marker_key not in marker_to_group:
            raise ValueError(f"Broadcast changed marker probability {marker!r}.")
        if support in support_to_group:
            raise ValueError("Broadcast DEM has duplicate error supports.")
        support_to_group[support] = marker_to_group[marker_key]

    full_support_set = set(full_supports)
    missing = [support for support in full_supports if support not in support_to_group]
    extra = [support for support in support_to_group if support not in full_support_set]
    if missing or extra:
        raise ValueError(
            f"Broadcast support mismatch: missing={len(missing)}, extra={len(extra)}."
        )

    full_to_group = np.asarray(
        [support_to_group[support] for support in full_supports],
        dtype=np.int64,
    )
    shared_true = get_error_rates(source_dem).astype(np.float64, copy=False)
    full_true = get_error_rates(full_dem).astype(np.float64, copy=False)
    expanded_true = shared_true[full_to_group]
    max_abs_diff = float(np.max(np.abs(expanded_true - full_true)))
    if not np.array_equal(expanded_true, full_true):
        raise ValueError(
            "Extracted shared probabilities do not exactly reconstruct the true r10 DEM: "
            f"max_abs_diff={max_abs_diff:.17g}."
        )
    if np.unique(full_to_group).size != source_dem.num_errors:
        raise ValueError("At least one shared parameter is unused by the full r10 DEM.")

    return {
        "source_dem": source_dem,
        "source_info": source_info,
        "full_to_group": full_to_group,
        "shared_true": shared_true,
        "full_true": full_true,
        "max_abs_reconstruction_diff": max_abs_diff,
    }


class TiedGroupTN(GroupTN):
    """GroupTN whose full-r10 priors are gathered from shared logits."""

    def __init__(self, *args, full_to_group: np.ndarray, **kwargs):
        super().__init__(*args, **kwargs)
        mapping = torch.as_tensor(
            full_to_group,
            dtype=torch.long,
            device=self.primary_dev,
        )
        self.register_buffer("full_to_group", mapping)

    def _refresh_priors_cache(self):
        # Rebuild this gather for every micro-batch. Reusing it after backward
        # would reuse an already-freed autograd graph.
        expanded = self.priors_logits.index_select(0, self.full_to_group)
        priors_by_device = {}
        for device in sorted(set(self.tn_devices)):
            if device == self.primary_dev:
                priors_by_device[device] = expanded
            else:
                priors_by_device[device] = expanded.to(device, non_blocking=True)
        return priors_by_device

    def shared_probabilities(self) -> np.ndarray:
        return torch.sigmoid(self.priors_logits.detach()).cpu().numpy()

    def full_probabilities(self) -> np.ndarray:
        shared = torch.sigmoid(self.priors_logits.detach())
        return shared.index_select(0, self.full_to_group).cpu().numpy()

    def sequential_sub_loss_and_grad(self, syndromes, loss_scale=1.0):
        """Backward each patch separately while rebuilding the tied gather graph."""
        syndromes_by_device = self._prepare_syndromes_by_device(syndromes)
        total_loss_value = 0.0
        for i in range(self.n_sub):
            priors_by_device = self._refresh_priors_cache()
            loss_i = self._forward_single_sub(i, syndromes_by_device, priors_by_device)
            scaled_loss = loss_i * (loss_scale / self.n_sub)
            scaled_loss.backward()
            total_loss_value += scaled_loss.detach().to(self.primary_dev).item()
        return torch.tensor(total_loss_value, device=self.primary_dev, dtype=self.dtype)


def mean_relative_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(estimate - truth) / truth))


def grouped_training_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    gradient: np.ndarray,
    bulk_mask: np.ndarray,
) -> dict[str, float]:
    boundary_mask = ~bulk_mask
    return {
        "boundary_mre": mean_relative_error(estimate[boundary_mask], truth[boundary_mask]),
        "bulk_mre": mean_relative_error(estimate[bulk_mask], truth[bulk_mask]),
        "boundary_grad_mean_abs": float(np.mean(np.abs(gradient[boundary_mask]))),
        "bulk_grad_mean_abs": float(np.mean(np.abs(gradient[bulk_mask]))),
    }


def _parse_devices(devices: str) -> list[str]:
    parsed = [item.strip() for item in devices.split(",") if item.strip()]
    if len(parsed) != 5:
        raise ValueError(f"Expected exactly five devices, got {parsed}.")
    return parsed


def _parse_mini_batches(value: str | None, devices: list[str], default: int) -> list[int]:
    if value is None:
        return [int(default)] * len(devices)
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(parsed) != len(devices) or any(item <= 0 for item in parsed):
        raise ValueError(
            f"Expected one positive mini-batch per device {devices}, got {parsed}."
        )
    return parsed


def _reset_cuda_peak_stats(devices: list[str]) -> None:
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.reset_peak_memory_stats(device)


def _cuda_peak_gb(devices: list[str]) -> dict[str, float]:
    return {
        device: torch.cuda.max_memory_allocated(device) / (1024**3)
        for device in devices
    }


def train_patch_parallel_batch(
    model: TiedGroupTN,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
    mini_batch: int,
) -> float:
    batch = batch.to(model.primary_dev, dtype=DTYPE, non_blocking=True)
    actual_batch_size = int(batch.size(0))
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for chunk in batch.split(int(mini_batch), dim=0):
        loss_scale = float(chunk.size(0)) / float(actual_batch_size)
        loss = model.manual_sync_loss_and_grad(chunk, loss_scale=loss_scale)
        total_loss += float(loss.cpu().item())
    optimizer.step()
    return total_loss


def train_data_parallel_batch(
    models: list[TiedGroupTN],
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
    mini_batches: list[int],
    partition_mode: str,
) -> tuple[float, list[dict[str, float | int | str]]]:
    actual_batch_size = int(batch.size(0))
    for replica in models:
        replica.zero_grad(set_to_none=True)

    def backward_chunks(
        replica: TiedGroupTN,
        chunks: list[torch.Tensor],
        mini_batch: int,
    ) -> dict[str, float | int | str]:
        started = time.perf_counter()
        local_loss = 0.0
        local_samples = 0
        local_chunks = 0
        for host_chunk in chunks:
            chunk = host_chunk.to(replica.primary_dev, dtype=DTYPE, non_blocking=True)
            loss_scale = float(chunk.size(0)) / float(actual_batch_size)
            loss = replica.sequential_sub_loss_and_grad(chunk, loss_scale=loss_scale)
            local_loss += float(loss.cpu().item())
            local_samples += int(chunk.size(0))
            local_chunks += 1
        return {
            "device": replica.primary_dev,
            "samples": local_samples,
            "chunks": local_chunks,
            "seconds": time.perf_counter() - started,
            "loss": local_loss,
            "mini_batch": int(mini_batch),
        }

    def backward_dynamic(
        replica: TiedGroupTN,
        mini_batch: int,
        cursor: list[int],
        cursor_lock: Lock,
    ) -> dict[str, float | int | str]:
        started = time.perf_counter()
        local_loss = 0.0
        local_samples = 0
        local_chunks = 0
        while True:
            with cursor_lock:
                start = cursor[0]
                stop = min(start + int(mini_batch), actual_batch_size)
                cursor[0] = stop
            if start >= actual_batch_size:
                break
            chunk = batch[start:stop].to(
                replica.primary_dev,
                dtype=DTYPE,
                non_blocking=True,
            )
            loss_scale = float(chunk.size(0)) / float(actual_batch_size)
            loss = replica.sequential_sub_loss_and_grad(chunk, loss_scale=loss_scale)
            local_loss += float(loss.cpu().item())
            local_samples += int(chunk.size(0))
            local_chunks += 1
        return {
            "device": replica.primary_dev,
            "samples": local_samples,
            "chunks": local_chunks,
            "seconds": time.perf_counter() - started,
            "loss": local_loss,
            "mini_batch": int(mini_batch),
        }

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        if partition_mode == "dynamic":
            cursor = [0]
            cursor_lock = Lock()
            futures = [
                executor.submit(
                    backward_dynamic,
                    replica,
                    mini_batch,
                    cursor,
                    cursor_lock,
                )
                for replica, mini_batch in zip(models, mini_batches)
            ]
        else:
            local_batches = torch.tensor_split(batch, len(models), dim=0)
            local_chunks = [
                list(local_batch.split(int(mini_batch), dim=0))
                for local_batch, mini_batch in zip(local_batches, mini_batches)
            ]
            futures = [
                executor.submit(backward_chunks, replica, chunks, mini_batch)
                for replica, chunks, mini_batch in zip(models, local_chunks, mini_batches)
            ]
        replica_stats = [future.result() for future in futures]
        total_loss = sum(float(row["loss"]) for row in replica_stats)

    assigned_samples = sum(int(row["samples"]) for row in replica_stats)
    if assigned_samples != actual_batch_size:
        raise RuntimeError(
            f"Data-parallel assignment covered {assigned_samples}/{actual_batch_size} samples."
        )

    master = models[0]
    master_grad = master.priors_logits.grad
    if master_grad is None:
        raise RuntimeError("Primary data-parallel replica produced no gradient.")
    for replica in models[1:]:
        replica_grad = replica.priors_logits.grad
        if replica_grad is None:
            raise RuntimeError(f"Replica {replica.primary_dev} produced no gradient.")
        master_grad.add_(replica_grad.to(master.primary_dev, non_blocking=True))

    optimizer.step()
    with torch.no_grad():
        updated_logits = master.priors_logits.detach()
        for replica in models[1:]:
            replica.priors_logits.copy_(updated_logits.to(replica.primary_dev, non_blocking=True))
    return total_loss, replica_stats


def _checkpoint_payload(
    *,
    epoch: int,
    model: TiedGroupTN,
    optimizer: torch.optim.Optimizer,
    loss: float | None,
    group_mre: float,
    full_mre: float,
    tesseract_eval: dict | None,
    params: dict,
    kind: str,
):
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": None if loss is None else float(loss),
        "group_mre": float(group_mre),
        "full_mre": float(full_mre),
        "tesseract_eval": tesseract_eval,
        "params": params,
        "checkpoint_kind": kind,
    }


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def run(args) -> None:
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or "d7r10_tied_fixed50k_mask"
    log_dir = Path(args.log_dir)
    checkpoint_dir = Path(args.checkpoint_dir) / timestamp
    log_path = log_dir / f"{run_name}_{timestamp}.log"
    logger = TeeLogger(log_path)

    try:
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))
        devices = _parse_devices(args.devices)
        mini_batches = _parse_mini_batches(args.mini_batches, devices, int(args.mini_batch))
        primary_dev = devices[0]
        if int(args.train_size) <= 0 or int(args.eval_size) <= 0:
            raise ValueError("train_size and eval_size must be positive.")
        if int(args.batch_size) <= 0 or int(args.mini_batch) <= 0:
            raise ValueError("batch_size and mini_batch must be positive.")
        if int(args.mini_batch) > int(args.batch_size):
            raise ValueError("mini_batch must not exceed batch_size.")
        if int(args.train_seed) == int(args.eval_seed):
            raise ValueError("Training and evaluation sampling seeds must differ.")

        data_root = Path(args.data_root)
        ideal_path = data_root / "circuit_ideal.stim"
        noisy_path = data_root / "circuit_noisy_si1000.stim"
        if not ideal_path.exists() or not noisy_path.exists():
            raise FileNotFoundError(f"Missing circuit files under {data_root}.")
        ideal_circuit = stim.Circuit.from_file(str(ideal_path))
        noisy_circuit = stim.Circuit.from_file(str(noisy_path))
        generated_true_dem = noisy_circuit.detector_error_model(
            decompose_errors=False,
            flatten_loops=True,
        )
        reuse_run_dir = Path(args.reuse_run_dir) if args.reuse_run_dir else None
        if reuse_run_dir is not None:
            true_dem = stim.DetectorErrorModel.from_file(str(reuse_run_dir / "true.dem"))
            reused_rows = _error_rows(true_dem)
            generated_rows = _error_rows(generated_true_dem)
            if [row[0] for row in reused_rows] != [row[0] for row in generated_rows]:
                raise ValueError("Reused true DEM supports differ from the generated true DEM.")
            max_true_diff = max(
                abs(reused[1] - generated[1])
                for reused, generated in zip(reused_rows, generated_rows)
            )
            if max_true_diff > 1e-15:
                raise ValueError(
                    f"Reused true DEM probabilities differ: max_abs_diff={max_true_diff}."
                )
        else:
            true_dem = generated_true_dem
        if true_dem.num_errors != EXPECTED_FULL_ERRORS:
            raise ValueError(
                f"Expected {EXPECTED_FULL_ERRORS} full-r10 errors, got {true_dem.num_errors}."
            )

        tied = build_tied_parameterization(true_dem)
        shared_true = tied["shared_true"]
        full_true = tied["full_true"]
        full_to_group = tied["full_to_group"]
        group_copy_counts = np.bincount(
            full_to_group,
            minlength=len(shared_true),
        )
        bulk_group_mask = group_copy_counts > 1

        if reuse_run_dir is not None:
            initial_checkpoints = sorted(reuse_run_dir.glob("*_initial.pt"))
            if len(initial_checkpoints) != 1:
                raise ValueError(
                    f"Expected one initial checkpoint under {reuse_run_dir}, "
                    f"got {initial_checkpoints}."
                )
            initial_payload = torch.load(
                initial_checkpoints[0],
                map_location="cpu",
                weights_only=False,
            )
            shared_initial_t = torch.sigmoid(
                initial_payload["model_state_dict"]["priors_logits"].detach().to(DTYPE)
            )
        else:
            _, shared_initial_t = add_perturbation(
                shared_true,
                float(args.perturbation_strength),
            )
        shared_initial = shared_initial_t.numpy()
        full_initial = shared_initial[full_to_group]
        if not np.all(full_initial == shared_initial[full_to_group]):
            raise ValueError("Repeated mechanisms did not receive identical initial parameters.")

        true_dem_path = checkpoint_dir / "true.dem"
        initial_dem_path = checkpoint_dir / "initial_perturbed.dem"
        write_dem(true_dem_path, true_dem)
        write_dem(initial_dem_path, update_dem(true_dem, full_initial))

        if reuse_run_dir is not None:
            source_dataset_path = reuse_run_dir / "fixed_synthetic_data.npz"
            with np.load(source_dataset_path) as saved:
                train_dets = torch.from_numpy(
                    saved["train_dets"].astype(np.float64, copy=False)
                )
                eval_dets = saved["eval_dets"].astype(bool, copy=False)
                eval_obs = saved["eval_obs"].astype(bool, copy=False)
            if len(train_dets) != int(args.train_size) or len(eval_dets) != int(args.eval_size):
                raise ValueError(
                    f"Reused dataset sizes are train={len(train_dets)}, eval={len(eval_dets)}; "
                    f"expected {args.train_size}/{args.eval_size}."
                )
            logger.write(f"# Reusing fixed train/eval datasets from: {source_dataset_path}")
        else:
            logger.write(
                f"# Sampling fixed independent datasets: train={args.train_size} "
                f"(seed={args.train_seed}), eval={args.eval_size} (seed={args.eval_seed})"
            )
            train_dets = sample_dem_dets(true_dem, int(args.train_size), int(args.train_seed))
            eval_dets, eval_obs = sample_dem_eval(
                true_dem,
                int(args.eval_size),
                int(args.eval_seed),
            )
        dataset_path = checkpoint_dir / "fixed_synthetic_data.npz"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dataset_path,
            train_dets=train_dets.numpy().astype(bool, copy=False),
            eval_dets=eval_dets,
            eval_obs=eval_obs,
            train_seed=np.asarray(int(args.train_seed)),
            eval_seed=np.asarray(int(args.eval_seed)),
        )

        sub_pcms, sub_dets, sub_errors, sub_full_masks = subsample_d5_pcms_from_circuit(
            ideal_circuit,
            true_dem,
            print_info=True,
            return_edge_masks=True,
        )
        path_dir = Path(args.path_dir)
        missing_trees = [
            str(path_dir / f"subsample_tree_{index}.json")
            for index in range(5)
            if not (path_dir / f"subsample_tree_{index}.json").exists()
        ]
        if missing_trees:
            raise FileNotFoundError(f"Missing contraction trees: {missing_trees}")

        def build_model(model_devices: list[str]) -> TiedGroupTN:
            return TiedGroupTN(
                d=7,
                r=10,
                sub_pcms=sub_pcms,
                sub_dets=sub_dets,
                sub_errors=sub_errors,
                init_priors=shared_initial_t,
                full_to_group=full_to_group,
                dev=model_devices[0],
                devices=model_devices,
                dtype=DTYPE,
                path_dir=str(path_dir),
                parallel_subs=len(model_devices) > 1,
                manual_sync_grads=False,
                sub_full_masks=sub_full_masks,
                stop_grad_partial=bool(args.stop_grad_partial),
                partial_only_grad=bool(args.partial_only_grad),
            )

        if args.parallel_mode == "data":
            models = [build_model([device]) for device in devices]
        else:
            models = [build_model(devices)]
        model = models[0]
        if tuple(model.priors_logits.shape) != (EXPECTED_SHARED_PARAMETERS,):
            raise ValueError(f"Unexpected shared-logit shape {tuple(model.priors_logits.shape)}.")
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
        start_epoch = 0
        resume_payload = None
        if args.resume_checkpoint:
            resume_path = Path(args.resume_checkpoint)
            resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
            for replica in models:
                replica.load_state_dict(resume_payload["model_state_dict"])
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(primary_dev)
            start_epoch = int(resume_payload["epoch"])
            logger.write(f"# Resuming model and Adam state from: {resume_path}")

        generator = torch.Generator().manual_seed(int(args.shuffle_seed))
        loader = DataLoader(
            TensorDataset(train_dets),
            batch_size=int(args.batch_size),
            shuffle=True,
            drop_last=False,
            num_workers=0,
            generator=generator,
        )
        if start_epoch > 0:
            logger.write(
                f"# Advancing deterministic DataLoader state through {start_epoch} completed epochs."
            )
            for _ in range(start_epoch):
                for _batch in loader:
                    pass

        initial_group_mre = mean_relative_error(shared_initial, shared_true)
        initial_full_mre = mean_relative_error(full_initial, full_true)
        start_shared = model.shared_probabilities()
        start_full = start_shared[full_to_group]
        start_group_mre = mean_relative_error(start_shared, shared_true)
        start_full_mre = mean_relative_error(start_full, full_true)
        params = {
            "run_name": run_name,
            "timestamp": timestamp,
            "code_task": "Google d7 X memory, SI1000 circuit",
            "distance": 7,
            "rounds": 10,
            "data_root": str(data_root),
            "decompose_errors": False,
            "flatten_loops": True,
            "full_dem_num_errors": int(true_dem.num_errors),
            "shared_parameter_count": int(len(shared_true)),
            "boundary_parameter_count": int(np.count_nonzero(~bulk_group_mask)),
            "bulk_parameter_count": int(np.count_nonzero(bulk_group_mask)),
            "bulk_copy_count": int(group_copy_counts[bulk_group_mask][0]),
            "parameterization": "exact r4 source groups tied across full r10 mechanisms",
            "max_abs_true_reconstruction_diff": tied["max_abs_reconstruction_diff"],
            "train_size": int(args.train_size),
            "train_seed": int(args.train_seed),
            "eval_size": int(args.eval_size),
            "eval_seed": int(args.eval_seed),
            "eval_source": "independent fixed sample from the same true r10 DEM",
            "epochs": int(args.epochs),
            "samples_per_epoch": int(args.train_size),
            "batch_size": int(args.batch_size),
            "mini_batch": int(args.mini_batch),
            "mini_batches": mini_batches,
            "parallel_mode": args.parallel_mode,
            "data_partition_mode": args.data_partition_mode,
            "optimizer_steps_per_epoch": int(len(loader)),
            "optimizer": "Adam",
            "lr": float(args.lr),
            "devices": devices,
            "dtype": "float64",
            "perturbation_strength": float(args.perturbation_strength),
            "perturbation_scope": "one draw per shared group, then expanded to full r10",
            "seed": int(args.seed),
            "shuffle_seed": int(args.shuffle_seed),
            "stop_grad_partial": bool(args.stop_grad_partial),
            "partial_only_grad": bool(args.partial_only_grad),
            "manual_sync_grads": False,
            "sequential_patch_backward": args.parallel_mode == "data",
            "decode_interval": int(args.decode_interval),
            "periodic_checkpoint_interval": int(args.periodic_checkpoint_interval),
            "keep_top_k": int(args.keep_top_k),
            "decoder": "TesseractSinterDecoder",
            "tesseract_det_beam": int(args.tesseract_det_beam),
            "tesseract_beam_climbing": bool(args.tesseract_beam_climbing),
            "tesseract_pqlimit": int(args.tesseract_pqlimit),
            "tesseract_num_det_orders": int(args.tesseract_num_det_orders),
            "tesseract_det_order_method": args.tesseract_det_order_method,
            "path_dir": str(path_dir),
            "true_dem_path": str(true_dem_path),
            "initial_perturbed_dem_path": str(initial_dem_path),
            "fixed_dataset_path": str(dataset_path),
            "log_path": str(log_path),
            "checkpoint_dir": str(checkpoint_dir),
            "partial_gradient_stats": getattr(model, "partial_gradient_stats", None),
            "reuse_run_dir": None if reuse_run_dir is None else str(reuse_run_dir),
            "resume_checkpoint": args.resume_checkpoint,
            "start_epoch": start_epoch,
        }

        logger.write(f"# {run_name} training")
        logger.write(f"# timestamp: {timestamp}")
        logger.write("# parameters:")
        for key, value in params.items():
            logger.write(f"#   {key}: {value}")
        logger.write(f"# Initial shared-parameter MRE: {initial_group_mre:.6f}")
        logger.write(f"# Initial full-r10 MRE: {initial_full_mre:.6f}")
        if resume_payload is not None:
            logger.write(f"# Resume shared-parameter MRE: {start_group_mre:.6f}")
            logger.write(f"# Resume full-r10 MRE: {start_full_mre:.6f}")

        initial_eval = None
        true_eval = None
        if not bool(args.skip_initial_decode):
            initial_eval = tesseract_ler(
                true_dem,
                full_initial,
                eval_dets,
                eval_obs,
                batch_size=int(args.tesseract_eval_batch_size),
                det_beam=int(args.tesseract_det_beam),
                beam_climbing=bool(args.tesseract_beam_climbing),
                pqlimit=int(args.tesseract_pqlimit),
                num_det_orders=int(args.tesseract_num_det_orders),
                det_order_method_name=args.tesseract_det_order_method,
                logger=logger,
                label="initial",
            )
            true_eval = tesseract_ler(
                true_dem,
                full_true,
                eval_dets,
                eval_obs,
                batch_size=int(args.tesseract_eval_batch_size),
                det_beam=int(args.tesseract_det_beam),
                beam_climbing=bool(args.tesseract_beam_climbing),
                pqlimit=int(args.tesseract_pqlimit),
                num_det_orders=int(args.tesseract_num_det_orders),
                det_order_method_name=args.tesseract_det_order_method,
                logger=logger,
                label="true",
            )
            logger.write(
                f"# Initial Tesseract LER: {initial_eval['ler']:.6f}, "
                f"wrong={initial_eval['wrong']}/{initial_eval['total']}"
            )
            logger.write(
                f"# True Tesseract LER: {true_eval['ler']:.6f}, "
                f"wrong={true_eval['wrong']}/{true_eval['total']}"
            )

        start_kind = "resume_start" if resume_payload is not None else "initial"
        _save_checkpoint(
            checkpoint_dir / f"{run_name}_{timestamp}_{start_kind}.pt",
            _checkpoint_payload(
                epoch=start_epoch,
                model=model,
                optimizer=optimizer,
                loss=None,
                group_mre=start_group_mre,
                full_mre=start_full_mre,
                tesseract_eval=(
                    resume_payload.get("tesseract_eval")
                    if resume_payload is not None
                    else initial_eval
                ),
                params=params,
                kind=start_kind,
            ),
        )

        best_checkpoints: list[tuple[float, int, Path]] = []
        best_group_mre = start_group_mre
        best_group_mre_epoch = start_epoch
        last_loss = None
        run_t0 = time.perf_counter()
        _reset_cuda_peak_stats(devices)

        for epoch in range(start_epoch + 1, int(args.epochs) + 1):
            epoch_t0 = time.perf_counter()
            epoch_weighted_loss = 0.0
            epoch_sample_count = 0
            for batch_index, (batch,) in enumerate(loader, start=1):
                batch_t0 = time.perf_counter()
                replica_stats = None
                if args.parallel_mode == "data":
                    last_loss, replica_stats = train_data_parallel_batch(
                        models,
                        optimizer,
                        batch,
                        mini_batches,
                        args.data_partition_mode,
                    )
                else:
                    last_loss = train_patch_parallel_batch(
                        model,
                        optimizer,
                        batch,
                        int(args.mini_batch),
                    )
                epoch_weighted_loss += last_loss * len(batch)
                epoch_sample_count += len(batch)
                shared_current = model.shared_probabilities()
                full_current = shared_current[full_to_group]
                group_mre = mean_relative_error(shared_current, shared_true)
                full_mre = mean_relative_error(full_current, full_true)
                gradient = model.priors_logits.grad.detach().cpu().numpy()
                grouped_metrics = grouped_training_metrics(
                    shared_current,
                    shared_true,
                    gradient,
                    bulk_group_mask,
                )
                if group_mre < best_group_mre:
                    best_group_mre = group_mre
                    best_group_mre_epoch = epoch
                logger.write(
                    f"epoch {epoch}/{args.epochs} batch {batch_index}/{len(loader)}: "
                    f"batch_size={len(batch)}, loss={last_loss:.6f}, "
                    f"group_MRE={group_mre:.6f}, full_MRE={full_mre:.6f}, "
                    f"boundary_MRE={grouped_metrics['boundary_mre']:.6f}, "
                    f"bulk_MRE={grouped_metrics['bulk_mre']:.6f}, "
                    f"boundary_grad_mean_abs={grouped_metrics['boundary_grad_mean_abs']:.6e}, "
                    f"bulk_grad_mean_abs={grouped_metrics['bulk_grad_mean_abs']:.6e}, "
                    f"lr={args.lr:.6g}, batch_sec={time.perf_counter() - batch_t0:.2f}, "
                    f"elapsed={(time.perf_counter() - run_t0) / 3600.0:.2f}h"
                )
                if replica_stats is not None:
                    assignment = {
                        str(row["device"]): {
                            "samples": int(row["samples"]),
                            "chunks": int(row["chunks"]),
                            "mini_batch": int(row["mini_batch"]),
                            "seconds": round(float(row["seconds"]), 2),
                        }
                        for row in replica_stats
                    }
                    logger.write(
                        f"epoch {epoch}/{args.epochs} batch {batch_index}/{len(loader)}: "
                        f"data_parallel_assignment={assignment}"
                    )
                if int(args.max_batches_per_epoch) > 0 and batch_index >= int(
                    args.max_batches_per_epoch
                ):
                    break

            epoch_sec = time.perf_counter() - epoch_t0
            peaks = _cuda_peak_gb(devices)
            average_loss = epoch_weighted_loss / epoch_sample_count
            logger.write(
                f"epoch {epoch}/{args.epochs}: train_sec={epoch_sec:.2f}, "
                f"samples={epoch_sample_count}, optimizer_steps={batch_index}, "
                f"average_loss={average_loss:.6f}, "
                f"peak_allocated_gb={peaks}"
            )

            eval_row = None
            if int(args.decode_interval) > 0 and epoch % int(args.decode_interval) == 0:
                eval_row = tesseract_ler(
                    true_dem,
                    full_current,
                    eval_dets,
                    eval_obs,
                    batch_size=int(args.tesseract_eval_batch_size),
                    det_beam=int(args.tesseract_det_beam),
                    beam_climbing=bool(args.tesseract_beam_climbing),
                    pqlimit=int(args.tesseract_pqlimit),
                    num_det_orders=int(args.tesseract_num_det_orders),
                    det_order_method_name=args.tesseract_det_order_method,
                    logger=logger,
                    label=f"epoch_{epoch}",
                )
                logger.write(
                    f"epoch {epoch}/{args.epochs}: Tesseract LER={eval_row['ler']:.6f}, "
                    f"wrong={eval_row['wrong']}/{eval_row['total']}, "
                    f"compile_sec={eval_row['compile_sec']:.3f}, "
                    f"decode_sec={eval_row['decode_sec']:.3f}"
                )
                candidate_path = checkpoint_dir / (
                    f"{run_name}_{timestamp}_epoch{epoch:04d}_"
                    f"ler{eval_row['ler']:.6f}_mre{group_mre:.6f}.pt"
                )
                _save_checkpoint(
                    candidate_path,
                    _checkpoint_payload(
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        loss=last_loss,
                        group_mre=group_mre,
                        full_mre=full_mre,
                        tesseract_eval=eval_row,
                        params=params,
                        kind="best_tesseract_candidate",
                    ),
                )
                best_checkpoints.append((eval_row["ler"], epoch, candidate_path))
                best_checkpoints.sort(key=lambda row: (row[0], row[1]))
                while len(best_checkpoints) > int(args.keep_top_k):
                    _, _, stale_path = best_checkpoints.pop()
                    stale_path.unlink(missing_ok=True)

            interval = int(args.periodic_checkpoint_interval)
            if interval > 0 and epoch % interval == 0:
                periodic_path = checkpoint_dir / (
                    f"{run_name}_{timestamp}_periodic_epoch{epoch:04d}_mre{group_mre:.6f}.pt"
                )
                _save_checkpoint(
                    periodic_path,
                    _checkpoint_payload(
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        loss=last_loss,
                        group_mre=group_mre,
                        full_mre=full_mre,
                        tesseract_eval=eval_row,
                        params=params,
                        kind="periodic",
                    ),
                )

        shared_final = model.shared_probabilities()
        full_final = shared_final[full_to_group]
        final_group_mre = mean_relative_error(shared_final, shared_true)
        final_full_mre = mean_relative_error(full_final, full_true)
        final_eval = None
        if bool(args.final_decode):
            final_eval = tesseract_ler(
                true_dem,
                full_final,
                eval_dets,
                eval_obs,
                batch_size=int(args.tesseract_eval_batch_size),
                det_beam=int(args.tesseract_det_beam),
                beam_climbing=bool(args.tesseract_beam_climbing),
                pqlimit=int(args.tesseract_pqlimit),
                num_det_orders=int(args.tesseract_num_det_orders),
                det_order_method_name=args.tesseract_det_order_method,
                logger=logger,
                label="final",
            )
        final_path = checkpoint_dir / (
            f"{run_name}_{timestamp}_final_epoch{args.epochs:04d}_mre{final_group_mre:.6f}.pt"
        )
        _save_checkpoint(
            final_path,
            _checkpoint_payload(
                epoch=int(args.epochs),
                model=model,
                optimizer=optimizer,
                loss=last_loss,
                group_mre=final_group_mre,
                full_mre=final_full_mre,
                tesseract_eval=final_eval,
                params=params,
                kind="final",
            ),
        )
        logger.write("# Final results:")
        logger.write(f"#   Initial shared-parameter MRE: {initial_group_mre:.6f}")
        logger.write(f"#   Final shared-parameter MRE: {final_group_mre:.6f}")
        logger.write(f"#   Final full-r10 MRE: {final_full_mre:.6f}")
        logger.write(f"#   Best shared-parameter MRE: {best_group_mre:.6f}")
        logger.write(f"#   Best shared-parameter MRE epoch: {best_group_mre_epoch}")
        logger.write(f"#   Final checkpoint: {final_path}")
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
        "--data_root",
        default=str(willow_root / "d7_at_q6_7/X/r10"),
    )
    parser.add_argument("--path_dir", default="path/d7r10_new")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4")
    parser.add_argument("--parallel_mode", choices=["patch", "data"], default="patch")
    parser.add_argument(
        "--data_partition_mode",
        choices=["equal", "dynamic"],
        default="equal",
    )
    parser.add_argument("--train_size", type=int, default=50000)
    parser.add_argument("--train_seed", type=int, default=75330)
    parser.add_argument("--eval_size", type=int, default=10000)
    parser.add_argument("--eval_seed", type=int, default=75331)
    parser.add_argument("--shuffle_seed", type=int, default=75332)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--mini_batch", type=int, default=18)
    parser.add_argument(
        "--mini_batches",
        default=None,
        help="Comma-separated per-device mini-batches for data parallel mode.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_batches_per_epoch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--perturbation_strength", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=75328)
    parser.add_argument("--stop_grad_partial", type=parse_bool, default=True)
    parser.add_argument("--partial_only_grad", type=parse_bool, default=False)
    parser.add_argument("--decode_interval", type=int, default=1)
    parser.add_argument("--periodic_checkpoint_interval", type=int, default=10)
    parser.add_argument("--keep_top_k", type=int, default=10)
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
    parser.add_argument("--skip_initial_decode", type=parse_bool, default=False)
    parser.add_argument("--final_decode", type=parse_bool, default=True)
    parser.add_argument("--reuse_run_dir", default=None)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--log_dir", default="log/sc_tn/simulation/d7r10_tied_fixed_pool")
    parser.add_argument("--checkpoint_dir", default="data/simulation/d7r10_tied_fixed_pool")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
