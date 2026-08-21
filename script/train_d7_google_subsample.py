"""
Train on Google experimental d=7 surface code data using GroupTN (5 subsampled d=5 TNs).

Set ``PATCHDMLE_WILLOW_DATA`` to the extracted Willow dataset root.

Assumptions:
  - use the requested Google round tag, e.g. r01 or r10
  - use the X basis by default
  - initial DEM is built from circuit_noisy_si1000.stim with decompose_errors=False
  - pre-generated Julia JSON contraction trees exist under path/d7r{r}_google{basis}_subsample
"""

from __future__ import annotations

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import stim
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WILLOW_ROOT = Path(
    os.environ.get("PATCHDMLE_WILLOW_DATA", REPO_ROOT / "dataset/willow_surface")
) / "d7_at_q6_7"
from src import (
    DetectorCoordinateProxy,
    GroupTN,
    MWPM_dem,
    PCM,
    broadcast_dem,
    get_error_rates,
    subsample_d5_pcms_from_circuit,
    update_dem,
)


def _load_google_round_root(root: str | Path):
    root = Path(root)
    ideal_circuit_path = root / "circuit_ideal.stim"
    noisy_circuit_path = root / "circuit_noisy_si1000.stim"
    cropped_dem_path = root / "initial_dem_non_decomposed.dem"
    detector_coords_path = root / "detector_coordinates.json"
    det_path = root / "detection_events.b8"
    obv_path = root / "obs_flips_actual.b8"

    if ideal_circuit_path.exists() and noisy_circuit_path.exists():
        ideal_circuit = stim.Circuit.from_file(str(ideal_circuit_path))
        noisy_circuit = stim.Circuit.from_file(str(noisy_circuit_path))
        initial_dem = noisy_circuit.detector_error_model(
            decompose_errors=False,
            flatten_loops=True,
        )
    elif cropped_dem_path.exists() and detector_coords_path.exists():
        with open(detector_coords_path) as f:
            detector_coordinates = {
                int(k): tuple(v)
                for k, v in json.load(f).items()
            }
        ideal_circuit = DetectorCoordinateProxy(detector_coordinates)
        noisy_circuit = ideal_circuit
        initial_dem = stim.DetectorErrorModel.from_file(str(cropped_dem_path))
    else:
        required_files = [det_path, obv_path]
        if not (ideal_circuit_path.exists() and noisy_circuit_path.exists()):
            required_files.extend([ideal_circuit_path, noisy_circuit_path])
        missing_files = [str(path) for path in required_files if not path.exists()]
        if cropped_dem_path.exists() or detector_coords_path.exists():
            missing_files.extend(
                str(path)
                for path in [cropped_dem_path, detector_coords_path]
                if not path.exists()
            )
        raise FileNotFoundError("Missing Google data files: " + ", ".join(sorted(set(missing_files))))

    num_detectors = initial_dem.num_detectors

    dets = stim.read_shot_data_file(
        path=str(det_path),
        format="b8",
        num_detectors=num_detectors,
        bit_packed=False,
    )
    obvs = stim.read_shot_data_file(
        path=str(obv_path),
        format="b8",
        num_detectors=1,
        bit_packed=False,
    ).flatten()

    return (
        ideal_circuit,
        noisy_circuit,
        torch.from_numpy(dets.astype(np.float64)),
        torch.from_numpy(obvs.astype(np.float64)),
        initial_dem,
        str(root),
    )


def _load_google_round(data_root: str, basis: str = "Z", round_tag: str = "r01"):
    return _load_google_round_root(Path(data_root) / basis.upper() / round_tag)


def _round_count(round_tag: str) -> int:
    if str(round_tag).startswith("r"):
        return int(str(round_tag)[1:])
    return int(round_tag)


def _parse_devices(dev: str = "cuda:2", devices=None):
    if devices is None:
        return [str(dev)]
    if isinstance(devices, str):
        parsed = [item.strip() for item in devices.split(",") if item.strip()]
        if parsed:
            return parsed
        return [str(dev)]
    return [str(item) for item in devices]


def _reset_cuda_peak_stats(devices):
    for device in devices:
        if str(device).startswith("cuda") and torch.cuda.is_available():
            with torch.cuda.device(device):
                torch.cuda.reset_peak_memory_stats(device)


def _cuda_peak_gb(devices):
    peaks = {}
    for device in devices:
        if str(device).startswith("cuda") and torch.cuda.is_available():
            peaks[str(device)] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return peaks


def _load_round_metadata(round_dir: str):
    metadata_path = Path(round_dir) / "metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path) as f:
        return json.load(f)


def _resolve_eval_round(
    train_round_dir: str,
    data_root: str,
    basis: str,
    round_tag: str,
    eval_data_root: str | None,
    eval_basis: str | None,
    eval_round_tag: str | None,
):
    metadata = _load_round_metadata(train_round_dir)
    if eval_round_tag is not None or eval_data_root is not None:
        resolved_basis = (eval_basis or basis).upper()
        resolved_round = eval_round_tag or round_tag
        resolved_root = Path(eval_data_root or data_root) / resolved_basis / resolved_round
        return resolved_root, resolved_basis, resolved_round, "explicit"

    source_root = metadata.get("source_root")
    source_round_tag = metadata.get("source_round_tag")
    if source_root:
        source_root = Path(source_root)
        return source_root, source_root.parent.name.upper(), source_root.name, "metadata.source_root"

    return Path(train_round_dir), basis.upper(), round_tag, "training_round"


def _baseline_dem_eval(dem_path: str, dets, obvs):
    dem = stim.DetectorErrorModel.from_file(dem_path)
    er = get_error_rates(dem)
    flag = False
    decoder = MWPM_dem(dem, enable_correlations=flag)
    ler = decoder.logical_error_rate(dets, obvs, er)
    return {
        "path": dem_path,
        "enable_correlations": flag,
        "ler": ler,
    }


def train(
    data_root=str(DEFAULT_WILLOW_ROOT),
    basis="X",
    round_tag="r01",
    run_name=None,
    timestamp=None,
    path_dir=None,
    epochs=1000,
    lr=0.001,
    batch_size=10000,
    mini_batch=10000,
    eval_size=5000,
    dev="cuda:2",
    devices=None,
    seed=75328,
    decode_interval=1,
    keep_top_k=5,
    enable_correlations=False,
    manual_sync_grads=False,
    baseline_dem_path=None,
    eval_data_root=None,
    eval_basis=None,
    eval_round_tag=None,
    broadcast_repeat_chunk=1,
    max_train_shots=None,
    resume_from=None,
    resume_optimizer=False,
    log_mini_batch=False,
    save_all_eval_checkpoints=False,
    save_checkpoint_interval=0,
    stop_grad_partial=False,
    partial_only_grad=True,
    sequential_patch_backward=False,
):
    d = 7
    r = _round_count(round_tag)
    dtype = torch.float64
    dtype_name = str(dtype).replace("torch.", "")
    basis = basis.upper()
    device_list = _parse_devices(dev=dev, devices=devices)
    primary_dev = device_list[0]
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    if run_name is None:
        run_name = f"d{d}r{r}_google{basis}_subsample"
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_size = int(batch_size)
    mini_batch = int(mini_batch)
    eval_size = int(eval_size)
    broadcast_repeat_chunk = int(broadcast_repeat_chunk)
    max_train_shots = None if max_train_shots is None else int(max_train_shots)
    save_checkpoint_interval = int(save_checkpoint_interval)

    if mini_batch <= 0:
        raise ValueError("mini_batch must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if mini_batch > batch_size:
        raise ValueError("mini_batch must be <= batch_size.")

    log_dir = f"log/sc_tn/google/{run_name}"
    ckpt_dir = f"data/google/{run_name}/{timestamp}"
    if path_dir is None:
        path_dir = f"path/d{d}r{r}_google{basis}_subsample"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    ideal_circuit, noisy_circuit, dets, obvs, initial_dem, round_dir = _load_google_round(
        data_root=data_root, basis=basis, round_tag=round_tag
    )
    metadata = _load_round_metadata(round_dir)
    eval_root, eval_basis_actual, eval_round_tag_actual, eval_source = _resolve_eval_round(
        train_round_dir=round_dir,
        data_root=data_root,
        basis=basis,
        round_tag=round_tag,
        eval_data_root=eval_data_root,
        eval_basis=eval_basis,
        eval_round_tag=eval_round_tag,
    )
    (
        _eval_ideal_circuit,
        _eval_noisy_circuit,
        eval_dets_all,
        eval_obvs_all,
        eval_initial_dem,
        eval_round_dir,
    ) = _load_google_round_root(eval_root)
    pcm, _ = PCM(initial_dem)
    print(f"Loaded Google experimental data from: {round_dir}")
    print(f"  shots: {dets.shape[0]:,}")
    print(f"  PCM shape: {pcm.shape}")
    print(f"  initial DEM error terms: {initial_dem.num_errors}")
    print(f"Loaded evaluation data from: {eval_round_dir}")
    print(f"  eval shots: {eval_dets_all.shape[0]:,}")
    print(f"  eval detectors: {eval_initial_dem.num_detectors}")
    if dets.shape[0] != eval_dets_all.shape[0]:
        raise ValueError(
            f"Training and evaluation shot counts differ: {dets.shape[0]} vs {eval_dets_all.shape[0]}"
        )
    if eval_size <= 0 or eval_size > eval_dets_all.shape[0]:
        raise ValueError(f"eval_size must be in [1, {eval_dets_all.shape[0]}].")

    er_ref = get_error_rates(initial_dem)
    init_er = torch.from_numpy(er_ref).to(torch.float64)
    print("  Initial priors come directly from circuit_noisy_si1000.stim")

    sub_pcms, sub_dets, sub_errors, sub_full_masks = subsample_d5_pcms_from_circuit(
        ideal_circuit,
        initial_dem,
        print_info=True,
        return_edge_masks=True,
    )

    if max_train_shots is not None:
        if max_train_shots <= 0 or max_train_shots > dets.shape[0]:
            raise ValueError(f"max_train_shots must be in [1, {dets.shape[0]}].")
        train_indices = torch.randperm(dets.shape[0], generator=generator)[:max_train_shots]
        train_dets = dets[train_indices]
    else:
        train_dets = dets

    eval_indices = torch.randperm(eval_dets_all.shape[0], generator=generator)[:eval_size]
    eval_dets = eval_dets_all[eval_indices]
    eval_obvs = eval_obvs_all[eval_indices]
    dataset = TensorDataset(train_dets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    optimizer_steps_per_epoch = len(dataloader)

    model = GroupTN(
        d=d,
        r=r,
        sub_pcms=sub_pcms,
        sub_dets=sub_dets,
        sub_errors=sub_errors,
        init_priors=init_er,
        dev=primary_dev,
        devices=device_list,
        dtype=dtype,
        use_tree=True,
        path_dir=path_dir,
        manual_sync_grads=manual_sync_grads,
        sub_full_masks=sub_full_masks,
        stop_grad_partial=stop_grad_partial,
        partial_only_grad=partial_only_grad,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    resume_epoch = 0
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=primary_dev, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        resume_epoch = int(checkpoint.get("epoch", 0))
        if resume_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for group in optimizer.param_groups:
                group["lr"] = lr
        print(f"Resumed model from: {resume_from}")
        print(f"  checkpoint epoch: {resume_epoch}")
        print(f"  checkpoint LER: {checkpoint.get('ler', 'unknown')}")
        print(f"  resume_optimizer: {resume_optimizer}")

    eval_round_count = _round_count(eval_round_tag_actual)
    train_round_count = _round_count(round_tag)
    broadcast_eval_dem = eval_initial_dem.num_detectors != initial_dem.num_detectors
    if broadcast_eval_dem and eval_round_count <= train_round_count:
        raise ValueError(
            f"Evaluation round {eval_round_tag_actual} must be longer than training round {round_tag} "
            "when DEM broadcast is required."
        )

    def _make_eval_dem(error_rates):
        short_dem = update_dem(initial_dem, error_rates)
        if broadcast_eval_dem:
            return broadcast_dem(
                short_dem,
                broadcast_time_layer=eval_round_count,
                repeat_chunk=broadcast_repeat_chunk,
            )
        return short_dem

    def _eval_ler(error_rates):
        eval_dem = _make_eval_dem(error_rates)
        decoder = MWPM_dem(eval_dem, enable_correlations=enable_correlations)
        return decoder.logical_error_rate(eval_dets, eval_obvs)

    initial_pymatching_ler = _eval_ler(er_ref)
    initial_ler = initial_pymatching_ler
    if baseline_dem_path is None:
        baseline_dem_path = (
            f"{eval_round_dir}/decoding_results/"
            "correlated_matching_decoder_with_rl_optimized_prior/error_model.dem"
        )
    baseline_eval = None
    if os.path.exists(baseline_dem_path):
        baseline_eval = _baseline_dem_eval(
            baseline_dem_path, eval_dets, eval_obvs
        )

    log_path = f"{log_dir}/{run_name}_{timestamp}.log"
    log_file = open(log_path, "w")
    log_file.write(f"# {run_name} training\n")
    log_file.write(f"# timestamp: {timestamp}\n")
    log_file.write("# parameters:\n")
    params = {
        "run_name": run_name,
        "timestamp": timestamp,
        "data_root": data_root,
        "basis": basis,
        "round_tag": round_tag,
        "metadata_source_root": metadata.get("source_root"),
        "d": d,
        "r": r,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "training_shots_total": int(dets.shape[0]),
        "max_train_shots": max_train_shots,
        "samples_per_epoch": int(train_dets.shape[0]),
        "mini_batch": mini_batch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "dev": primary_dev,
        "devices": device_list,
        "dtype": dtype_name,
        "seed": seed,
        "eval_size": eval_size,
        "eval_data_root": eval_data_root,
        "eval_basis": eval_basis_actual,
        "eval_round_tag": eval_round_tag_actual,
        "eval_round_dir": eval_round_dir,
        "eval_source": eval_source,
        "eval_detector_count": int(eval_initial_dem.num_detectors),
        "broadcast_eval_dem": broadcast_eval_dem,
        "broadcast_repeat_chunk": broadcast_repeat_chunk,
        "broadcast_time_layer": eval_round_count if broadcast_eval_dem else None,
        "decode_interval": decode_interval,
        "keep_top_k": keep_top_k,
        "decoder": "MWPM_dem",
        "enable_correlations": enable_correlations,
        "manual_sync_grads": manual_sync_grads,
        "sequential_patch_backward": sequential_patch_backward,
        "stop_grad_partial": stop_grad_partial,
        "partial_only_grad": partial_only_grad,
        "partial_gradient_stats": getattr(model, "partial_gradient_stats", None),
        "baseline_dem_path": baseline_dem_path,
        "resume_from": resume_from,
        "resume_epoch": resume_epoch,
        "resume_optimizer": resume_optimizer,
        "log_mini_batch": log_mini_batch,
        "save_all_eval_checkpoints": save_all_eval_checkpoints,
        "save_checkpoint_interval": save_checkpoint_interval,
        "initial_dem_source": metadata.get(
            "dem_source",
            "circuit_noisy_si1000.stim decompose_errors=False",
        ),
        "initial_dem_decompose_errors": False,
        "contraction_tree_source": "OMEinsumContractionOrders.TreeSA JSON",
        "log_dir": log_dir,
        "checkpoint_dir": ckpt_dir,
        "path_dir": path_dir,
    }
    for key, value in params.items():
        log_file.write(f"#   {key}: {value}\n")
    if baseline_eval is not None:
        log_file.write(
            f"# Baseline DEM LER (path={baseline_eval['path']}, enable_correlations={baseline_eval['enable_correlations']}, eval_size={eval_size}): {baseline_eval['ler']:.6f}\n"
        )
    log_file.write(
        f"# Initial DEM PyMatching-compatible LER (MWPM_dem, enable_correlations={enable_correlations}, eval_size={eval_size}, broadcast_eval_dem={broadcast_eval_dem}, eval_round_tag={eval_round_tag_actual}): {initial_pymatching_ler:.6f}\n"
    )
    log_file.write(
        f"# Initial Eval LER: {initial_ler:.6f}\n"
    )
    if resume_from:
        log_file.write(f"# Resumed checkpoint: {resume_from}\n")
    log_file.flush()

    loss_list = []
    best_checkpoints = []

    def _checkpoint_payload(epoch, loss, ler, oer, *, checkpoint_kind):
        return {
            "epoch": epoch,
            "resume_epoch": resume_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "ler": ler,
            "er_ref": er_ref,
            "oer": oer,
            "decoder": "MWPM_dem",
            "enable_correlations": enable_correlations,
            "checkpoint_kind": checkpoint_kind,
            "params": params,
        }

    run_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        losses = []
        epoch_start = time.perf_counter()
        for batch_idx, (syndrome_batch,) in enumerate(dataloader, start=1):
            batch_start = time.perf_counter()
            actual_batch_size = int(syndrome_batch.size(0))
            _reset_cuda_peak_stats(device_list)
            inputs = syndrome_batch.split(mini_batch, dim=0)
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for chunk_idx, input_chunk in enumerate(inputs, start=1):
                input_chunk = input_chunk.to(primary_dev, non_blocking=True).to(dtype)
                loss_scale = float(input_chunk.size(0)) / float(actual_batch_size)
                if sequential_patch_backward:
                    loss_k = model.sequential_sub_loss_and_grad(
                        input_chunk, loss_scale=loss_scale
                    )
                else:
                    loss_k = model.manual_sync_loss_and_grad(
                        input_chunk, loss_scale=loss_scale
                    )
                chunk_loss = loss_k.detach().item()
                step_loss += chunk_loss
                if log_mini_batch:
                    elapsed_hours = (time.perf_counter() - run_start) / 3600.0
                    mini_msg = (
                        f"epoch {epoch:4d} batch {batch_idx:3d}/{optimizer_steps_per_epoch} "
                        f"mini {chunk_idx:2d}/{len(inputs)}: "
                        f"loss={chunk_loss:.6f}, elapsed={elapsed_hours:.2f}h"
                    )
                    print(mini_msg, flush=True)
                    log_file.write(mini_msg + "\n")
                    log_file.flush()
            optimizer.step()
            for device in device_list:
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize(device)
            peak_gb = _cuda_peak_gb(device_list)
            losses.append(step_loss)
            batch_sec = time.perf_counter() - batch_start
            elapsed_hours = (time.perf_counter() - run_start) / 3600.0
            batch_msg = (
                f"epoch {epoch:4d} batch {batch_idx:3d}/{optimizer_steps_per_epoch}: "
                f"loss={step_loss:.6f}, batch_size={actual_batch_size}, "
                f"mini_batch={mini_batch}, batch_sec={batch_sec:.2f}, "
                f"peak_gb={peak_gb}, elapsed={elapsed_hours:.2f}h"
            )
            print(batch_msg, flush=True)
            log_file.write(batch_msg + "\n")
            log_file.flush()

        avg_loss = float(np.mean(losses))
        loss_list.append(avg_loss)
        oer = torch.sigmoid(model.priors_logits.detach().cpu())

        print(f"epoch {epoch:4d}: loss={avg_loss:.6f}")
        log_file.write(f"epoch {epoch}: loss={avg_loss:.6f}\n")
        log_file.flush()

        if decode_interval and epoch % decode_interval == 0:
            current_ler = _eval_ler(oer.numpy())
            separator = "-" * 45
            print(separator)
            print(
                f"epoch {epoch:4d}: MWPM_dem(eval_size={eval_size}, enable_correlations={enable_correlations}, broadcast_eval_dem={broadcast_eval_dem}, eval_round_tag={eval_round_tag_actual}) LER={current_ler:.6f}"
            )
            print(separator)
            log_file.write(separator + "\n")
            log_file.write(
                f"epoch {epoch}: MWPM_dem(eval_size={eval_size}, enable_correlations={enable_correlations}, broadcast_eval_dem={broadcast_eval_dem}, eval_round_tag={eval_round_tag_actual}) LER={current_ler:.6f}\n"
            )
            log_file.write(separator + "\n")
            log_file.flush()

            ckpt_path = (
                f"{ckpt_dir}/{run_name}_{timestamp}_"
                f"epoch{epoch}_ler{current_ler:.6f}.pt"
            )
            torch.save(
                _checkpoint_payload(
                    epoch,
                    avg_loss,
                    current_ler,
                    oer,
                    checkpoint_kind="eval",
                ),
                ckpt_path,
            )
            best_checkpoints.append((current_ler, epoch, ckpt_path))
            best_checkpoints.sort(key=lambda item: (item[0], item[1]))
            while len(best_checkpoints) > keep_top_k:
                _, _, stale_path = best_checkpoints.pop()
                if not save_all_eval_checkpoints and os.path.exists(stale_path):
                    os.remove(stale_path)

        saved_by_eval = bool(decode_interval and epoch % decode_interval == 0)
        if (
            save_checkpoint_interval > 0
            and epoch % save_checkpoint_interval == 0
            and not saved_by_eval
        ):
            periodic_ckpt_path = (
                f"{ckpt_dir}/{run_name}_{timestamp}_"
                f"epoch{epoch}_periodic.pt"
            )
            torch.save(
                _checkpoint_payload(
                    epoch,
                    avg_loss,
                    None,
                    oer,
                    checkpoint_kind="periodic",
                ),
                periodic_ckpt_path,
            )
            log_file.write(
                f"# Periodic checkpoint saved: epoch {epoch}, file={os.path.basename(periodic_ckpt_path)}\n"
            )
            log_file.flush()

    final_er = torch.sigmoid(model.priors_logits.detach().cpu())
    final_ler = _eval_ler(final_er.numpy())
    final_ckpt_path = (
        f"{ckpt_dir}/{run_name}_{timestamp}_"
        f"final_epoch{epochs}_ler{final_ler:.6f}.pt"
    )
    torch.save(
        _checkpoint_payload(
            epochs,
            loss_list[-1] if loss_list else None,
            final_ler,
            final_er,
            checkpoint_kind="final",
        ),
        final_ckpt_path,
    )
    print("\n=== Final Results ===")
    if baseline_eval is not None:
        print(
            f"Baseline DEM LER:               {baseline_eval['ler']:.6f} "
            f"(enable_correlations={baseline_eval['enable_correlations']})"
        )
    print(
        f"Initial DEM PyMatching-compatible LER: {initial_pymatching_ler:.6f} "
        f"(broadcast_eval_dem={broadcast_eval_dem}, eval_round_tag={eval_round_tag_actual})"
    )
    print(f"Initial Eval LER:   {initial_ler:.6f}")
    print(f"Final Eval LER:   {final_ler:.6f}")

    log_file.write("# Final results:\n")
    if baseline_eval is not None:
        log_file.write(
            f"#   Baseline DEM LER: {baseline_eval['ler']:.6f} "
            f"(enable_correlations={baseline_eval['enable_correlations']})\n"
        )
    log_file.write(
        f"#   Initial DEM PyMatching-compatible LER: {initial_pymatching_ler:.6f} "
        f"(broadcast_eval_dem={broadcast_eval_dem}, eval_round_tag={eval_round_tag_actual})\n"
    )
    log_file.write(f"#   Initial Eval LER: {initial_ler:.6f}\n")
    log_file.write(f"#   Final Eval LER: {final_ler:.6f}\n")
    log_file.write(f"#   Final checkpoint: {os.path.basename(final_ckpt_path)}\n")
    if best_checkpoints:
        log_file.write("# Best checkpoints by LER:\n")
        for ler, epoch, ckpt_path in best_checkpoints:
            log_file.write(f"#   epoch {epoch}: LER={ler:.6f}, file={os.path.basename(ckpt_path)}\n")
    log_file.close()

    return final_ler


if __name__ == "__main__":
    import fire

    fire.Fire(train)
