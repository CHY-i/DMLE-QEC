from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import opt_einsum as oe
import stim
import torch
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import MWPM_dem, PCM, TensorNetwork, get_error_rates


DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "PATCHDMLE_SYCAMORE_DATA",
        REPO_ROOT / "dataset/sycamore/sample_00/d3_at_q6_5/X/r05",
    )
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/google/d3r5_googleX_dmle"


class Tee:
    def __init__(self, stream, path: Path):
        self.stream = stream
        self.file = path.open("a", encoding="utf-8", buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a full-TN dMLE DEM to d3 Sycamore X-memory data."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--seed", type=int, default=75328)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=15_000)
    parser.add_argument("--minibatch", type=int, default=50)
    parser.add_argument("--decode-interval", type=int, default=10)
    parser.add_argument("--keep-top-k", type=int, default=5)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Checkpoint whose model and optimizer states are resumed.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=0,
        help="Save an additional checkpoint every N global epochs (0 disables).",
    )
    parser.add_argument("--perturbation-strength", type=float, default=1.0)
    parser.add_argument("--path-search-batch", type=int, default=10)
    parser.add_argument("--path-search-seconds", type=int, default=120)
    parser.add_argument("--expected-shots", type=int, default=75_000)
    parser.add_argument("--expected-detectors", type=int, default=56)
    parser.add_argument("--expected-errors", type=int, default=669)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_error_targets(dem: stim.DetectorErrorModel) -> list[tuple[str, ...]]:
    return [
        tuple(str(target) for target in instruction.targets_copy())
        for instruction in dem.flattened()
        if instruction.type == "error"
    ]


def dem_with_probabilities(
    base_dem: stim.DetectorErrorModel,
    probabilities: np.ndarray,
) -> stim.DetectorErrorModel:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (base_dem.num_errors,):
        raise ValueError(
            f"Expected {base_dem.num_errors} probabilities, got "
            f"{probabilities.shape}."
        )

    output = stim.DetectorErrorModel()
    error_index = 0
    for instruction in base_dem.flattened():
        if instruction.type == "error":
            output.append(
                stim.DemInstruction(
                    "error",
                    [float(probabilities[error_index])],
                    instruction.targets_copy(),
                )
            )
            error_index += 1
        else:
            output.append(instruction)
    if error_index != base_dem.num_errors:
        raise RuntimeError(
            f"Updated {error_index} errors, expected {base_dem.num_errors}."
        )
    if ordered_error_targets(output) != ordered_error_targets(base_dem):
        raise RuntimeError("Output DEM error targets changed during probability update.")
    return output


def atomic_write_dem(path: Path, dem: stim.DetectorErrorModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(str(dem), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_metric(path: Path, row: dict) -> None:
    fields = [
        "epoch",
        "nll",
        "eval_ler",
        "learning_rate",
        "grad_nonzero_ratio",
        "probability_min",
        "probability_mean",
        "probability_max",
        "epoch_seconds",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row[field] for field in fields})


def find_full_tn_path(
    model: TensorNetwork,
    pcm: np.ndarray,
    batch_size: int,
    max_time: int,
):
    import cotengra as ctg

    shapes = [(2, 2) for _ in model.xor_list]
    shapes.extend(
        [(2,) * int(pcm[:, j].sum()) for j in range(pcm.shape[1])]
    )
    shapes.extend([(batch_size, 2) for _ in range(pcm.shape[0])])
    optimizer = ctg.HyperOptimizer(
        methods=(
            "kahypar-agglom",
            "labels-agglom",
            "greedy-span",
            "greedy-span-max",
            "betweenness",
            "labelprop",
        ),
        minimize="size",
        max_repeats=64,
        max_time=max_time,
        parallel=min(32, os.cpu_count() or 1),
        progbar=True,
    )
    path, info = oe.contract_path(
        model.eq_str,
        *shapes,
        optimize=optimizer,
        shapes=True,
    )
    stats = {
        "flops_log10": math.log10(info.opt_cost),
        "space_log2": math.log2(info.largest_intermediate),
        "peak_gib_float64": (
            float(info.largest_intermediate) * 8 / (1024**3)
        ),
    }
    if stats["space_log2"] >= 30:
        raise RuntimeError(f"No safe contraction path found: {stats}")
    return path, stats


def load_inputs(args: argparse.Namespace):
    simple_path = args.data_root / "dems/simple.dem"
    correlation_path = args.data_root / "dems/correlations.dem"
    det_path = args.data_root / "detection_events.b8"
    obs_path = args.data_root / "obs_flips_actual.b8"
    ideal_path = args.data_root / "circuit_ideal.stim"
    for path in [
        simple_path,
        correlation_path,
        det_path,
        obs_path,
        ideal_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    dem = stim.DetectorErrorModel.from_file(str(simple_path))
    correlation_dem = stim.DetectorErrorModel.from_file(str(correlation_path))
    if dem.num_detectors != args.expected_detectors:
        raise ValueError(
            f"Expected {args.expected_detectors} detectors, got {dem.num_detectors}."
        )
    if dem.num_errors != args.expected_errors:
        raise ValueError(
            f"Expected {args.expected_errors} errors, got {dem.num_errors}."
        )
    if dem.num_observables != 1:
        raise ValueError(f"Expected one observable, got {dem.num_observables}.")
    if ordered_error_targets(dem) != ordered_error_targets(correlation_dem):
        raise ValueError("simple.dem and correlations.dem error targets differ.")

    dets_np = stim.read_shot_data_file(
        path=str(det_path),
        format="b8",
        num_detectors=dem.num_detectors,
        bit_packed=False,
    ).astype(np.float64)
    obs_np = stim.read_shot_data_file(
        path=str(obs_path),
        format="b8",
        num_detectors=1,
        bit_packed=False,
    ).reshape(-1)
    if len(dets_np) != args.expected_shots or len(obs_np) != args.expected_shots:
        raise ValueError(
            f"Expected {args.expected_shots} shots, got "
            f"{len(dets_np)} detector and {len(obs_np)} observable shots."
        )

    paths = {
        "simple_dem": simple_path,
        "correlation_dem": correlation_path,
        "detection_events": det_path,
        "observable_flips": obs_path,
        "ideal_circuit": ideal_path,
    }
    return dem, dets_np, obs_np, paths


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "d3r5_googleX_dmle"
    run_dir = args.output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "metadata.json").exists():
        raise FileExistsError(f"Run directory already contains a run: {run_dir}")

    tee = Tee(sys.stdout, run_dir / "train.log")
    sys.stdout = tee
    sys.stderr = tee

    print("# d3 full-TN dMLE training", flush=True)
    print(f"run_name: {run_name}", flush=True)
    print(f"timestamp: {timestamp}", flush=True)
    print(f"run_dir: {run_dir}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    dem, dets_np, obs_np, source_paths = load_inputs(args)
    pcm, _logical = PCM(dem)
    if pcm.shape != (args.expected_detectors, args.expected_errors):
        raise ValueError(
            f"Expected PCM {(args.expected_detectors, args.expected_errors)}, "
            f"got {pcm.shape}."
        )

    source_hashes = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    pcm_hash = hashlib.sha256(
        np.asarray(pcm, dtype=np.uint8).tobytes()
    ).hexdigest()
    params = {
        "run_name": run_name,
        "timestamp": timestamp,
        "method": "full_tensor_network_dMLE",
        "subsample": False,
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "run_dir": str(run_dir),
        "device": args.device,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "minibatch": args.minibatch,
        "decode_interval": args.decode_interval,
        "keep_top_k": args.keep_top_k,
        "resume_from": str(args.resume_from) if args.resume_from else None,
        "save_interval": args.save_interval,
        "perturbation_strength": args.perturbation_strength,
        "path_search_batch": args.path_search_batch,
        "path_search_seconds": args.path_search_seconds,
        "dtype": "float64",
        "shots": int(len(dets_np)),
        "num_detectors": int(dem.num_detectors),
        "num_errors": int(dem.num_errors),
        "num_observables": int(dem.num_observables),
        "pcm_sha256": pcm_hash,
        "source_paths": {
            name: str(path) for name, path in source_paths.items()
        },
        "source_sha256": source_hashes,
        "evaluation": (
            "transductive: MWPM evaluation uses the same 75000 experimental "
            "shots used by dMLE fitting"
        ),
    }
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps({"status": "initialized", "params": params}, indent=2),
        encoding="utf-8",
    )
    for key, value in params.items():
        print(f"{key}: {value}", flush=True)

    if args.validate_only:
        print("validation complete", flush=True)
        metadata_path.write_text(
            json.dumps({"status": "validated", "params": params}, indent=2),
            encoding="utf-8",
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this fit.")
    device = torch.device(args.device)
    base_probabilities = torch.from_numpy(get_error_rates(dem)).to(torch.float64)
    resume_payload = None
    if args.resume_from is not None:
        resume_payload = torch.load(
            args.resume_from,
            map_location="cpu",
            weights_only=False,
        )
        initial_logits = resume_payload["model_state_dict"]["priors_logits"].to(
            torch.float64
        )
        start_epoch = int(resume_payload["epoch"])
        loss_history = list(resume_payload.get("loss_history", []))
        print(
            f"resuming_from: {args.resume_from} at global_epoch={start_epoch}",
            flush=True,
        )
    else:
        perturbation = torch.rand(
            base_probabilities.shape,
            dtype=torch.float64,
            generator=generator,
        )
        directions = (
            2
            * torch.bernoulli(
                torch.full(
                    base_probabilities.shape,
                    0.5,
                    dtype=torch.float64,
                ),
                generator=generator,
            )
            - 1
        )
        initial_probabilities = base_probabilities * (
            1 + args.perturbation_strength * directions * perturbation
        )
        initial_probabilities = initial_probabilities.clamp(1e-10, 1 - 1e-10)
        initial_logits = torch.logit(initial_probabilities)
        start_epoch = 0
        loss_history = []
    initial_probabilities = torch.sigmoid(initial_logits.detach().cpu())

    model = TensorNetwork(
        pcm=pcm,
        priors_logits=initial_logits,
        dtype=torch.float64,
        dev=str(device),
    )
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"])
    path_file = args.output_root / "contraction_path_full_tn.pkl"
    path_stats_file = args.output_root / "contraction_path_full_tn.json"
    if not path_file.exists():
        path_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"generating contraction path: {path_file}", flush=True)
        contraction_path, path_stats = find_full_tn_path(
            model=model,
            pcm=pcm,
            batch_size=args.path_search_batch,
            max_time=args.path_search_seconds,
        )
        temporary = path_file.with_suffix(path_file.suffix + ".tmp")
        with temporary.open("wb") as file:
            pickle.dump(contraction_path, file)
        os.replace(temporary, path_file)
        path_stats_file.write_text(
            json.dumps(path_stats, indent=2),
            encoding="utf-8",
        )
        print(f"contraction path stats: {path_stats}", flush=True)
    model.load_path(str(path_file))

    dets = torch.from_numpy(dets_np)
    dataset = TensorDataset(dets)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
    matcher = MWPM_dem(dem, enable_correlations=True)
    initial_ler = float(
        matcher.logical_error_rate(
            dets_np.astype(np.uint8),
            obs_np,
            initial_probabilities.numpy(),
        )
    )
    params["contraction_path"] = str(path_file)
    params["initial_ler"] = initial_ler
    metadata_path.write_text(
        json.dumps({"status": "training", "params": params}, indent=2),
        encoding="utf-8",
    )
    print(f"initial_ler: {initial_ler:.10f}", flush=True)

    metrics_path = run_dir / "metrics.csv"
    best_checkpoints = []
    best_loss = float("inf")

    final_epoch = start_epoch
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        epoch_start = time.perf_counter()
        batch_losses = []
        for batch_index, (syndrome_batch,) in enumerate(dataloader, start=1):
            chunks = syndrome_batch.split(args.minibatch, dim=0)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for chunk in chunks:
                loss = model(chunk) / len(chunks)
                loss.backward()
                batch_loss += float(loss.detach().cpu())
            optimizer.step()
            batch_losses.append(batch_loss)
            print(
                f"epoch {epoch:03d} batch {batch_index}/{len(dataloader)} "
                f"nll={batch_loss:.10f}",
                flush=True,
            )

        average_loss = float(np.mean(batch_losses))
        loss_history.append(average_loss)
        probabilities = torch.sigmoid(model.priors_logits.detach().cpu())
        gradient = model.priors_logits.grad
        gradient_ratio = (
            float(torch.count_nonzero(gradient).item() / gradient.numel())
            if gradient is not None
            else 0.0
        )
        eval_ler = None
        if args.decode_interval and epoch % args.decode_interval == 0:
            eval_ler = float(
                matcher.logical_error_rate(
                    dets_np.astype(np.uint8),
                    obs_np,
                    probabilities.numpy(),
                )
            )

        epoch_seconds = time.perf_counter() - epoch_start
        print(
            f"epoch {epoch:03d}: nll={average_loss:.10f}, "
            f"ler={eval_ler if eval_ler is not None else 'not_evaluated'}, "
            f"grad_nonzero={gradient_ratio:.6f}, "
            f"time={epoch_seconds:.2f}s",
            flush=True,
        )
        append_metric(
            metrics_path,
            {
                "epoch": epoch,
                "nll": f"{average_loss:.12f}",
                "eval_ler": (
                    f"{eval_ler:.12f}" if eval_ler is not None else ""
                ),
                "learning_rate": f"{optimizer.param_groups[0]['lr']:.12e}",
                "grad_nonzero_ratio": f"{gradient_ratio:.12f}",
                "probability_min": f"{probabilities.min().item():.12e}",
                "probability_mean": f"{probabilities.mean().item():.12e}",
                "probability_max": f"{probabilities.max().item():.12e}",
                "epoch_seconds": f"{epoch_seconds:.6f}",
            },
        )

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "nll": average_loss,
            "eval_ler": eval_ler,
            "oer": probabilities,
            "loss_history": loss_history,
            "params": params,
        }
        atomic_torch_save(run_dir / "last.pt", payload)
        current_dem = dem_with_probabilities(dem, probabilities.numpy())

        if average_loss < best_loss:
            best_loss = average_loss
            atomic_torch_save(run_dir / "best_nll.pt", payload)
            atomic_write_dem(run_dir / "best_nll.dem", current_dem)

        if eval_ler is not None:
            checkpoint_path = run_dir / (
                f"{run_name}_{timestamp}_epoch{epoch}_ler{eval_ler:.6f}.pt"
            )
            atomic_torch_save(checkpoint_path, payload)
            best_checkpoints.append((eval_ler, epoch, checkpoint_path))
            best_checkpoints.sort(key=lambda item: (item[0], item[1]))
            while len(best_checkpoints) > args.keep_top_k:
                _, _, stale_path = best_checkpoints.pop()
                stale_path.unlink(missing_ok=True)

        if args.save_interval and epoch % args.save_interval == 0:
            periodic_path = run_dir / (
                f"{run_name}_{timestamp}_epoch{epoch}_periodic.pt"
            )
            atomic_torch_save(periodic_path, payload)

        final_epoch = epoch
        if (
            epoch >= 10
            and abs(loss_history[-1] - loss_history[-2])
            / max(abs(loss_history[-2]), 1e-30)
            < 1e-12
        ):
            print(f"loss converged at epoch {epoch}", flush=True)
            break

    final_probabilities = torch.sigmoid(model.priors_logits.detach().cpu())
    final_dem = dem_with_probabilities(dem, final_probabilities.numpy())
    final_ler = float(
        matcher.logical_error_rate(
            dets_np.astype(np.uint8),
            obs_np,
            final_probabilities.numpy(),
        )
    )
    final_payload = {
        "epoch": final_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "nll": loss_history[-1],
        "eval_ler": final_ler,
        "oer": final_probabilities,
        "loss_history": loss_history,
        "params": params,
    }
    final_checkpoint = run_dir / (
        f"{run_name}_{timestamp}_final_epoch{final_epoch}_ler{final_ler:.6f}.pt"
    )
    atomic_torch_save(final_checkpoint, final_payload)
    atomic_write_dem(run_dir / "final.dem", final_dem)

    metadata = {
        "status": "complete",
        "params": params,
        "result": {
            "final_epoch": final_epoch,
            "final_nll": loss_history[-1],
            "initial_ler": initial_ler,
            "final_ler": final_ler,
            "final_checkpoint": str(final_checkpoint),
            "final_dem": str(run_dir / "final.dem"),
            "final_dem_sha256": sha256_file(run_dir / "final.dem"),
            "ordered_error_targets_unchanged": (
                ordered_error_targets(final_dem) == ordered_error_targets(dem)
            ),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print("# training complete", flush=True)
    print(f"final_epoch: {final_epoch}", flush=True)
    print(f"final_nll: {loss_history[-1]:.12f}", flush=True)
    print(f"final_ler: {final_ler:.12f}", flush=True)
    print(f"final_dem: {run_dir / 'final.dem'}", flush=True)


if __name__ == "__main__":
    main()
