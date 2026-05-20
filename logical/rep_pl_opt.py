"""
Optimize DMLE (PlanarNet) priors on repetition-code *experimental* detection events.

Data and decoding match :mod:`logical.repetition_code_exp_data.data_prcess` /
``test_subsampling_ler_mwpm_dem.py``: ``build_stim_circuit(..., add_noise=False)`` + Stim ``m2d``.

The abstract code / init DEM use the same experimental circuit template as PyMatching
(``build_stim_circuit`` with default noise, ``ini_state=0``).

Run::

    conda run -n qec python logical/rep_pl_opt.py --round 2 --circuit-type phase_flip --reset 0
    conda run -n qec python logical/rep_pl_opt.py --hdf5 /path/to/dataset.hdf5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import stim
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if not hasattr(np, "bool8"):
    np.bool8 = bool

from logical.process_data import (  # noqa: E402
    DataLab,
    default_raw_data_dir,
    load_dets_obs_for_cycle_from_raw,
    measurements_to_detection_and_observables,
)
from logical.repetition_code_exp_data.circuits import build_stim_circuit  # noqa: E402
from src.decoder import MWPM_dem, Planar  # noqa: E402
from src.model import PlanarNet  # noqa: E402
from src.utils import get_error_rates, rep_dem  # noqa: E402

LER_REPORT_DECIMALS = 6


def _ler_round(x: float) -> float:
    if x != x:
        return x
    return round(float(x), LER_REPORT_DECIMALS)


def _dataset_number_from_hdf5_path(filepath: Path) -> int:
    num_part = ""
    for char in filepath.stem:
        if char.isdigit():
            num_part += char
        else:
            break
    if not num_part:
        raise ValueError(f"No dataset number in filename: {filepath}")
    return int(num_part)


def repetition_code_hdf5_glob_pattern(cycle: int, circuit_type: str, reset: int) -> str:
    ct = circuit_type.replace("_", " ")
    reset_str = "True" if bool(reset) else "False"
    return f"* - * Repetition code {ct} reset={reset_str} cycle={int(cycle)}.hdf5"


def resolve_repetition_hdf5_path(
    raw_dir: Path,
    cycle: int,
    circuit_type: str,
    reset: int,
) -> Path:
    pat = repetition_code_hdf5_glob_pattern(cycle, circuit_type, reset)
    matches = sorted(raw_dir.glob(pat))
    if not matches:
        raise FileNotFoundError(
            f"No HDF5 matching {pat!r} under {raw_dir}. "
            "Check --round, --circuit-type, --reset, and --raw-dir, or pass --hdf5."
        )
    if len(matches) > 1:
        raise FileNotFoundError(
            "Multiple HDF5 files match; narrow parameters or use --hdf5:\n  "
            + "\n  ".join(m.name for m in matches)
        )
    return matches[0]


def load_dets_obs_from_hdf5(hdf5_path: Path, *, raw_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Same conversion as subsampling / ``load_dets_obs_for_cycle_from_raw`` per file."""
    dl = DataLab(data_dir=raw_dir)
    dl.load_dataset(_dataset_number_from_hdf5_path(hdf5_path), noisy=False)
    if "shape" not in dl.parameters:
        dl.parameters["shape"] = [
            round(dl.data.shape[0] / dl.parameters["stats"]),
            dl.parameters["stats"],
        ]
    cycle = int(dl.parameters["cycle"])
    circuit_type = str(dl.parameters["circuit_type"])
    reset = bool(dl.parameters["reset_after_measure"])
    ini_state_idx, _, state = dl.get_data(dep_index=np.arange(dl.data.shape[1] - 2))
    qubits = np.arange(len(dl.parameters["qubits"]))
    ini_states_param = dl.parameters["ini_state"]

    dets_parts: list[np.ndarray] = []
    obs_parts: list[np.ndarray] = []
    for i in range(len(ini_state_idx)):
        idx = int(round(ini_state_idx[i]))
        dets, obs, _ = measurements_to_detection_and_observables(
            state[i],
            qubits=qubits,
            ini_state=ini_states_param[idx],
            cycle=cycle,
            circuit_type=circuit_type,
            reset=reset,
            add_noise=False,
        )
        dets_parts.append(np.asarray(dets, dtype=np.uint8))
        obs_parts.append(np.asarray(obs, dtype=np.uint8).reshape(-1))

    return np.vstack(dets_parts), np.concatenate(obs_parts)


def load_experimental_dets_obs(
    *,
    round: int,
    circuit_type: str,
    reset: int,
    raw_dir: Path | None,
    hdf5: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    raw = Path(raw_dir) if raw_dir is not None else default_raw_data_dir()
    if hdf5 is not None:
        hdf5 = hdf5.resolve()
        return load_dets_obs_from_hdf5(hdf5, raw_dir=hdf5.parent)
    dets, obs = load_dets_obs_for_cycle_from_raw(
        cycle=round,
        circuit_type=circuit_type,
        reset=reset,
        raw_dir=raw,
    )
    return dets, obs


def build_experiment_reference_circuit(
    *,
    distance: int,
    round: int,
    circuit_type: str,
    reset: int,
    add_noise: bool = True,
) -> stim.Circuit:
    """Circuit template used for init DEM (same qubit layout as subsampling decode)."""
    q_num = 2 * distance - 1
    qubits = np.arange(q_num)
    ini_state = np.zeros(distance, dtype=int)
    circuit_type_pd = circuit_type.replace("_", " ")
    return build_stim_circuit(
        qubits=qubits,
        ini_state=ini_state,
        cycle=round,
        circuit_type=circuit_type_pd,
        reset=bool(reset),
        add_noise=add_noise,
    )


def initialize_from_experiment_circuit(
    *,
    distance: int,
    round: int,
    circuit_type: str,
    reset: int,
    device: str,
    dtype: torch.dtype,
    perturb_init: bool,
) -> tuple[torch.Tensor, object, stim.DetectorErrorModel]:
    circuit = build_experiment_reference_circuit(
        distance=distance,
        round=round,
        circuit_type=circuit_type,
        reset=reset,
        add_noise=True,
    )
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    er = torch.tensor(get_error_rates(dem), dtype=dtype, device=device)

    if perturb_init:
        pertub = torch.rand_like(er)
        sign = 2 * torch.bernoulli(torch.ones(len(er), dtype=dtype, device=device) / 2) - 1
        init_er = er + sign * er * pertub
        init_er = init_er.clamp(1e-12, 1.0 - 1e-12)
    else:
        init_er = er.clone()

    rep = rep_dem(dem)
    return init_er, rep, dem


def decoding_benchmark(
    *,
    dets: np.ndarray,
    obs: np.ndarray,
    rep: object,
    dem: stim.DetectorErrorModel,
    error_rates: torch.Tensor,
    mini_batch: int,
    max_shots: int | None = None,
    device: str = "cpu",
) -> dict[str, float | int]:
    n = dets.shape[0]
    if max_shots is not None and n > max_shots:
        dets = dets[:max_shots]
        obs = obs[:max_shots]

    n = int(dets.shape[0])
    mb = max(1, int(mini_batch))
    obs = np.asarray(obs, dtype=np.uint8).reshape(-1)

    pl = Planar(rep, dev=device)
    mw = MWPM_dem(dem)
    er_np = error_rates.detach().cpu().numpy()

    total_eq_pl = 0.0
    total_eq_mw = 0.0
    for s in range(0, n, mb):
        e = min(s + mb, n)
        nb = e - s
        ler_b = pl.logical_error_rate(dets[s:e], obs[s:e], error_rates=error_rates)
        total_eq_pl += (1.0 - float(ler_b.detach().cpu().item())) * nb
        ler_m = mw.logical_error_rate(dets[s:e], obs[s:e], error_rates=er_np)
        total_eq_mw += (1.0 - float(ler_m)) * nb

    ler_pl = 1.0 - total_eq_pl / n if n else float("nan")
    ler_mw = 1.0 - total_eq_mw / n if n else float("nan")
    return {
        "shots": n,
        "planar_ler": _ler_round(ler_pl),
        "mwpm_dem_ler": _ler_round(ler_mw),
    }


def log_decode_block(
    log,
    *,
    tag: str,
    epoch: int,
    nll: float | None,
    decode: dict,
    priors: np.ndarray | None = None,
) -> None:
    lines = [f"[{tag}] epoch={epoch}"]
    if nll is not None:
        lines[0] += f" nll={nll:.6f}"
    lines.append(
        f"  decode shots={decode['shots']} "
        f"planar_ler={decode['planar_ler']} mwpm_dem_ler={decode['mwpm_dem_ler']}"
    )
    if priors is not None:
        lines.append(f"  priors: {priors!r}")
    block = "\n".join(lines) + "\n"
    log.write(block)
    log.flush()
    print(block, end="", flush=True)


def save_checkpoint(
    ckpt_dir: Path,
    *,
    epoch: int,
    planar: PlanarNet,
    dem: stim.DetectorErrorModel,
    decode: dict,
    nll: float | None,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"epoch_{epoch:04d}.pt"
    with torch.no_grad():
        priors = planar.get_priors().detach().cpu()
    torch.save(
        {
            "epoch": epoch,
            "planar_state_dict": planar.state_dict(),
            "priors": priors,
            "nll": nll,
            "decode": decode,
            "dem_text": str(dem),
        },
        path,
    )
    meta = ckpt_dir / f"epoch_{epoch:04d}_decode.json"
    meta.write_text(json.dumps(decode, indent=2) + "\n", encoding="utf-8")
    return path


def train_dmle_on_det(
    *,
    dets: np.ndarray,
    obs: np.ndarray,
    distance: int,
    round: int,
    circuit_type: str,
    reset: int,
    epochs: int,
    batch_size: int,
    mini_batch: int,
    lr: float,
    perturb_init: bool,
    log_path: Path,
    checkpoint_every: int,
    decode_mini_batch: int,
    max_decode_shots: int | None,
    device: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = log_path.parent / "checkpoints" / log_path.stem
    dtype = torch.float64

    if dets.shape[0] != obs.shape[0]:
        raise ValueError(f"dets rows {dets.shape[0]} != obs length {obs.shape[0]}")

    init_er_train, rep, dem = initialize_from_experiment_circuit(
        distance=distance,
        round=round,
        circuit_type=circuit_type,
        reset=reset,
        device=device,
        dtype=dtype,
        perturb_init=perturb_init,
    )
    baseline_er = torch.tensor(get_error_rates(dem), dtype=dtype, device=device)

    if dem.num_detectors != dets.shape[1]:
        raise ValueError(
            f"DEM num_detectors={dem.num_detectors} != dets width {dets.shape[1]}"
        )

    dets_t = torch.from_numpy(dets.astype(np.float64, copy=False))
    dataset = TensorDataset(dets_t)
    planar = PlanarNet(abstract_code=rep, init_priors=init_er_train, dev=device)
    optim = torch.optim.AdamW(planar.parameters(), lr=lr, weight_decay=0.01)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    nb = batch_size // mini_batch
    if batch_size % mini_batch != 0:
        raise ValueError(f"batch_size={batch_size} must be divisible by mini_batch={mini_batch}")

    decode_kw = dict(
        dets=dets,
        obs=obs,
        rep=rep,
        dem=dem,
        mini_batch=decode_mini_batch,
        max_shots=max_decode_shots,
        device=device,
    )

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(
            f"# Data: round={round} circuit_type={circuit_type} reset={reset}\n"
            f"# dets {dets.shape}, obs {obs.shape}\n"
            f"# Code distance={distance}, dem errors={dem.num_errors}\n"
            f"# Train: epochs={epochs} batch_size={batch_size} mini_batch={mini_batch} "
            f"lr={lr} device={device} perturb_init={int(perturb_init)}\n"
            f"# baseline_er (unperturbed circuit DEM):\n{repr(baseline_er.detach().cpu().numpy())}\n"
        )
        log.flush()

        print("[baseline] unperturbed circuit DEM + init rates …", flush=True)
        bm0 = decoding_benchmark(**decode_kw, error_rates=baseline_er)
        log_decode_block(
            log,
            tag="baseline",
            epoch=0,
            nll=None,
            decode=bm0,
            priors=baseline_er.detach().cpu().numpy(),
        )
        save_checkpoint(
            ckpt_dir,
            epoch=0,
            planar=planar,
            dem=dem,
            decode=bm0,
            nll=None,
        )

        for epoch in range(1, epochs + 1):
            planar.train()
            loss_list: list[float] = []

            for (det_batch,) in dataloader:
                det_batch = det_batch.reshape(nb, mini_batch, -1)
                optim.zero_grad(set_to_none=True)
                loss_item = 0.0
                for i in range(nb):
                    x = det_batch[i].to(device).to(dtype)
                    loss = planar.forward(x) / nb
                    loss.backward()
                    loss_item += float(loss.detach().cpu().item())
                optim.step()
                loss_list.append(loss_item)

            nll = float(np.mean(loss_list)) if loss_list else float("nan")
            msg = f"epoch:{epoch}, nll:{nll:.6f}"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()

            is_ckpt = checkpoint_every > 0 and epoch % checkpoint_every == 0
            if is_ckpt or epoch == epochs:
                with torch.no_grad():
                    priors = planar.get_priors()
                decode = decoding_benchmark(**decode_kw, error_rates=priors)
                log_decode_block(
                    log,
                    tag="checkpoint",
                    epoch=epoch,
                    nll=nll,
                    decode=decode,
                    priors=priors.detach().cpu().numpy(),
                )
                ckpt_path = save_checkpoint(
                    ckpt_dir,
                    epoch=epoch,
                    planar=planar,
                    dem=dem,
                    decode=decode,
                    nll=nll,
                )
                print(f"[checkpoint] saved {ckpt_path}", flush=True)

        with torch.no_grad():
            opt_er = planar.get_priors().detach().cpu().numpy()
        log.write("optimized error rates:\n")
        log.write(repr(opt_er) + "\n")
        log.flush()

        final_path = log_path.with_suffix(".pt")
        torch.save(
            {
                "distance": distance,
                "round": round,
                "circuit_type": circuit_type,
                "reset": reset,
                "baseline_er": baseline_er.detach().cpu(),
                "optimized_er": torch.tensor(opt_er),
                "planar_state_dict": planar.state_dict(),
                "dem_text": str(dem),
            },
            final_path,
        )
        print(f"[done] weights → {final_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=2, help="experiment cycle (HDF5 cycle=)")
    ap.add_argument("--distance", type=int, default=9)
    ap.add_argument("--circuit-type", type=str, default="phase_flip", choices=["phase_flip", "bit_flip"])
    ap.add_argument("--reset", type=int, default=0, choices=[0, 1])
    ap.add_argument("--raw-dir", type=str, default=None, help="HDF5 directory (default: public raw)")
    ap.add_argument("--hdf5", type=str, default=None, help="single HDF5 file instead of merging by cycle")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=10000)
    ap.add_argument("--mini-batch", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--perturb-init",
        action="store_true",
        help="randomly perturb init priors for PlanarNet (baseline decode still uses unperturbed DEM rates)",
    )
    ap.add_argument("--checkpoint-every", type=int, default=1, help="decode eval + save every N epochs (0=final only)")
    ap.add_argument("--decode-mini-batch", type=int, default=1000)
    ap.add_argument("--max-decode-shots", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    hdf5 = Path(args.hdf5).resolve() if args.hdf5 else None

    if hdf5 is None and raw_dir is None:
        print(f"[data] raw_dir={default_raw_data_dir()}", flush=True)
    elif hdf5 is not None:
        print(f"[data] HDF5: {hdf5}", flush=True)
    else:
        print(f"[data] raw_dir={raw_dir}", flush=True)

    dets, obs = load_experimental_dets_obs(
        round=args.round,
        circuit_type=args.circuit_type,
        reset=args.reset,
        raw_dir=raw_dir,
        hdf5=hdf5,
    )
    print(f"[data] dets {dets.shape}, obs {obs.shape}", flush=True)

    log_dir = REPO_ROOT / "log" / "rep" / "logical"
    log_path = log_dir / f"dmle_r{args.round:02d}_{args.circuit_type}_reset{args.reset}_e{args.epochs}.txt"

    train_dmle_on_det(
        dets=dets,
        obs=obs,
        distance=args.distance,
        round=args.round,
        circuit_type=args.circuit_type,
        reset=args.reset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mini_batch=args.mini_batch,
        lr=args.lr,
        perturb_init=args.perturb_init,
        log_path=log_path,
        checkpoint_every=args.checkpoint_every,
        decode_mini_batch=args.decode_mini_batch,
        max_decode_shots=args.max_decode_shots,
        device=device,
    )


if __name__ == "__main__":
    main()
