"""
Gate-level DMLE on Sycamore hardware surface-code data (``data/sycamore_old``).

Unlike :mod:`gate.tn_opt`, this script:

- loads ``circuit_noisy.stim`` from an experiment subdirectory as the circuit (no Stim
  ``Circuit.generated`` and **no** init perturbation);
- uses **all** shots in ``detection_events.b8`` as training data (no DEM sampling);
- defaults ``param_sharing`` to ``time_shared_dep12_M_DD`` (dep12 + ancilla pre-M
  X_ERROR + data DD after syndrome measurement);
- uses ``circuit.detector_error_model(decompose_errors=False)`` throughout;
- runs TN decoding (LER) at init and each checkpoint on ``obs_flips_actual.01``.

Directory layout (see ``data/sycamore_old/README.txt``)::

    surface_code_bX_d3_r03_center_5_7/circuit_noisy.stim
    surface_code_bX_d3_r03_center_5_7/detection_events.b8

Run from repo root::

    python gate/tn_syc_old.py --distance 3 --rounds 3 --center 5_7
    python gate/tn_syc_old.py --center 3_5 --param-sharing dep2 --device cuda:0
    python gate/tn_syc_old.py -d 5 -r 11 --center-row 5 --center-col 5 --device cuda:0
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import get_args

import numpy as np
import stim
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "sycamore_old"
LOG_DIR = REPO_ROOT / "gate" / "log" / "sur" / "syc_old"
CHECKPOINT_DIR = LOG_DIR / "checkpoint"
PATH_DIR = REPO_ROOT / "gate" / "path"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gate.tn_opt import relative_error_stats  # noqa: E402
from src import PCM, TensorNetwork, TensorNetworkDecoder  # noqa: E402
from src.g2dem import GateNoiseToDEM, ParamSharing  # noqa: E402


def parse_center(center: str) -> tuple[int, int]:
    parts = center.strip().split("_")
    if len(parts) != 2:
        raise ValueError(f"--center must be ROW_COL, got {center!r}")
    return int(parts[0]), int(parts[1])


def experiment_dir_name(
    *,
    basis: str,
    distance: int,
    rounds: int,
    center_row: int,
    center_col: int,
) -> str:
    return f"surface_code_b{basis}_d{distance}_r{rounds:02d}_center_{center_row}_{center_col}"


def resolve_experiment_dir(
    *,
    data_root: Path,
    basis: str,
    distance: int,
    rounds: int,
    center_row: int,
    center_col: int,
) -> Path:
    subdir = experiment_dir_name(
        basis=basis,
        distance=distance,
        rounds=rounds,
        center_row=center_row,
        center_col=center_col,
    )
    path = data_root / subdir
    if not path.is_dir():
        raise FileNotFoundError(f"experiment directory not found: {path}")
    return path


def load_noisy_circuit(experiment_dir: Path) -> stim.Circuit:
    circuit_path = experiment_dir / "circuit_noisy.stim"
    if not circuit_path.is_file():
        raise FileNotFoundError(f"missing circuit: {circuit_path}")
    return stim.Circuit.from_file(str(circuit_path))


def load_observable_flips(experiment_dir: Path) -> np.ndarray:
    """Load hardware logical observable flips (``obs_flips_actual.01``)."""
    obvs_path = experiment_dir / "obs_flips_actual.01"
    if not obvs_path.is_file():
        raise FileNotFoundError(f"missing observable flips: {obvs_path}")
    obvs = stim.read_shot_data_file(
        path=str(obvs_path),
        format="01",
        num_detectors=1,
        bit_packed=False,
    )
    return np.asarray(obvs, dtype=np.uint8).flatten()


def load_detection_events(
    experiment_dir: Path,
    circuit: stim.Circuit,
) -> np.ndarray:
    """Load all detection events from ``detection_events.b8``."""
    b8_path = experiment_dir / "detection_events.b8"
    if not b8_path.is_file():
        raise FileNotFoundError(f"missing detection events: {b8_path}")

    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    num_detectors = dem.num_detectors
    dets = stim.read_shot_data_file(
        path=str(b8_path),
        format="b8",
        num_detectors=num_detectors,
        bit_packed=False,
    )
    return np.asarray(dets, dtype=np.float64)


def align_dets_to_pcm(
    dets: np.ndarray,
    circuit: stim.Circuit,
    pcm: np.ndarray,
) -> np.ndarray:
    """Drop detector columns removed by :func:`src.utils.PCM` zero-row filtering."""
    if pcm.shape[0] == dets.shape[1]:
        return dets

    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    raw_pcm = np.zeros((dem.num_detectors, dem.num_errors), dtype=np.uint8)
    for i in range(dem.num_errors):
        inst = dem[i]
        if inst.type != "error":
            continue
        for t in inst.targets_copy():
            if str(t).startswith("D"):
                raw_pcm[int(str(t)[1:]), i] = 1

    non_zero_rows = np.where(raw_pcm.sum(axis=1) != 0)[0]
    if len(non_zero_rows) != pcm.shape[0]:
        raise RuntimeError(
            f"cannot align detection events: pcm rows={pcm.shape[0]}, "
            f"non-zero dem detector rows={len(non_zero_rows)}, dets cols={dets.shape[1]}"
        )
    print(
        f"[warning] PCM filtered zero-rows. Aligning detection events "
        f"{dets.shape[1]} -> {pcm.shape[0]} detectors.",
        flush=True,
    )
    return dets[:, non_zero_rows]


def default_log_filename(
    *,
    basis: str,
    distance: int,
    rounds: int,
    center_row: int,
    center_col: int,
    param_sharing: ParamSharing,
) -> str:
    return (
        f"syc_old_b{basis}_d{distance}r{rounds:02d}_"
        f"c{center_row}_{center_col}_{param_sharing}.txt"
    )


def resolve_log_path(
    user_log: str | None,
    *,
    basis: str,
    distance: int,
    rounds: int,
    center_row: int,
    center_col: int,
    param_sharing: ParamSharing,
) -> Path:
    name = (
        Path(user_log).name
        if user_log
        else default_log_filename(
            basis=basis,
            distance=distance,
            rounds=rounds,
            center_row=center_row,
            center_col=center_col,
            param_sharing=param_sharing,
        )
    )
    return LOG_DIR / name


def resolve_path_file(log_path: Path) -> Path:
    """``gate/path/{log_stem}_path.pkl`` — paired with the training log filename."""
    return PATH_DIR / f"{log_path.stem}_path.pkl"


def get_or_create_contraction_path(
    tn: TensorNetwork,
    path_file: Path,
    *,
    minibatch: int,
    max_time: int,
) -> None:
    """Load cached cotengra path, or search, save, and free the path object."""
    if path_file.is_file():
        print(f"  --> Path file '{path_file}' exists; loading cached contraction path.")
        tn.load_path(str(path_file))
        return

    print(
        f"  --> Path file '{path_file}' not found; searching with cotengra "
        f"(minibatch={minibatch}, max_time={max_time}s)..."
    )
    path_file.parent.mkdir(parents=True, exist_ok=True)

    path = tn.find_contraction_path(batch_size=minibatch, max_time=max_time)
    if path is None:
        raise RuntimeError(
            "Cotengra could not find a feasible TN contraction path "
            f"(space complexity >= 30 within max_time={max_time}s).\n"
            "Try increasing --ctg-max-time or decreasing --mini-batch."
        )

    tn.save_path(path, filename=str(path_file))
    print("  --> Contraction path found and saved.")
    del path
    gc.collect()
    tn.load_path(str(path_file))


def setup_models_with_cached_path(
    circuit: stim.Circuit,
    *,
    distance: int,
    rounds: int,
    device: str,
    dtype: torch.dtype,
    param_sharing: ParamSharing,
    mini_batch: int,
    ctg_max_time: int,
    path_file: Path,
) -> tuple[GateNoiseToDEM, TensorNetwork, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Like :func:`gate.tn_opt.setup_models`, but reuse ``path_file`` when present."""
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    pcm, l = PCM(dem)

    g2d = GateNoiseToDEM(
        circuit,
        learnable=True,
        param_mode="logit",
        param_sharing=param_sharing,
        dtype=dtype,
        device=device,
    )
    if g2d.num_dem != pcm.shape[1]:
        raise RuntimeError(f"g2d.num_dem={g2d.num_dem} != pcm.cols={pcm.shape[1]}")

    true_gate = g2d.gate_probs().detach().clone()
    with torch.no_grad():
        true_dem = g2d().detach().clone()
    stim_dem = g2d.stim_ref_dem_probs.detach().clone()

    tn = TensorNetwork(
        pcm=pcm,
        l=l[0] if l.shape[0] > 0 else None,
        dev=device,
        dtype=dtype,
        learn_priors=False,
    )
    get_or_create_contraction_path(
        tn,
        path_file,
        minibatch=mini_batch,
        max_time=ctg_max_time,
    )
    if tn.path is None:
        raise RuntimeError(f"failed to load contraction path from {path_file}")

    return g2d, tn, true_gate, true_dem, stim_dem, pcm


def setup_tn_decoder(
    pcm: np.ndarray,
    l: np.ndarray,
    tn_train: TensorNetwork,
    *,
    device: str,
    dtype: torch.dtype,
) -> TensorNetworkDecoder:
    """TN decoder sharing the training contraction path (see ``tn_opt_sycamore_old``)."""
    logical = l.flatten() if l.size > 0 else None
    tn_dec = TensorNetwork(
        pcm=pcm,
        l=logical,
        dev=device,
        dtype=dtype,
        decoding=True,
        learn_priors=False,
    )
    if tn_train.path is None:
        raise RuntimeError("training TensorNetwork has no contraction path")
    tn_dec.path = tn_train.path
    return TensorNetworkDecoder(model=tn_dec, dev=device)


def compute_tn_logical_error_rate(
    decoder: TensorNetworkDecoder,
    dets: np.ndarray,
    obvs: np.ndarray,
    dem_probs: torch.Tensor,
    *,
    device: str,
    decode_batch_size: int,
) -> float:
    """Logical error rate on all shots using current DEM error probabilities."""
    n = int(dets.shape[0])
    if n == 0:
        return float("nan")
    if obvs.shape[0] != n:
        raise ValueError(f"dets shots={n} != obvs shots={obvs.shape[0]}")

    dem_probs = dem_probs.detach().to(device=device, dtype=torch.float64)
    if n <= decode_batch_size:
        return float(
            decoder.logical_error_rate(
                torch.from_numpy(dets).to(device=device, dtype=torch.float64),
                torch.from_numpy(obvs).to(device=device),
                dem_probs,
            )
        )

    total_errors = 0.0
    for start in range(0, n, decode_batch_size):
        end = min(start + decode_batch_size, n)
        batch_size = end - start
        batch_ler = decoder.logical_error_rate(
            torch.from_numpy(dets[start:end]).to(device=device, dtype=torch.float64),
            torch.from_numpy(obvs[start:end]).to(device=device),
            dem_probs,
        )
        total_errors += float(batch_ler) * batch_size
    return total_errors / n


def log_checkpoint(
    log,
    *,
    tag: str,
    epoch: int,
    nll: float | None,
    g2d: GateNoiseToDEM,
    true_gate: torch.Tensor,
    true_dem: torch.Tensor,
    stim_dem: torch.Tensor,
    ler_tn: float | None = None,
) -> None:
    """Log summary stats only (no full gate/DEM probability vectors)."""
    with torch.no_grad():
        gate_p = g2d.gate_probs().detach().cpu()
        dem_p = g2d().detach().cpu()
    gate_err = relative_error_stats(gate_p, true_gate.cpu())
    dem_err = relative_error_stats(dem_p, true_dem.cpu())
    stim_err = relative_error_stats(dem_p, stim_dem.cpu())

    line = f"[{tag}] epoch={epoch}"
    if nll is not None:
        line += f" nll={nll:.6f}"
    if ler_tn is not None:
        line += f" ler_tn={ler_tn:.8f}"
    lines = [
        line,
        (
            f"  gate vs true_gate: max_rel={gate_err['max_rel']:.3e} "
            f"mean_rel={gate_err['mean_rel']:.3e}"
        ),
        (
            f"  dem vs true_dem (g2d@ref gate): max_rel={dem_err['max_rel']:.3e} "
            f"mean_rel={dem_err['mean_rel']:.3e}"
        ),
        (
            f"  dem vs stim_dem (circuit, decompose=False): "
            f"max_rel={stim_err['max_rel']:.3e} mean_rel={stim_err['mean_rel']:.3e}"
        ),
    ]
    block = "\n".join(lines) + "\n"
    log.write(block)
    log.flush()
    print(block, end="", flush=True)


def train(
    *,
    experiment_dir: Path,
    basis: str,
    distance: int,
    rounds: int,
    center_row: int,
    center_col: int,
    epochs: int,
    batch_size: int,
    mini_batch: int,
    ctg_max_time: int,
    lr: float,
    device: str,
    log_path: Path,
    checkpoint_every: int,
    param_sharing: ParamSharing,
    seed: int,
) -> None:
    dtype = torch.float64
    torch.manual_seed(seed)

    circuit = load_noisy_circuit(experiment_dir)
    dets = load_detection_events(experiment_dir, circuit)
    obvs = load_observable_flips(experiment_dir)
    num_shots = dets.shape[0]
    if obvs.shape[0] != num_shots:
        raise ValueError(
            f"shot count mismatch: detection_events={num_shots} obs_flips={obvs.shape[0]}"
        )

    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    pcm_full, l = PCM(dem)

    path_file = resolve_path_file(log_path)
    g2d, tn, true_gate, true_dem, stim_dem, pcm = setup_models_with_cached_path(
        circuit,
        distance=distance,
        rounds=rounds,
        device=device,
        dtype=dtype,
        param_sharing=param_sharing,
        mini_batch=mini_batch,
        ctg_max_time=ctg_max_time,
        path_file=path_file,
    )

    dets = align_dets_to_pcm(dets, circuit, pcm)
    if pcm.shape != pcm_full.shape:
        raise RuntimeError(
            f"PCM shape mismatch: setup_models {pcm.shape} vs PCM(dem) {pcm_full.shape}"
        )

    decoder = setup_tn_decoder(
        pcm_full, l, tn, device=device, dtype=dtype
    )
    print(
        f"[data] experiment={experiment_dir.name} shots={num_shots:,} "
        f"detectors={dets.shape[1]} learnable={g2d.num_slots} "
        f"dem_errors={g2d.num_dem} decompose_errors=False",
        flush=True,
    )

    optim = torch.optim.AdamW(g2d.parameters(), lr=lr, weight_decay=0.01)

    dets_t = torch.from_numpy(dets)
    dataset = TensorDataset(dets_t)
    if batch_size % mini_batch != 0:
        raise ValueError(f"batch_size={batch_size} must divide mini_batch={mini_batch}")
    nb = batch_size // mini_batch
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(
            f"# sycamore_old {experiment_dir.name}\n"
            f"# circuit={experiment_dir / 'circuit_noisy.stim'}\n"
            f"# detection_events={experiment_dir / 'detection_events.b8'}\n"
            f"# observables={experiment_dir / 'obs_flips_actual.01'}\n"
            f"# param_sharing={param_sharing} gate_params={g2d.num_slots} "
            f"elementary_sites={g2d.num_elementary}\n"
            f"# shots={num_shots} epochs={epochs} batch_size={batch_size} "
            f"mini_batch={mini_batch} decode_batch={mini_batch} lr={lr}\n"
            f"# contraction_path={path_file}\n"
            f"# dem_errors={g2d.num_dem} perturb_init=0 decompose_errors=False\n"
        )
        with torch.no_grad():
            stim_gap = relative_error_stats(true_dem.cpu(), stim_dem.cpu())
        log.write(
            f"# at init: g2d(init gate) vs stim_dem: max_rel={stim_gap['max_rel']:.3e} "
            f"mean_rel={stim_gap['mean_rel']:.3e}\n"
        )
        log.flush()

        with torch.no_grad():
            probe = torch.from_numpy(dets[:mini_batch]).to(device=device, dtype=dtype)
            init_nll = float(tn(probe, priors=g2d()).item())
            init_dem = g2d()
            init_ler = compute_tn_logical_error_rate(
                decoder,
                dets,
                obvs,
                init_dem,
                device=device,
                decode_batch_size=mini_batch,
            )
        print(f"[init] nll~{init_nll:.4f} ler_tn={init_ler:.8f}", flush=True)
        log_checkpoint(
            log,
            tag="init",
            epoch=0,
            nll=init_nll,
            g2d=g2d,
            true_gate=true_gate,
            true_dem=true_dem,
            stim_dem=stim_dem,
            ler_tn=init_ler,
        )

        for epoch in range(1, epochs + 1):
            g2d.train()
            tn.train()
            losses: list[float] = []

            for (det_batch,) in dataloader:
                det_batch = det_batch.reshape(nb, mini_batch, -1)
                optim.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for i in range(nb):
                    x = det_batch[i].to(device=device, dtype=dtype)
                    dem_p = g2d()
                    loss = tn(x, priors=dem_p) / nb
                    loss.backward()
                    loss_sum += float(loss.detach().cpu().item())
                optim.step()
                losses.append(loss_sum)

            nll = float(np.mean(losses))
            is_ckpt = checkpoint_every > 0 and epoch % checkpoint_every == 0
            if is_ckpt or epoch == epochs:
                with torch.no_grad():
                    ckpt_dem = g2d()
                    ler_tn = compute_tn_logical_error_rate(
                        decoder,
                        dets,
                        obvs,
                        ckpt_dem,
                        device=device,
                        decode_batch_size=mini_batch,
                    )
                log_checkpoint(
                    log,
                    tag="checkpoint",
                    epoch=epoch,
                    nll=nll,
                    g2d=g2d,
                    true_gate=true_gate,
                    true_dem=true_dem,
                    stim_dem=stim_dem,
                    ler_tn=ler_tn,
                )

        with torch.no_grad():
            final_dem = g2d().detach().cpu()
            final_gate = g2d.gate_probs().detach().cpu()
        save_path = CHECKPOINT_DIR / f"{log_path.stem}.pt"
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "experiment_dir": str(experiment_dir),
                "basis": basis,
                "distance": distance,
                "rounds": rounds,
                "center_row": center_row,
                "center_col": center_col,
                "num_shots": num_shots,
                "true_gate_probs": true_gate.detach().cpu(),
                "true_dem_probs": true_dem.detach().cpu(),
                "stim_dem_probs": stim_dem.detach().cpu(),
                "gate_probs": final_gate,
                "dem_probs": final_dem,
                "param_sharing": param_sharing,
                "slot_keys": g2d.slot_keys,
            },
            save_path,
        )
        log.write(f"# saved {save_path}\n")
        print(f"[done] checkpoint → {save_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Gate-level DMLE on sycamore_old hardware surface-code data."
    )
    ap.add_argument("--distance", "-d", type=int, default=3)
    ap.add_argument("--rounds", "-r", type=int, default=3)
    ap.add_argument(
        "--center",
        type=str,
        default="5_7",
        help="center data qubit as ROW_COL (default: 5_7)",
    )
    ap.add_argument("--center-row", type=int, default=None)
    ap.add_argument("--center-col", type=int, default=None)
    ap.add_argument("--basis", type=str, default="X", choices=("X", "Z"))
    ap.add_argument(
        "--data-root",
        type=str,
        default=str(DATA_ROOT),
        help="root directory containing experiment subfolders",
    )
    ap.add_argument("--seed", type=int, default=75328)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=10_000)
    ap.add_argument("--mini-batch", type=int, default=1_000)
    ap.add_argument("--ctg-max-time", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--log", type=str, default=None)
    ap.add_argument(
        "--param-sharing",
        type=str,
        default="time_shared_dep12_M_DD",
        choices=get_args(ParamSharing),
        help=(
            "gate noise tying (default time_shared_dep12_M_DD: dep12 + ancilla pre-M "
            "X_ERROR + data DD after syndrome M)"
        ),
    )
    args = ap.parse_args()
    param_sharing: ParamSharing = args.param_sharing  # type: ignore[assignment]

    if args.center_row is not None and args.center_col is not None:
        center_row, center_col = args.center_row, args.center_col
    else:
        center_row, center_col = parse_center(args.center)

    experiment_dir = resolve_experiment_dir(
        data_root=Path(args.data_root),
        basis=args.basis,
        distance=args.distance,
        rounds=args.rounds,
        center_row=center_row,
        center_col=center_col,
    )

    log_path = resolve_log_path(
        args.log,
        basis=args.basis,
        distance=args.distance,
        rounds=args.rounds,
        center_row=center_row,
        center_col=center_col,
        param_sharing=param_sharing,
    )

    train(
        experiment_dir=experiment_dir,
        basis=args.basis,
        distance=args.distance,
        rounds=args.rounds,
        center_row=center_row,
        center_col=center_col,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mini_batch=args.mini_batch,
        ctg_max_time=args.ctg_max_time,
        lr=args.lr,
        device=args.device,
        log_path=log_path,
        checkpoint_every=args.checkpoint_every,
        param_sharing=param_sharing,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
