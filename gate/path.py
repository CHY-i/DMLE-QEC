"""
Find and cache TN contraction paths for sycamore_old decoding (cotengra).

Saves ``gate/path/syc_old_b{basis}_d{distance}r{rounds:02d}_c{center}_{param_sharing}_path.pkl``,
using the same naming as :mod:`gate.tn_syc_old` / :mod:`gate.decoding`.

Run from repo root (``qec`` environment)::

    conda activate qec
    python gate/path.py --basis X --rounds 5 7 9 11 --mini-batch 10000
    python gate/path.py --basis Z -r 11 --mini-batch 10000
    python gate/path.py --basis X --rounds 13-25 --mini-batch 10000
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import stim
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "sycamore_old"
LOG_DIR = REPO_ROOT / "gate" / "log" / "sur" / "syc_old"
PATH_DIR = REPO_ROOT / "gate" / "path"

DEFAULT_DISTANCE = 3
DEFAULT_CENTER = (3, 5)
DEFAULT_PARAM_SHARING = "time_shared_dep12_M_DD"
DEFAULT_MINI_BATCH = 10_000
DEFAULT_CTG_MAX_TIME = 60
DEFAULT_DEVICE = "cpu"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gate.tn_syc_old import (  # noqa: E402
    get_or_create_contraction_path,
    load_noisy_circuit,
    resolve_experiment_dir,
    resolve_log_path,
    resolve_path_file,
)
from src import PCM, TensorNetwork  # noqa: E402


def log_path_for_round(
    *,
    basis: str,
    distance: int,
    rounds: int,
    center_row: int,
    center_col: int,
    param_sharing: str,
) -> Path:
    return resolve_log_path(
        None,
        basis=basis,
        distance=distance,
        rounds=rounds,
        center_row=center_row,
        center_col=center_col,
        param_sharing=param_sharing,  # type: ignore[arg-type]
    )


def parse_round_list(text: str) -> list[int]:
    """Accept ``5``, ``5 7 11``, or ``13-25``."""
    text = text.strip()
    if "-" in text and " " not in text:
        lo, hi = text.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.replace(",", " ").split()]


def find_path_for_round(
    *,
    basis: str,
    rounds: int,
    distance: int,
    center_row: int,
    center_col: int,
    param_sharing: str,
    data_root: Path,
    device: str,
    mini_batch: int,
    ctg_max_time: int,
) -> Path:
    log_path = log_path_for_round(
        basis=basis,
        distance=distance,
        rounds=rounds,
        center_row=center_row,
        center_col=center_col,
        param_sharing=param_sharing,
    )
    path_file = resolve_path_file(log_path)

    if path_file.is_file():
        print(f"[paths] r{rounds:02d} basis={basis} exists -> {path_file}", flush=True)
        return path_file

    experiment_dir = resolve_experiment_dir(
        data_root=data_root,
        basis=basis,
        distance=distance,
        rounds=rounds,
        center_row=center_row,
        center_col=center_col,
    )
    circuit = load_noisy_circuit(experiment_dir)
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    pcm, l = PCM(dem)
    logical = l.flatten() if l.size > 0 else None

    dtype = torch.float64
    tn = TensorNetwork(
        pcm=pcm,
        l=logical,
        dev=device,
        dtype=dtype,
        learn_priors=False,
    )
    print(
        f"[paths] searching r{rounds:02d} basis={basis} -> {path_file.name} "
        f"(minibatch={mini_batch}, max_time={ctg_max_time}s)",
        flush=True,
    )
    get_or_create_contraction_path(
        tn,
        path_file,
        minibatch=mini_batch,
        max_time=ctg_max_time,
    )
    del tn
    gc.collect()
    print(f"[paths] saved {path_file}", flush=True)
    return path_file


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Search/cache cotengra TN contraction paths for sycamore_old rounds."
    )
    ap.add_argument("--basis", "-b", type=str, required=True, choices=("X", "Z"))
    ap.add_argument(
        "--rounds",
        "-r",
        type=str,
        required=True,
        help="code round(s): single (11), list (5 7 11), or range (13-25)",
    )
    ap.add_argument(
        "--mini-batch",
        type=int,
        default=DEFAULT_MINI_BATCH,
        help=f"cotengra minibatch (default: {DEFAULT_MINI_BATCH})",
    )
    args = ap.parse_args()

    center_row, center_col = DEFAULT_CENTER
    round_list = parse_round_list(args.rounds)

    print(
        f"[config] basis={args.basis} rounds={round_list} "
        f"center={center_row}_{center_col} d={DEFAULT_DISTANCE} "
        f"sharing={DEFAULT_PARAM_SHARING} device={DEFAULT_DEVICE} "
        f"mini_batch={args.mini_batch} ctg_max_time={DEFAULT_CTG_MAX_TIME}",
        flush=True,
    )

    for r in sorted(round_list):
        find_path_for_round(
            basis=args.basis,
            rounds=r,
            distance=DEFAULT_DISTANCE,
            center_row=center_row,
            center_col=center_col,
            param_sharing=DEFAULT_PARAM_SHARING,
            data_root=DATA_ROOT,
            device=DEFAULT_DEVICE,
            mini_batch=args.mini_batch,
            ctg_max_time=DEFAULT_CTG_MAX_TIME,
        )

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
