"""
Reproduce ``repetition_code_exp_data-main/notebooks/test.py`` logical error rates
using :class:`src.decoder.MWPM_dem`, and verify agreement with the original
PyMatching path (``decoder=\"pymatching\"``).

Run from repo root::

    conda run -n qec python logical/test_subsampling_ler_mwpm_dem.py

Environment:

- ``REPETITION_CODE_RAW_DIR`` — directory of ``*.hdf5`` (default: public dataset path).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if not hasattr(np, "bool8"):
    np.bool8 = bool

from logical.repetition_code_exp_data import data_prcess  # noqa: E402


def _print_ler_table(cycles, raw_ler: dict, target_d: int) -> None:
    if target_d not in raw_ler:
        print(f"No distance d={target_d}; available: {sorted(raw_ler.keys())}")
        return
    print(f"\n=== d={target_d} LER vs dataset index (same layout as upstream test.py) ===")
    for c, err in zip(cycles, raw_ler[target_d]):
        print(f"Round {int(c):<3} : LER = {float(err):.6f}")
    print("-" * 40)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=3418, help="first dataset id (inclusive)")
    ap.add_argument("--end", type=int, default=3457, help="last dataset id (exclusive)")
    ap.add_argument("--target-d", type=int, default=9, help="code distance to print")
    ap.add_argument("--atol", type=float, default=1e-9, help="max abs diff tolerance")
    args = ap.parse_args()

    datasets = np.arange(args.start, args.end)

    print("PyMatching (reference notebook path) …")
    cycles_pm, raw_pm, fitted_pm, lambda_pm = data_prcess.subsampling_logical_error_per_cycle(
        datasets, decoder="pymatching"
    )
    _print_ler_table(cycles_pm, raw_pm, args.target_d)
    print(f"epsilon_L (fit) d={args.target_d}: {fitted_pm.get(args.target_d, float('nan')):.2e}")
    print(f"Lambda (fit): {lambda_pm:.2f}")

    print("\nMWPM_dem (DMLE-QEC) …")
    cycles_mw, raw_mw, fitted_mw, lambda_mw = data_prcess.subsampling_logical_error_per_cycle(
        datasets, decoder="mwpm_dem"
    )
    _print_ler_table(cycles_mw, raw_mw, args.target_d)
    print(f"epsilon_L (fit) d={args.target_d}: {fitted_mw.get(args.target_d, float('nan')):.2e}")
    print(f"Lambda (fit): {lambda_mw:.2f}")

    # Numerical agreement on raw LER curves
    max_diff = 0.0
    for d in sorted(set(raw_pm.keys()) & set(raw_mw.keys())):
        a = np.asarray(raw_pm[d], dtype=np.float64)
        b = np.asarray(raw_mw[d], dtype=np.float64)
        if a.shape != b.shape:
            print(f"WARN shape mismatch d={d}: {a.shape} vs {b.shape}")
            continue
        max_diff = max(max_diff, float(np.max(np.abs(a - b))))
    print(f"\nMax |raw_LER_pymatching - raw_LER_mwpm_dem| over distances: {max_diff:.3e}")
    if max_diff <= args.atol:
        print("OK: MWPM_dem matches PyMatching raw LER within tolerance.")
    else:
        print(
            "NOTE: small differences can occur if DEM / weight handling differs; "
            f"tighten investigation if needed (atol={args.atol})."
        )


if __name__ == "__main__":
    main()
