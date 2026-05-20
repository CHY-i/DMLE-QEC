"""
Test the new ``src.utils.rep_dem`` construction against commutation relations.

Run:

    conda run -n qec python logical/test_rep_dem_commutation.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import stim

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import rep_dem  # noqa: E402


def main() -> None:
    # Use a generated repetition-code circuit just to obtain a representative DEM.
    d = 9
    r = 2
    circuit = stim.Circuit.generated(
        code_task="repetition_code:memory",
        distance=d,
        rounds=r,
        after_clifford_depolarization=1e-3,
        before_measure_flip_probability=1e-3,
        after_reset_flip_probability=1e-3,
    )
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)

    rep = rep_dem(dem)

    hx, hz, lx, lz, pebz = rep.hx, rep.hz, rep.lx, rep.lz, rep.pebz

    checks = {
        "hx_hz_commute_sum": int(((hx @ hz.T) % 2).sum()),
        "hx_lz_commute_sum": int(((hx @ lz.reshape(-1).T) % 2).sum()),
        "lx_lz_anticommute": int((lx.reshape(-1) @ lz.reshape(-1)) % 2),
        "pebz_is_pinv": int(((pebz @ hx.T) % 2 != np.eye(hx.shape[0], hx.shape[0])).sum()),
    }

    print("d, r:", d, r)
    print("hx shape:", hx.shape, "hz shape:", hz.shape, "lx shape:", lx.shape, "lz shape:", lz.shape)
    for k, v in checks.items():
        print(k, "=", v)

    # Hard assertions: should hold exactly.
    assert checks["hx_hz_commute_sum"] == 0
    assert checks["hx_lz_commute_sum"] == 0
    assert checks["lx_lz_anticommute"] == 1
    assert checks["pebz_is_pinv"] == 0
    print("OK")


if __name__ == "__main__":
    main()

