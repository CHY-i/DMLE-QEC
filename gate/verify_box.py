"""
Numerical verification of gate→DEM on an open box (p_lo, p_hi)^n for ``time_shared``.

Checks at random interior points:
  - Jacobian rank / nullity
  - sign of entries (monotonicity proxy)
  - coordinate-wise DEM increases
  - random collision search: F(p1) ≈ F(p2) with p1 ≠ p2

Run::

    python gate/verify_box.py --code rep --distance 5 --rounds 3
    python gate/verify_box.py --code rep --distance 7 --rounds 7 --p-hi 0.1
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import stim
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gate.check import (  # noqa: E402
    build_repetition_circuit,
    build_surface_code_circuit,
    dem_from_gate_probs,
    jacobian_dem_wrt_gate_probs,
)
from src.g2dem import compile_circuit  # noqa: E402


@dataclass(frozen=True)
class BoxVerifyReport:
    label: str
    num_params: int
    num_dem: int
    p_lo: float
    p_hi: float
    n_jacobian_samples: int
    n_collision_trials: int
    min_rank: int
    max_rank: int
    min_nullity: int
    max_nullity: int
    neg_jacobian_entries: int
    coord_mono_failures: int
    collision_hits: int
    best_collision_dem_l2: float
    best_collision_gate_linf: float

    @property
    def full_rank_everywhere(self) -> bool:
        return self.min_rank == self.num_params

    @property
    def no_neg_jacobian(self) -> bool:
        return self.neg_jacobian_entries == 0

    @property
    def no_collisions_found(self) -> bool:
        return self.collision_hits == 0


def _sample_gate_probs(
    n: int,
    *,
    p_lo: float,
    p_hi: float,
    gen: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    return p_lo + (p_hi - p_lo) * torch.rand(n, generator=gen, dtype=dtype)


def verify_box(
    circuit: stim.Circuit,
    *,
    label: str,
    p_lo: float,
    p_hi: float,
    n_jacobian_samples: int,
    n_collision_trials: int,
    dem_atol: float,
    gate_atol: float,
    coord_eps: float,
    seed: int,
    verbose: bool = True,
) -> BoxVerifyReport:
    meta = compile_circuit(circuit, param_sharing="time_shared")
    channels = meta["channels"]
    num_dem = meta["num_dem"]
    n = meta["num_slots"]
    gen = torch.Generator().manual_seed(seed)

    min_rank = n
    max_rank = 0
    neg_entries = 0
    coord_fail = 0

    if verbose:
        print("\n" + "=" * 72)
        print(f"box verification (time_shared) — {label}")
        print("=" * 72)
        print(f"  domain: ({p_lo}, {p_hi})^{n}  dem={num_dem}  jacobian_samples={n_jacobian_samples}")

    for s in range(n_jacobian_samples):
        p = _sample_gate_probs(n, p_lo=p_lo, p_hi=p_hi, gen=gen)
        J = jacobian_dem_wrt_gate_probs(p, channels=channels, num_dem=num_dem)
        rank = int(torch.linalg.matrix_rank(J.double(), tol=1e-9).item())
        min_rank = min(min_rank, rank)
        max_rank = max(max_rank, rank)
        neg_entries += int((J < -1e-14).sum().item())

        base = dem_from_gate_probs(p, channels=channels, num_dem=num_dem)
        for j in range(n):
            pp = p.clone()
            pp[j] = (pp[j] + coord_eps).clamp(p_lo + 1e-12, p_hi - 1e-12)
            if (dem_from_gate_probs(pp, channels=channels, num_dem=num_dem) - base).min() < -1e-11:
                coord_fail += 1

        if verbose and (s + 1) % max(1, n_jacobian_samples // 5) == 0:
            print(f"  [jacobian] {s + 1}/{n_jacobian_samples}  min_rank_so_far={min_rank}")

    collision_hits = 0
    best_dem_l2 = float("inf")
    best_gate_linf = 0.0

    if verbose:
        print(f"  collision_trials={n_collision_trials}  dem_atol={dem_atol}  gate_atol={gate_atol}")

    for _ in range(n_collision_trials):
        p1 = _sample_gate_probs(n, p_lo=p_lo, p_hi=p_hi, gen=gen)
        p2 = _sample_gate_probs(n, p_lo=p_lo, p_hi=p_hi, gen=gen)
        gate_dist = float((p1 - p2).abs().max().item())
        if gate_dist <= gate_atol:
            continue
        dem1 = dem_from_gate_probs(p1, channels=channels, num_dem=num_dem)
        dem2 = dem_from_gate_probs(p2, channels=channels, num_dem=num_dem)
        dem_l2 = float((dem1 - dem2).pow(2).sum().sqrt().item())
        if dem_l2 < best_dem_l2:
            best_dem_l2 = dem_l2
            best_gate_linf = gate_dist
        if dem_l2 <= dem_atol * (num_dem**0.5):
            collision_hits += 1

    report = BoxVerifyReport(
        label=label,
        num_params=n,
        num_dem=num_dem,
        p_lo=p_lo,
        p_hi=p_hi,
        n_jacobian_samples=n_jacobian_samples,
        n_collision_trials=n_collision_trials,
        min_rank=min_rank,
        max_rank=max_rank,
        min_nullity=n - max_rank,
        max_nullity=n - min_rank,
        neg_jacobian_entries=neg_entries,
        coord_mono_failures=coord_fail,
        collision_hits=collision_hits,
        best_collision_dem_l2=best_dem_l2,
        best_collision_gate_linf=best_gate_linf,
    )

    if verbose:
        _print_report(report)
    return report


def _print_report(r: BoxVerifyReport) -> None:
    print("\n--- summary ---")
    print(f"  Jacobian rank over samples: min={r.min_rank} max={r.max_rank} (need {r.num_params})")
    print(f"  nullity range: [{r.min_nullity}, {r.max_nullity}]")
    print(f"  negative Jacobian entries (total over samples): {r.neg_jacobian_entries}")
    print(f"  coordinate monotonicity failures: {r.coord_mono_failures}")
    print(
        f"  random collisions (||Δdem||_2 < tol): {r.collision_hits} / {r.n_collision_trials}"
    )
    print(
        f"  closest random pair: ||Δdem||_2={r.best_collision_dem_l2:.3e}  "
        f"||Δgate||_inf={r.best_collision_gate_linf:.3e}"
    )
    if r.full_rank_everywhere and r.no_neg_jacobian and r.no_collisions_found:
        print(
            "\n  Numerical evidence: on sampled interior points, full column rank, "
            "nonnegative Jacobian, no random DEM collisions."
        )
        print("  (Does NOT constitute a mathematical proof on the full open box.)")
    else:
        print("\n  At least one check failed or collisions were found — global injectivity not supported.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify time_shared gate→DEM on (p_lo, p_hi)^n.")
    ap.add_argument("--code", choices=("rep", "surface"), default="rep")
    ap.add_argument("--distance", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--error-prob", type=float, default=0.001)
    ap.add_argument("--p-lo", type=float, default=1e-6)
    ap.add_argument("--p-hi", type=float, default=0.1)
    ap.add_argument("--jacobian-samples", type=int, default=40)
    ap.add_argument("--collision-trials", type=int, default=8000)
    ap.add_argument("--dem-atol", type=float, default=1e-12)
    ap.add_argument("--gate-atol", type=float, default=1e-5)
    ap.add_argument("--coord-eps", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.code == "rep":
        circuit = build_repetition_circuit(
            distance=args.distance, rounds=args.rounds, error_prob=args.error_prob
        )
        label = f"repetition_code:memory d={args.distance} r={args.rounds}"
    else:
        circuit = build_surface_code_circuit(
            distance=args.distance, rounds=args.rounds, error_prob=args.error_prob
        )
        label = f"surface d={args.distance} r={args.rounds}"

    verify_box(
        circuit,
        label=label,
        p_lo=args.p_lo,
        p_hi=args.p_hi,
        n_jacobian_samples=args.jacobian_samples,
        n_collision_trials=args.collision_trials,
        dem_atol=args.dem_atol,
        gate_atol=args.gate_atol,
        coord_eps=args.coord_eps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
