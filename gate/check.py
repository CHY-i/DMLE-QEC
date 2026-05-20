"""
Check whether multiple gate-probability vectors yield the same DEM, for each
``param_sharing`` mode in :mod:`src.g2dem` (``elementary``, ``dep``, ``time_shared``).

Run all three and compare which setting makes DEM (locally) determine gate params::

    python gate/check.py --code rep --distance 5 --rounds 3
    python gate/check.py --code both
    python gate/check.py --code surface --surface-distance 5 --surface-rounds 5
    python gate/check.py --param-sharing elementary
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.g2dem import (  # noqa: E402
    GateNoiseToDEM,
    ParamSharing,
    _merge_channels,
    compile_circuit,
)

ALL_PARAM_SHARINGS: tuple[ParamSharing, ...] = ("elementary", "dep", "time_shared")


def build_repetition_circuit(*, distance: int, rounds: int, error_prob: float) -> stim.Circuit:
    return stim.Circuit.generated(
        code_task="repetition_code:memory",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=error_prob,
        before_measure_flip_probability=error_prob,
        after_reset_flip_probability=error_prob,
        before_round_data_depolarization=error_prob,
    )


def build_surface_code_circuit(
    *,
    distance: int = 3,
    rounds: int = 3,
    error_prob: float = 0.001,
    task: str = "surface_code:rotated_memory_z",
) -> stim.Circuit:
    return stim.Circuit.generated(
        task,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=error_prob,
        after_reset_flip_probability=error_prob,
        before_measure_flip_probability=error_prob,
        before_round_data_depolarization=error_prob,
    )


def probs_to_logit(p: torch.Tensor, p_min: float, p_max: float) -> torch.Tensor:
    p = p.clamp(p_min, p_max)
    return torch.log(p / (1.0 - p))


def logit_to_probs(x: torch.Tensor, p_min: float, p_max: float) -> torch.Tensor:
    return torch.sigmoid(x).clamp(p_min, p_max)


def dem_from_gate_probs(
    gate_probs: torch.Tensor,
    *,
    channels,
    num_dem: int,
) -> torch.Tensor:
    """Forward map used in g2dem (independent per-slot gate probabilities)."""
    return _merge_channels(gate_probs, channels, num_dem)


def dem_from_logit(
    logit: torch.Tensor,
    *,
    channels,
    num_dem: int,
    p_min: float,
    p_max: float,
) -> torch.Tensor:
    return dem_from_gate_probs(logit_to_probs(logit, p_min, p_max), channels=channels, num_dem=num_dem)


def fit_gate_to_dem(
    target_dem: torch.Tensor,
    *,
    channels,
    num_dem: int,
    init_logit: torch.Tensor,
    p_min: float,
    p_max: float,
    repulse_logit: torch.Tensor | None = None,
    repulse_weight: float = 0.0,
    steps: int = 2000,
    lr: float = 0.05,
    tol: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Find gate probabilities whose DEM equals ``target_dem`` (if possible)."""
    logit = init_logit.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([logit], lr=lr)

    best_loss = float("inf")
    best_logit = logit.detach().clone()
    target = target_dem.detach()

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = dem_from_logit(logit, channels=channels, num_dem=num_dem, p_min=p_min, p_max=p_max)
        loss = F.mse_loss(pred, target)
        if repulse_logit is not None and repulse_weight > 0:
            loss = loss - repulse_weight * (logit - repulse_logit).pow(2).mean()
        loss.backward()
        opt.step()
        lv = float(loss.detach().cpu())
        if lv < best_loss:
            best_loss = lv
            best_logit = logit.detach().clone()
        if lv < tol:
            break

    with torch.no_grad():
        p = logit_to_probs(best_logit, p_min, p_max)
        pred = dem_from_gate_probs(p, channels=channels, num_dem=num_dem)
        dem_max = float((pred - target).abs().max().cpu())
    return p, pred, dem_max


def jacobian_dem_wrt_gate_probs(
    gate_probs: torch.Tensor,
    *,
    channels,
    num_dem: int,
) -> torch.Tensor:
    """Shape ``(num_dem, num_slots)``."""
    p = gate_probs.detach().clone().requires_grad_(True)
    dem = dem_from_gate_probs(p, channels=channels, num_dem=num_dem)
    rows = []
    for i in range(dem.numel()):
        if p.grad is not None:
            p.grad.zero_()
        dem[i].backward(retain_graph=True)
        rows.append(p.grad.detach().clone())
    return torch.stack(rows, dim=0)


def null_space_basis(jac: torch.Tensor, *, rtol: float = 1e-10) -> tuple[int, torch.Tensor | None]:
    """Return rank and orthonormal basis for null(J), shape ``(n_slots, k)``."""
    j = jac.detach().cpu().double()
    u, s, vh = torch.linalg.svd(j, full_matrices=True)
    rank = int((s > rtol * s[0].item()).sum().item()) if s.numel() else 0
    n = vh.shape[1]
    if rank >= n:
        return rank, None
    return rank, vh[rank:, :].T.contiguous()


def verify_null_direction(
    gate_probs: torch.Tensor,
    direction: torch.Tensor,
    *,
    channels,
    num_dem: int,
    eps: float = 1e-4,
) -> float:
    """Max |Δdem| after perturbing gate_probs along ``direction``."""
    d = direction / direction.norm().clamp(min=1e-30)
    with torch.no_grad():
        p_plus = (gate_probs + eps * d).clamp(1e-8, 1.0 - 1e-8)
        p_minus = (gate_probs - eps * d).clamp(1e-8, 1.0 - 1e-8)
        dem0 = dem_from_gate_probs(gate_probs, channels=channels, num_dem=num_dem)
        dem_p = dem_from_gate_probs(p_plus, channels=channels, num_dem=num_dem)
        dem_m = dem_from_gate_probs(p_minus, channels=channels, num_dem=num_dem)
        return float(torch.max((dem_p - dem0).abs().max(), (dem_m - dem0).abs().max()).cpu())


@dataclass(frozen=True)
class SharingCheckResult:
    param_sharing: ParamSharing
    num_slots: int
    num_elementary: int
    num_dem: int
    rank: int
    nullity: int
    locally_injective: bool
    null_certified: bool
    distinct_second_preimage: bool
    null_walk_gate_dist: float
    null_walk_dem_err: float
    opt_gate_dist: float
    opt_dem_err: float

    @property
    def dem_determines_gate(self) -> bool:
        """Heuristic: no 1st-order ambiguity and no certified distinct preimage."""
        return self.locally_injective and not self.null_certified and not self.distinct_second_preimage


def run_check(
    circuit: stim.Circuit,
    *,
    label: str,
    param_sharing: ParamSharing,
    dem_target: str,
    p_min: float,
    p_max: float,
    dem_atol: float,
    gate_atol: float,
    opt_steps: int,
    seed: int,
    verbose: bool = True,
) -> SharingCheckResult:
    dtype = torch.float64
    meta = compile_circuit(circuit, param_sharing=param_sharing)
    channels = meta["channels"]
    num_dem = meta["num_dem"]
    num_slots = meta["num_slots"]

    g2d = GateNoiseToDEM(
        circuit, learnable=False, param_sharing=param_sharing, dtype=dtype, device="cpu"
    )
    p_ref = g2d.gate_probs().detach()
    logit_ref = probs_to_logit(p_ref, p_min, p_max)

    if dem_target == "stim":
        target_dem = g2d.ref_dem_probs.clone()
    elif dem_target == "circuit":
        target_dem = torch.tensor(
            dem_from_gate_probs(p_ref, channels=channels, num_dem=num_dem),
            dtype=dtype,
        )
    elif dem_target == "random":
        gen = torch.Generator().manual_seed(seed)
        rand_p = p_min + (p_max - p_min) * torch.rand(num_slots, generator=gen, dtype=dtype)
        target_dem = dem_from_gate_probs(rand_p, channels=channels, num_dem=num_dem)
    else:
        raise ValueError(f"unknown dem_target={dem_target!r}")

    with torch.no_grad():
        ref_fit = float((dem_from_gate_probs(p_ref, channels=channels, num_dem=num_dem) - target_dem).abs().max())

    if verbose:
        print("\n" + "-" * 72)
        print(f"  param_sharing={param_sharing}")
        print(
            f"  learnable_params={num_slots}  elementary_sites={meta['num_elementary']}"
            f"  dem_errors={num_dem}  channels={meta['num_channels']}"
        )
        print(f"  dem_target={dem_target}  |dem(p_ref)-target|_inf={ref_fit:.3e}")
        print(f"  gate bounds: [{p_min}, {p_max}]")

    # --- 1) Jacobian rank / null space ---
    if verbose:
        print("\n[1] Jacobian  d(dem)/d(gate_probs) at reference gate_probs")
    jac = jacobian_dem_wrt_gate_probs(p_ref, channels=channels, num_dem=num_dem)
    rank, null_basis = null_space_basis(jac)
    nullity = num_slots - rank
    if verbose:
        print(f"  rank(J)={rank}  nullity={nullity}")
    if verbose and null_basis is not None:
        for k in range(min(3, null_basis.shape[1])):
            delta_dem = verify_null_direction(
                p_ref, null_basis[:, k], channels=channels, num_dem=num_dem
            )
            print(f"  null vector {k}: |Δdem| along unit direction (eps=1e-4) ≈ {delta_dem:.3e}")
        if null_basis.shape[1] > 0:
            print("  => positive nullity: **local** non-uniqueness (first-order ambiguity exists)")

    # --- 2) Null-space walk from p_ref (exact DEM at circuit gate probs) ---
    if verbose:
        print("\n[2] Null-space walk from reference gate_probs (exact DEM preimage)")
    p_a = p_ref.clone()
    dem_a = dem_from_gate_probs(p_a, channels=channels, num_dem=num_dem)
    err_a = float((dem_a - target_dem).abs().max().cpu())

    best_alt_p: torch.Tensor | None = None
    best_alt_dem_err = float("inf")
    best_alt_gate_dist = 0.0

    if null_basis is not None:
        for k in range(null_basis.shape[1]):
            direction = null_basis[:, k].to(dtype=dtype)
            for scale in (1e-3, 1e-2, 5e-2, 0.1, 0.2):
                p_try = (p_ref + scale * direction).clamp(p_min, p_max)
                dem_try = dem_from_gate_probs(p_try, channels=channels, num_dem=num_dem)
                err_try = float((dem_try - target_dem).abs().max().cpu())
                gdist = float((p_try - p_ref).abs().max().cpu())
                if err_try < best_alt_dem_err and gdist > gate_atol:
                    best_alt_dem_err = err_try
                    best_alt_gate_dist = gdist
                    best_alt_p = p_try.clone()

    # --- 3) Second fit: start from null perturbation, repulse away from p_ref ---
    if verbose:
        print("\n[3] Optimized second preimage (repulse from p_ref)")
    if null_basis is not None and null_basis.shape[1] > 0:
        init_b = probs_to_logit(
            (p_ref + 0.05 * null_basis[:, 0].to(dtype=dtype)).clamp(p_min, p_max), p_min, p_max
        )
    else:
        gen = torch.Generator().manual_seed(seed)
        init_b = probs_to_logit(
            p_min + (p_max - p_min) * torch.rand(num_slots, generator=gen, dtype=dtype), p_min, p_max
        )

    p_b, dem_b, err_b = fit_gate_to_dem(
        target_dem,
        channels=channels,
        num_dem=num_dem,
        init_logit=init_b,
        p_min=p_min,
        p_max=p_max,
        repulse_logit=probs_to_logit(p_ref, p_min, p_max),
        repulse_weight=5.0,
        steps=opt_steps,
        lr=0.08,
    )

    gate_dist = float((p_a - p_b).abs().max().cpu())
    dem_agree = float((dem_a - dem_b).abs().max().cpu())
    match_a = err_a <= dem_atol
    match_b = err_b <= dem_atol
    different = gate_dist > gate_atol

    null_certified = best_alt_p is not None and best_alt_dem_err <= dem_atol
    locally_injective = nullity == 0
    distinct_second = match_a and match_b and different

    result = SharingCheckResult(
        param_sharing=param_sharing,
        num_slots=num_slots,
        num_elementary=meta["num_elementary"],
        num_dem=num_dem,
        rank=rank,
        nullity=nullity,
        locally_injective=locally_injective,
        null_certified=null_certified,
        distinct_second_preimage=distinct_second,
        null_walk_gate_dist=best_alt_gate_dist,
        null_walk_dem_err=best_alt_dem_err,
        opt_gate_dist=gate_dist,
        opt_dem_err=err_b,
    )

    if verbose:
        print(f"  reference p_ref: max|dem-target|={err_a:.3e}")
        print(
            f"  null-walk best: max|dem-target|={best_alt_dem_err:.3e}"
            f"  max|p-p_ref|={best_alt_gate_dist:.3e}"
        )
        print(f"  optimized B:   max|dem-target|={err_b:.3e}  gate[min,max]=({p_b.min():.4g},{p_b.max():.4g})")
        print(f"  max|p_ref - p_B|={gate_dist:.3e}  max|dem_ref - dem_B|={dem_agree:.3e}")

        if null_certified:
            print(
                f"\n  **Null-space walk:** gate changed by {best_alt_gate_dist:.3e} "
                f"with max|Δdem|={best_alt_dem_err:.3e} (same DEM)."
            )
        if distinct_second:
            print("  **Optimized B:** distinct gate vector with (nearly) the same DEM.")
        elif null_certified:
            print("  The gate→DEM map is **not injective** (Jacobian null space).")
        elif locally_injective:
            print("  At p_ref: full rank (nullity=0); no distinct preimage found in this test.")
        else:
            print("  Second distinct preimage not certified at current tolerances / steps.")

    return result


def _verdict_label(r: SharingCheckResult) -> str:
    if r.dem_determines_gate:
        return "DEM → gate (local, tested)"
    if r.locally_injective and (r.null_certified or r.distinct_second_preimage):
        return "full rank but global ambiguity"
    if r.locally_injective:
        return "full rank only"
    return "NOT unique (nullity>0 or walk)"


def print_sharing_comparison(label: str, results: list[SharingCheckResult]) -> None:
    print("\n" + "=" * 72)
    print(f"g2dem injectivity comparison — {label}")
    print("=" * 72)
    header = (
        f"{'sharing':<14} {'#params':>7} {'rank':>5} {'null':>5} "
        f"{'local inj.':>10} {'null walk':>10} {'2nd preim.':>11}  verdict"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.param_sharing:<14} {r.num_slots:>7} {r.rank:>5} {r.nullity:>5} "
            f"{'yes' if r.locally_injective else 'no':>10} "
            f"{'yes' if r.null_certified else 'no':>10} "
            f"{'yes' if r.distinct_second_preimage else 'no':>11}  {_verdict_label(r)}"
        )

    candidates = [r for r in results if r.dem_determines_gate]
    print()
    if len(candidates) == 1:
        print(
            f"Conclusion: only ``{candidates[0].param_sharing}`` passes "
            f"(nullity=0, no null-walk / second-preimage found)."
        )
    elif candidates:
        names = ", ".join(f"``{r.param_sharing}``" for r in candidates)
        print(f"Conclusion: {names} pass the local uniqueness test at p_ref.")
    else:
        print("Conclusion: none of the three modes yield a unique gate vector from DEM (at p_ref).")
        best = min(results, key=lambda r: r.nullity)
        print(
            f"  Smallest nullity: ``{best.param_sharing}`` (nullity={best.nullity}, "
            f"{best.num_slots} learnable params)."
        )
    print()


def run_check_all_sharings(
    circuit: stim.Circuit,
    *,
    label: str,
    sharings: tuple[ParamSharing, ...],
    **check_kw,
) -> list[SharingCheckResult]:
    print("\n" + "=" * 72)
    print(f"g2dem non-injectivity check — {label}")
    print("=" * 72)
    results: list[SharingCheckResult] = []
    for sharing in sharings:
        results.append(
            run_check(circuit, label=label, param_sharing=sharing, verbose=True, **check_kw)
        )
    print_sharing_comparison(label, results)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Check gate→DEM non-injectivity via g2dem.")
    ap.add_argument(
        "--code",
        choices=("rep", "surface", "both"),
        default="both",
        help="repetition (--distance/--rounds), surface (--surface-distance/--surface-rounds), or both",
    )
    ap.add_argument("--distance", type=int, default=5, help="repetition-code distance")
    ap.add_argument("--rounds", type=int, default=3, help="repetition-code rounds")
    ap.add_argument(
        "--surface-distance",
        type=int,
        default=3,
        help="surface_code:rotated_memory_z distance (default 3, matches prior script)",
    )
    ap.add_argument(
        "--surface-rounds",
        type=int,
        default=3,
        help="surface_code:rotated_memory_z measurement rounds",
    )
    ap.add_argument("--error-prob", type=float, default=0.001)
    ap.add_argument(
        "--dem-target",
        choices=("stim", "circuit", "random"),
        default="stim",
        help="target DEM: Stim ref_dem, g2dem(p_ref), or random gate pushforward",
    )
    ap.add_argument("--p-min", type=float, default=1e-6)
    ap.add_argument("--p-max", type=float, default=0.49)
    ap.add_argument("--dem-atol", type=float, default=1e-9, help="treat DEM as matched below this")
    ap.add_argument("--gate-atol", type=float, default=1e-8, help="min |p_A-p_B| to call gate vectors distinct")
    ap.add_argument("--opt-steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--param-sharing",
        type=str,
        default="all",
        choices=("all", *ALL_PARAM_SHARINGS),
        help="run one sharing mode or compare elementary / dep / time_shared (default: all)",
    )
    args = ap.parse_args()

    check_kw = dict(
        dem_target=args.dem_target,
        p_min=args.p_min,
        p_max=args.p_max,
        dem_atol=args.dem_atol,
        gate_atol=args.gate_atol,
        opt_steps=args.opt_steps,
        seed=args.seed,
    )

    if args.param_sharing == "all":
        sharings = ALL_PARAM_SHARINGS
    else:
        sharings = (args.param_sharing,)  # type: ignore[assignment]

    def run_for_circuit(circuit: stim.Circuit, label: str) -> None:
        if args.param_sharing == "all":
            run_check_all_sharings(circuit, label=label, sharings=sharings, **check_kw)
        else:
            run_check(circuit, label=label, param_sharing=sharings[0], verbose=True, **check_kw)

    if args.code in ("rep", "both"):
        run_for_circuit(
            build_repetition_circuit(
                distance=args.distance, rounds=args.rounds, error_prob=args.error_prob
            ),
            label=f"repetition_code:memory d={args.distance} r={args.rounds}",
        )

    if args.code in ("surface", "both"):
        sd, sr = args.surface_distance, args.surface_rounds
        run_for_circuit(
            build_surface_code_circuit(distance=sd, rounds=sr, error_prob=args.error_prob),
            label=f"surface_code:rotated_memory_z d={sd} r={sr}",
        )


if __name__ == "__main__":
    main()
