"""
Check g2dem vs Stim DEM alignment and gate→DEM injectivity (Jacobian rank / null space).

Sections per ``param_sharing``:

  [0] ``max|g2d() - stim_ref_dem_probs|`` and ``max|g2d() - ref_dem_probs|`` at init gate probs
  [1] Jacobian ``d(dem)/d(gate_probs)``, rank, nullity
  [2] Null-space walk
  [3] Optimized second preimage (non-injectivity probe)

Run::

    python gate/check.py --code rep --distance 5 --rounds 3
    python gate/check.py --code both --noise-model si1000
    python gate/check.py --code surface --surface-distance 3 --surface-rounds 3
    python gate/check.py --param-sharing dep2
    python gate/check.py --nullity-sweep
    python gate/check.py --code syc_old
    python gate/check.py --code syc_old --center 5_7 --distance 3 --rounds 3
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import stim
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gate.pl_opt import (  # noqa: E402
    NoiseModelKind,
    build_repetition_circuit_depolarizing,
    build_repetition_circuit_si1000,
    noise_model_si1000,
)
from src.g2dem import GateNoiseToDEM, ParamSharing, compile_circuit, dem_from_gate_probs  # noqa: E402

ALL_PARAM_SHARINGS: tuple[ParamSharing, ...] = (
    "elementary",
    "dep",
    "time_shared",
    "time_shared_dep12",
    "time_shared_dep12_M_DD",
    "time_shared_dep12_M_DD_I",
    "time_shared_dep12_M_R_DD_I",
    "time_shared_dep2",
    "dep2",
)

RESTRICTED_PARAM_SHARINGS: tuple[ParamSharing, ...] = (
    "time_shared_dep12",
    "time_shared_dep12_M_DD",
    "time_shared_dep12_M_DD_I",
    "time_shared_dep12_M_R_DD_I",
    "time_shared_dep2",
    "dep2",
)

SYC_OLD_DATA_ROOT = REPO_ROOT / "data" / "sycamore_old"
SYC_OLD_BASES: tuple[str, ...] = ("X", "Z")


def build_surface_code_circuit_depolarizing(
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


def build_surface_code_circuit_si1000(
    *,
    distance: int = 3,
    rounds: int = 3,
    error_prob: float = 0.001,
    task: str = "surface_code:rotated_memory_z",
) -> stim.Circuit:
    ideal = stim.Circuit.generated(task, distance=distance, rounds=rounds)
    return noise_model_si1000(error_prob).noisy_circuit(ideal)


def build_repetition_circuit(
    *,
    distance: int,
    rounds: int,
    error_prob: float,
    noise_model: NoiseModelKind = "si1000",
) -> stim.Circuit:
    if noise_model == "depolarizing":
        return build_repetition_circuit_depolarizing(
            distance=distance, rounds=rounds, error_prob=error_prob
        )
    if noise_model == "si1000":
        return build_repetition_circuit_si1000(
            distance=distance, rounds=rounds, error_prob=error_prob
        )
    raise ValueError(f"unknown noise_model={noise_model!r}")


def build_surface_code_circuit(
    *,
    distance: int = 3,
    rounds: int = 3,
    error_prob: float = 0.001,
    task: str = "surface_code:rotated_memory_z",
    noise_model: NoiseModelKind = "si1000",
) -> stim.Circuit:
    if noise_model == "depolarizing":
        return build_surface_code_circuit_depolarizing(
            distance=distance,
            rounds=rounds,
            error_prob=error_prob,
            task=task,
        )
    if noise_model == "si1000":
        return build_surface_code_circuit_si1000(
            distance=distance,
            rounds=rounds,
            error_prob=error_prob,
            task=task,
        )
    raise ValueError(f"unknown noise_model={noise_model!r}")


def probs_to_logit(p: torch.Tensor, p_min: float, p_max: float) -> torch.Tensor:
    p = p.clamp(p_min, p_max)
    return torch.log(p / (1.0 - p))


def logit_to_probs(x: torch.Tensor, p_min: float, p_max: float) -> torch.Tensor:
    return torch.sigmoid(x).clamp(p_min, p_max)


def dem_from_logit(
    logit: torch.Tensor,
    *,
    merge_ops,
    stim_dem: stim.DetectorErrorModel,
    num_dem: int,
    p_min: float,
    p_max: float,
) -> torch.Tensor:
    return dem_from_gate_probs(
        logit_to_probs(logit, p_min, p_max),
        merge_ops=merge_ops,
        stim_dem=stim_dem,
        num_dem=num_dem,
    )


def fit_gate_to_dem(
    target_dem: torch.Tensor,
    *,
    merge_ops,
    stim_dem: stim.DetectorErrorModel,
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
        pred = dem_from_logit(
            logit,
            merge_ops=merge_ops,
            stim_dem=stim_dem,
            num_dem=num_dem,
            p_min=p_min,
            p_max=p_max,
        )
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
        pred = dem_from_gate_probs(p, merge_ops=merge_ops, stim_dem=stim_dem, num_dem=num_dem)
        dem_max = float((pred - target).abs().max().cpu())
    return p, pred, dem_max


def dem_from_learn_param(
    g2d: GateNoiseToDEM,
    learn_param: torch.Tensor,
    *,
    merge_ops,
    stim_dem: stim.DetectorErrorModel,
    num_dem: int,
) -> torch.Tensor:
    return dem_from_gate_probs(
        g2d.gate_probs(learn_param),
        merge_ops=merge_ops,
        stim_dem=stim_dem,
        num_dem=num_dem,
    )


def jacobian_dem_wrt_gate_probs(
    gate_probs: torch.Tensor,
    *,
    merge_ops,
    stim_dem: stim.DetectorErrorModel,
    num_dem: int,
) -> torch.Tensor:
    """Shape ``(num_dem, num_elementary_gate_probs)`` — expanded gate vector, not learnable params."""
    p = gate_probs.detach().clone().requires_grad_(True)
    dem = dem_from_gate_probs(p, merge_ops=merge_ops, stim_dem=stim_dem, num_dem=num_dem)
    rows = []
    for i in range(dem.numel()):
        if p.grad is not None:
            p.grad.zero_()
        dem[i].backward(retain_graph=True)
        rows.append(p.grad.detach().clone())
    return torch.stack(rows, dim=0)


def jacobian_dem_wrt_learn_params(
    g2d: GateNoiseToDEM,
    *,
    merge_ops,
    stim_dem: stim.DetectorErrorModel,
    num_dem: int,
) -> torch.Tensor:
    """Shape ``(num_dem, num_learnable)`` — respects ``param_sharing`` (dep / time_shared / elementary)."""
    param = g2d.gate_param.detach().clone().requires_grad_(True)
    dem = dem_from_learn_param(g2d, param, merge_ops=merge_ops, stim_dem=stim_dem, num_dem=num_dem)
    rows = []
    for i in range(dem.numel()):
        if param.grad is not None:
            param.grad.zero_()
        dem[i].backward(retain_graph=True)
        rows.append(param.grad.detach().clone())
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
    merge_ops,
    stim_dem: stim.DetectorErrorModel,
    num_dem: int,
    eps: float = 1e-4,
) -> float:
    """Max |Δdem| after perturbing expanded gate_probs along ``direction``."""
    d = direction / direction.norm().clamp(min=1e-30)
    with torch.no_grad():
        p_plus = (gate_probs + eps * d).clamp(1e-8, 1.0 - 1e-8)
        p_minus = (gate_probs - eps * d).clamp(1e-8, 1.0 - 1e-8)
        dem0 = dem_from_gate_probs(gate_probs, merge_ops=merge_ops, stim_dem=stim_dem, num_dem=num_dem)
        dem_p = dem_from_gate_probs(p_plus, merge_ops=merge_ops, stim_dem=stim_dem, num_dem=num_dem)
        dem_m = dem_from_gate_probs(p_minus, merge_ops=merge_ops, stim_dem=stim_dem, num_dem=num_dem)
        return float(torch.max((dem_p - dem0).abs().max(), (dem_m - dem0).abs().max()).cpu())


def verify_null_direction_learn(
    g2d: GateNoiseToDEM,
    learn_param: torch.Tensor,
    direction: torch.Tensor,
    *,
    merge_ops,
    stim_dem: stim.DetectorErrorModel,
    num_dem: int,
    eps: float = 1e-4,
) -> float:
    """Max |Δdem| after perturbing learnable ``gate_param`` along ``direction``."""
    d = direction / direction.norm().clamp(min=1e-30)
    with torch.no_grad():
        dem0 = dem_from_learn_param(
            g2d, learn_param, merge_ops=merge_ops, stim_dem=stim_dem, num_dem=num_dem
        )
        dem_p = dem_from_learn_param(
            g2d,
            learn_param + eps * d,
            merge_ops=merge_ops,
            stim_dem=stim_dem,
            num_dem=num_dem,
        )
        dem_m = dem_from_learn_param(
            g2d,
            learn_param - eps * d,
            merge_ops=merge_ops,
            stim_dem=stim_dem,
            num_dem=num_dem,
        )
        return float(torch.max((dem_p - dem0).abs().max(), (dem_m - dem0).abs().max()).cpu())


def fit_learn_param_to_dem(
    g2d: GateNoiseToDEM,
    target_dem: torch.Tensor,
    *,
    merge_ops,
    stim_dem: stim.DetectorErrorModel,
    num_dem: int,
    init_param: torch.Tensor,
    repulse_param: torch.Tensor | None = None,
    repulse_weight: float = 0.0,
    steps: int = 2000,
    lr: float = 0.05,
    tol: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Find ``gate_param`` whose DEM equals ``target_dem`` (if possible)."""
    param = init_param.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([param], lr=lr)

    best_loss = float("inf")
    best_param = param.detach().clone()
    target = target_dem.detach()

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = dem_from_learn_param(g2d, param, merge_ops=merge_ops, stim_dem=stim_dem, num_dem=num_dem)
        loss = F.mse_loss(pred, target)
        if repulse_param is not None and repulse_weight > 0:
            loss = loss - repulse_weight * (param - repulse_param).pow(2).mean()
        loss.backward()
        opt.step()
        lv = float(loss.detach().cpu())
        if lv < best_loss:
            best_loss = lv
            best_param = param.detach().clone()
        if lv < tol:
            break

    with torch.no_grad():
        p = g2d.gate_probs(best_param)
        pred = dem_from_gate_probs(p, merge_ops=merge_ops, stim_dem=stim_dem, num_dem=num_dem)
        dem_max = float((pred - target).abs().max().cpu())
    return best_param, pred, dem_max


@dataclass(frozen=True)
class DemAlignmentResult:
    max_abs_vs_stim: float
    max_abs_vs_ref: float
    worst_line_stim: int
    matches_stim: bool
    matches_ref: bool


@dataclass(frozen=True)
class SharingCheckResult:
    param_sharing: ParamSharing
    num_slots: int
    num_elementary: int
    num_dem: int
    num_gate_prob_slots: int
    dem_alignment: DemAlignmentResult
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


def check_dem_alignment(
    g2d: GateNoiseToDEM,
    *,
    dem_atol: float,
    verbose: bool = True,
) -> DemAlignmentResult:
    """Compare ``g2d()`` to Stim reference and patched-circuit ``ref_dem_probs``."""
    with torch.no_grad():
        pred = g2d()
        stim_ref = g2d.stim_ref_dem_probs
        ref = g2d.ref_dem_probs
        diff_stim = (pred - stim_ref).abs()
        diff_ref = (pred - ref).abs()
        max_stim = float(diff_stim.max().cpu())
        max_ref = float(diff_ref.max().cpu())
        worst = int(diff_stim.argmax().cpu())

    result = DemAlignmentResult(
        max_abs_vs_stim=max_stim,
        max_abs_vs_ref=max_ref,
        worst_line_stim=worst,
        matches_stim=max_stim <= dem_atol,
        matches_ref=max_ref <= dem_atol,
    )
    if verbose:
        print("\n[0] DEM alignment (g2d vs Stim / ref at init gate_probs)")
        print(
            f"  learnable={g2d.num_slots}  elementary={g2d.num_elementary}  "
            f"dem_errors={g2d.num_dem}"
        )
        print(
            f"  max|g2d - stim_ref|={result.max_abs_vs_stim:.3e}  "
            f"max|g2d - ref_dem|={result.max_abs_vs_ref:.3e}  "
            f"worst_line={result.worst_line_stim}"
        )
        ok = "OK" if result.matches_stim else "FAIL"
        print(f"  stim match (atol={dem_atol:.0e}): {ok}")
    return result


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
    merge_ops = meta["merge_ops"]
    stim_dem = meta["stim_dem"]
    num_dem = meta["num_dem"]
    num_slots = meta["num_slots"]

    g2d = GateNoiseToDEM(
        circuit, learnable=False, param_sharing=param_sharing, dtype=dtype, device="cpu"
    )
    dem_align = check_dem_alignment(g2d, dem_atol=dem_atol, verbose=verbose)
    if verbose and param_sharing in RESTRICTED_PARAM_SHARINGS:
        if param_sharing == "dep2":
            tag = "independent per post-2Q DEPOLARIZE2"
        elif param_sharing == "time_shared_dep2":
            tag = "time_shared grouping per post-2Q DEPOLARIZE2"
        elif param_sharing == "time_shared_dep12_M_DD":
            tag = "time_shared dep12 + ancilla pre-M X_ERROR + data DD"
        elif param_sharing == "time_shared_dep12_M_DD_I":
            tag = "time_shared dep12_M_DD + idle DEP1 (excl. DD and reset-sync idle)"
        elif param_sharing == "time_shared_dep12_M_R_DD_I":
            tag = "time_shared dep12_M_DD_I + reset X_ERROR (X_ERROR_R)"
        else:
            tag = "time_shared dep1(after 1Q) + dep2(after 2Q)"
        print(f"  learnable restricted slots ({tag}): {meta['slot_keys']}")

    param_ref = g2d.gate_param.detach()
    p_ref = g2d.gate_probs().detach()
    n_gate = int(p_ref.numel())

    if dem_target == "stim":
        target_dem = torch.tensor(meta["stim_dem_probs"], dtype=dtype)
    elif dem_target == "circuit":
        target_dem = g2d.ref_dem_probs.clone()
    elif dem_target == "random":
        gen = torch.Generator().manual_seed(seed)
        rand_param = param_ref + 0.25 * torch.randn(num_slots, generator=gen, dtype=dtype)
        target_dem = dem_from_learn_param(
            g2d, rand_param, merge_ops=merge_ops, stim_dem=stim_dem, num_dem=num_dem
        )
    else:
        raise ValueError(f"unknown dem_target={dem_target!r}")

    with torch.no_grad():
        ref_fit = float(
            (
                dem_from_learn_param(
                    g2d, param_ref, merge_ops=merge_ops, stim_dem=stim_dem, num_dem=num_dem
                )
                - target_dem
            )
            .abs()
            .max()
        )

    if verbose:
        print("\n" + "-" * 72)
        print(f"  param_sharing={param_sharing}")
        print(
            f"  learnable_params={num_slots}  gate_prob_slots={n_gate}  "
            f"elementary_sites={meta['num_elementary']}  dem_errors={num_dem}  "
            f"merge_ops={len(merge_ops)}"
        )
        print(f"  dem_target={dem_target}  |dem(p_ref)-target|_inf={ref_fit:.3e}")
        print(f"  gate bounds: [{p_min}, {p_max}]")

    # --- 1) Jacobian rank / null space (w.r.t. learnable gate_param) ---
    if verbose:
        print("\n[1] Jacobian  d(dem)/d(gate_param) at reference learnable params")
    jac = jacobian_dem_wrt_learn_params(
        g2d, merge_ops=merge_ops, stim_dem=stim_dem, num_dem=num_dem
    )
    rank, null_basis = null_space_basis(jac)
    nullity = num_slots - rank
    if verbose:
        print(f"  J shape={tuple(jac.shape)}  rank(J)={rank}  nullity={nullity}  (learnable={num_slots})")
        jac_gate = jacobian_dem_wrt_gate_probs(
            p_ref, merge_ops=merge_ops, stim_dem=stim_dem, num_dem=num_dem
        )
        rank_g = int(
            torch.linalg.matrix_rank(jac_gate.double(), tol=1e-9).item()
        )
        print(
            f"  (expanded gate_probs only: J_gate shape={tuple(jac_gate.shape)}  "
            f"rank={rank_g}  — same for all param_sharing at fixed init rates)"
        )
    if verbose and null_basis is not None:
        for k in range(min(3, null_basis.shape[1])):
            delta_dem = verify_null_direction_learn(
                g2d,
                param_ref,
                null_basis[:, k],
                merge_ops=merge_ops,
                stim_dem=stim_dem,
                num_dem=num_dem,
            )
            print(f"  null vector {k}: |Δdem| along unit direction (eps=1e-4) ≈ {delta_dem:.3e}")
        if null_basis.shape[1] > 0:
            print("  => positive nullity: **local** non-uniqueness in learnable coordinates")

    # --- 2) Null-space walk from reference gate_param ---
    if verbose:
        print("\n[2] Null-space walk from reference gate_param (exact DEM preimage)")
    dem_a = dem_from_learn_param(
        g2d, param_ref, merge_ops=merge_ops, stim_dem=stim_dem, num_dem=num_dem
    )
    err_a = float((dem_a - target_dem).abs().max().cpu())

    best_alt_param: torch.Tensor | None = None
    best_alt_dem_err = float("inf")
    best_alt_gate_dist = 0.0

    if null_basis is not None:
        for k in range(null_basis.shape[1]):
            direction = null_basis[:, k].to(dtype=dtype)
            for scale in (1e-3, 1e-2, 5e-2, 0.1, 0.2):
                param_try = param_ref + scale * direction
                dem_try = dem_from_learn_param(
                    g2d, param_try, merge_ops=merge_ops, stim_dem=stim_dem, num_dem=num_dem
                )
                err_try = float((dem_try - target_dem).abs().max().cpu())
                gdist = float((param_try - param_ref).abs().max().cpu())
                if err_try < best_alt_dem_err and gdist > gate_atol:
                    best_alt_dem_err = err_try
                    best_alt_gate_dist = gdist
                    best_alt_param = param_try.clone()

    # --- 3) Second fit: start from null perturbation, repulse away from param_ref ---
    if verbose:
        print("\n[3] Optimized second preimage (repulse from gate_param)")
    if null_basis is not None and null_basis.shape[1] > 0:
        init_b = param_ref + 0.05 * null_basis[:, 0].to(dtype=dtype)
    else:
        gen = torch.Generator().manual_seed(seed)
        init_b = param_ref + 0.25 * torch.randn(num_slots, generator=gen, dtype=dtype)

    param_b, dem_b, err_b = fit_learn_param_to_dem(
        g2d,
        target_dem,
        merge_ops=merge_ops,
        stim_dem=stim_dem,
        num_dem=num_dem,
        init_param=init_b,
        repulse_param=param_ref,
        repulse_weight=5.0,
        steps=opt_steps,
        lr=0.08,
    )
    p_b = g2d.gate_probs(param_b).detach()

    gate_dist = float((param_ref - param_b).abs().max().cpu())
    dem_agree = float((dem_a - dem_b).abs().max().cpu())
    match_a = err_a <= dem_atol
    match_b = err_b <= dem_atol
    different = gate_dist > gate_atol

    null_certified = best_alt_param is not None and best_alt_dem_err <= dem_atol
    locally_injective = nullity == 0
    distinct_second = match_a and match_b and different

    result = SharingCheckResult(
        param_sharing=param_sharing,
        num_slots=num_slots,
        num_elementary=meta["num_elementary"],
        num_dem=num_dem,
        num_gate_prob_slots=n_gate,
        dem_alignment=dem_align,
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
        f"{'sharing':<14} {'#learn':>6} {'|g2d-stim|':>11} {'|g2d-ref|':>11} "
        f"{'rank':>5} {'null':>5} {'DEM':>4} "
        f"{'local inj.':>10} {'null walk':>10} {'2nd preim.':>11}  verdict"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        d = r.dem_alignment
        dem_ok = "ok" if d.matches_stim else "FAIL"
        print(
            f"{r.param_sharing:<14} {r.num_slots:>6} {d.max_abs_vs_stim:11.3e} {d.max_abs_vs_ref:11.3e} "
            f"{r.rank:5d} {r.nullity:5d} {dem_ok:>4} "
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


@dataclass(frozen=True)
class NullityReport:
    label: str
    noise_model: str
    num_learnable: int
    num_dem: int
    rank: int
    nullity: int
    learnable_keys: tuple


def measure_nullity(
    circuit: stim.Circuit,
    *,
    param_sharing: ParamSharing = "time_shared_dep2",
    verbose: bool = False,
) -> NullityReport:
    """Jacobian rank / nullity w.r.t. learnable ``gate_param`` only."""
    dtype = torch.float64
    meta = compile_circuit(circuit, param_sharing=param_sharing)
    g2d = GateNoiseToDEM(
        circuit, learnable=False, param_sharing=param_sharing, dtype=dtype, device="cpu"
    )
    jac = jacobian_dem_wrt_learn_params(
        g2d,
        merge_ops=meta["merge_ops"],
        stim_dem=meta["stim_dem"],
        num_dem=meta["num_dem"],
    )
    rank, _ = null_space_basis(jac)
    num_learn = g2d.num_slots
    nullity = num_learn - rank
    report = NullityReport(
        label="",
        noise_model="",
        num_learnable=num_learn,
        num_dem=meta["num_dem"],
        rank=rank,
        nullity=nullity,
        learnable_keys=tuple(meta["slot_keys"]),
    )
    if verbose:
        print(
            f"  learnable={num_learn}  dem={meta['num_dem']}  "
            f"rank(J)={rank}  nullity={nullity}  J={tuple(jac.shape)}"
        )
        if param_sharing in RESTRICTED_PARAM_SHARINGS:
            if param_sharing == "dep2":
                desc = "independent DEPOLARIZE2 after 2Q"
            elif param_sharing == "time_shared_dep2":
                desc = "time-shared DEPOLARIZE2 after 2Q"
            elif param_sharing == "time_shared_dep12_M_DD":
                desc = "time-shared dep12 + ancilla pre-M + data DD"
            elif param_sharing == "time_shared_dep12_M_DD_I":
                desc = "time-shared dep12_M_DD + idle DEP1 (excl. reset-sync)"
            elif param_sharing == "time_shared_dep12_M_R_DD_I":
                desc = "time-shared dep12_M_DD_I + reset X_ERROR"
            else:
                desc = "time-shared (DEPOLARIZE1 after 1Q + DEPOLARIZE2 after 2Q)"
            print(f"  learnable restricted-noise slots ({desc}): {report.learnable_keys}")
    return report


def run_nullity_sweep_dep2_modes(
    *,
    rep_distance: int = 3,
    rep_rounds: int = 3,
    surface_distance: int = 3,
    surface_rounds: int = 3,
    error_prob: float = 0.001,
    sharings: tuple[ParamSharing, ...] = RESTRICTED_PARAM_SHARINGS,
) -> list[NullityReport]:
    """Nullity for dep1/dep2-restricted sharing modes."""
    noise_models: tuple[NoiseModelKind, ...] = ("si1000", "depolarizing")
    cases: list[tuple[str, stim.Circuit, str]] = []
    for nm in noise_models:
        try:
            cases.append(
                (
                    f"rep d{rep_distance}r{rep_rounds}",
                    build_repetition_circuit(
                        distance=rep_distance,
                        rounds=rep_rounds,
                        error_prob=error_prob,
                        noise_model=nm,
                    ),
                    nm,
                )
            )
            cases.append(
                (
                    f"surface d{surface_distance}r{surface_rounds}",
                    build_surface_code_circuit(
                        distance=surface_distance,
                        rounds=surface_rounds,
                        error_prob=error_prob,
                        noise_model=nm,
                    ),
                    nm,
                )
            )
        except ModuleNotFoundError as exc:
            print(f"[skip] noise_model={nm}: missing dependency ({exc})")

    print("=" * 88)
    print("Nullity sweep — restricted-noise sharing modes")
    print("  time_shared_dep12: DEPOLARIZE1-after-1Q + DEPOLARIZE2-after-2Q (time-shared)")
    print("  time_shared_dep12_M_DD: dep12 + ancilla pre-M X_ERROR + data DD")
    print("  time_shared_dep12_M_DD_I: time_shared_dep12_M_DD + idle DEP1 (excl. reset-sync idle)")
    print("  time_shared_dep12_M_R_DD_I: time_shared_dep12_M_DD_I + reset X_ERROR (X_ERROR_R)")
    print("  dep2: each post-2Q DEPOLARIZE2 is an independent learnable parameter")
    print("  time_shared_dep2: same qubit pair + rate bucket shared across rounds")
    print("=" * 88)
    print(
        f"{'circuit':<22} {'noise':<12} {'sharing':<18} {'#learn':>6} {'rank':>5} "
        f"{'null':>5} {'dem':>5}"
    )
    print("-" * 88)

    reports: list[NullityReport] = []
    for label, circuit, nm in cases:
        for sharing in sharings:
            row = measure_nullity(circuit, param_sharing=sharing)
            row = NullityReport(
                label=label,
                noise_model=nm,
                num_learnable=row.num_learnable,
                num_dem=row.num_dem,
                rank=row.rank,
                nullity=row.nullity,
                learnable_keys=row.learnable_keys,
            )
            reports.append(row)
            print(
                f"{label:<22} {nm:<12} {sharing:<18} {row.num_learnable:6d} "
                f"{row.rank:5d} {row.nullity:5d} {row.num_dem:5d}"
            )
    print()
    return reports


def run_nullity_sweep_two_dep2(
    *,
    rep_distance: int = 3,
    rep_rounds: int = 3,
    surface_distance: int = 3,
    surface_rounds: int = 3,
    error_prob: float = 0.001,
) -> list[NullityReport]:
    """Backward-compatible alias: sweep ``time_shared_dep2`` only."""
    return run_nullity_sweep_dep2_modes(
        rep_distance=rep_distance,
        rep_rounds=rep_rounds,
        surface_distance=surface_distance,
        surface_rounds=surface_rounds,
        error_prob=error_prob,
        sharings=("time_shared_dep2",),
    )


def parse_syc_old_center(center: str) -> tuple[int, int]:
    parts = center.strip().split("_")
    if len(parts) != 2:
        raise ValueError(f"--center must be ROW_COL, got {center!r}")
    return int(parts[0]), int(parts[1])


def syc_old_experiment_dir(
    *,
    basis: str,
    distance: int,
    rounds: int,
    center_row: int,
    center_col: int,
    data_root: Path = SYC_OLD_DATA_ROOT,
) -> Path:
    subdir = (
        f"surface_code_b{basis}_d{distance}_r{rounds:02d}_"
        f"center_{center_row}_{center_col}"
    )
    path = data_root / subdir
    if not path.is_dir():
        raise FileNotFoundError(f"sycamore_old experiment not found: {path}")
    return path


def load_syc_old_circuit(
    *,
    basis: str,
    distance: int = 3,
    rounds: int = 3,
    center_row: int = 5,
    center_col: int = 7,
    data_root: Path = SYC_OLD_DATA_ROOT,
) -> stim.Circuit:
    """Load ``circuit_noisy.stim`` from ``data/sycamore_old`` (hardware calibration)."""
    exp_dir = syc_old_experiment_dir(
        basis=basis,
        distance=distance,
        rounds=rounds,
        center_row=center_row,
        center_col=center_col,
        data_root=data_root,
    )
    circuit_path = exp_dir / "circuit_noisy.stim"
    if not circuit_path.is_file():
        raise FileNotFoundError(f"missing circuit: {circuit_path}")
    return stim.Circuit.from_file(str(circuit_path))


def run_syc_old_checks(
    *,
    distance: int = 3,
    rounds: int = 3,
    center_row: int = 5,
    center_col: int = 7,
    bases: tuple[str, ...] = SYC_OLD_BASES,
    sharings: tuple[ParamSharing, ...] = ALL_PARAM_SHARINGS,
    data_root: Path = SYC_OLD_DATA_ROOT,
    **check_kw,
) -> dict[str, list[SharingCheckResult]]:
    """Nullity / injectivity for X and Z hardware surface-code circuits (d3 r3 default)."""
    all_results: dict[str, list[SharingCheckResult]] = {}
    print("\n" + "=" * 72)
    print(
        f"sycamore_old hardware surface code — d={distance} r={rounds:02d} "
        f"center={center_row}_{center_col}"
    )
    print(f"data_root={data_root}")
    print("=" * 72)

    for basis in bases:
        circuit = load_syc_old_circuit(
            basis=basis,
            distance=distance,
            rounds=rounds,
            center_row=center_row,
            center_col=center_col,
            data_root=data_root,
        )
        label = (
            f"syc_old b{basis} d={distance} r={rounds:02d} "
            f"center={center_row}_{center_col} (circuit_noisy.stim)"
        )
        results = run_check_all_sharings(
            circuit, label=label, sharings=sharings, **check_kw
        )
        all_results[basis] = results

    print_syc_old_basis_summary(all_results)
    return all_results


def print_syc_old_basis_summary(
    by_basis: dict[str, list[SharingCheckResult]],
) -> None:
    """Cross-basis table: nullity and injectivity per param_sharing."""
    print("\n" + "=" * 72)
    print("syc_old summary — X vs Z (all param_sharing modes)")
    print("=" * 72)
    header = (
        f"{'basis':<5} {'sharing':<18} {'#learn':>6} {'|g2d-stim|':>11} "
        f"{'rank':>5} {'null':>5} {'local inj.':>10} {'null walk':>10} "
        f"{'2nd preim.':>11}  verdict"
    )
    print(header)
    print("-" * len(header))
    for basis in sorted(by_basis):
        for r in by_basis[basis]:
            d = r.dem_alignment
            print(
                f"{basis:<5} {r.param_sharing:<18} {r.num_slots:>6} "
                f"{d.max_abs_vs_stim:11.3e} {r.rank:5d} {r.nullity:5d} "
                f"{'yes' if r.locally_injective else 'no':>10} "
                f"{'yes' if r.null_certified else 'no':>10} "
                f"{'yes' if r.distinct_second_preimage else 'no':>11}  "
                f"{_verdict_label(r)}"
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
    ap = argparse.ArgumentParser(
        description="g2dem DEM vs Stim alignment + gate→DEM injectivity (Jacobian rank)."
    )
    ap.add_argument(
        "--code",
        choices=("rep", "surface", "both", "syc_old"),
        default="both",
        help=(
            "repetition (--distance/--rounds), surface (--surface-distance/--surface-rounds), "
            "both, or sycamore_old hardware (X+Z basis)"
        ),
    )
    ap.add_argument(
        "--center",
        default="5_7",
        help="syc_old only: data-qubit center as ROW_COL (default 5_7)",
    )
    ap.add_argument(
        "--syc-distance",
        type=int,
        default=3,
        help="syc_old only: code distance (default 3)",
    )
    ap.add_argument(
        "--syc-rounds",
        type=int,
        default=3,
        help="syc_old only: measurement rounds (default 3)",
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
        "--noise-model",
        type=str,
        default="si1000",
        choices=("depolarizing", "si1000"),
        help="depolarizing (Stim generated) or si1000 (tqec on ideal circuit)",
    )
    ap.add_argument(
        "--dem-target",
        choices=("stim", "circuit", "random"),
        default="stim",
        help="target DEM: Stim ref_dem, g2dem(p_ref), or random gate pushforward",
    )
    ap.add_argument("--p-min", type=float, default=1e-6)
    ap.add_argument("--p-max", type=float, default=0.49)
    ap.add_argument(
        "--dem-atol",
        type=float,
        default=1e-12,
        help="max |g2d - stim_ref| / injectivity DEM match tolerance",
    )
    ap.add_argument("--gate-atol", type=float, default=1e-8, help="min |p_A-p_B| to call gate vectors distinct")
    ap.add_argument("--opt-steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--nullity-sweep",
        action="store_true",
        help="print nullity for time_shared_dep12 / dep2 / time_shared_dep2 across rep + surface circuits",
    )
    ap.add_argument(
        "--param-sharing",
        type=str,
        default="all",
        choices=("all", *ALL_PARAM_SHARINGS),
        help="sharing mode to check, or all (default: all)",
    )
    args = ap.parse_args()

    if args.nullity_sweep:
        run_nullity_sweep_dep2_modes(
            rep_distance=args.distance,
            rep_rounds=args.rounds,
            surface_distance=args.surface_distance,
            surface_rounds=args.surface_rounds,
            error_prob=args.error_prob,
        )
        return

    if args.code == "syc_old":
        center_row, center_col = parse_syc_old_center(args.center)
        if args.param_sharing == "all":
            sharings_syc = ALL_PARAM_SHARINGS
        else:
            sharings_syc = (args.param_sharing,)  # type: ignore[assignment]
        run_syc_old_checks(
            distance=args.syc_distance,
            rounds=args.syc_rounds,
            center_row=center_row,
            center_col=center_col,
            sharings=sharings_syc,
            dem_target=args.dem_target,
            p_min=args.p_min,
            p_max=args.p_max,
            dem_atol=args.dem_atol,
            gate_atol=args.gate_atol,
            opt_steps=args.opt_steps,
            seed=args.seed,
        )
        return

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

    nm: NoiseModelKind = args.noise_model

    if args.code in ("rep", "both"):
        run_for_circuit(
            build_repetition_circuit(
                distance=args.distance,
                rounds=args.rounds,
                error_prob=args.error_prob,
                noise_model=nm,
            ),
            label=f"repetition_code:memory d={args.distance} r={args.rounds} ({nm})",
        )

    if args.code in ("surface", "both"):
        sd, sr = args.surface_distance, args.surface_rounds
        run_for_circuit(
            build_surface_code_circuit(
                distance=sd, rounds=sr, error_prob=args.error_prob, noise_model=nm
            ),
            label=f"surface_code:rotated_memory_z d={sd} r={sr} ({nm})",
        )


if __name__ == "__main__":
    main()
