"""
Perturb non-trainable gate probabilities and compare DEM drift.

For a trained sycamore_old checkpoint, this script:
- fixes the gate probabilities corresponding to learnable slots (e.g. dep2);
- perturbs all other elementary gate probabilities by multiplicative noise;
- runs g2dem replay to get perturbed DEM probabilities;
- reports relative-error stats against the unperturbed checkpoint DEM.

Run from repo root::

    python gate/perturb_check.py --center 3_5 --basis X --param-sharing dep2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "gate" / "log" / "sur" / "syc_old" / "checkpoint"
DATA_ROOT = REPO_ROOT / "data" / "sycamore_old"
LOG_DIR = REPO_ROOT / "gate" / "log" / "sur" / "syc_old"

if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from gate.tn_syc_old import parse_center, resolve_experiment_dir, load_noisy_circuit  # noqa: E402
from src.g2dem import (  # noqa: E402
    GateNoiseToDEM,
    ParamSharing,
    compile_circuit,
    dem_from_gate_probs,
    solve_dep2_gate_probs_from_dem,
)


@dataclass(frozen=True)
class RatioStats:
    ratio: float
    n_samples: int
    max_rel_mean: float
    max_rel_std: float
    max_rel_median: float
    max_rel_p95: float
    max_rel_worst: float
    mean_rel_mean: float
    mean_rel_std: float
    mean_rel_median: float
    mean_rel_p95: float
    mean_rel_worst: float


def checkpoint_stem(
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
        f"c{center_row}_{center_col}_{param_sharing}"
    )


def default_log_path(*, distance: int, rounds: int, center_row: int, center_col: int) -> Path:
    return LOG_DIR / f"d{distance}r{rounds:02d}_c{center_row}_{center_col}_perturb_check.txt"


def resolve_checkpoint_path(
    *,
    basis: str,
    distance: int,
    rounds: int,
    center_row: int,
    center_col: int,
    param_sharing: str,
) -> tuple[Path, ParamSharing]:
    if param_sharing in ("dep2", "time_shared_dep2"):
        sharing: ParamSharing = param_sharing  # type: ignore[assignment]
        path = CHECKPOINT_DIR / (
            checkpoint_stem(
                basis=basis,
                distance=distance,
                rounds=rounds,
                center_row=center_row,
                center_col=center_col,
                param_sharing=sharing,
            )
            + ".pt"
        )
        return path, sharing

    # auto mode: prefer dep2, then time_shared_dep2
    for sharing in ("dep2", "time_shared_dep2"):
        path = CHECKPOINT_DIR / (
            checkpoint_stem(
                basis=basis,
                distance=distance,
                rounds=rounds,
                center_row=center_row,
                center_col=center_col,
                param_sharing=sharing,
            )
            + ".pt"
        )
        if path.is_file():
            return path, sharing  # type: ignore[return-value]
    # if none exists, return preferred one for clear error message
    return (
        CHECKPOINT_DIR
        / (
            checkpoint_stem(
                basis=basis,
                distance=distance,
                rounds=rounds,
                center_row=center_row,
                center_col=center_col,
                param_sharing="dep2",
            )
            + ".pt"
        ),
        "dep2",
    )


def parse_ratios(text: str) -> list[float]:
    vals: list[float] = []
    for chunk in text.split(","):
        v = float(chunk.strip())
        if v < 0:
            raise ValueError(f"ratio must be >=0, got {v}")
        vals.append(v)
    if not vals:
        raise ValueError("empty ratio list")
    return vals


def relative_error_stats(pred: np.ndarray, ref: np.ndarray, eps: float) -> tuple[float, float]:
    rel = np.abs(pred - ref) / np.maximum(np.abs(ref), eps)
    return float(rel.max()), float(rel.mean())


def summarize_ratio(ratio: float, max_rel: np.ndarray, mean_rel: np.ndarray) -> RatioStats:
    return RatioStats(
        ratio=ratio,
        n_samples=int(max_rel.size),
        max_rel_mean=float(np.mean(max_rel)),
        max_rel_std=float(np.std(max_rel)),
        max_rel_median=float(np.median(max_rel)),
        max_rel_p95=float(np.percentile(max_rel, 95)),
        max_rel_worst=float(np.max(max_rel)),
        mean_rel_mean=float(np.mean(mean_rel)),
        mean_rel_std=float(np.std(mean_rel)),
        mean_rel_median=float(np.median(mean_rel)),
        mean_rel_p95=float(np.percentile(mean_rel, 95)),
        mean_rel_worst=float(np.max(mean_rel)),
    )


def run_perturbation(
    *,
    gate_probs_ref: np.ndarray,
    dem_ref: np.ndarray,
    non_train_idx: np.ndarray,
    ratios: Sequence[float],
    n_samples: int,
    seed: int,
    merge_ops,
    stim_dem,
    num_dem: int,
    dem_eps: float,
) -> list[RatioStats]:
    rng = np.random.default_rng(seed)
    out: list[RatioStats] = []

    for ratio in ratios:
        max_rel_vals = np.empty(n_samples, dtype=np.float64)
        mean_rel_vals = np.empty(n_samples, dtype=np.float64)

        for i in range(n_samples):
            perturbed = gate_probs_ref.copy()
            scale = 1.0 + rng.uniform(-ratio, ratio, size=non_train_idx.size)
            perturbed[non_train_idx] *= scale
            np.clip(perturbed, 1e-12, 1.0 - 1e-12, out=perturbed)

            dem_new = (
                dem_from_gate_probs(
                    torch.from_numpy(perturbed),
                    merge_ops=merge_ops,
                    stim_dem=stim_dem,
                    num_dem=num_dem,
                )
                .detach()
                .cpu()
                .numpy()
            )
            max_rel, mean_rel = relative_error_stats(dem_new, dem_ref, dem_eps)
            max_rel_vals[i] = max_rel
            mean_rel_vals[i] = mean_rel

        out.append(summarize_ratio(ratio, max_rel_vals, mean_rel_vals))
    return out


def run_perturbation_on_indices(
    *,
    gate_probs_ref: np.ndarray,
    dem_ref: np.ndarray,
    perturb_idx: np.ndarray,
    ratios: Sequence[float],
    n_samples: int,
    seed: int,
    merge_ops,
    stim_dem,
    num_dem: int,
    dem_eps: float,
) -> list[RatioStats]:
    rng = np.random.default_rng(seed)
    out: list[RatioStats] = []

    for ratio in ratios:
        max_rel_vals = np.empty(n_samples, dtype=np.float64)
        mean_rel_vals = np.empty(n_samples, dtype=np.float64)

        for i in range(n_samples):
            perturbed = gate_probs_ref.copy()
            scale = 1.0 + rng.uniform(-ratio, ratio, size=perturb_idx.size)
            perturbed[perturb_idx] *= scale
            np.clip(perturbed, 1e-12, 1.0 - 1e-12, out=perturbed)

            dem_new = (
                dem_from_gate_probs(
                    torch.from_numpy(perturbed),
                    merge_ops=merge_ops,
                    stim_dem=stim_dem,
                    num_dem=num_dem,
                )
                .detach()
                .cpu()
                .numpy()
            )
            max_rel, mean_rel = relative_error_stats(dem_new, dem_ref, dem_eps)
            max_rel_vals[i] = max_rel
            mean_rel_vals[i] = mean_rel

        out.append(summarize_ratio(ratio, max_rel_vals, mean_rel_vals))
    return out


def run_inverse_dep2_after_non_dep2_perturb(
    *,
    g2d: GateNoiseToDEM,
    gate_probs_ref: np.ndarray,
    dem_ref: np.ndarray,
    dep2_ref: np.ndarray,
    non_train_idx: np.ndarray,
    ratios: Sequence[float],
    n_samples: int,
    seed: int,
    inverse_steps: int,
    inverse_lr: float,
    gate_eps: float,
) -> list[RatioStats]:
    """Perturb non-dep2, then solve dep2 to recover ``dem_ref``; measure dep2 vs reference."""
    rng = np.random.default_rng(seed)
    out: list[RatioStats] = []

    for ratio in ratios:
        max_rel_vals = np.empty(n_samples, dtype=np.float64)
        mean_rel_vals = np.empty(n_samples, dtype=np.float64)

        for i in range(n_samples):
            perturbed = gate_probs_ref.copy()
            scale = 1.0 + rng.uniform(-ratio, ratio, size=non_train_idx.size)
            perturbed[non_train_idx] *= scale
            np.clip(perturbed, 1e-12, 1.0 - 1e-12, out=perturbed)

            solved = solve_dep2_gate_probs_from_dem(
                g2d,
                dem_ref,
                perturbed,
                steps=inverse_steps,
                lr=inverse_lr,
            )
            dep2_solved = solved.dep2_elementary_probs.detach().cpu().numpy()
            max_rel, mean_rel = relative_error_stats(dep2_solved, dep2_ref, gate_eps)
            max_rel_vals[i] = max_rel
            mean_rel_vals[i] = mean_rel

        out.append(summarize_ratio(ratio, max_rel_vals, mean_rel_vals))
    return out


def print_summary(
    *,
    ckpt_path: Path,
    basis: str,
    distance: int,
    rounds: int,
    center_row: int,
    center_col: int,
    param_sharing: ParamSharing,
    n_trainable: int,
    n_non_trainable: int,
    stats: Sequence[RatioStats],
) -> None:
    print(
        render_summary(
            ckpt_path=ckpt_path,
            basis=basis,
            distance=distance,
            rounds=rounds,
            center_row=center_row,
            center_col=center_col,
            param_sharing=param_sharing,
            n_trainable=n_trainable,
            n_non_trainable=n_non_trainable,
            stats=stats,
        )
    )


def render_summary(
    *,
    ckpt_path: Path,
    basis: str,
    distance: int,
    rounds: int,
    center_row: int,
    center_col: int,
    param_sharing: ParamSharing,
    n_trainable: int,
    n_non_trainable: int,
    stats: Sequence[RatioStats],
    title: str = "Perturbation check: non-trainable gate probs -> DEM drift",
    subtitle: str | None = None,
) -> str:
    lines: list[str] = []
    lines.append("=" * 108)
    lines.append(title)
    lines.append("=" * 108)
    lines.append(
        f"checkpoint={ckpt_path} | basis={basis} d={distance} r={rounds:02d} "
        f"center={center_row}_{center_col} sharing={param_sharing}"
    )
    if subtitle:
        lines.append(subtitle)
    else:
        lines.append(
            f"fixed trainable dep2 elementary sites={n_trainable} | "
            f"perturbed non-trainable elementary sites={n_non_trainable}"
        )
    lines.append("-" * 108)
    lines.append(
        f"{'ratio':>8} {'samples':>8} "
        f"{'max_rel(mean±std/med/p95/worst)':>45} "
        f"{'mean_rel(mean±std/med/p95/worst)':>45}"
    )
    lines.append("-" * 108)
    for s in stats:
        lines.append(
            f"{s.ratio:8.2f} {s.n_samples:8d} "
            f"{s.max_rel_mean:9.3e}±{s.max_rel_std:9.3e}/{s.max_rel_median:9.3e}/{s.max_rel_p95:9.3e}/{s.max_rel_worst:9.3e} "
            f"{s.mean_rel_mean:9.3e}±{s.mean_rel_std:9.3e}/{s.mean_rel_median:9.3e}/{s.mean_rel_p95:9.3e}/{s.mean_rel_worst:9.3e}"
        )
    lines.append("")
    return "\n".join(lines)


def write_summary_log(log_path: Path, summary_text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(summary_text)
        if not summary_text.endswith("\n"):
            f.write("\n")
        f.write("\n")


def render_comparison_summary(
    *,
    title: str,
    stats_a: Sequence[RatioStats],
    stats_b: Sequence[RatioStats],
    label_a: str,
    label_b: str,
) -> str:
    if len(stats_a) != len(stats_b):
        raise ValueError("comparison stats length mismatch")
    lines: list[str] = []
    lines.append("=" * 108)
    lines.append(title)
    lines.append("=" * 108)
    lines.append(
        f"{'ratio':>8}  {'mean_rel_mean(' + label_a + ')':>22}  "
        f"{'mean_rel_mean(' + label_b + ')':>22}  {'A/B':>10}"
    )
    lines.append("-" * 108)
    for a, b in zip(stats_a, stats_b):
        denom = b.mean_rel_mean if b.mean_rel_mean > 0 else np.nan
        ratio = a.mean_rel_mean / denom if np.isfinite(denom) else np.nan
        lines.append(
            f"{a.ratio:8.2f}  {a.mean_rel_mean:22.3e}  {b.mean_rel_mean:22.3e}  {ratio:10.3f}"
        )
    lines.append("")
    return "\n".join(lines)


def print_summary_legacy(
    *,
    ckpt_path: Path,
    basis: str,
    distance: int,
    rounds: int,
    center_row: int,
    center_col: int,
    param_sharing: ParamSharing,
    n_trainable: int,
    n_non_trainable: int,
    stats: Sequence[RatioStats],
) -> None:
    print("=" * 108)
    print("Perturbation check: non-trainable gate probs -> DEM drift")
    print("=" * 108)
    print(
        f"checkpoint={ckpt_path} | basis={basis} d={distance} r={rounds:02d} "
        f"center={center_row}_{center_col} sharing={param_sharing}"
    )
    print(
        f"fixed trainable dep2 elementary sites={n_trainable} | "
        f"perturbed non-trainable elementary sites={n_non_trainable}"
    )
    print("-" * 108)
    print(
        f"{'ratio':>8} {'samples':>8} "
        f"{'max_rel(mean/med/p95/worst)':>39} "
        f"{'mean_rel(mean/med/p95/worst)':>39}"
    )
    print("-" * 108)
    for s in stats:
        print(
            f"{s.ratio:8.2f} {s.n_samples:8d} "
            f"{s.max_rel_mean:9.3e}/{s.max_rel_median:9.3e}/{s.max_rel_p95:9.3e}/{s.max_rel_worst:9.3e} "
            f"{s.mean_rel_mean:9.3e}/{s.mean_rel_median:9.3e}/{s.mean_rel_p95:9.3e}/{s.mean_rel_worst:9.3e}"
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basis", choices=("X", "Z"), default="X")
    ap.add_argument("--distance", "-d", type=int, default=3)
    ap.add_argument("--rounds", "-r", type=int, default=3)
    ap.add_argument("--center", type=str, default="3_5", help="ROW_COL")
    ap.add_argument(
        "--param-sharing",
        type=str,
        default="auto",
        choices=("auto", "dep2", "time_shared_dep2"),
        help="checkpoint/sharing mode (auto: prefer dep2, fallback time_shared_dep2)",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="optional explicit checkpoint path (.pt); overrides auto path",
    )
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument(
        "--ratios",
        type=str,
        default="0.01,0.1,0.5,0.8,1.0,2.0",
        help="comma-separated multiplicative perturb ranges",
    )
    ap.add_argument("--seed", type=int, default=75328)
    ap.add_argument("--dem-rel-eps", type=float, default=1e-12)
    ap.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="output log file (default: gate/log/sur/syc_old/d{d}r{r}_c{row}_{col}_perturb_check.txt)",
    )
    ap.add_argument(
        "--both-directions",
        action="store_true",
        help="also run reverse perturbation: perturb dep2 while fixing non-dep2",
    )
    ap.add_argument(
        "--skip-inverse-dep2",
        action="store_true",
        help="skip inverse dep2 solve after non-dep2 perturbation",
    )
    ap.add_argument("--inverse-steps", type=int, default=400)
    ap.add_argument("--inverse-lr", type=float, default=0.08)
    ap.add_argument(
        "--gate-rel-eps",
        type=float,
        default=1e-12,
        help="epsilon for dep2 gate-probability relative errors",
    )
    args = ap.parse_args()

    center_row, center_col = parse_center(args.center)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if args.param_sharing == "auto":
            raise ValueError("when --checkpoint is set, please also set --param-sharing")
        param_sharing: ParamSharing = args.param_sharing  # type: ignore[assignment]
    else:
        ckpt_path, param_sharing = resolve_checkpoint_path(
            basis=args.basis,
            distance=args.distance,
            rounds=args.rounds,
            center_row=center_row,
            center_col=center_col,
            param_sharing=args.param_sharing,
        )
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "gate_probs" not in ckpt:
        raise KeyError(f"'gate_probs' missing in checkpoint: {ckpt_path}")
    gate_probs_ref = np.asarray(ckpt["gate_probs"], dtype=np.float64)

    exp_dir = resolve_experiment_dir(
        data_root=DATA_ROOT,
        basis=args.basis,
        distance=args.distance,
        rounds=args.rounds,
        center_row=center_row,
        center_col=center_col,
    )
    circuit = load_noisy_circuit(exp_dir)
    meta = compile_circuit(circuit, param_sharing=param_sharing)
    elem_to_learn = np.asarray(meta["elem_to_learn"], dtype=np.int64)
    if gate_probs_ref.shape[0] != elem_to_learn.shape[0]:
        raise ValueError(
            f"gate_probs len={gate_probs_ref.shape[0]} != num_elementary={elem_to_learn.shape[0]}"
        )

    trainable_mask = elem_to_learn >= 0
    non_train_idx = np.where(~trainable_mask)[0]
    if non_train_idx.size == 0:
        raise RuntimeError("no non-trainable probabilities to perturb under this sharing mode")

    dem_ref = (
        dem_from_gate_probs(
            torch.from_numpy(gate_probs_ref),
            merge_ops=meta["merge_ops"],
            stim_dem=meta["stim_dem"],
            num_dem=meta["num_dem"],
        )
        .detach()
        .cpu()
        .numpy()
    )

    train_idx = np.where(trainable_mask)[0]
    dep2_ref = gate_probs_ref[train_idx]
    g2d = GateNoiseToDEM(
        circuit,
        learnable=False,
        param_sharing=param_sharing,
        dtype=torch.float64,
        device="cpu",
    )

    stats_non_dep2 = run_perturbation_on_indices(
        gate_probs_ref=gate_probs_ref,
        dem_ref=dem_ref,
        perturb_idx=non_train_idx,
        ratios=parse_ratios(args.ratios),
        n_samples=args.n_samples,
        seed=args.seed,
        merge_ops=meta["merge_ops"],
        stim_dem=meta["stim_dem"],
        num_dem=meta["num_dem"],
        dem_eps=args.dem_rel_eps,
    )

    summary_non_dep2 = render_summary(
        ckpt_path=ckpt_path,
        basis=args.basis,
        distance=args.distance,
        rounds=args.rounds,
        center_row=center_row,
        center_col=center_col,
        param_sharing=param_sharing,
        n_trainable=int(trainable_mask.sum()),
        n_non_trainable=int(non_train_idx.size),
        stats=stats_non_dep2,
    )
    print(summary_non_dep2)
    log_path = Path(args.log_path) if args.log_path else default_log_path(
        distance=args.distance,
        rounds=args.rounds,
        center_row=center_row,
        center_col=center_col,
    )
    write_summary_log(log_path, summary_non_dep2)

    if not args.skip_inverse_dep2:
        stats_inverse_dep2 = run_inverse_dep2_after_non_dep2_perturb(
            g2d=g2d,
            gate_probs_ref=gate_probs_ref,
            dem_ref=dem_ref,
            dep2_ref=dep2_ref,
            non_train_idx=non_train_idx,
            ratios=parse_ratios(args.ratios),
            n_samples=args.n_samples,
            seed=args.seed + 100,
            inverse_steps=args.inverse_steps,
            inverse_lr=args.inverse_lr,
            gate_eps=args.gate_rel_eps,
        )
        summary_inverse = render_summary(
            ckpt_path=ckpt_path,
            basis=args.basis,
            distance=args.distance,
            rounds=args.rounds,
            center_row=center_row,
            center_col=center_col,
            param_sharing=param_sharing,
            n_trainable=int(trainable_mask.sum()),
            n_non_trainable=int(non_train_idx.size),
            stats=stats_inverse_dep2,
            title="Inverse dep2 solve after non-dep2 perturb (target DEM = checkpoint DEM)",
            subtitle=(
                f"perturb non-dep2 ({non_train_idx.size} sites), solve dep2 ({train_idx.size} sites) "
                f"| compare solved dep2 vs reference dep2 gate probs"
            ),
        )
        print(summary_inverse)
        write_summary_log(log_path, summary_inverse)

    if args.both_directions:
        stats_dep2 = run_perturbation_on_indices(
            gate_probs_ref=gate_probs_ref,
            dem_ref=dem_ref,
            perturb_idx=train_idx,
            ratios=parse_ratios(args.ratios),
            n_samples=args.n_samples,
            seed=args.seed + 1,
            merge_ops=meta["merge_ops"],
            stim_dem=meta["stim_dem"],
            num_dem=meta["num_dem"],
            dem_eps=args.dem_rel_eps,
        )
        summary_dep2 = render_summary(
            ckpt_path=ckpt_path,
            basis=args.basis,
            distance=args.distance,
            rounds=args.rounds,
            center_row=center_row,
            center_col=center_col,
            param_sharing=param_sharing,
            n_trainable=int(non_train_idx.size),
            n_non_trainable=int(train_idx.size),
            stats=stats_dep2,
        ).replace(
            "fixed trainable dep2 elementary sites=", "fixed non-dep2 elementary sites="
        ).replace(
            "perturbed non-trainable elementary sites=", "perturbed dep2 elementary sites="
        )
        comp = render_comparison_summary(
            title="Direction comparison (mean_rel): non-dep2 perturb vs dep2 perturb",
            stats_a=stats_non_dep2,
            stats_b=stats_dep2,
            label_a="non_dep2",
            label_b="dep2",
        )
        print(summary_dep2)
        print(comp)
        write_summary_log(log_path, summary_dep2)
        write_summary_log(log_path, comp)

    print(f"[log] wrote {log_path}")


if __name__ == "__main__":
    main()

