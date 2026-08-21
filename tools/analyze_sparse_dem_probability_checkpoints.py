#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMMON_KEYS = [
    "oer",
    "probs",
    "probabilities",
    "error_probs",
    "error_rates",
    "er",
    "theta",
    "params",
    "state_dict",
    "model_state_dict",
]


def parse_epochs(text: str) -> list[int]:
    epochs = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = item.split("-", 1)
            epochs.extend(range(int(lo), int(hi) + 1))
        else:
            epochs.append(int(item))
    return sorted(dict.fromkeys(epochs))


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def logit(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value)
        except Exception:
            return None
    if np.isscalar(value):
        return None
    return None


def flatten_numeric_array(value: Any) -> np.ndarray | None:
    arr = to_numpy(value)
    if arr is None:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def find_numeric_vector(obj: Any, path: str = "") -> tuple[np.ndarray, str] | None:
    arr = flatten_numeric_array(obj)
    if arr is not None and arr.size > 1:
        return arr, path or "<root>"

    if isinstance(obj, dict):
        for key in COMMON_KEYS:
            if key in obj:
                found = find_numeric_vector(obj[key], f"{path}.{key}" if path else key)
                if found is not None:
                    return found
        candidates = []
        for key, value in obj.items():
            found = find_numeric_vector(value, f"{path}.{key}" if path else str(key))
            if found is not None:
                candidates.append(found)
        if candidates:
            candidates.sort(key=lambda item: item[0].size, reverse=True)
            return candidates[0]
    return None


def load_checkpoint_object(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    if suffix in {".pt", ".pth"}:
        import torch

        return torch.load(path, map_location="cpu", weights_only=False)
    if suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported checkpoint suffix: {path}")


def values_to_probability(values: np.ndarray, mode: str, eps: float) -> tuple[np.ndarray, str]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if mode == "probability":
        p = values
        inferred = "probability"
    elif mode == "logit":
        p = sigmoid(values)
        inferred = "logit"
    elif mode == "auto":
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("No finite values in checkpoint vector.")
        if np.nanmin(finite) >= 0.0 and np.nanmax(finite) <= 1.0:
            p = values
            inferred = "probability"
        else:
            p = sigmoid(values)
            inferred = "logit"
    else:
        raise ValueError(f"Unknown checkpoint value mode: {mode}")
    return np.clip(p, eps, 1.0 - eps), inferred


def parse_epoch_from_name(path: Path) -> int | None:
    patterns = [
        r"(?:^|[_-])epoch[_-]?(\d+)(?:[_\.-]|$)",
        r"(?:^|[_-])ep[_-]?(\d+)(?:[_\.-]|$)",
        r"(?:^|[_-])(\d+)(?:[_\.-]|$)",
    ]
    name = path.name
    for pattern in patterns:
        m = re.search(pattern, name, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def find_checkpoint_files(checkpoint_dir: Path, requested_epochs: list[int]) -> dict[int, Path]:
    suffixes = {".npy", ".npz", ".pt", ".pth", ".pkl"}
    files = [p for p in checkpoint_dir.iterdir() if p.is_file() and p.suffix.lower() in suffixes]
    by_epoch: dict[int, list[Path]] = {}
    for path in files:
        epoch = parse_epoch_from_name(path)
        if epoch is None:
            continue
        by_epoch.setdefault(epoch, []).append(path)

    selected = {}
    for epoch in requested_epochs:
        candidates = by_epoch.get(epoch, [])
        if not candidates:
            continue
        candidates.sort(key=lambda p: (len(p.name), p.name))
        selected[epoch] = candidates[0]
    return selected


def load_probability_matrix(
    checkpoint_dir: Path,
    requested_epochs: list[int],
    mode: str,
    eps: float,
) -> tuple[list[int], np.ndarray, pd.DataFrame]:
    found_files = find_checkpoint_files(checkpoint_dir, requested_epochs)
    rows = []
    vectors = []
    loaded_epochs = []
    expected_len = None

    for epoch in requested_epochs:
        path = found_files.get(epoch)
        if path is None:
            rows.append({"epoch": epoch, "status": "missing", "path": "", "source_key": "", "inferred_values": ""})
            continue
        obj = load_checkpoint_object(path)
        found = find_numeric_vector(obj)
        if found is None:
            rows.append({"epoch": epoch, "status": "no_numeric_vector", "path": str(path), "source_key": "", "inferred_values": ""})
            continue
        values, source_key = found
        p, inferred = values_to_probability(values, mode, eps)
        if expected_len is None:
            expected_len = len(p)
        elif len(p) != expected_len:
            rows.append(
                {
                    "epoch": epoch,
                    "status": f"length_mismatch_expected_{expected_len}_got_{len(p)}",
                    "path": str(path),
                    "source_key": source_key,
                    "inferred_values": inferred,
                }
            )
            continue
        loaded_epochs.append(epoch)
        vectors.append(p)
        rows.append({"epoch": epoch, "status": "loaded", "path": str(path), "source_key": source_key, "inferred_values": inferred})

    if not vectors:
        raise RuntimeError(f"No checkpoints could be loaded from {checkpoint_dir}")
    return loaded_epochs, np.vstack(vectors), pd.DataFrame(rows)


def unknown_edge_metadata(num_edges: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "edge_id": np.arange(num_edges),
            "detector_support_size": np.nan,
            "detectors": "unknown",
            "observables": "unknown",
            "has_observable": False,
            "min_detector_time": np.nan,
            "max_detector_time": np.nan,
            "detector_coords_summary": "unknown",
            "edge_type_or_group": "unknown",
            "time_region": "unknown",
            "spatial_region": "unknown",
            "boundary_seam_bulk": "unknown",
        }
    )


def load_json_metadata(path: Path | None, num_edges: int) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        print(f"[warn] metadata file missing: {path}")
        return None
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        print(f"[warn] failed to read metadata JSON {path}: {exc}")
        return None

    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        for key in ["edges", "error_mechanisms", "metadata"]:
            if isinstance(data.get(key), list):
                df = pd.DataFrame(data[key])
                break
        else:
            return None
    else:
        return None
    if "edge_id" not in df.columns:
        df.insert(0, "edge_id", np.arange(len(df)))
    return normalize_metadata_df(df, num_edges)


def dem_target_to_text(target: Any) -> str:
    if target.is_relative_detector_id():
        return f"D{target.val}"
    if target.is_logical_observable_id():
        return f"L{target.val}"
    if target.is_separator():
        return "^"
    return str(target)


def parse_stim_dem_metadata(path: Path | None, num_edges: int) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        print(f"[warn] DEM file missing: {path}")
        return None
    try:
        import stim
    except Exception as exc:
        print(f"[warn] stim is unavailable; cannot parse DEM metadata: {exc}")
        return None
    try:
        dem = stim.DetectorErrorModel.from_file(str(path))
    except Exception as exc:
        print(f"[warn] failed to parse DEM {path}: {exc}")
        return None

    detector_coords = {}
    errors = []
    for inst in dem.flattened():
        if inst.type == "detector":
            args = inst.args_copy()
            det_id = None
            for target in inst.targets_copy():
                if target.is_relative_detector_id():
                    det_id = int(target.val)
                    break
            if det_id is not None:
                detector_coords[det_id] = tuple(float(x) for x in args)
        elif inst.type == "error":
            errors.append(inst)

    rows = []
    all_times = []
    for coords in detector_coords.values():
        if len(coords) >= 3:
            all_times.append(coords[2])
    max_time = max(all_times) if all_times else None

    for edge_id in range(num_edges):
        if edge_id >= len(errors):
            rows.append({"edge_id": edge_id})
            continue
        inst = errors[edge_id]
        dets = []
        obs = []
        separators = 0
        for target in inst.targets_copy():
            if target.is_relative_detector_id():
                dets.append(int(target.val))
            elif target.is_logical_observable_id():
                obs.append(int(target.val))
            elif target.is_separator():
                separators += 1
        coords = [detector_coords.get(det) for det in dets if det in detector_coords]
        times = [coord[2] for coord in coords if len(coord) >= 3]
        xs = [coord[0] for coord in coords if len(coord) >= 1]
        ys = [coord[1] for coord in coords if len(coord) >= 2]
        min_t = min(times) if times else np.nan
        max_t = max(times) if times else np.nan
        summary = "unknown"
        if coords:
            summary = (
                f"x=[{min(xs):.3g},{max(xs):.3g}], "
                f"y=[{min(ys):.3g},{max(ys):.3g}], "
                f"t=[{min_t:.3g},{max_t:.3g}]"
            )
        if not times or max_time is None:
            time_region = "unknown"
        elif max_t <= 1:
            time_region = "init"
        elif min_t >= max_time - 1:
            time_region = "final"
        else:
            time_region = "bulk"
        if xs and ys:
            spatial_region = f"xmid={np.mean(xs):.3g},ymid={np.mean(ys):.3g}"
            min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
            is_boundary = any(
                abs(v) < 1e-9 for v in [min_x, min_y]
            ) or (max(xs) == min(xs)) or (max(ys) == min(ys))
            boundary_seam_bulk = "boundary" if is_boundary else "bulk"
        else:
            spatial_region = "unknown"
            boundary_seam_bulk = "unknown"
        if obs:
            group = "observable"
        elif separators:
            group = f"correlated_group_{separators + 1}_components"
        else:
            group = f"{len(dets)}det"
        rows.append(
            {
                "edge_id": edge_id,
                "detector_support_size": len(dets),
                "detectors": " ".join(f"D{x}" for x in dets) if dets else "",
                "observables": " ".join(f"L{x}" for x in obs) if obs else "",
                "has_observable": bool(obs),
                "min_detector_time": min_t,
                "max_detector_time": max_t,
                "detector_coords_summary": summary,
                "edge_type_or_group": group,
                "time_region": time_region,
                "spatial_region": spatial_region,
                "boundary_seam_bulk": boundary_seam_bulk,
            }
        )
    return normalize_metadata_df(pd.DataFrame(rows), num_edges)


def normalize_metadata_df(df: pd.DataFrame, num_edges: int) -> pd.DataFrame:
    base = unknown_edge_metadata(num_edges).drop(columns=["edge_id"])
    out = pd.DataFrame({"edge_id": np.arange(num_edges)}).join(base)
    if "edge_id" not in df.columns:
        df = df.copy()
        df["edge_id"] = np.arange(len(df))
    df = df.copy()
    df["edge_id"] = df["edge_id"].astype(int)
    df = df.drop_duplicates("edge_id", keep="first")
    for col in df.columns:
        if col == "edge_id":
            continue
        mapped = out["edge_id"].map(df.set_index("edge_id")[col])
        mask = mapped.notna()
        out.loc[mask, col] = mapped.loc[mask]
    return out


def load_edge_metadata(dem: Path | None, metadata: Path | None, num_edges: int) -> pd.DataFrame:
    df = load_json_metadata(metadata, num_edges)
    if df is not None:
        print(f"[info] loaded edge metadata from {metadata}")
        return df
    df = parse_stim_dem_metadata(dem, num_edges)
    if df is not None:
        print(f"[info] parsed edge metadata from DEM {dem}")
        return df
    print("[warn] no usable DEM/metadata provided; metadata columns will be unknown.")
    return unknown_edge_metadata(num_edges)


def require_epoch(epoch_to_idx: dict[int, int], epoch: int, label: str):
    if epoch not in epoch_to_idx:
        raise RuntimeError(f"Required {label} epoch {epoch} was not loaded. Available epochs: {sorted(epoch_to_idx)}")
    return epoch_to_idx[epoch]


def quantile_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"median": np.nan, "p90": np.nan, "p99": np.nan, "max": np.nan, "mean": np.nan}
    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def save_hist(values: np.ndarray, path: Path, title: str, xlabel: str):
    plt.figure(figsize=(7, 4.5))
    plt.hist(values, bins=80, color="#4c78a8", edgecolor="none")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_scatter(x: np.ndarray, y: np.ndarray, path: Path, title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(x, y, s=5, alpha=0.45, color="#4c78a8")
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def step_summary(epochs: list[int], Z: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in range(1, len(epochs)):
        prev_e, curr_e = epochs[i - 1], epochs[i]
        step = np.abs(Z[i] - Z[i - 1])
        stats = quantile_stats(step)
        rows.append(
            {
                "epoch_prev": prev_e,
                "epoch_curr": curr_e,
                "is_gap": bool(curr_e - prev_e > 1),
                "median_abs_step": stats["median"],
                "p90_abs_step": stats["p90"],
                "p99_abs_step": stats["p99"],
                "max_abs_step": stats["max"],
                "mean_abs_step": stats["mean"],
            }
        )
    return pd.DataFrame(rows)


def plot_step_df(df: pd.DataFrame, path: Path, title: str, omit_gaps: bool):
    plot_df = df.copy()
    if omit_gaps:
        plot_df = plot_df[~plot_df["is_gap"]]
    plt.figure(figsize=(8, 4.5))
    if not plot_df.empty:
        x = plot_df["epoch_curr"].to_numpy()
        for col, color in [("median_abs_step", "#4c78a8"), ("p90_abs_step", "#f58518"), ("p99_abs_step", "#e45756")]:
            plt.plot(x, plot_df[col].to_numpy(), marker="o", label=col, color=color)
    plt.title(title)
    plt.xlabel("current epoch")
    plt.ylabel("abs logit step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x0 = x - x.mean()
    denom = np.sum(x0 * x0)
    if denom == 0:
        return np.nan
    return float(np.sum(x0 * (y - y.mean())) / denom)


def late_status(late_drift: np.ndarray) -> np.ndarray:
    out = np.full(late_drift.shape, "severe_drifting", dtype=object)
    out[late_drift < 1.0] = "strong_drifting"
    out[late_drift < 0.5] = "slow_drifting"
    out[late_drift < 0.1] = "converged"
    return out


def group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group, sub in df.groupby(group_col, dropna=False):
        abs_delta = sub["abs_delta_z_22_97"].to_numpy(dtype=np.float64)
        late = sub["late_drift"].to_numpy(dtype=np.float64) if "late_drift" in sub else np.full(len(sub), np.nan)
        rows.append(
            {
                "grouping": group_col,
                "group": str(group),
                "count": int(len(sub)),
                "median_abs_delta_z": float(np.nanmedian(abs_delta)) if len(abs_delta) else np.nan,
                "mean_abs_delta_z": float(np.nanmean(abs_delta)) if len(abs_delta) else np.nan,
                "p90_abs_delta_z": float(np.nanquantile(abs_delta, 0.90)) if len(abs_delta) else np.nan,
                "max_abs_delta_z": float(np.nanmax(abs_delta)) if len(abs_delta) else np.nan,
                "fraction_abs_delta_gt_0.5": float(np.mean(abs_delta > 0.5)) if len(abs_delta) else np.nan,
                "fraction_abs_delta_gt_1.0": float(np.mean(abs_delta > 1.0)) if len(abs_delta) else np.nan,
                "median_late_drift": float(np.nanmedian(late)) if len(late) else np.nan,
                "p90_late_drift": float(np.nanquantile(late, 0.90)) if len(late) else np.nan,
                "fraction_late_drift_gt_0.5": float(np.nanmean(late > 0.5)) if len(late) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_group_boxplot(df: pd.DataFrame, path: Path):
    candidates = ["time_region", "has_observable", "edge_type_or_group", "detector_support_size"]
    plt.figure(figsize=(10, 6))
    data = []
    labels = []
    for col in candidates:
        if col not in df.columns:
            continue
        med = df.groupby(col)["abs_delta_z_22_97"].median().sort_values(ascending=False)
        for group in med.index[:8]:
            vals = df.loc[df[col] == group, "abs_delta_z_22_97"].to_numpy(dtype=np.float64)
            if len(vals) >= 2:
                data.append(vals)
                labels.append(f"{col}={group}")
        if data:
            break
    if data:
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("|delta z| 22->97")
    else:
        plt.text(0.5, 0.5, "No group metadata available", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_top_trajectories(epochs: list[int], Z: np.ndarray, top_edges: np.ndarray, path: Path):
    plt.figure(figsize=(10, 6))
    epochs_arr = np.asarray(epochs)
    early_mask = epochs_arr <= 30
    late_mask = epochs_arr >= 90
    for edge_id in top_edges:
        color = None
        if early_mask.any():
            line = plt.plot(epochs_arr[early_mask], Z[early_mask, edge_id], marker="o", linewidth=1, alpha=0.8)
            color = line[0].get_color()
        if late_mask.any():
            plt.plot(epochs_arr[late_mask], Z[late_mask, edge_id], marker="o", linewidth=1, alpha=0.8, color=color)
    if early_mask.any() and late_mask.any():
        plt.axvspan(30, 90, color="gray", alpha=0.12, label="missing 31-89")
    plt.title("Top 20 drifting edge logit trajectories")
    plt.xlabel("epoch")
    plt.ylabel("z = logit(p)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--epochs", required=True)
    parser.add_argument("--epoch-best", type=int, required=True)
    parser.add_argument("--epoch-final", type=int, required=True)
    parser.add_argument("--late-start", type=int, required=True)
    parser.add_argument("--early-start", type=int, required=True)
    parser.add_argument("--early-end", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dem", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--checkpoint-values", choices=["auto", "probability", "logit"], default="auto")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_epochs = parse_epochs(args.epochs)
    print(f"[info] requested epochs: {requested_epochs}")
    print(f"[info] loading checkpoints from {checkpoint_dir}")
    epochs, P, manifest = load_probability_matrix(
        checkpoint_dir=checkpoint_dir,
        requested_epochs=requested_epochs,
        mode=args.checkpoint_values,
        eps=args.eps,
    )
    manifest.to_csv(output_dir / "loaded_checkpoint_manifest.csv", index=False)
    missing_epochs = [int(row.epoch) for row in manifest.itertuples() if row.status != "loaded"]
    print(f"[info] loaded epochs: {epochs}")
    if missing_epochs:
        print(f"[warn] missing/unloaded epochs: {missing_epochs}")

    Z = logit(P, args.eps)
    num_edges = P.shape[1]
    epoch_to_idx = {epoch: i for i, epoch in enumerate(epochs)}
    metadata_df = load_edge_metadata(
        Path(args.dem) if args.dem else None,
        Path(args.metadata) if args.metadata else None,
        num_edges,
    )

    i_best = require_epoch(epoch_to_idx, args.epoch_best, "epoch-best")
    i_final = require_epoch(epoch_to_idx, args.epoch_final, "epoch-final")
    delta = Z[i_final] - Z[i_best]
    abs_delta = np.abs(delta)
    ratio = np.exp(np.clip(delta, -700, 700))

    drift_df = pd.DataFrame(
        {
            "edge_id": np.arange(num_edges),
            f"p{args.epoch_best}": P[i_best],
            f"p{args.epoch_final}": P[i_final],
            f"z{args.epoch_best}": Z[i_best],
            f"z{args.epoch_final}": Z[i_final],
            f"delta_z_{args.epoch_best}_{args.epoch_final}": delta,
            f"abs_delta_z_{args.epoch_best}_{args.epoch_final}": abs_delta,
            "delta_z_22_97": delta,
            "abs_delta_z_22_97": abs_delta,
            "probability_ratio_approx": ratio,
        }
    ).merge(metadata_df, on="edge_id", how="left")
    drift_df.to_csv(output_dir / "per_edge_drift_epoch22_to_epoch97.csv", index=False)
    top_df = drift_df.sort_values("abs_delta_z_22_97", ascending=False).head(args.top_k)
    top_df.to_csv(output_dir / "top_drifting_edges_epoch22_to_epoch97.csv", index=False)
    save_hist(abs_delta, output_dir / "delta_z_epoch22_to_epoch97_hist.png", "abs delta z: epoch 22 to 97", "|z97 - z22|")
    save_scatter(Z[i_best], Z[i_final], output_dir / "z_epoch22_vs_epoch97_scatter.png", "z epoch 22 vs epoch 97", "z22", "z97")

    late_epochs = [e for e in epochs if args.late_start <= e <= args.epoch_final]
    late_idx = [epoch_to_idx[e] for e in late_epochs]
    if args.late_start in epoch_to_idx and args.epoch_final in epoch_to_idx:
        late_drift = np.abs(Z[epoch_to_idx[args.epoch_final]] - Z[epoch_to_idx[args.late_start]])
    elif len(late_idx) >= 2:
        late_drift = np.abs(Z[late_idx[-1]] - Z[late_idx[0]])
        print(f"[warn] late-start {args.late_start} unavailable; using {late_epochs[0]}->{late_epochs[-1]} for late_drift.")
    else:
        late_drift = np.full(num_edges, np.nan)
        print("[warn] fewer than two late checkpoints available; late convergence is mostly unavailable.")

    late_steps = []
    if len(late_idx) >= 2:
        for j in range(1, len(late_idx)):
            late_steps.append(np.abs(Z[late_idx[j]] - Z[late_idx[j - 1]]))
        late_mean_step = np.mean(np.vstack(late_steps), axis=0)
    else:
        late_mean_step = np.full(num_edges, np.nan)
    late_slopes = np.array([linear_slope(np.asarray(late_epochs), Z[late_idx, edge]) if len(late_idx) >= 2 else np.nan for edge in range(num_edges)])
    late_df = pd.DataFrame(
        {
            "edge_id": np.arange(num_edges),
            "late_drift": late_drift,
            "late_mean_step": late_mean_step,
            "late_slope": late_slopes,
            "late_status": late_status(np.nan_to_num(late_drift, nan=np.inf)),
        }
    ).merge(metadata_df, on="edge_id", how="left")
    late_df.to_csv(output_dir / "per_edge_late_convergence_status_epoch90_to_epoch97.csv", index=False)
    late_summary = late_df["late_status"].value_counts().rename_axis("status").reset_index(name="count")
    late_summary["fraction"] = late_summary["count"] / num_edges
    late_summary.to_csv(output_dir / "late_convergence_status_summary.csv", index=False)
    late_step_df = step_summary(late_epochs, Z[late_idx]) if len(late_idx) >= 2 else pd.DataFrame()
    late_step_df.to_csv(output_dir / "late_step_size_vs_epoch.csv", index=False)
    if not late_step_df.empty:
        plot_step_df(late_step_df, output_dir / "late_step_size_vs_epoch.png", "Late-window available step sizes", omit_gaps=False)

    for col in ["late_drift", "late_mean_step", "late_slope", "late_status"]:
        drift_df[col] = late_df[col]

    i_early_start = require_epoch(epoch_to_idx, args.early_start, "early-start")
    i_early_end = require_epoch(epoch_to_idx, args.early_end, "early-end")
    dz_10_22 = Z[i_best] - Z[i_early_start]
    dz_22_30 = Z[i_early_end] - Z[i_best]
    early_df = pd.DataFrame(
        {
            "edge_id": np.arange(num_edges),
            "delta_z_10_22": dz_10_22,
            "delta_z_22_30": dz_22_30,
            "abs_delta_z_10_22": np.abs(dz_10_22),
            "abs_delta_z_22_30": np.abs(dz_22_30),
        }
    ).merge(metadata_df, on="edge_id", how="left")
    early_df.to_csv(output_dir / "per_edge_early_drift_epoch10_to_epoch22_and_epoch22_to_epoch30.csv", index=False)
    early_epochs = [e for e in epochs if args.early_start <= e <= args.early_end]
    early_idx = [epoch_to_idx[e] for e in early_epochs]
    early_step_df = step_summary(early_epochs, Z[early_idx]) if len(early_idx) >= 2 else pd.DataFrame()
    early_step_df.to_csv(output_dir / "early_step_size_vs_epoch.csv", index=False)
    if not early_step_df.empty:
        plot_step_df(early_step_df, output_dir / "early_step_size_vs_epoch.png", "Early-window available step sizes", omit_gaps=False)

    sparse_step_df = step_summary(epochs, Z)
    sparse_step_df.to_csv(output_dir / "sparse_available_step_size_vs_epoch.csv", index=False)
    plot_step_df(sparse_step_df, output_dir / "sparse_available_step_size_vs_epoch.png", "Sparse available step sizes (gaps omitted)", omit_gaps=True)

    top20_edges = drift_df.sort_values("abs_delta_z_22_97", ascending=False)["edge_id"].head(20).to_numpy(dtype=int)
    traj_rows = []
    for edge_id in top20_edges:
        for epoch, idx in epoch_to_idx.items():
            traj_rows.append({"edge_id": int(edge_id), "epoch": int(epoch), "p": float(P[idx, edge_id]), "z": float(Z[idx, edge_id])})
    pd.DataFrame(traj_rows).to_csv(output_dir / "top20_drifting_edge_trajectories_sparse.csv", index=False)
    plot_top_trajectories(epochs, Z, top20_edges, output_dir / "top20_drifting_edge_trajectories_sparse.png")

    group_cols = [
        "detector_support_size",
        "has_observable",
        "edge_type_or_group",
        "time_region",
        "spatial_region",
        "boundary_seam_bulk",
    ]
    group_tables = [group_summary(drift_df, col) for col in group_cols if col in drift_df.columns]
    groupwise = pd.concat(group_tables, ignore_index=True) if group_tables else pd.DataFrame()
    groupwise.to_csv(output_dir / "groupwise_drift_summary_sparse.csv", index=False)
    save_group_boxplot(drift_df, output_dir / "groupwise_abs_delta_z_boxplot.png")

    (output_dir / "proxy_signature_groups_epoch22_to_epoch97.csv").write_text(
        "status,message\nskipped,patch projection maps were not provided to this standalone script\n"
    )
    print("[info] proxy signature grouping skipped: patch projection maps unavailable.")

    stats_22_97 = quantile_stats(abs_delta)
    clip_mask = (P <= args.eps * 1.0001) | (P >= 1.0 - args.eps * 1.0001)
    clipped_edges = np.where(np.any(clip_mask, axis=0))[0]
    warnings = []
    if np.sum(abs_delta > 1.0) > 0.05 * num_edges:
        warnings.append("many edges drift strongly from 22 to 97")
    if np.sum(late_drift > 0.5) > 0.05 * num_edges:
        warnings.append("many edges still drifting during 90-97")
    if len(clipped_edges) > 0:
        warnings.append("many probabilities hit clipping boundary" if len(clipped_edges) > 0.01 * num_edges else "some probabilities hit clipping boundary")
    obs_rows = drift_df[drift_df["has_observable"].astype(bool)] if "has_observable" in drift_df else pd.DataFrame()
    non_obs_rows = drift_df[~drift_df["has_observable"].astype(bool)] if "has_observable" in drift_df else pd.DataFrame()
    if len(obs_rows) and len(non_obs_rows):
        if obs_rows["abs_delta_z_22_97"].median() > non_obs_rows["abs_delta_z_22_97"].median():
            warnings.append("observable-related edges drift more than non-observable edges")

    lines = ["# Sparse DEM Probability Checkpoint Analysis", ""]
    lines.append(f"- checkpoint_dir: `{checkpoint_dir}`")
    lines.append(f"- available epochs loaded: `{epochs}`")
    if missing_epochs:
        lines.append(f"- missing/unloaded requested epochs: `{missing_epochs}`")
        if any(31 <= e <= 89 for e in missing_epochs) or (30 in epochs and 90 in epochs):
            lines.append("- note: epochs 31-89 are missing and are treated as a sparse gap; no interpolation was used.")
    lines.append(f"- number of error mechanisms: `{num_edges}`")
    lines.append("")
    lines.append("## Epoch 22 to 97 Drift")
    lines.append("")
    lines.append(f"- median |delta z|: `{stats_22_97['median']:.6g}`")
    lines.append(f"- p90 |delta z|: `{stats_22_97['p90']:.6g}`")
    lines.append(f"- p99 |delta z|: `{stats_22_97['p99']:.6g}`")
    lines.append(f"- edges with |delta z| > 0.5: `{int(np.sum(abs_delta > 0.5))}`")
    lines.append(f"- edges with |delta z| > 1.0: `{int(np.sum(abs_delta > 1.0))}`")
    lines.append("")
    lines.append("## Late Convergence 90 to 97")
    lines.append("")
    for row in late_summary.itertuples():
        lines.append(f"- {row.status}: `{row.count}` (`{row.fraction:.3%}`)")
    stable_fraction = float(np.mean(late_df["late_status"] == "converged"))
    lines.append(f"- late checkpoints appear stable: `{'yes' if stable_fraction > 0.8 else 'no'}`")
    lines.append("")
    lines.append("## Top 10 Drifting Edges")
    lines.append("")
    top10_cols = [
        "edge_id",
        "p22",
        "p97",
        "delta_z_22_97",
        "abs_delta_z_22_97",
        "detector_support_size",
        "detectors",
        "observables",
        "edge_type_or_group",
    ]
    for _, row in top_df.head(10).iterrows():
        parts = [f"{col}={row[col]}" for col in top10_cols if col in row]
        lines.append("- " + ", ".join(parts))
    lines.append("")
    lines.append("## Groupwise Highlights")
    lines.append("")
    if not groupwise.empty:
        hi = groupwise.sort_values("median_abs_delta_z", ascending=False).head(10)
        for _, row in hi.iterrows():
            lines.append(
                f"- {row['grouping']}={row['group']}: count={row['count']}, "
                f"median_abs_delta_z={row['median_abs_delta_z']:.6g}, "
                f"p90_abs_delta_z={row['p90_abs_delta_z']:.6g}"
            )
    else:
        lines.append("- no group metadata available.")
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- no automatic warnings triggered.")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")

    print(f"[info] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
