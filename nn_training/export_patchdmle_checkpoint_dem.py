from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import stim
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from script.train_d7_google_subsample import _load_google_round_root
from src import broadcast_dem, update_dem


def max_detector_time(dem: stim.DetectorErrorModel) -> int:
    max_t = 0
    for inst in dem.flattened():
        if inst.type != "detector":
            continue
        args = inst.args_copy()
        if len(args) >= 3:
            max_t = max(max_t, int(round(args[2])))
    return max_t


def parse_args():
    p = argparse.ArgumentParser(description="Export a patchDMLE checkpoint probability vector as a DEM.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--round-root",
        default=str(REPO_ROOT / "dataset/google_broadcast_source/d7_at_q6_7_bulk2/X/r04"),
        help="Round root containing initial_dem_non_decomposed.dem and detector_coordinates.json.",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--broadcast-time-layer", type=int, default=None, help="Example: 10 to broadcast r04 source to r10.")
    p.add_argument("--repeat-chunk", type=int, default=1)
    p.add_argument("--eps", type=float, default=1e-15)
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    _ideal, _noisy, _dets, _obs, initial_dem, _round_dir = _load_google_round_root(args.round_root)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "oer" not in ckpt:
        raise KeyError(f"Checkpoint has no 'oer' key: {ckpt_path}")
    oer = ckpt["oer"]
    if isinstance(oer, torch.Tensor):
        oer = oer.detach().cpu().numpy()
    error_rates = np.clip(np.asarray(oer, dtype=np.float64), args.eps, 1.0 - args.eps)
    if len(error_rates) != initial_dem.num_errors:
        raise ValueError(f"checkpoint error count {len(error_rates)} != DEM errors {initial_dem.num_errors}")

    dem = update_dem(initial_dem, error_rates)
    source_max_time = max_detector_time(dem)
    if args.broadcast_time_layer is not None:
        if args.broadcast_time_layer <= source_max_time:
            raise ValueError(
                f"broadcast-time-layer={args.broadcast_time_layer} must exceed source max time {source_max_time}"
            )
        dem = broadcast_dem(dem, broadcast_time_layer=args.broadcast_time_layer, repeat_chunk=args.repeat_chunk)

    dem.to_file(str(output))
    metadata = {
        "checkpoint": str(ckpt_path),
        "round_root": str(args.round_root),
        "output": str(output),
        "source_max_time": source_max_time,
        "broadcast_time_layer": args.broadcast_time_layer,
        "repeat_chunk": args.repeat_chunk,
        "num_detectors": dem.num_detectors,
        "num_errors": dem.num_errors,
        "num_observables": dem.num_observables,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_ler": float(ckpt["ler"]) if "ler" in ckpt else None,
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
