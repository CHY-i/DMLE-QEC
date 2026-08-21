from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import stim
import torch

from model import AlphaQubits
from surface_utils import point_cloud_from_dem, read_experimental_round, surface_layout_from_dem


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained surface-code neural decoder.")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--checkpoint-dir", default=None, help="If set, use the checkpoint with lowest *_ler_*.pt value.")
    p.add_argument("--dem-path", required=True, help="DEM defining detector coordinates/layout.")
    p.add_argument("--eval-data-root", required=True, help="Round dir with detection_events.b8 and obs_flips_actual.b8.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument("--max-shots", type=int, default=0, help="0 means evaluate all shots.")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--save-predictions",
        default=None,
        help="Optional .npz path for per-shot predictions, observables, and errors.",
    )
    return p.parse_args()


def find_best_checkpoint(path: str | Path) -> Path:
    files = glob.glob(str(Path(path) / "step_*_ler_*.pt"))
    if not files:
        raise FileNotFoundError(f"No step_*_ler_*.pt checkpoints under {path}")

    def score(filename):
        m = re.search(r"_ler_([0-9.]+)\.pt$", Path(filename).name)
        return float(m.group(1)) if m else float("inf")

    return Path(min(files, key=score))


def main():
    args = parse_args()
    if not args.checkpoint and not args.checkpoint_dir:
        raise ValueError("Provide --checkpoint or --checkpoint-dir")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_best_checkpoint(args.checkpoint_dir)
    device = torch.device(args.device)

    dem = stim.DetectorErrorModel.from_file(args.dem_path)
    layout = surface_layout_from_dem(dem)
    coords_t = torch.from_numpy(layout["coords"]).float().to(device)
    det_types_t = torch.from_numpy(layout["det_types"]).long().to(device)
    dets, obs = read_experimental_round(args.eval_data_root, dem)
    if args.max_shots and args.max_shots < len(dets):
        dets = dets[: args.max_shots]
        obs = obs[: args.max_shots]

    model = AlphaQubits(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        dropout=args.dropout,
        use_conv=False,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    wrong = 0
    predictions = []
    with torch.no_grad():
        for start in range(0, len(dets), args.batch_size):
            stop = min(start + args.batch_size, len(dets))
            pc = point_cloud_from_dem(dets[start:stop], dem, layout)
            x = torch.from_numpy(pc).float().to(device)
            logits = model(x, coords_t, det_types_t)[:, -1]
            pred = (logits >= 0).to(torch.int32).cpu().numpy()
            predictions.append(pred)
            wrong += int(np.sum(pred != obs[start:stop]))
            print(f"decoded {stop}/{len(dets)}", flush=True)

    ler = wrong / len(dets)
    predictions = np.concatenate(predictions).astype(np.uint8, copy=False)
    observables = np.asarray(obs, dtype=np.uint8)
    errors = (predictions != observables).astype(np.uint8)
    if args.save_predictions:
        output_path = Path(args.save_predictions)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            predictions=predictions,
            observables=observables,
            errors=errors,
            checkpoint=np.asarray(str(ckpt_path)),
            dem_path=np.asarray(args.dem_path),
            eval_data_root=np.asarray(args.eval_data_root),
        )
        print(f"saved_predictions: {output_path}", flush=True)
    print("# Neural decoder evaluation", flush=True)
    print(f"checkpoint: {ckpt_path}", flush=True)
    print(f"dem_path: {args.dem_path}", flush=True)
    print(f"eval_data_root: {args.eval_data_root}", flush=True)
    print(f"LER={ler:.6f} ({wrong}/{len(dets)})", flush=True)


if __name__ == "__main__":
    main()
