from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import stim
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from model import AlphaQubits
from surface_utils import (
    point_cloud_from_dem,
    read_experimental_round,
    sample_dem_batch,
    save_json,
    setup_logger,
    surface_layout_from_dem,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train AlphaQubits-style neural decoder on a surface-code DEM.")
    p.add_argument("--dem-path", required=True, help="DEM used to generate online training samples.")
    p.add_argument("--run-name", default=None)
    p.add_argument("--output-root", default="nn_training/checkpoints")
    p.add_argument("--timestamp", default=None, help="Optional fixed timestamp shared by all DDP ranks.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=71534)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=5000, help="Per-process/per-GPU training batch size.")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument(
        "--scheduler-steps",
        type=int,
        default=None,
        help="Cosine scheduler horizon; defaults to --steps.",
    )
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--eval-shots", type=int, default=5000, help="Sampled DEM eval shots if --eval-data-root is absent.")
    p.add_argument("--eval-data-root", default=None, help="Optional experimental round dir containing detection_events.b8 and obs_flips_actual.b8.")
    p.add_argument("--eval-max-shots", type=int, default=0, help="0 means all experimental shots.")
    p.add_argument(
        "--expected-eval-shots",
        type=int,
        default=0,
        help="Fail unless the fixed experimental evaluation set has this many shots.",
    )
    p.add_argument("--eval-batch-size", type=int, default=5000)
    p.add_argument("--keep-top-k", type=int, default=5)
    p.add_argument("--pretrained", default=None)
    p.add_argument("--resume", default=None, help="Resume from a checkpoint, including optimizer/scheduler state if available.")
    p.add_argument(
        "--reset-scheduler",
        action="store_true",
        help="Keep resumed optimizer moments but restart LR scheduling from --lr.",
    )
    return p.parse_args()


class NullLogger:
    def info(self, *_args, **_kwargs):
        pass


def init_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA for this script.")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
    else:
        device = torch.device(args.device)

    return distributed, rank, local_rank, world_size, device


def evaluate_model(model, dem, layout, coords_t, det_types_t, device, args, sampler, eval_data):
    model.eval()
    total_errors = 0
    total = 0
    with torch.no_grad():
        if eval_data is not None:
            dets, obs = eval_data
            for start in range(0, len(dets), args.eval_batch_size):
                stop = min(start + args.eval_batch_size, len(dets))
                pc = point_cloud_from_dem(dets[start:stop], dem, layout)
                x = torch.from_numpy(pc).float().to(device)
                logits = model(x, coords_t, det_types_t)[:, -1]
                pred = (logits >= 0).to(torch.int32).cpu().numpy()
                total_errors += int(np.sum(pred != obs[start:stop]))
                total += stop - start
        else:
            remaining = args.eval_shots
            while remaining > 0:
                shots = min(args.eval_batch_size, remaining)
                dets, obs = sample_dem_batch(sampler, shots)
                pc = point_cloud_from_dem(dets, dem, layout)
                x = torch.from_numpy(pc).float().to(device)
                logits = model(x, coords_t, det_types_t)[:, -1]
                pred = (logits >= 0).to(torch.int32).cpu().numpy()
                total_errors += int(np.sum(pred != obs))
                total += shots
                remaining -= shots
    model.train()
    return total_errors / total, total_errors, total


def append_eval_metric(path: Path, row: dict) -> None:
    fieldnames = [
        "step",
        "local_step",
        "eval_ler",
        "wrong",
        "eval_shots",
        "loss",
        "lr",
        "elapsed_seconds",
        "time_per_step",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row[key] for key in fieldnames})


def main():
    args = parse_args()
    if args.pretrained and args.resume:
        raise ValueError("Use only one of --pretrained or --resume.")
    if args.expected_eval_shots and not args.eval_data_root:
        raise ValueError(
            "--expected-eval-shots requires --eval-data-root; sampled DEM "
            "evaluation is not a fixed dataset."
        )
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive.")
    if args.scheduler_steps is not None and args.scheduler_steps <= 0:
        raise ValueError("--scheduler-steps must be positive.")

    distributed, rank, local_rank, world_size, device = init_distributed(args)
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    dem_path = Path(args.dem_path)
    dem = stim.DetectorErrorModel.from_file(str(dem_path))
    if dem.num_observables != 1:
        raise RuntimeError(f"Expected one logical observable, got {dem.num_observables}.")

    run_name = args.run_name or dem_path.stem
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / run_name / timestamp
    logger = setup_logger(out_dir / "train.log") if is_main else NullLogger()

    logger.info("# Surface neural decoder training")
    logger.info(f"dem_path: {dem_path}")
    logger.info(f"detectors/errors/observables: {dem.num_detectors}/{dem.num_errors}/{dem.num_observables}")
    logger.info(f"distributed: {distributed}")
    logger.info(f"rank/world_size: {rank}/{world_size}")
    logger.info(f"local_rank: {local_rank}")
    logger.info(f"per_device_batch_size: {args.batch_size}")
    logger.info(f"global_batch_size: {args.batch_size * world_size}")
    for key, value in sorted(vars(args).items()):
        logger.info(f"{key}: {value}")

    layout = surface_layout_from_dem(dem)
    coords_t = torch.from_numpy(layout["coords"]).float().to(device)
    det_types_t = torch.from_numpy(layout["det_types"]).long().to(device)
    logger.info(f"point_cloud cycles={layout['cycles']} slots={layout['num_slots']}")

    eval_data = None
    if is_main and args.eval_data_root:
        eval_dets, eval_obs = read_experimental_round(args.eval_data_root, dem)
        if args.eval_max_shots and args.eval_max_shots < len(eval_dets):
            eval_dets = eval_dets[: args.eval_max_shots]
            eval_obs = eval_obs[: args.eval_max_shots]
        if args.expected_eval_shots and len(eval_dets) != args.expected_eval_shots:
            raise ValueError(
                f"Expected {args.expected_eval_shots} fixed evaluation shots, "
                f"loaded {len(eval_dets)} from {args.eval_data_root}."
            )
        eval_data = (eval_dets, eval_obs)
        logger.info(
            f"loaded fixed experimental evaluation data once: "
            f"shots={len(eval_dets)}, detectors={eval_dets.shape[1]}, "
            f"observable_flip_rate={float(np.mean(eval_obs)):.8f}"
        )

    sampler = dem.compile_sampler(seed=args.seed + rank * 1000003)
    model = AlphaQubits(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        dropout=args.dropout,
        use_conv=False,
    ).to(device)

    resume_ckpt = None
    resume_step = 0
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"], strict=True)
        resume_step = int(resume_ckpt.get("step", 0))
        logger.info(f"loaded resume checkpoint: {args.resume}")
        logger.info(f"resume_step: {resume_step}")
    elif args.pretrained:
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        logger.info(f"loaded pretrained checkpoint: {args.pretrained}")

    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable_params: {nparams}")
    train_model = DDP(model, device_ids=[local_rank], output_device=local_rank) if distributed else model

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_steps = args.scheduler_steps or args.steps
    if resume_ckpt is not None:
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            logger.info("loaded optimizer_state_dict from resume checkpoint")
        else:
            logger.info("resume checkpoint has no optimizer_state_dict; optimizer reinitialized")

    if args.reset_scheduler:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr
            param_group["initial_lr"] = args.lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_steps, eta_min=args.min_lr
        )
        logger.info(
            f"scheduler reset: CosineAnnealingLR lr={args.lr:.3e} -> "
            f"{args.min_lr:.3e} over {scheduler_steps} steps"
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_steps, eta_min=args.min_lr
        )
        if resume_ckpt is not None and "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
            logger.info("loaded scheduler_state_dict from resume checkpoint")
        elif resume_ckpt is not None:
            logger.info("resume checkpoint has no scheduler_state_dict; scheduler reinitialized")

    best_models = []
    start_time = time.time()
    target_step = resume_step + args.steps

    if is_main:
        save_json(
            out_dir / "metadata.json",
            {
                "dem_path": str(dem_path),
                "run_name": run_name,
                "timestamp": timestamp,
                "args": vars(args),
                "distributed": distributed,
                "world_size": int(world_size),
                "per_device_batch_size": int(args.batch_size),
                "global_batch_size": int(args.batch_size * world_size),
                "num_detectors": dem.num_detectors,
                "num_errors": dem.num_errors,
                "num_observables": dem.num_observables,
                "cycles": int(layout["cycles"]),
                "num_slots": int(layout["num_slots"]),
                "trainable_params": int(nparams),
                "resume_step": int(resume_step),
                "target_step": int(target_step),
                "fixed_eval_shots": int(len(eval_data[0])) if eval_data is not None else 0,
            },
        )

    metrics_path = out_dir / "eval_metrics.csv"
    for local_step in range(1, args.steps + 1):
        step = resume_step + local_step
        train_model.train()
        dets, obs = sample_dem_batch(sampler, args.batch_size)
        pc = point_cloud_from_dem(dets, dem, layout)
        x = torch.from_numpy(pc).float().to(device)
        labels = torch.from_numpy(obs.astype(np.float32)).to(device)

        logits = train_model(x, coords_t, det_types_t)[:, -1]
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if is_main and local_step % args.log_interval == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"step {step}/{target_step}: loss={loss.item():.6f}, "
                f"lr={optimizer.param_groups[0]['lr']:.3e}, grad_norm={float(grad_norm):.4f}, "
                f"time_per_step={elapsed / local_step:.4f}s, global_batch={args.batch_size * world_size}"
            )

        if args.eval_interval and local_step % args.eval_interval == 0:
            if distributed:
                dist.barrier()
            if not is_main:
                if distributed:
                    dist.barrier()
                continue

            eval_ler, wrong, total = evaluate_model(
                model,
                dem,
                layout,
                coords_t,
                det_types_t,
                device,
                args,
                sampler,
                eval_data,
            )
            elapsed = time.time() - start_time
            logger.info("-" * 60)
            logger.info(f"eval step {step}: LER={eval_ler:.6f} ({wrong}/{total})")
            logger.info("-" * 60)
            append_eval_metric(
                metrics_path,
                {
                    "step": int(step),
                    "local_step": int(local_step),
                    "eval_ler": f"{eval_ler:.10f}",
                    "wrong": int(wrong),
                    "eval_shots": int(total),
                    "loss": f"{loss.item():.10f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.12e}",
                    "elapsed_seconds": f"{elapsed:.6f}",
                    "time_per_step": f"{elapsed / local_step:.8f}",
                },
            )

            ckpt_name = f"step_{step}_ler_{eval_ler:.6f}.pt"
            ckpt_path = out_dir / ckpt_name
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": int(step),
                "local_step": int(local_step),
                "resume_step": int(resume_step),
                "eval_ler": float(eval_ler),
                "wrong": int(wrong),
                "eval_shots": int(total),
                "dem_path": str(dem_path),
                "args": vars(args),
            }
            torch.save(checkpoint, ckpt_path)
            last_tmp_path = out_dir / "last.pt.tmp"
            torch.save(checkpoint, last_tmp_path)
            os.replace(last_tmp_path, out_dir / "last.pt")
            best_models.append((eval_ler, step, ckpt_path))
            best_models.sort(key=lambda item: item[0])
            while len(best_models) > args.keep_top_k:
                _ler, _step, old_path = best_models.pop()
                if old_path.exists():
                    old_path.unlink()
            logger.info(f"checkpoint: {ckpt_path.name}")
            if distributed:
                dist.barrier()

    if is_main:
        logger.info("training complete")
        for rank, (ler, step, path) in enumerate(best_models, start=1):
            logger.info(f"top{rank}: step={step}, LER={ler:.6f}, file={path.name}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
