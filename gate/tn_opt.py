"""
Optimize surface-code **gate** noise via GateNoiseToDEM + TensorNetwork.

Uses Stim ``surface_code:rotated_memory_z`` + noise from ``tqec.NoiseModel`` (no HDF5).
Detection events are sampled from the circuit DEM; learnable parameters are the
gate noise rates in :class:`src.g2dem.GateNoiseToDEM` (default: ``time_shared``). Each step:

    gate_probs → g2d() → DEM error probabilities → TensorNetwork NLL

Run from repo root::

    python gate/tn_opt_surf.py --distance 3 --rounds 3 --epochs 500 --device cuda:0
    python gate/tn_opt_surf.py --noise-model si1000 --error-prob 0.001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import stim
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "gate" / "log" / "sur" / "sim"
CHECKPOINT_DIR = REPO_ROOT / "gate" / "log" / "sur" / "sim" / "checkpoint"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.g2dem import GateNoiseToDEM, ParamSharing  # noqa: E402
from src.model import TensorNetwork  # noqa: E402
from src.utils import PCM  # 直接使用你的 PCM 函数


NoiseModelKind = Literal["depolarizing", "si1000"]


def build_surface_circuit_depolarizing(
    *,
    distance: int,
    rounds: int,
    error_prob: float,
) -> stim.Circuit:
    """Uniform depolarizing noise via ``stim.Circuit.generated``."""
    return stim.Circuit.generated(
        code_task="surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=error_prob,
        before_measure_flip_probability=error_prob,
        after_reset_flip_probability=error_prob,
        before_round_data_depolarization=error_prob,
    )


def noise_model_si1000(p: float):
    """``tqec.NoiseModel.si1000`` plus ``MR``/``M`` rules."""
    from tqec import NoiseModel
    from tqec.utils.noise_model import NoiseRule

    nm = NoiseModel.si1000(p)
    flip = NoiseRule(after={}, flip_result=p * 5)
    extra = {"MR": flip, "M": flip}
    if nm.gate_rules is None:
        nm.gate_rules = extra
    else:
        nm.gate_rules = {**nm.gate_rules, **extra}
    return nm


def build_surface_circuit_si1000(
    *,
    distance: int,
    rounds: int,
    error_prob: float,
) -> stim.Circuit:
    """Ideal surface memory circuit + SI1000 noise from ``tqec``."""
    ideal = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
    )
    return noise_model_si1000(error_prob).noisy_circuit(ideal)


def build_circuit(
    *,
    distance: int,
    rounds: int,
    error_prob: float,
    noise_model: NoiseModelKind,
) -> stim.Circuit:
    if noise_model == "depolarizing":
        return build_surface_circuit_depolarizing(
            distance=distance, rounds=rounds, error_prob=error_prob
        )
    if noise_model == "si1000":
        return build_surface_circuit_si1000(
            distance=distance, rounds=rounds, error_prob=error_prob
        )
    raise ValueError(f"unknown noise_model={noise_model!r}")


def default_log_filename(
    *,
    distance: int,
    rounds: int,
    error_prob: float,
    param_sharing: ParamSharing,
    noise_model: NoiseModelKind,
) -> str:
    return f"surf_tn_d{distance}r{rounds}_p{error_prob}_{noise_model}_{param_sharing}.txt"


def resolve_log_path(
    user_log: str | None,
    *,
    distance: int,
    rounds: int,
    error_prob: float,
    param_sharing: ParamSharing,
    noise_model: NoiseModelKind,
) -> Path:
    name = Path(user_log).name if user_log else default_log_filename(
        distance=distance,
        rounds=rounds,
        error_prob=error_prob,
        param_sharing=param_sharing,
        noise_model=noise_model,
    )
    return LOG_DIR / name


def sample_detection_events(
    circuit: stim.Circuit,
    *,
    num_shots: int,
    seed: int,
) -> np.ndarray:
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    sampler = dem.compile_sampler(seed=seed)
    dets, _obs, _ = sampler.sample(shots=num_shots)
    return np.asarray(dets, dtype=np.float64)


def perturb_probabilities_like_pl_opt(p: torch.Tensor) -> torch.Tensor:
    """Same random relative perturbation as ``script/pl_opt.py`` DEM init."""
    pertub = torch.rand_like(p)
    sign = 2.0 * torch.bernoulli(torch.ones_like(p) / 2.0) - 1.0
    return (p + sign * p * pertub).clamp(1e-12, 1.0 - 1e-12)


def prob_to_logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-12, 1.0 - 1e-12)
    return torch.log(p / (1.0 - p))


def relative_error_stats(pred: torch.Tensor, ref: torch.Tensor, *, eps: float = 1e-30) -> dict[str, float]:
    diff = (pred - ref).abs()
    rel = diff / ref.abs().clamp(min=eps)
    return {
        "max_rel": float(rel.max().cpu()),
        "mean_rel": float(rel.mean().cpu()),
    }


def log_checkpoint(
    log,
    *,
    tag: str,
    epoch: int,
    nll: float | None,
    g2d: GateNoiseToDEM,
    true_gate: torch.Tensor,
    true_dem: torch.Tensor,
    stim_dem: torch.Tensor,
) -> None:
    with torch.no_grad():
        gate_p = g2d.gate_probs().detach().cpu().numpy()
        dem_p = g2d().detach().cpu().numpy()
    gate_err = relative_error_stats(
        torch.tensor(gate_p), true_gate.cpu()
    )
    dem_err = relative_error_stats(
        torch.tensor(dem_p), true_dem.cpu()
    )
    stim_err = relative_error_stats(
        torch.tensor(dem_p), stim_dem.cpu()
    )

    lines = [f"[{tag}] epoch={epoch}"]
    if nll is not None:
        lines[0] += f" nll={nll:.6f}"
    lines.append(f"  gate_error_rates: {gate_p!r}")
    lines.append(f"  dem_error_rates: {dem_p!r}")
    lines.append(
        f"  gate vs true_gate: max_rel={gate_err['max_rel']:.3e} "
        f"mean_rel={gate_err['mean_rel']:.3e}"
    )
    lines.append(
        f"  dem vs true_dem (g2d@ref gate): max_rel={dem_err['max_rel']:.3e} "
        f"mean_rel={dem_err['mean_rel']:.3e}"
    )
    lines.append(
        f"  dem vs stim_dem (circuit): max_rel={stim_err['max_rel']:.3e} "
        f"mean_rel={stim_err['mean_rel']:.3e}"
    )
    block = "\n".join(lines) + "\n"

    log.write(block)
    log.flush()
    print(block, end="", flush=True)


def setup_models(
    circuit: stim.Circuit,
    *,
    distance: int,
    rounds: int,
    device: str,
    dtype: torch.dtype,
    perturb_init: bool,
    param_sharing: ParamSharing,
    mini_batch: int,
    ctg_max_time: int,
) -> tuple[GateNoiseToDEM, TensorNetwork, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    
    # ----------------------------------------------------
    # 调用你的 src.utils 中的 PCM 函数
    pcm, l = PCM(dem)
    # ----------------------------------------------------

    g2d = GateNoiseToDEM(
        circuit,
        learnable=True,
        param_mode="logit",
        param_sharing=param_sharing,
        dtype=dtype,
        device=device,
    )
    
    # 维度一致性检查 (验证 GateNoiseToDEM 的错误数是否匹配 PCM 的列数)
    if g2d.num_dem != pcm.shape[1]:
        raise RuntimeError(f"g2d.num_dem={g2d.num_dem} != pcm.cols={pcm.shape[1]}")

    true_gate = g2d.gate_probs().detach().clone()
    with torch.no_grad():
        true_dem = g2d().detach().clone()
    stim_dem = g2d.stim_ref_dem_probs.detach().clone()

    if perturb_init:
        with torch.no_grad():
            # 1. 提取当前 24 个可学习参数（Slots）的实际概率
            init_slot_probs = torch.sigmoid(g2d.gate_param)
            
            # 2. 仅对这 24 个独立参数进行随机扰动
            perturbed_slots = perturb_probabilities_like_pl_opt(init_slot_probs)
            
            # 3. 将扰动后的概率转回 logit 空间并写回 gate_param
            g2d.gate_param.copy_(prob_to_logit(perturbed_slots))
            
    # Construct TensorNetwork
    # 设置 learn_priors=False，因为底层的 GateNoiseToDEM 将提供可微的先验概率
    tn = TensorNetwork(
        pcm=pcm,
        l=l[0] if l.shape[0] > 0 else None, 
        dev=device,
        dtype=dtype,
        learn_priors=False 
    )
    
    # 自动搜寻张量收缩路径
    tn.path = tn.find_contraction_path(batch_size=mini_batch, max_time=ctg_max_time)
    if tn.path is None:
        raise RuntimeError("Cotengra could not find a feasible TN contraction path (memory limit exceeded).")

    return g2d, tn, true_gate, true_dem, stim_dem, pcm


def train(
    *,
    distance: int,
    rounds: int,
    error_prob: float,
    noise_model: NoiseModelKind,
    num_shots: int,
    seed: int,
    epochs: int,
    batch_size: int,
    mini_batch: int,
    ctg_max_time: int,
    lr: float,
    perturb_init: bool,
    device: str,
    log_path: Path,
    checkpoint_every: int,
    param_sharing: ParamSharing,
) -> None:
    dtype = torch.float64
    torch.manual_seed(seed)

    circuit = build_circuit(
        distance=distance,
        rounds=rounds,
        error_prob=error_prob,
        noise_model=noise_model,
    )
    dets = sample_detection_events(circuit, num_shots=num_shots, seed=seed)
    
    g2d, tn, true_gate, true_dem, stim_dem, pcm = setup_models(
        circuit,
        distance=distance,
        rounds=rounds,
        device=device,
        dtype=dtype,
        perturb_init=perturb_init,
        param_sharing=param_sharing,
        mini_batch=mini_batch,
        ctg_max_time=ctg_max_time,
    )

    # ----------------------------------------------------
    # 【对齐数据维度】
    # 你的 PCM 函数自动过滤了全 0 行，这意味着 pcm.shape[0] 可能小于 dem.num_detectors
    # 如果发生了过滤，我们必须在送入 Dataset 之前切片掉 dets 中无效的对应列，否则网络维度会崩溃
    original_num_detectors = dets.shape[1]
    if pcm.shape[0] != original_num_detectors:
        print(f"[warning] PCM filtered out zero-rows. Aligning detection events from {original_num_detectors} -> {pcm.shape[0]} detectors.")
        # 重新提取一下非零行索引来切片 dets
        dem_temp = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
        raw_pcm = np.zeros([dem_temp.num_detectors, dem_temp.num_errors])
        for i, e in enumerate(dem_temp[:dem_temp.num_errors]):
            for t in e.targets_copy():
                D = str(t)
                if D.startswith('D'):
                    raw_pcm[int(D[1:]), i] = 1
        non_zero_rows = np.where(raw_pcm.sum(axis=1) != 0)[0]
        dets = dets[:, non_zero_rows]
    # ----------------------------------------------------
    print(f"[data] {num_shots} shots, {dets.shape[1]} detectors (aligned with PCM)", flush=True)

    optim = torch.optim.AdamW(g2d.parameters(), lr=lr, weight_decay=0.01)

    dets_t = torch.from_numpy(dets)
    dataset = TensorDataset(dets_t)
    if batch_size % mini_batch != 0:
        raise ValueError(f"batch_size={batch_size} must divide mini_batch={mini_batch}")
    nb = batch_size // mini_batch
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(
            f"# surface_code:rotated_memory_z d={distance} r={rounds} p={error_prob} "
            f"noise_model={noise_model}\n"
            f"# param_sharing={param_sharing} gate_params={g2d.num_slots} "
            f"elementary_sites={g2d.num_elementary}\n"
            f"# shots={num_shots} epochs={epochs} batch_size={batch_size} "
            f"mini_batch={mini_batch} lr={lr}\n"
            f"# dem errors={g2d.num_dem} perturb_init={int(perturb_init)}\n"
            f"# true gate (circuit gate probs):\n{repr(true_gate.detach().cpu().numpy())}\n"
            f"# true dem (g2d @ true gate, used for dem vs true in checkpoints):\n"
            f"{repr(true_dem.detach().cpu().numpy())}\n"
            f"# stim dem (circuit.detector_error_model):\n"
            f"{repr(stim_dem.detach().cpu().numpy())}\n"
        )
        with torch.no_grad():
            stim_gap = relative_error_stats(true_dem.cpu(), stim_dem.cpu())
        log.write(
            f"# at init: g2d(true_gate) vs stim_dem: max_rel={stim_gap['max_rel']:.3e} "
            f"mean_rel={stim_gap['mean_rel']:.3e}\n"
        )
        if perturb_init:
            log.write(
                f"# init gate (perturbed):\n{repr(g2d.gate_probs().detach().cpu().numpy())}\n"
            )
        log.flush()

        with torch.no_grad():
            probe = torch.from_numpy(dets[:mini_batch]).to(device=device, dtype=dtype)
            init_nll = float(tn(probe, priors=g2d()).item())
        print(f"[init] nll~{init_nll:.4f}", flush=True)
        log_checkpoint(
            log,
            tag="init",
            epoch=0,
            nll=init_nll,
            g2d=g2d,
            true_gate=true_gate,
            true_dem=true_dem,
            stim_dem=stim_dem,
        )

        for epoch in range(1, epochs + 1):
            g2d.train()
            tn.train()
            losses: list[float] = []

            for (det_batch,) in dataloader:
                # 调整 batch 形状
                det_batch = det_batch.reshape(nb, mini_batch, -1)
                optim.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for i in range(nb):
                    x = det_batch[i].to(device=device, dtype=dtype)
                    # 从 g2d 获取底层 gate 映射出的 error model 概率
                    dem_p = g2d()
                    # 传入 TN，此时作为 DEM 先验
                    loss = tn(x, priors=dem_p) / nb
                    loss.backward()
                    loss_sum += float(loss.detach().cpu().item())
                optim.step()
                losses.append(loss_sum)

            nll = float(np.mean(losses))
            is_ckpt = checkpoint_every > 0 and epoch % checkpoint_every == 0
            if is_ckpt or epoch == epochs:
                log_checkpoint(
                    log,
                    tag="checkpoint",
                    epoch=epoch,
                    nll=nll,
                    g2d=g2d,
                    true_gate=true_gate,
                    true_dem=true_dem,
                    stim_dem=stim_dem,
                )

        with torch.no_grad():
            final_dem = g2d().detach().cpu()
            final_gate = g2d.gate_probs().detach().cpu()
        save_path = CHECKPOINT_DIR / f"{log_path.stem}.pt"
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "distance": distance,
                "rounds": rounds,
                "error_prob": error_prob,
                "noise_model": noise_model,
                "true_gate_probs": true_gate.detach().cpu(),
                "true_dem_probs": true_dem.detach().cpu(),
                "stim_dem_probs": stim_dem.detach().cpu(),
                "gate_probs": final_gate,
                "dem_probs": final_dem,
                "param_sharing": param_sharing,
                "slot_keys": g2d.slot_keys,
            },
            save_path,
        )
        log.write(f"# saved {save_path}\n")
        print(f"[done] checkpoint → {save_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate-level DMLE on surface code via TN (Stim circuit).")
    ap.add_argument("--distance", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--error-prob", type=float, default=0.001)
    ap.add_argument(
        "--noise-model",
        type=str,
        default="si1000",
        choices=("depolarizing", "si1000"),
    )
    ap.add_argument("--num-shots", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=75328)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=10_000)
    ap.add_argument("--mini-batch", type=int, default=1_000)
    ap.add_argument(
        "--ctg-max-time",
        type=int,
        default=60,
        help="Max time (seconds) for cotengra to search for contraction path.",
    )
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument(
        "--no-perturb-init",
        action="store_true",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
    )
    ap.add_argument(
        "--log",
        type=str,
        default=None,
    )
    ap.add_argument(
        "--param-sharing",
        type=str,
        default="time_shared_dep2",
        choices=("elementary", "dep", "time_shared","time_shared_dep2"),
    )
    args = ap.parse_args()

    log_path = resolve_log_path(
        args.log,
        distance=args.distance,
        rounds=args.rounds,
        error_prob=args.error_prob,
        param_sharing=args.param_sharing,
        noise_model=args.noise_model,
    )

    train(
        distance=args.distance,
        rounds=args.rounds,
        error_prob=args.error_prob,
        noise_model=args.noise_model,
        num_shots=args.num_shots,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mini_batch=args.mini_batch,
        ctg_max_time=args.ctg_max_time,
        lr=args.lr,
        perturb_init=not args.no_perturb_init,
        device=args.device,
        log_path=log_path,
        checkpoint_every=args.checkpoint_every,
        param_sharing=args.param_sharing,
    )


if __name__ == "__main__":
    main()