"""
Optimize repetition-code **gate** noise via GateNoiseToDEM + PlanarNet.

Uses Stim's standard ``repetition_code:memory`` circuit (no experimental HDF5).
Detection events are sampled from the circuit DEM; learnable parameters are the
gate noise rates in :class:`src.g2dem.GateNoiseToDEM` (default: ``time_shared``). Each step:

    gate_probs → g2d() → DEM error probabilities → PlanarNet NLL

Run from repo root::

    python gate/pl_opt.py --distance 3 --rounds 5 --epochs 500 --device cuda:7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import stim
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "gate" / "log" / "rep" / "sim"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.g2dem import GateNoiseToDEM, ParamSharing  # noqa: E402
from src.model import PlanarNet  # noqa: E402
from src.utils import get_error_rates, rep_cir  # noqa: E402


def build_repetition_circuit(
    *,
    distance: int,
    rounds: int,
    error_prob: float,
) -> stim.Circuit:
    return stim.Circuit.generated(
        code_task="repetition_code:memory",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=error_prob,
        before_measure_flip_probability=error_prob,
        after_reset_flip_probability=error_prob,
        before_round_data_depolarization=error_prob,
    )


def default_log_filename(
    *,
    distance: int,
    rounds: int,
    error_prob: float,
    param_sharing: ParamSharing,
) -> str:
    return f"rep_d{distance}r{rounds}_p{error_prob}_{param_sharing}.txt"


def resolve_log_path(
    user_log: str | None,
    *,
    distance: int,
    rounds: int,
    error_prob: float,
    param_sharing: ParamSharing,
) -> Path:
    name = Path(user_log).name if user_log else default_log_filename(
        distance=distance,
        rounds=rounds,
        error_prob=error_prob,
        param_sharing=param_sharing,
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

    lines = [f"[{tag}] epoch={epoch}"]
    if nll is not None:
        lines[0] += f" nll={nll:.6f}"
    lines.append(f"  gate_error_rates: {gate_p!r}")
    lines.append(f"  dem_error_rates: {dem_p!r}")
    lines.append(
        f"  gate vs true: max_rel={gate_err['max_rel']:.3e} mean_rel={gate_err['mean_rel']:.3e}"
    )
    lines.append(
        f"  dem vs true: max_rel={dem_err['max_rel']:.3e} mean_rel={dem_err['mean_rel']:.3e}"
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
) -> tuple[GateNoiseToDEM, PlanarNet, torch.Tensor, torch.Tensor]:
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    rep = rep_cir(distance, rounds)
    rep.reorder(dem)

    ref_er = torch.tensor(get_error_rates(dem), dtype=dtype, device=device)
    if ref_er.numel() != rep.n:
        raise RuntimeError(f"DEM has {ref_er.numel()} errors but rep_cir.n={rep.n}")

    g2d = GateNoiseToDEM(
        circuit,
        learnable=True,
        param_mode="logit",
        param_sharing=param_sharing,
        dtype=dtype,
        device=device,
    )
    if g2d.num_dem != rep.n:
        raise RuntimeError(f"g2d.num_dem={g2d.num_dem} != rep.n={rep.n}")

    true_gate = g2d.gate_probs().detach().clone()
    true_dem = g2d.ref_dem_probs.detach().clone()

    if perturb_init:
        with torch.no_grad():
            init_gate = perturb_probabilities_like_pl_opt(true_gate)
            g2d.gate_param.copy_(prob_to_logit(init_gate))

    planar = PlanarNet(
        abstract_code=rep,
        init_priors=ref_er,
        dev=device,
        learn_priors=False,
    )
    return g2d, planar, true_gate, true_dem


def train(
    *,
    distance: int,
    rounds: int,
    error_prob: float,
    num_shots: int,
    seed: int,
    epochs: int,
    batch_size: int,
    mini_batch: int,
    lr: float,
    perturb_init: bool,
    device: str,
    log_path: Path,
    checkpoint_every: int,
    param_sharing: ParamSharing,
) -> None:
    dtype = torch.float64
    torch.manual_seed(seed)

    circuit = build_repetition_circuit(distance=distance, rounds=rounds, error_prob=error_prob)
    dets = sample_detection_events(circuit, num_shots=num_shots, seed=seed)
    print(f"[data] {num_shots} shots, {dets.shape[1]} detectors", flush=True)

    g2d, planar, true_gate, true_dem = setup_models(
        circuit,
        distance=distance,
        rounds=rounds,
        device=device,
        dtype=dtype,
        perturb_init=perturb_init,
        param_sharing=param_sharing,
    )

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
            f"# repetition_code:memory d={distance} r={rounds} p={error_prob}\n"
            f"# param_sharing={param_sharing} gate_params={g2d.num_slots} "
            f"elementary_sites={g2d.num_elementary}\n"
            f"# shots={num_shots} epochs={epochs} batch_size={batch_size} "
            f"mini_batch={mini_batch} lr={lr}\n"
            f"# dem errors={g2d.num_dem} perturb_init={int(perturb_init)}\n"
            f"# true gate (circuit):\n{repr(true_gate.detach().cpu().numpy())}\n"
            f"# true dem (Stim):\n{repr(true_dem.detach().cpu().numpy())}\n"
        )
        if perturb_init:
            log.write(
                f"# init gate (perturbed):\n{repr(g2d.gate_probs().detach().cpu().numpy())}\n"
            )
        log.flush()

        with torch.no_grad():
            probe = torch.from_numpy(dets[:mini_batch]).to(device=device, dtype=dtype)
            init_nll = float(planar(probe, priors=g2d()).item())
        print(f"[init] nll~{init_nll:.4f}", flush=True)
        log_checkpoint(
            log,
            tag="init",
            epoch=0,
            nll=init_nll,
            g2d=g2d,
            true_gate=true_gate,
            true_dem=true_dem,
        )

        for epoch in range(1, epochs + 1):
            g2d.train()
            planar.train()
            losses: list[float] = []

            for (det_batch,) in dataloader:
                det_batch = det_batch.reshape(nb, mini_batch, -1)
                optim.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for i in range(nb):
                    x = det_batch[i].to(device=device, dtype=dtype)
                    dem_p = g2d()
                    loss = planar(x, priors=dem_p) / nb
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
                )

        with torch.no_grad():
            final_dem = g2d().detach().cpu()
            final_gate = g2d.gate_probs().detach().cpu()
        save_path = log_path.with_suffix(".pt")
        torch.save(
            {
                "distance": distance,
                "rounds": rounds,
                "error_prob": error_prob,
                "true_gate_probs": true_gate.detach().cpu(),
                "true_dem_probs": true_dem.detach().cpu(),
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
    ap = argparse.ArgumentParser(description="Gate-level DMLE on repetition code (Stim circuit).")
    ap.add_argument("--distance", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--error-prob", type=float, default=0.001)
    ap.add_argument("--num-shots", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=75328)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=10_000)
    ap.add_argument("--mini-batch", type=int, default=1_000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument(
        "--no-perturb-init",
        action="store_true",
        help="skip pl_opt-style random perturbation on gate probabilities at init",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="log optimized gate/dem error rates every N epochs (0 = final only)",
    )
    ap.add_argument(
        "--log",
        type=str,
        default=None,
        help=(
            "log filename under gate/log/rep/sim/ "
            "(default: rep_d{d}r{r}_p{p}_{param_sharing}.txt)"
        ),
    )
    ap.add_argument(
        "--param-sharing",
        type=str,
        default="time_shared",
        choices=("elementary", "dep", "time_shared"),
        help="how gate noise scalars are tied across circuit sites (see g2dem.param_sharing)",
    )
    args = ap.parse_args()

    log_path = resolve_log_path(
        args.log,
        distance=args.distance,
        rounds=args.rounds,
        error_prob=args.error_prob,
        param_sharing=args.param_sharing,
    )

    train(
        distance=args.distance,
        rounds=args.rounds,
        error_prob=args.error_prob,
        num_shots=args.num_shots,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mini_batch=args.mini_batch,
        lr=args.lr,
        perturb_init=not args.no_perturb_init,
        device=args.device,
        log_path=log_path,
        checkpoint_every=args.checkpoint_every,
        param_sharing=args.param_sharing,
    )


if __name__ == "__main__":
    main()
