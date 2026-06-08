"""
Decode sycamore_old experiments using cross-round averaged gate checkpoints.

Workflow (per experiment group, e.g. ``syc_old_bX_d3r*_c3_5_time_shared_dep12_M_DD``):

1. Scan ``gate/log/sur/syc_old/checkpoint/`` for checkpoints that share basis, distance,
   center, and ``param_sharing`` but differ in code round ``r``.
2. For each round, check ``gate/path/{log_stem}_path.pkl``; search missing paths with
   cotengra in **ascending** round order (small ``r`` first).
3. Average learnable slot probabilities across all discovered round checkpoints.
4. Replay the average into each round's dataset ``circuit_noisy.stim`` → patched noisy
   circuit per round.
5. From each patched circuit build two DEMs for decoding:

   - ``dem_without_decompose`` (``decompose_errors=False``) → TN decoder (g2dem replay
     error probs on the non-decomposed stim DEM topology);
   - ``dem_decompose`` (``decompose_errors=True``) → :class:`src.decoder.MWPM_dem`.

6. Also decode with the original dataset noisy circuit stim DEMs (no learned gate).
7. Per code round, decode with that round's **trained checkpoint** gate (no cross-round
   average).
8. For code rounds **r13–r25** (if present under ``data/sycamore_old``), decode **stim**
   and **avg** only; ``avg`` uses the same checkpoint-round mean gate applied to each
   larger-round ``circuit_noisy.stim`` (no per-round checkpoint for r13+).

Logs: ``gate/log/sur/syc_old/decoding/decoding_d{d}_c{row}_{col}_{sharing}.txt``
(one file per distance/center/sharing, all bases and code rounds inside)

Run from repo root (``qec`` conda environment)::

    conda activate qec
    python gate/decoding.py
    python gate/decoding.py --center 3_5
    python gate/decoding.py --extended-only --extended-rounds 13-25

After checkpoint rounds (e.g. r05–r11) are logged, continue extended rounds only::

    python gate/decoding.py --extended-only

``--extended-only`` skips checkpoint-round decoding, runs r13–r25 stim+avg only,
and **appends** to the existing combined log (does not overwrite).

By default both ``--basis X`` and ``--basis Z`` are processed, with
``param_sharing=time_shared_dep12_M_DD``.
"""

from __future__ import annotations

DEFAULT_BASES: tuple[str, ...] = ("X", "Z")

import argparse
import datetime
import gc
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import get_args

import numpy as np
import stim
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "sycamore_old"
LOG_DIR = REPO_ROOT / "gate" / "log" / "sur" / "syc_old"
CHECKPOINT_DIR = LOG_DIR / "checkpoint"
DECODING_DIR = LOG_DIR / "decoding"
CIRCUIT_DIR = DECODING_DIR / "circuits"
PATH_DIR = REPO_ROOT / "gate" / "path"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gate.tn_syc_old import (  # noqa: E402
    align_dets_to_pcm,
    compute_tn_logical_error_rate,
    get_or_create_contraction_path,
    load_detection_events,
    load_noisy_circuit,
    load_observable_flips,
    parse_center,
    resolve_experiment_dir,
    resolve_log_path,
    resolve_path_file,
    setup_models_with_cached_path,
    setup_tn_decoder,
)
from src import MWPM_dem, PCM, TensorNetwork, TensorNetworkDecoder  # noqa: E402
from src.g2dem import (  # noqa: E402
    GateNoiseToDEM,
    ParamSharing,
    compile_circuit,
    dep2_learn_probs_from_gate_probs,
)
from src.g2dem import _circuit_with_gate_probs  # noqa: E402
from src.utils import get_error_rates  # noqa: E402

DEFAULT_PARAM_SHARING: ParamSharing = "time_shared_dep12_M_DD"
DEFAULT_MINI_BATCH: int = 10_000
DEFAULT_EXTENDED_ROUNDS: tuple[int, ...] = tuple(range(13, 26))

CHECKPOINT_STEM_RE = re.compile(
    r"^syc_old_b([XZ])_d(\d+)r(\d{2})_c(\d+)_(\d+)_(.+)$"
)


@dataclass(frozen=True)
class CheckpointGroup:
    basis: str
    distance: int
    center_row: int
    center_col: int
    param_sharing: str
    round_ckpts: dict[int, Path] = field(default_factory=dict)

    @property
    def rounds(self) -> list[int]:
        return sorted(self.round_ckpts)

    def stem_for_round(self, rounds: int) -> str:
        return checkpoint_stem(
            basis=self.basis,
            distance=self.distance,
            rounds=rounds,
            center_row=self.center_row,
            center_col=self.center_col,
            param_sharing=self.param_sharing,
        )


@dataclass(frozen=True)
class ExperimentBundle:
    """All basis / code-round groups for one distance, center, and param_sharing."""

    distance: int
    center_row: int
    center_col: int
    param_sharing: str
    groups_by_basis: dict[str, CheckpointGroup]

    @property
    def bases(self) -> list[str]:
        return sorted(self.groups_by_basis)


@dataclass
class RoundDecodeResult:
    basis: str
    rounds: int
    stem: str
    experiment_dir: Path
    results: dict[str, float]
    init_stim_max_rel: float
    avg_rounds: list[int]
    group: CheckpointGroup
    art: RoundDemArtifacts


@dataclass
class RoundDemArtifactsStimAvg:
    """DEM artifacts for extended rounds (stim + cross-checkpoint avg gate only)."""

    rounds: int
    dem_decompose: stim.DetectorErrorModel
    tn_error_rates: torch.Tensor
    mwpm_error_rates: np.ndarray
    patched_circuit_path: Path | None = None


@dataclass
class RoundDemArtifacts:
    rounds: int
    patched_circuit: stim.Circuit
    dem_without_decompose: stim.DetectorErrorModel
    dem_decompose: stim.DetectorErrorModel
    tn_error_rates: torch.Tensor
    mwpm_error_rates: np.ndarray
    init_tn_error_rates: torch.Tensor
    init_mwpm_error_rates: np.ndarray
    init_dem_decompose: stim.DetectorErrorModel
    ckpt_tn_error_rates: torch.Tensor
    ckpt_mwpm_error_rates: np.ndarray
    ckpt_dem_decompose: stim.DetectorErrorModel
    ckpt_path: Path
    patched_circuit_path: Path | None = None
    ckpt_patched_circuit_path: Path | None = None


def checkpoint_stem(
    *,
    basis: str,
    distance: int,
    rounds: int,
    center_row: int,
    center_col: int,
    param_sharing: str,
) -> str:
    return (
        f"syc_old_b{basis}_d{distance}r{rounds:02d}_"
        f"c{center_row}_{center_col}_{param_sharing}"
    )


def parse_checkpoint_stem(stem: str) -> dict[str, int | str]:
    m = CHECKPOINT_STEM_RE.match(stem)
    if m is None:
        raise ValueError(f"cannot parse checkpoint stem: {stem!r}")
    basis, distance, rounds, center_row, center_col, param_sharing = m.groups()
    return {
        "basis": basis,
        "distance": int(distance),
        "rounds": int(rounds),
        "center_row": int(center_row),
        "center_col": int(center_col),
        "param_sharing": param_sharing,
    }


def resolve_bases(basis: str | None) -> tuple[str, ...]:
    """Default: process both X and Z."""
    if basis is None:
        return DEFAULT_BASES
    return (basis,)


def discover_checkpoint_groups(
    checkpoint_dir: Path = CHECKPOINT_DIR,
    *,
    bases: tuple[str, ...] = DEFAULT_BASES,
    distance: int | None = None,
    center_row: int | None = None,
    center_col: int | None = None,
    param_sharing: str = DEFAULT_PARAM_SHARING,
) -> list[CheckpointGroup]:
    """Group ``syc_old_b*_d*r*_c*_*_*.pt`` files by everything except code round."""
    buckets: dict[tuple[str, int, int, int, str], dict[int, Path]] = {}

    for path in sorted(checkpoint_dir.glob("syc_old_b*.pt")):
        try:
            meta = parse_checkpoint_stem(path.stem)
        except ValueError:
            continue
        if meta["basis"] not in bases:
            continue
        if distance is not None and meta["distance"] != distance:
            continue
        if center_row is not None and meta["center_row"] != center_row:
            continue
        if center_col is not None and meta["center_col"] != center_col:
            continue
        if meta["param_sharing"] != param_sharing:
            continue

        key = (
            str(meta["basis"]),
            int(meta["distance"]),
            int(meta["center_row"]),
            int(meta["center_col"]),
            str(meta["param_sharing"]),
        )
        buckets.setdefault(key, {})[int(meta["rounds"])] = path

    groups: list[CheckpointGroup] = []
    for (b, d, cr, cc, sharing), round_ckpts in sorted(buckets.items()):
        if not round_ckpts:
            continue
        groups.append(
            CheckpointGroup(
                basis=b,
                distance=d,
                center_row=cr,
                center_col=cc,
                param_sharing=sharing,
                round_ckpts=round_ckpts,
            )
        )
    return groups


def cluster_experiment_bundles(groups: list[CheckpointGroup]) -> list[ExperimentBundle]:
    buckets: dict[tuple[int, int, int, str], dict[str, CheckpointGroup]] = {}
    for group in groups:
        key = (group.distance, group.center_row, group.center_col, group.param_sharing)
        buckets.setdefault(key, {})[group.basis] = group
    return [
        ExperimentBundle(
            distance=d,
            center_row=cr,
            center_col=cc,
            param_sharing=sharing,
            groups_by_basis=by_basis,
        )
        for (d, cr, cc, sharing), by_basis in sorted(buckets.items())
    ]


def combined_log_path(bundle: ExperimentBundle) -> Path:
    return (
        DECODING_DIR
        / f"decoding_d{bundle.distance}_c{bundle.center_row}_{bundle.center_col}_"
        f"{bundle.param_sharing}.txt"
    )


def log_path_for_group(group: CheckpointGroup, rounds: int) -> Path:
    return resolve_log_path(
        None,
        basis=group.basis,
        distance=group.distance,
        rounds=rounds,
        center_row=group.center_row,
        center_col=group.center_col,
        param_sharing=group.param_sharing,  # type: ignore[arg-type]
    )


def missing_contraction_path_rounds(group: CheckpointGroup) -> list[int]:
    return missing_contraction_path_rounds_for(group, group.rounds)


def missing_contraction_path_rounds_for(
    group: CheckpointGroup,
    rounds: list[int],
) -> list[int]:
    missing: list[int] = []
    for r in rounds:
        path_file = resolve_path_file(log_path_for_group(group, r))
        if not path_file.is_file():
            missing.append(r)
    return missing


def discover_extended_rounds(
    group: CheckpointGroup,
    *,
    data_root: Path,
    rounds: tuple[int, ...] = DEFAULT_EXTENDED_ROUNDS,
) -> list[int]:
    """Code rounds present under ``data_root`` but not necessarily in checkpoints."""
    found: list[int] = []
    for r in rounds:
        if r in group.round_ckpts:
            continue
        try:
            resolve_experiment_dir(
                data_root=data_root,
                basis=group.basis,
                distance=group.distance,
                rounds=r,
                center_row=group.center_row,
                center_col=group.center_col,
            )
        except FileNotFoundError:
            continue
        found.append(r)
    return found


def ensure_contraction_paths_ascending(
    group: CheckpointGroup,
    *,
    data_root: Path,
    device: str,
    dtype: torch.dtype,
    mini_batch: int,
    ctg_max_time: int,
    rounds: list[int] | None = None,
) -> list[int]:
    """Find and save missing TN paths, processing rounds from smallest to largest."""
    param_sharing: ParamSharing = group.param_sharing  # type: ignore[assignment]
    round_list = sorted(rounds if rounds is not None else group.rounds)
    missing = missing_contraction_path_rounds_for(group, round_list)
    if not missing:
        label = "extended" if rounds is not None else group.param_sharing
        print(f"[paths] all contraction paths exist ({label}, basis={group.basis})", flush=True)
        return []

    print(
        f"[paths] basis={group.basis} missing rounds (ascending): "
        f"{[f'r{r:02d}' for r in missing]}",
        flush=True,
    )

    for r in missing:
        stem = group.stem_for_round(r)
        log_path = log_path_for_group(group, r)
        path_file = resolve_path_file(log_path)
        experiment_dir = resolve_experiment_dir(
            data_root=data_root,
            basis=group.basis,
            distance=group.distance,
            rounds=r,
            center_row=group.center_row,
            center_col=group.center_col,
        )
        circuit = load_noisy_circuit(experiment_dir)
        dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
        pcm, l = PCM(dem)
        logical = l.flatten() if l.size > 0 else None
        tn = TensorNetwork(
            pcm=pcm,
            l=logical,
            dev=device,
            dtype=dtype,
            learn_priors=False,
        )
        print(
            f"[paths] searching r{r:02d} -> {path_file.name} "
            f"(minibatch={mini_batch}, max_time={ctg_max_time}s)",
            flush=True,
        )
        get_or_create_contraction_path(
            tn,
            path_file,
            minibatch=mini_batch,
            max_time=ctg_max_time,
        )
        del tn
        gc.collect()
        print(f"[paths] saved {path_file}", flush=True)

    return missing


def _align_learn_probs_to_ref(
    learn: torch.Tensor,
    ck_slot_keys: list,
    ref_slot_keys: list,
    *,
    round_label: str,
) -> torch.Tensor:
    key_to_val = {str(k): learn[i] for i, k in enumerate(ck_slot_keys)}
    ref_str = [str(k) for k in ref_slot_keys]
    missing = sorted(set(ref_str) - set(key_to_val))
    if missing:
        raise ValueError(
            f"{round_label}: checkpoint slot_keys missing {len(missing)} keys "
            f"required by reference, e.g. {missing[0]!r}"
        )
    return torch.stack([key_to_val[k] for k in ref_str], dim=0)


def average_learnable_probs(
    group: CheckpointGroup,
    ref_slot_keys: list,
    *,
    data_root: Path,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[int]]:
    """Average learnable-slot probabilities over all rounds in the group."""
    param_sharing: ParamSharing = group.param_sharing  # type: ignore[assignment]
    learn_stack: list[torch.Tensor] = []
    used_rounds: list[int] = []

    for r in group.rounds:
        ckpt_path = group.round_ckpts[r]
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        exp_dir = resolve_experiment_dir(
            data_root=data_root,
            basis=group.basis,
            distance=group.distance,
            rounds=r,
            center_row=group.center_row,
            center_col=group.center_col,
        )
        circuit = load_noisy_circuit(exp_dir)
        g2d = GateNoiseToDEM(
            circuit,
            learnable=False,
            param_sharing=param_sharing,
            dtype=dtype,
            device=device,
        )
        learn = dep2_learn_probs_from_gate_probs(g2d, ck["gate_probs"])
        aligned = _align_learn_probs_to_ref(
            learn,
            ck["slot_keys"],
            ref_slot_keys,
            round_label=f"r{r:02d}",
        )
        learn_stack.append(aligned)
        used_rounds.append(r)

    avg_learn = torch.stack(learn_stack, dim=0).mean(dim=0)
    print(
        f"[average] rounds={used_rounds} n_slots={avg_learn.numel()}",
        flush=True,
    )
    return avg_learn, used_rounds


def learn_probs_to_gate_param(g2d: GateNoiseToDEM, learn_probs: torch.Tensor) -> torch.Tensor:
    learn_probs = learn_probs.clamp(1e-12, 1.0 - 1e-12)
    if g2d.param_mode == "logit":
        return torch.log(learn_probs / (1.0 - learn_probs))
    return learn_probs.clone()


def apply_learn_probs(g2d: GateNoiseToDEM, learn_probs: torch.Tensor) -> None:
    with torch.no_grad():
        g2d.gate_param.copy_(learn_probs_to_gate_param(g2d, learn_probs))


def build_patched_noisy_circuit(
    base_circuit: stim.Circuit,
    gate_probs: torch.Tensor,
    *,
    param_key_to_id: dict,
) -> stim.Circuit:
    return _circuit_with_gate_probs(
        base_circuit,
        gate_probs.detach().cpu().numpy(),
        param_key_to_id=param_key_to_id,
    )


def dem_probs_decompose_true(
    circuit: stim.Circuit,
    gate_probs: torch.Tensor,
    *,
    param_key_to_id: dict,
) -> np.ndarray:
    """Error rates from ``decompose_errors=True`` DEM on a gate-patched circuit."""
    patched = build_patched_noisy_circuit(
        circuit,
        gate_probs,
        param_key_to_id=param_key_to_id,
    )
    dem = patched.detector_error_model(decompose_errors=True, flatten_loops=True)
    return get_error_rates(dem)


def _patched_dem_pair(
    base_circuit: stim.Circuit,
    gate_probs: torch.Tensor,
    *,
    param_key_to_id: dict,
) -> tuple[stim.Circuit, stim.DetectorErrorModel, stim.DetectorErrorModel]:
    patched = build_patched_noisy_circuit(
        base_circuit,
        gate_probs,
        param_key_to_id=param_key_to_id,
    )
    dem_false = patched.detector_error_model(
        decompose_errors=False, flatten_loops=True
    )
    dem_true = patched.detector_error_model(
        decompose_errors=True, flatten_loops=True
    )
    return patched, dem_false, dem_true


def build_round_dem_artifacts(
    group: CheckpointGroup,
    avg_learn: torch.Tensor,
    *,
    data_root: Path,
    device: str,
    dtype: torch.dtype,
    save_circuits: bool,
) -> dict[int, RoundDemArtifacts]:
    """Build per-round patched noisy circuits and both decoding DEMs."""
    param_sharing: ParamSharing = group.param_sharing  # type: ignore[assignment]
    artifacts: dict[int, RoundDemArtifacts] = {}

    if save_circuits:
        CIRCUIT_DIR.mkdir(parents=True, exist_ok=True)

    for r in group.rounds:
        experiment_dir = resolve_experiment_dir(
            data_root=data_root,
            basis=group.basis,
            distance=group.distance,
            rounds=r,
            center_row=group.center_row,
            center_col=group.center_col,
        )
        base_circuit = load_noisy_circuit(experiment_dir)
        meta = compile_circuit(base_circuit, param_sharing=param_sharing)
        g2d = GateNoiseToDEM(
            base_circuit,
            learnable=True,
            param_sharing=param_sharing,
            dtype=dtype,
            device=device,
        )
        with torch.no_grad():
            init_gate_p = g2d.gate_probs()
            init_tn_er = g2d()
        init_patched = build_patched_noisy_circuit(
            base_circuit,
            init_gate_p,
            param_key_to_id=meta["param_key_to_id"],
        )
        init_dem_decompose = init_patched.detector_error_model(
            decompose_errors=True, flatten_loops=True
        )
        init_mwpm_er = get_error_rates(init_dem_decompose)

        apply_learn_probs(g2d, avg_learn)

        with torch.no_grad():
            gate_p = g2d.gate_probs()
            tn_er = g2d()

        patched, dem_without_decompose, dem_decompose = _patched_dem_pair(
            base_circuit,
            gate_p,
            param_key_to_id=meta["param_key_to_id"],
        )

        ckpt_path = group.round_ckpts[r]
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "gate_probs" not in ck:
            raise KeyError(f"'gate_probs' missing in checkpoint: {ckpt_path}")
        if "dem_probs" not in ck:
            raise KeyError(f"'dem_probs' missing in checkpoint: {ckpt_path}")
        ckpt_gate = ck["gate_probs"].to(dtype=dtype, device=device)
        ckpt_tn_er = ck["dem_probs"].to(dtype=dtype, device=device)
        if ckpt_tn_er.shape[0] != g2d.num_dem:
            raise ValueError(
                f"r{r:02d} checkpoint dem_probs length {ckpt_tn_er.shape[0]} "
                f"!= g2d.num_dem {g2d.num_dem} ({ckpt_path})"
            )

        ckpt_patched, _, ckpt_dem_decompose = _patched_dem_pair(
            base_circuit,
            ckpt_gate,
            param_key_to_id=meta["param_key_to_id"],
        )
        ckpt_mwpm_er = get_error_rates(ckpt_dem_decompose)

        circuit_path: Path | None = None
        ckpt_circuit_path: Path | None = None
        if save_circuits:
            stem = group.stem_for_round(r)
            circuit_path = CIRCUIT_DIR / f"{stem}_avg_gate.stim"
            with open(circuit_path, "w", encoding="utf-8") as f:
                print(patched, file=f)
            ckpt_circuit_path = CIRCUIT_DIR / f"{stem}_ckpt_gate.stim"
            with open(ckpt_circuit_path, "w", encoding="utf-8") as f:
                print(ckpt_patched, file=f)

        artifacts[r] = RoundDemArtifacts(
            rounds=r,
            patched_circuit=patched,
            dem_without_decompose=dem_without_decompose,
            dem_decompose=dem_decompose,
            tn_error_rates=tn_er.detach(),
            mwpm_error_rates=get_error_rates(dem_decompose),
            init_tn_error_rates=init_tn_er.detach(),
            init_mwpm_error_rates=init_mwpm_er,
            init_dem_decompose=init_dem_decompose,
            ckpt_tn_error_rates=ckpt_tn_er.detach(),
            ckpt_mwpm_error_rates=ckpt_mwpm_er,
            ckpt_dem_decompose=ckpt_dem_decompose,
            ckpt_path=ckpt_path,
            patched_circuit_path=circuit_path,
            ckpt_patched_circuit_path=ckpt_circuit_path,
        )
        print(
            f"[circuit] r{r:02d} avg_gate dem_false={dem_without_decompose.num_errors} "
            f"dem_true={dem_decompose.num_errors} | "
            f"ckpt_gate dem_true={ckpt_dem_decompose.num_errors} "
            f"<- {ckpt_path.name}",
            flush=True,
        )

    return artifacts


def build_stim_avg_artifacts(
    group: CheckpointGroup,
    rounds: list[int],
    avg_learn: torch.Tensor,
    *,
    data_root: Path,
    device: str,
    dtype: torch.dtype,
    save_circuits: bool,
) -> dict[int, RoundDemArtifactsStimAvg]:
    """Apply checkpoint-round-averaged gate probs to larger-round circuits (stim+avg only)."""
    param_sharing: ParamSharing = group.param_sharing  # type: ignore[assignment]
    artifacts: dict[int, RoundDemArtifactsStimAvg] = {}

    if save_circuits:
        CIRCUIT_DIR.mkdir(parents=True, exist_ok=True)

    for r in sorted(rounds):
        experiment_dir = resolve_experiment_dir(
            data_root=data_root,
            basis=group.basis,
            distance=group.distance,
            rounds=r,
            center_row=group.center_row,
            center_col=group.center_col,
        )
        base_circuit = load_noisy_circuit(experiment_dir)
        meta = compile_circuit(base_circuit, param_sharing=param_sharing)
        g2d = GateNoiseToDEM(
            base_circuit,
            learnable=True,
            param_sharing=param_sharing,
            dtype=dtype,
            device=device,
        )
        apply_learn_probs(g2d, avg_learn)
        with torch.no_grad():
            gate_p = g2d.gate_probs()
            tn_er = g2d()
        _, _, dem_decompose = _patched_dem_pair(
            base_circuit,
            gate_p,
            param_key_to_id=meta["param_key_to_id"],
        )

        circuit_path: Path | None = None
        if save_circuits:
            stem = group.stem_for_round(r)
            circuit_path = CIRCUIT_DIR / f"{stem}_avg_gate.stim"
            patched, _, _ = _patched_dem_pair(
                base_circuit,
                gate_p,
                param_key_to_id=meta["param_key_to_id"],
            )
            with open(circuit_path, "w", encoding="utf-8") as f:
                print(patched, file=f)

        artifacts[r] = RoundDemArtifactsStimAvg(
            rounds=r,
            dem_decompose=dem_decompose,
            tn_error_rates=tn_er.detach(),
            mwpm_error_rates=get_error_rates(dem_decompose),
            patched_circuit_path=circuit_path,
        )
        print(
            f"[circuit] r{r:02d} extended avg_gate (from checkpoint-mean slots) "
            f"dem_decompose_errors={dem_decompose.num_errors}",
            flush=True,
        )
    return artifacts


def compute_mwpm_logical_error_rate(
    mwpm: MWPM_dem,
    dets: np.ndarray,
    obvs: np.ndarray,
    error_rates: np.ndarray,
    *,
    mini_batch: int,
) -> float:
    n = int(dets.shape[0])
    if n == 0:
        return float("nan")
    if n <= mini_batch:
        return float(mwpm.logical_error_rate(dets, obvs, error_rates=error_rates))

    total_errors = 0.0
    for start in range(0, n, mini_batch):
        end = min(start + mini_batch, n)
        batch_size = end - start
        batch_ler = mwpm.logical_error_rate(
            dets[start:end], obvs[start:end], error_rates=error_rates
        )
        total_errors += float(batch_ler) * batch_size
    return total_errors / n


def setup_tn_decoder_for_round(
    group: CheckpointGroup,
    rounds: int,
    *,
    data_root: Path,
    device: str,
    dtype: torch.dtype,
    mini_batch: int,
    ctg_max_time: int,
) -> tuple[TensorNetworkDecoder, np.ndarray, torch.Tensor]:
    """TN decoder on **dataset** ``circuit_noisy.stim`` PCM (same as ``tn_syc_old`` training)."""
    param_sharing: ParamSharing = group.param_sharing  # type: ignore[assignment]
    experiment_dir = resolve_experiment_dir(
        data_root=data_root,
        basis=group.basis,
        distance=group.distance,
        rounds=rounds,
        center_row=group.center_row,
        center_col=group.center_col,
    )
    circuit = load_noisy_circuit(experiment_dir)
    log_path = log_path_for_group(group, rounds)
    path_file = resolve_path_file(log_path)
    _, tn, _, _, stim_dem_probs, pcm = setup_models_with_cached_path(
        circuit,
        distance=group.distance,
        rounds=rounds,
        device=device,
        dtype=dtype,
        param_sharing=param_sharing,
        mini_batch=mini_batch,
        ctg_max_time=ctg_max_time,
        path_file=path_file,
    )
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    _, l = PCM(dem)
    decoder = setup_tn_decoder(pcm, l, tn, device=device, dtype=dtype)
    return decoder, pcm, stim_dem_probs


def write_bundle_header(
    log,
    bundle: ExperimentBundle,
    *,
    avg_rounds_by_basis: dict[str, list[int]],
    mwpm_correlations: bool,
) -> None:
    log.write(
        f"# decoding d={bundle.distance} center={bundle.center_row}_{bundle.center_col} "
        f"param_sharing={bundle.param_sharing}\n"
    )
    log.write(f"# bases={bundle.bases}\n")
    for basis in bundle.bases:
        group = bundle.groups_by_basis[basis]
        avg_rounds = avg_rounds_by_basis[basis]
        log.write(f"# basis={basis} code_rounds={group.rounds} ")
        log.write(f"gate_average_rounds={avg_rounds}\n")
        for ar in avg_rounds:
            log.write(f"#   r{ar:02d} -> {group.round_ckpts[ar]}\n")
    log.write(f"# mwpm_enable_correlations={mwpm_correlations}\n")
    log.write(
        "# tn_pcm=dataset circuit_noisy.stim; priors: stim_ref | init | avg | ckpt\n"
    )
    log.write(
        "# ckpt=per-round trained checkpoint gate (no cross-round average)\n"
    )
    log.write("# mwpm_dem: stim=base | init/avg/ckpt=patched (decompose=True)\n\n")
    log.flush()


def write_continuation_marker(log, *, label: str) -> None:
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    log.write(f"\n# === {label} @ {ts} ===\n\n")
    log.flush()


def write_round_section(log, result: RoundDecodeResult) -> None:
    log.write(f"[basis={result.basis} round=r{result.rounds:02d}]\n")
    log.write(f"experiment={result.experiment_dir}\n")
    log.write(f"circuit={result.experiment_dir / 'circuit_noisy.stim'}\n")
    if result.art.patched_circuit_path is not None:
        log.write(f"patched_circuit_avg={result.art.patched_circuit_path}\n")
    if result.art.ckpt_patched_circuit_path is not None:
        log.write(f"patched_circuit_ckpt={result.art.ckpt_patched_circuit_path}\n")
    log.write(f"checkpoint={result.art.ckpt_path}\n")
    log.write(f"init_vs_stim_ref_tn_priors_max_rel={result.init_stim_max_rel:.6e}\n")
    log.write(
        f"dem_false_errors={result.art.dem_without_decompose.num_errors} "
        f"dem_decompose_errors={result.art.dem_decompose.num_errors}\n"
    )
    log_path = log_path_for_group(result.group, result.rounds)
    log.write(f"contraction_path={resolve_path_file(log_path)}\n")
    for name, val in result.results.items():
        log.write(f"{name}={val:.8f}\n")
    log.write("\n")
    log.flush()
    for name, val in result.results.items():
        print(f"  [{result.stem}] {name}={val:.8f}", flush=True)


def decode_round(
    group: CheckpointGroup,
    art: RoundDemArtifacts,
    *,
    data_root: Path,
    device: str,
    dtype: torch.dtype,
    mini_batch: int,
    ctg_max_time: int,
    mwpm_correlations: bool,
    avg_rounds: list[int],
) -> RoundDecodeResult:
    r = art.rounds
    stem = group.stem_for_round(r)

    experiment_dir = resolve_experiment_dir(
        data_root=data_root,
        basis=group.basis,
        distance=group.distance,
        rounds=r,
        center_row=group.center_row,
        center_col=group.center_col,
    )
    base_circuit = load_noisy_circuit(experiment_dir)
    dets = load_detection_events(experiment_dir, base_circuit)
    obvs = load_observable_flips(experiment_dir)

    decoder, pcm, stim_ref_dem = setup_tn_decoder_for_round(
        group,
        r,
        data_root=data_root,
        device=device,
        dtype=dtype,
        mini_batch=mini_batch,
        ctg_max_time=ctg_max_time,
    )
    dets = align_dets_to_pcm(dets, base_circuit, pcm)

    stim_dem_true = base_circuit.detector_error_model(
        decompose_errors=True, flatten_loops=True
    )
    mwpm_stim = MWPM_dem(stim_dem_true, enable_correlations=mwpm_correlations)
    mwpm_init = MWPM_dem(art.init_dem_decompose, enable_correlations=mwpm_correlations)
    mwpm_avg = MWPM_dem(art.dem_decompose, enable_correlations=mwpm_correlations)
    mwpm_ckpt = MWPM_dem(art.ckpt_dem_decompose, enable_correlations=mwpm_correlations)

    stim_er_tn = stim_ref_dem.to(device=device, dtype=dtype)
    stim_er_true = get_error_rates(stim_dem_true)

    tn_er = art.tn_error_rates.to(device=device, dtype=dtype)
    init_tn_er = art.init_tn_error_rates.to(device=device, dtype=dtype)
    ckpt_tn_er = art.ckpt_tn_error_rates.to(device=device, dtype=dtype)

    init_stim_max_rel = float(
        (init_tn_er - stim_er_tn).abs().max()
        / stim_er_tn.abs().clamp_min(1e-30).max()
    )
    print(
        f"  [{stem}] init vs stim_ref dem (TN priors) max_rel={init_stim_max_rel:.3e}",
        flush=True,
    )

    results: dict[str, float] = {}

    results["ler_tn_init_gate_dem_without_decompose"] = compute_tn_logical_error_rate(
        decoder,
        dets,
        obvs,
        init_tn_er,
        device=device,
        decode_batch_size=mini_batch,
    )
    results["ler_mwpm_init_gate_dem_decompose"] = compute_mwpm_logical_error_rate(
        mwpm_init,
        dets,
        obvs,
        art.init_mwpm_error_rates,
        mini_batch=mini_batch,
    )
    results["ler_tn_stim_dem_without_decompose"] = compute_tn_logical_error_rate(
        decoder,
        dets,
        obvs,
        stim_er_tn,
        device=device,
        decode_batch_size=mini_batch,
    )
    results["ler_mwpm_stim_dem_decompose"] = compute_mwpm_logical_error_rate(
        mwpm_stim,
        dets,
        obvs,
        stim_er_true,
        mini_batch=mini_batch,
    )
    results["ler_tn_avg_gate_dem_without_decompose"] = compute_tn_logical_error_rate(
        decoder,
        dets,
        obvs,
        tn_er,
        device=device,
        decode_batch_size=mini_batch,
    )
    results["ler_mwpm_avg_gate_dem_decompose"] = compute_mwpm_logical_error_rate(
        mwpm_avg,
        dets,
        obvs,
        art.mwpm_error_rates,
        mini_batch=mini_batch,
    )
    results["ler_tn_ckpt_gate_dem_without_decompose"] = compute_tn_logical_error_rate(
        decoder,
        dets,
        obvs,
        ckpt_tn_er,
        device=device,
        decode_batch_size=mini_batch,
    )
    results["ler_mwpm_ckpt_gate_dem_decompose"] = compute_mwpm_logical_error_rate(
        mwpm_ckpt,
        dets,
        obvs,
        art.ckpt_mwpm_error_rates,
        mini_batch=mini_batch,
    )

    del decoder
    gc.collect()

    return RoundDecodeResult(
        basis=group.basis,
        rounds=r,
        stem=stem,
        experiment_dir=experiment_dir,
        results=results,
        init_stim_max_rel=init_stim_max_rel,
        avg_rounds=avg_rounds,
        group=group,
        art=art,
    )


def write_round_section_stim_avg(
    log,
    *,
    basis: str,
    rounds: int,
    stem: str,
    experiment_dir: Path,
    results: dict[str, float],
    avg_rounds: list[int],
    group: CheckpointGroup,
    art: RoundDemArtifactsStimAvg,
) -> None:
    log.write(f"[basis={basis} round=r{rounds:02d} mode=stim_avg_extended]\n")
    log.write(f"experiment={experiment_dir}\n")
    log.write(f"circuit={experiment_dir / 'circuit_noisy.stim'}\n")
    log.write(f"gate_average_rounds={avg_rounds}\n")
    if art.patched_circuit_path is not None:
        log.write(f"patched_circuit_avg={art.patched_circuit_path}\n")
    log.write(
        f"dem_decompose_errors={art.dem_decompose.num_errors} "
        "(avg gate from checkpoint-round mean)\n"
    )
    log_path = log_path_for_group(group, rounds)
    log.write(f"contraction_path={resolve_path_file(log_path)}\n")
    for name, val in results.items():
        log.write(f"{name}={val:.8f}\n")
    log.write("\n")
    log.flush()
    for name, val in results.items():
        print(f"  [{stem}] {name}={val:.8f}", flush=True)


def decode_round_stim_avg(
    group: CheckpointGroup,
    art: RoundDemArtifactsStimAvg,
    *,
    data_root: Path,
    device: str,
    dtype: torch.dtype,
    mini_batch: int,
    ctg_max_time: int,
    mwpm_correlations: bool,
    avg_rounds: list[int],
) -> dict[str, object]:
    r = art.rounds
    stem = group.stem_for_round(r)
    experiment_dir = resolve_experiment_dir(
        data_root=data_root,
        basis=group.basis,
        distance=group.distance,
        rounds=r,
        center_row=group.center_row,
        center_col=group.center_col,
    )
    base_circuit = load_noisy_circuit(experiment_dir)
    dets = load_detection_events(experiment_dir, base_circuit)
    obvs = load_observable_flips(experiment_dir)

    decoder, pcm, stim_ref_dem = setup_tn_decoder_for_round(
        group,
        r,
        data_root=data_root,
        device=device,
        dtype=dtype,
        mini_batch=mini_batch,
        ctg_max_time=ctg_max_time,
    )
    dets = align_dets_to_pcm(dets, base_circuit, pcm)

    stim_dem_true = base_circuit.detector_error_model(
        decompose_errors=True, flatten_loops=True
    )
    mwpm_stim = MWPM_dem(stim_dem_true, enable_correlations=mwpm_correlations)
    mwpm_avg = MWPM_dem(art.dem_decompose, enable_correlations=mwpm_correlations)

    stim_er_tn = stim_ref_dem.to(device=device, dtype=dtype)
    stim_er_true = get_error_rates(stim_dem_true)
    tn_er = art.tn_error_rates.to(device=device, dtype=dtype)

    results = {
        "ler_tn_stim_dem_without_decompose": compute_tn_logical_error_rate(
            decoder,
            dets,
            obvs,
            stim_er_tn,
            device=device,
            decode_batch_size=mini_batch,
        ),
        "ler_mwpm_stim_dem_decompose": compute_mwpm_logical_error_rate(
            mwpm_stim,
            dets,
            obvs,
            stim_er_true,
            mini_batch=mini_batch,
        ),
        "ler_tn_avg_gate_dem_without_decompose": compute_tn_logical_error_rate(
            decoder,
            dets,
            obvs,
            tn_er,
            device=device,
            decode_batch_size=mini_batch,
        ),
        "ler_mwpm_avg_gate_dem_decompose": compute_mwpm_logical_error_rate(
            mwpm_avg,
            dets,
            obvs,
            art.mwpm_error_rates,
            mini_batch=mini_batch,
        ),
    }

    del decoder
    gc.collect()

    return {
        "stem": stem,
        "experiment_dir": experiment_dir,
        "results": results,
    }


def run_extended_decoding(
    log,
    bundle: ExperimentBundle,
    *,
    avg_rounds_by_basis: dict[str, list[int]],
    avg_learn_by_basis: dict[str, torch.Tensor],
    data_root: Path,
    device: str,
    dtype: torch.dtype,
    mini_batch: int,
    ctg_max_time: int,
    mwpm_correlations: bool,
    save_circuits: bool,
    extended_rounds: tuple[int, ...],
) -> None:
    log.write("\n# === extended code rounds (stim + avg only) ===\n")
    log.write(f"# extended_rounds_config={list(extended_rounds)}\n")
    log.write("# avg gate: mean learnable slots over checkpoint rounds (e.g. r05–r11)\n\n")
    log.flush()

    for basis in bundle.bases:
        group = bundle.groups_by_basis[basis]
        ext = discover_extended_rounds(
            group, data_root=data_root, rounds=extended_rounds
        )
        if not ext:
            print(f"[extended] basis={basis}: no extra rounds found, skip", flush=True)
            continue

        print(
            f"[extended] basis={basis} rounds={ext} "
            f"(avg from checkpoint rounds {avg_rounds_by_basis[basis]})",
            flush=True,
        )
        ensure_contraction_paths_ascending(
            group,
            data_root=data_root,
            device=device,
            dtype=dtype,
            mini_batch=mini_batch,
            ctg_max_time=ctg_max_time,
            rounds=ext,
        )
        arts = build_stim_avg_artifacts(
            group,
            ext,
            avg_learn_by_basis[basis],
            data_root=data_root,
            device=device,
            dtype=dtype,
            save_circuits=save_circuits,
        )
        for r in ext:
            out = decode_round_stim_avg(
                group,
                arts[r],
                data_root=data_root,
                device=device,
                dtype=dtype,
                mini_batch=mini_batch,
                ctg_max_time=ctg_max_time,
                mwpm_correlations=mwpm_correlations,
                avg_rounds=avg_rounds_by_basis[basis],
            )
            write_round_section_stim_avg(
                log,
                basis=basis,
                rounds=r,
                stem=out["stem"],
                experiment_dir=out["experiment_dir"],
                results=out["results"],
                avg_rounds=avg_rounds_by_basis[basis],
                group=group,
                art=arts[r],
            )


def run_experiment_bundle(
    bundle: ExperimentBundle,
    *,
    data_root: Path,
    device: str,
    mini_batch: int,
    ctg_max_time: int,
    mwpm_correlations: bool,
    save_circuits: bool,
    paths_only: bool,
    run_extended: bool = True,
    extended_rounds: tuple[int, ...] = DEFAULT_EXTENDED_ROUNDS,
    extended_only: bool = False,
    append_log: bool | None = None,
) -> Path:
    dtype = torch.float64
    out_path = combined_log_path(bundle)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if append_log is None:
        append_log = extended_only
    log_exists = out_path.is_file() and out_path.stat().st_size > 0
    log_mode = "a" if append_log else "w"

    print(
        f"\n[bundle] d={bundle.distance} center={bundle.center_row}_{bundle.center_col} "
        f"sharing={bundle.param_sharing} bases={bundle.bases} -> {out_path.name} "
        f"({'append' if append_log else 'overwrite'})"
        + (" extended-only" if extended_only else ""),
        flush=True,
    )

    avg_rounds_by_basis: dict[str, list[int]] = {}
    avg_learn_by_basis: dict[str, torch.Tensor] = {}

    for basis in bundle.bases:
        group = bundle.groups_by_basis[basis]
        if not extended_only:
            ensure_contraction_paths_ascending(
                group,
                data_root=data_root,
                device=device,
                dtype=dtype,
                mini_batch=mini_batch,
                ctg_max_time=ctg_max_time,
            )
        min_r = group.rounds[0]
        ref_circuit = load_noisy_circuit(
            resolve_experiment_dir(
                data_root=data_root,
                basis=group.basis,
                distance=group.distance,
                rounds=min_r,
                center_row=group.center_row,
                center_col=group.center_col,
            )
        )
        ref_g2d = GateNoiseToDEM(
            ref_circuit,
            learnable=False,
            param_sharing=group.param_sharing,  # type: ignore[arg-type]
            dtype=dtype,
            device=device,
        )
        avg_learn, avg_rounds = average_learnable_probs(
            group,
            ref_g2d.slot_keys,
            data_root=data_root,
            device=device,
            dtype=dtype,
        )
        avg_rounds_by_basis[basis] = avg_rounds
        avg_learn_by_basis[basis] = avg_learn

    if run_extended:
        for basis in bundle.bases:
            group = bundle.groups_by_basis[basis]
            ext = discover_extended_rounds(
                group, data_root=data_root, rounds=extended_rounds
            )
            if ext:
                ensure_contraction_paths_ascending(
                    group,
                    data_root=data_root,
                    device=device,
                    dtype=dtype,
                    mini_batch=mini_batch,
                    ctg_max_time=ctg_max_time,
                    rounds=ext,
                )

    if paths_only:
        with open(out_path, log_mode, encoding="utf-8") as log:
            if append_log and log_exists:
                write_continuation_marker(log, label="paths-only continuation")
            log.write(f"# paths-only d={bundle.distance} c={bundle.center_row}_{bundle.center_col}\n")
            log.write(f"# bases={bundle.bases}\n")
            if run_extended:
                log.write(f"# extended_rounds={list(extended_rounds)}\n")
        print(f"[done] paths only → {out_path}", flush=True)
        return out_path

    if extended_only:
        if not run_extended:
            raise ValueError("--extended-only requires extended decoding (do not pass --no-extended)")
        with open(out_path, log_mode, encoding="utf-8") as log:
            if append_log and log_exists:
                write_continuation_marker(log, label="extended rounds only")
            elif not log_exists:
                write_bundle_header(
                    log,
                    bundle,
                    avg_rounds_by_basis=avg_rounds_by_basis,
                    mwpm_correlations=mwpm_correlations,
                )
            run_extended_decoding(
                log,
                bundle,
                avg_rounds_by_basis=avg_rounds_by_basis,
                avg_learn_by_basis=avg_learn_by_basis,
                data_root=data_root,
                device=device,
                dtype=dtype,
                mini_batch=mini_batch,
                ctg_max_time=ctg_max_time,
                mwpm_correlations=mwpm_correlations,
                save_circuits=save_circuits,
                extended_rounds=extended_rounds,
            )
        print(f"[done] extended decoding appended → {out_path}", flush=True)
        return out_path

    artifacts_by_basis: dict[str, dict[int, RoundDemArtifacts]] = {}

    for basis in bundle.bases:
        group = bundle.groups_by_basis[basis]
        artifacts_by_basis[basis] = build_round_dem_artifacts(
            group,
            avg_learn_by_basis[basis],
            data_root=data_root,
            device=device,
            dtype=dtype,
            save_circuits=save_circuits,
        )

    with open(out_path, log_mode, encoding="utf-8") as log:
        if append_log and log_exists:
            write_continuation_marker(log, label="decoding continuation")
        else:
            write_bundle_header(
                log,
                bundle,
                avg_rounds_by_basis=avg_rounds_by_basis,
                mwpm_correlations=mwpm_correlations,
            )
        for basis in bundle.bases:
            group = bundle.groups_by_basis[basis]
            avg_rounds = avg_rounds_by_basis[basis]
            for r in group.rounds:
                result = decode_round(
                    group,
                    artifacts_by_basis[basis][r],
                    data_root=data_root,
                    device=device,
                    dtype=dtype,
                    mini_batch=mini_batch,
                    ctg_max_time=ctg_max_time,
                    mwpm_correlations=mwpm_correlations,
                    avg_rounds=avg_rounds,
                )
                write_round_section(log, result)

        if run_extended:
            run_extended_decoding(
                log,
                bundle,
                avg_rounds_by_basis=avg_rounds_by_basis,
                avg_learn_by_basis=avg_learn_by_basis,
                data_root=data_root,
                device=device,
                dtype=dtype,
                mini_batch=mini_batch,
                ctg_max_time=ctg_max_time,
                mwpm_correlations=mwpm_correlations,
                save_circuits=save_circuits,
                extended_rounds=extended_rounds,
            )

    print(f"[done] decoding log → {out_path}", flush=True)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Cross-round average gate checkpoints, build per-round DEMs, "
            "decode with TN (no decompose) and MWPM (decompose)."
        )
    )
    ap.add_argument(
        "--basis",
        type=str,
        default=None,
        choices=("X", "Z"),
        help="process a single basis only (default: both X and Z)",
    )
    ap.add_argument("--distance", "-d", type=int, default=None)
    ap.add_argument("--center", type=str, default=None)
    ap.add_argument("--center-row", type=int, default=None)
    ap.add_argument("--center-col", type=int, default=None)
    ap.add_argument(
        "--param-sharing",
        type=str,
        default=DEFAULT_PARAM_SHARING,
        choices=get_args(ParamSharing),
        help=f"gate noise tying (default: {DEFAULT_PARAM_SHARING})",
    )
    ap.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--mini-batch",
        type=int,
        default=DEFAULT_MINI_BATCH,
        help=f"cotengra path search and TN/MWPM decoding batch size (default: {DEFAULT_MINI_BATCH})",
    )
    ap.add_argument("--ctg-max-time", type=int, default=60)
    ap.add_argument(
        "--mwpm-correlations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument(
        "--no-save-circuits",
        action="store_true",
        help="do not write patched noisy circuits to decoding/circuits/",
    )
    ap.add_argument(
        "--paths-only",
        action="store_true",
        help="only discover missing contraction paths (ascending r), skip decoding",
    )
    ap.add_argument(
        "--no-extended",
        action="store_true",
        help="skip extended-round decoding (r13–r25 stim+avg)",
    )
    ap.add_argument(
        "--extended-rounds",
        type=str,
        default="13-25",
        help="inclusive round range for extended decoding, e.g. 13-25",
    )
    ap.add_argument(
        "--extended-only",
        action="store_true",
        help=(
            "decode extended rounds (r13–r25 stim+avg) only; skip checkpoint rounds; "
            "append to the existing combined log"
        ),
    )
    ap.add_argument(
        "--append-log",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="append to the combined log instead of overwriting (default: on with --extended-only)",
    )
    args = ap.parse_args()

    def _parse_round_range(text: str) -> tuple[int, ...]:
        if "-" in text:
            a, b = text.split("-", 1)
            return tuple(range(int(a), int(b) + 1))
        return (int(text),)

    extended_rounds = _parse_round_range(args.extended_rounds)

    center_row: int | None = args.center_row
    center_col: int | None = args.center_col
    if args.center is not None:
        if center_row is not None or center_col is not None:
            raise ValueError("use either --center or --center-row/--center-col")
        center_row, center_col = parse_center(args.center)

    bases = resolve_bases(args.basis)
    param_sharing: ParamSharing = args.param_sharing  # type: ignore[assignment]
    groups = discover_checkpoint_groups(
        bases=bases,
        distance=args.distance,
        center_row=center_row,
        center_col=center_col,
        param_sharing=param_sharing,
    )
    if not groups:
        raise RuntimeError(
            f"no checkpoint groups found under {CHECKPOINT_DIR} "
            f"(bases={list(bases)}, param_sharing={param_sharing})"
        )

    print(
        f"[scan] bases={list(bases)} param_sharing={param_sharing} -> "
        f"{len(groups)} experiment group(s) under {CHECKPOINT_DIR}",
        flush=True,
    )
    bundles = cluster_experiment_bundles(groups)
    for bundle in bundles:
        print(
            f"  bundle bases={bundle.bases} rounds="
            f"{{{', '.join(f'{b}:{bundle.groups_by_basis[b].rounds}' for b in bundle.bases)}}}",
            flush=True,
        )

    if args.extended_only and args.no_extended:
        raise ValueError("--extended-only conflicts with --no-extended")

    for bundle in bundles:
        run_experiment_bundle(
            bundle,
            data_root=Path(args.data_root),
            device=args.device,
            mini_batch=args.mini_batch,
            ctg_max_time=args.ctg_max_time,
            mwpm_correlations=args.mwpm_correlations,
            save_circuits=not args.no_save_circuits,
            paths_only=args.paths_only,
            run_extended=not args.no_extended,
            extended_rounds=extended_rounds,
            extended_only=args.extended_only,
            append_log=args.append_log,
        )


if __name__ == "__main__":
    main()
