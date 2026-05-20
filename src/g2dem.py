"""
可微分的门级噪声 → DEM 概率（Gate → DEM）。

对固定 ``stim.Circuit``，由 ``param_sharing`` 决定可学习标量个数：

  - ``elementary``：每个 (时刻, 门, qubit/对) 独立；
  - ``dep``：全电路共用一个噪声概率；
  - ``time_shared``：按 (门类型, 空间 qubit 或对) 共享，不同测量轮共用同一标量。
  - 编译期用 ``explain_detector_error_model_errors`` 分通道，Hadamard + parity 逆合并；
  - 与 ``circuit.detector_error_model(decompose_errors=False, flatten_loops=True)``
    **逐行一致**（机器精度）。

用法::

    from src.g2dem import GateNoiseToDEM, build_d3r3_surface_code_circuit

    g2d = GateNoiseToDEM(circuit)
    dem_p = g2d()
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

ParamSharing = Literal["elementary", "dep", "time_shared"]

import numpy as np
import stim
import torch
import torch.nn as nn

NOISE_GATES = frozenset({"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR"})
EXPLAINED_GATES = NOISE_GATES | {"Y_ERROR"}
SINGLE_QUBIT_NOISE = frozenset({"DEPOLARIZE1", "X_ERROR", "Z_ERROR", "Y_ERROR"})

_INSTRUCTION_RE = re.compile(r"^([A-Z0-9_]+)\(([^)]*)\)")
_QUBIT_TARGET_RE = re.compile(r"\b(\d+)\s*(?:\[|$)")
_EXPLAINED_TARGET_RE = re.compile(r"(\d+)(?:\[[^\]]*\])?")

NoiseOp = tuple[int, int, str, list[int], float]

# Elementary: (flat_index, tick, gate_name, "q"|"pair", ...)
# time_shared: (gate_name, "q", qubit) or (gate_name, "pair", q0, q1)
# dep: ("dep",)
ParamKey = tuple
LearnableKey = tuple


@dataclass(frozen=True)
class _Channel:
    listed_param_ids: tuple[int, ...]
    gate_name: str
    listed_local_ids: tuple[int, ...]
    dem_indices: tuple[int, ...]
    hadamard: np.ndarray
    inverse_system: np.ndarray


def build_d3r3_surface_code_circuit(
    *,
    task: str = "surface_code:rotated_memory_z",
    distance: int = 3,
    rounds: int = 3,
    noise_rate: float = 0.001,
) -> stim.Circuit:
    return stim.Circuit.generated(
        task,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise_rate,
        after_reset_flip_probability=noise_rate,
        before_measure_flip_probability=noise_rate,
        before_round_data_depolarization=noise_rate,
    )


def _reference_dem(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    return circuit.detector_error_model(decompose_errors=False, flatten_loops=True)


def _dem_probs_from_stim(circuit: stim.Circuit) -> np.ndarray:
    dem = _reference_dem(circuit)
    return np.array(
        [float(dem[i].args_copy()[0]) for i in range(dem.num_errors) if dem[i].type == "error"],
        dtype=np.float64,
    )


def _depolarize2_pairs(targets: list[int]) -> list[tuple[int, int]]:
    if len(targets) % 2 != 0:
        raise ValueError(f"DEPOLARIZE2 requires an even number of targets, got {targets}")
    return [(targets[i], targets[i + 1]) for i in range(0, len(targets), 2)]


def _elementary_param_key(flat_idx: int, tick: int, gate: str, targets: list[int]) -> list[ParamKey]:
    """One learnable scalar per single-qubit site or per DEPOLARIZE2 pair."""
    keys: list[ParamKey] = []
    if gate in SINGLE_QUBIT_NOISE:
        for q in targets:
            keys.append((flat_idx, tick, gate, "q", q))
    elif gate == "DEPOLARIZE2":
        for a, b in _depolarize2_pairs(targets):
            lo, hi = (a, b) if a < b else (b, a)
            keys.append((flat_idx, tick, gate, "pair", lo, hi))
    else:
        raise ValueError(f"unsupported noise gate {gate!r}")
    return keys


def _scan_elementary_gate_params(
    circuit: stim.Circuit,
) -> tuple[dict[ParamKey, int], list[ParamKey], dict[int, list[int]], np.ndarray]:
    flat = list(circuit.flattened())
    param_key_to_id: dict[ParamKey, int] = {}
    param_keys: list[ParamKey] = []
    flat_index_to_param_ids: dict[int, list[int]] = {}
    init_probs: list[float] = []
    tick = 0

    for i, op in enumerate(flat):
        if op.name == "TICK":
            tick += 1
            continue
        if op.name not in NOISE_GATES:
            continue
        p_inst = float(op.gate_args_copy()[0]) if op.gate_args_copy() else 0.0
        targets = [t.value for t in op.targets_copy()]
        ids_on_inst: list[int] = []
        for pk in _elementary_param_key(i, tick, op.name, targets):
            if pk not in param_key_to_id:
                param_key_to_id[pk] = len(param_keys)
                param_keys.append(pk)
                init_probs.append(p_inst)
            ids_on_inst.append(param_key_to_id[pk])
        flat_index_to_param_ids[i] = ids_on_inst

    return (
        param_key_to_id,
        param_keys,
        flat_index_to_param_ids,
        np.array(init_probs, dtype=np.float64),
    )


def _elementary_to_learnable_key(elem_key: ParamKey, sharing: ParamSharing) -> LearnableKey:
    if sharing == "dep":
        return ("dep",)
    if sharing == "elementary":
        return elem_key
    if sharing == "time_shared":
        _fi, _tick, gate, kind = elem_key[0], elem_key[1], elem_key[2], elem_key[3]
        if kind == "q":
            return (gate, "q", elem_key[4])
        if kind == "pair":
            return (gate, "pair", elem_key[4], elem_key[5])
        raise ValueError(f"unknown elementary target kind {kind!r}")
    raise ValueError(f"unknown param_sharing {sharing!r}")


def _build_learnable_layout(
    elem_keys: list[ParamKey],
    init_elem_probs: np.ndarray,
    sharing: ParamSharing,
) -> tuple[list[LearnableKey], np.ndarray, list[int]]:
    """Map each elementary index → learnable index; init = mean over merged sites."""
    learn_key_to_id: dict[LearnableKey, int] = {}
    learn_keys: list[LearnableKey] = []
    init_sums: list[float] = []
    init_counts: list[int] = []
    elem_to_learn: list[int] = []

    for eid, ek in enumerate(elem_keys):
        lk = _elementary_to_learnable_key(ek, sharing)
        if lk not in learn_key_to_id:
            learn_key_to_id[lk] = len(learn_keys)
            learn_keys.append(lk)
            init_sums.append(0.0)
            init_counts.append(0)
        lid = learn_key_to_id[lk]
        elem_to_learn.append(lid)
        init_sums[lid] += float(init_elem_probs[eid])
        init_counts[lid] += 1

    init_learn = np.array(
        [s / c for s, c in zip(init_sums, init_counts)],
        dtype=np.float64,
    )
    return learn_keys, init_learn, elem_to_learn


def _signature_from_targets(
    targets: Iterable[stim.DemTarget | stim.DemTargetWithCoords],
    *,
    num_detectors: int,
) -> int:
    sig = 0
    for target in targets:
        if isinstance(target, stim.DemTargetWithCoords):
            name = str(target.dem_target)
        else:
            if target.is_separator():
                continue
            name = str(target)
        if name.startswith("D"):
            sig ^= 1 << int(name[1:])
        elif name.startswith("L"):
            sig ^= 1 << (num_detectors + int(name[1:]))
        else:
            raise ValueError(f"unsupported DEM target {name!r}")
    return sig


def _parse_instruction(instruction_targets: str) -> tuple[str, float]:
    m = _INSTRUCTION_RE.match(instruction_targets)
    if m is None:
        raise ValueError(f"could not parse explained instruction: {instruction_targets!r}")
    args = m.group(2).strip()
    if "," in args:
        raise ValueError(f"only single-parameter noise supported: {instruction_targets!r}")
    return m.group(1), float(args)


def _explained_qubits(instruction_targets: str) -> tuple[int, ...]:
    """Qubit indices from explain string, e.g. ``X_ERROR(0.001) 3`` or ``DEPOLARIZE2(0.001) 0 1``."""
    m = _INSTRUCTION_RE.match(instruction_targets.strip())
    if m is None:
        return tuple(int(x.group(1)) for x in _QUBIT_TARGET_RE.finditer(instruction_targets))
    tail = instruction_targets[m.end() :].strip()
    if not tail:
        return ()
    return tuple(int(m2.group(1)) for m2 in _EXPLAINED_TARGET_RE.finditer(tail))


def _instruction_matches_explained(
    flat: list[stim.CircuitInstruction],
    flat_idx: int,
    gate: str,
    qubits: tuple[int, ...],
) -> bool:
    if flat_idx < 0 or flat_idx >= len(flat) or flat[flat_idx].name != gate:
        return False
    targets = [t.value for t in flat[flat_idx].targets_copy()]
    if gate in SINGLE_QUBIT_NOISE:
        return len(qubits) == 1 and qubits[0] in targets
    if gate == "DEPOLARIZE2" and len(qubits) == 2:
        pairs = _depolarize2_pairs(targets)
        return (qubits[0], qubits[1]) in pairs or (qubits[1], qubits[0]) in pairs
    return bool(qubits) and set(qubits).issubset(targets)


def _lookup_param_id(
    elem_key_to_id: dict[ParamKey, int],
    elem_to_learn: list[int],
    flat_idx: int,
    tick: int,
    gate: str,
    targets: list[int],
) -> int:
    if gate in SINGLE_QUBIT_NOISE:
        if len(targets) != 1:
            raise KeyError(f"expected one qubit for {gate}, got {targets}")
        key: ParamKey = (flat_idx, tick, gate, "q", targets[0])
    elif gate == "DEPOLARIZE2":
        if len(targets) != 2:
            raise KeyError(f"expected two qubits for DEPOLARIZE2, got {targets}")
        a, b = targets
        lo, hi = (a, b) if a < b else (b, a)
        key = (flat_idx, tick, gate, "pair", lo, hi)
    else:
        raise KeyError(gate)
    if key not in elem_key_to_id:
        raise KeyError(f"elementary param missing for {key!r}")
    return elem_to_learn[elem_key_to_id[key]]


def _flat_index_from_stack(
    stack_frames: tuple,
    *,
    flat: list[stim.CircuitInstruction],
    gate_name: str,
    qubits: tuple[int, ...],
    flat_index_to_param_ids: dict[int, list[int]],
) -> int:
    for frame in reversed(stack_frames):
        off = int(frame.instruction_offset)
        if off in flat_index_to_param_ids and _instruction_matches_explained(
            flat, off, gate_name, qubits
        ):
            return off
    search_offs: list[int] = []
    for frame in stack_frames:
        search_offs.append(int(frame.instruction_offset))
    for base in search_offs:
        for delta in range(48):
            for off in (base + delta, base - delta):
                if off in flat_index_to_param_ids and _instruction_matches_explained(
                    flat, off, gate_name, qubits
                ):
                    return off
    raise KeyError(f"no flat noise instruction for {gate_name!r} {qubits!r}")


def _resolve_channel_param_id(
    elem_key_to_id: dict[ParamKey, int],
    elem_to_learn: list[int],
    flat_index_to_param_ids: dict[int, list[int]],
    noise_ops: list[NoiseOp],
    instruction_targets: str,
    gate_name: str,
    *,
    flat: list[stim.CircuitInstruction],
    stack_frames: tuple,
) -> int:
    qubits = list(_explained_qubits(instruction_targets))
    try:
        flat_idx = _flat_index_from_stack(
            stack_frames,
            flat=flat,
            gate_name=gate_name,
            qubits=tuple(qubits),
            flat_index_to_param_ids=flat_index_to_param_ids,
        )
        tick = next(t for fi, t, g, _tg, _p in noise_ops if fi == flat_idx)
        return _lookup_param_id(elem_key_to_id, elem_to_learn, flat_idx, tick, gate_name, qubits)
    except KeyError:
        pass

    ref_offs = [int(f.instruction_offset) for f in stack_frames] or [0]
    candidates: list[tuple[int, int]] = []
    for fi, tick, gate, targets, _p in noise_ops:
        if gate != gate_name:
            continue
        if not _instruction_matches_explained(flat, fi, gate_name, tuple(qubits)):
            continue
        if gate in SINGLE_QUBIT_NOISE and len(qubits) == 1:
            pid = _lookup_param_id(elem_key_to_id, elem_to_learn, fi, tick, gate, [qubits[0]])
        elif gate == "DEPOLARIZE2" and len(qubits) == 2:
            pid = _lookup_param_id(elem_key_to_id, elem_to_learn, fi, tick, gate, list(qubits))
        else:
            continue
        dist = min(abs(fi - off) for off in ref_offs)
        candidates.append((pid, dist))

    if not candidates:
        raise KeyError(f"no learnable param for channel {gate_name!r} {instruction_targets!r}")
    candidates.sort(key=lambda x: x[1])
    if len(candidates) >= 2 and candidates[0][1] == candidates[1][1]:
        raise KeyError(
            f"ambiguous learnable param for {gate_name!r} {instruction_targets!r}: "
            f"{candidates[:3]}"
        )
    return candidates[0][0]


def _hadamard_matrix(rank: int) -> np.ndarray:
    states = np.arange(1 << rank, dtype=np.int32)
    bits = np.arange(rank, dtype=np.int32)
    sb = ((states[:, None] >> bits[None, :]) & 1).astype(np.int8)
    return (1.0 - 2.0 * ((sb @ sb.T) & 1)).astype(np.float64)


def _parity_system_matrix(rank: int) -> np.ndarray:
    if rank == 0:
        return np.zeros((0, 0), dtype=np.float64)
    states = np.arange(1, 1 << rank, dtype=np.int32)
    bits = np.arange(rank, dtype=np.int32)
    sb = ((states[:, None] >> bits[None, :]) & 1).astype(np.int8)
    return ((sb @ sb.T) & 1).astype(np.float64)


def _xor_basis(vectors: Iterable[int]) -> tuple[int, ...]:
    rows: list[int] = []
    pivots: list[int] = []
    for vector in vectors:
        reduced = int(vector)
        for row, pivot in zip(rows, pivots):
            if (reduced >> pivot) & 1:
                reduced ^= row
        if reduced == 0:
            continue
        pivot = reduced.bit_length() - 1
        insert_at = sum(1 for p in pivots if p > pivot)
        rows.insert(insert_at, reduced)
        pivots.insert(insert_at, pivot)
    return tuple(rows)


def _enumerate_span(basis: tuple[int, ...]) -> tuple[int, ...]:
    states = [0]
    for mask in range(1, 1 << len(basis)):
        sig = 0
        for i, row in enumerate(basis):
            if (mask >> i) & 1:
                sig ^= row
        states.append(sig)
    return tuple(states)


def _build_signature_to_line(
    reference_dem: stim.DetectorErrorModel,
    *,
    num_detectors: int,
) -> dict[int, int]:
    signature_to_line: dict[int, int] = {}
    for i in range(reference_dem.num_errors):
        sig = _signature_from_targets(reference_dem[i].targets_copy(), num_detectors=num_detectors)
        if sig in signature_to_line:
            raise ValueError(f"duplicate DEM signature at lines {signature_to_line[sig]} and {i}")
        signature_to_line[sig] = i
    return signature_to_line


def _compile_channels(
    circuit: stim.Circuit,
    elem_key_to_id: dict[ParamKey, int],
    elem_to_learn: list[int],
    flat_index_to_param_ids: dict[int, list[int]],
    noise_ops: list[NoiseOp],
) -> list[_Channel]:
    reference_dem = _reference_dem(circuit)
    num_detectors = int(reference_dem.num_detectors)
    sig_to_line = _build_signature_to_line(reference_dem, num_detectors=num_detectors)

    flat = list(circuit.flattened())
    explained = circuit.explain_detector_error_model_errors(
        dem_filter=reference_dem,
        reduce_to_one_representative_error=False,
    )

    ch_sigs: dict[tuple[tuple[int, int, int], str], list[int]] = {}
    ch_gate: dict[tuple[tuple[int, int, int], str], str] = {}
    ch_frames: dict[tuple[tuple[int, int, int], str], tuple] = {}

    for item in explained:
        sig = _signature_from_targets(item.dem_error_terms, num_detectors=num_detectors)
        for loc in item.circuit_error_locations:
            stack_key = tuple(
                (int(f.instruction_offset), int(f.iteration_index), int(f.instruction_repetitions_arg))
                for f in loc.stack_frames
            )
            inst = str(loc.instruction_targets)
            gate, _ = _parse_instruction(inst)
            if gate not in EXPLAINED_GATES:
                raise ValueError(f"unsupported explained gate {gate!r}")
            key = (stack_key, inst)
            ch_sigs.setdefault(key, []).append(sig)
            ch_gate[key] = gate
            ch_frames[key] = loc.stack_frames

    channels: list[_Channel] = []
    for key, signatures in sorted(ch_sigs.items()):
        gate = ch_gate[key]
        branch_pid = _resolve_channel_param_id(
            elem_key_to_id,
            elem_to_learn,
            flat_index_to_param_ids,
            noise_ops,
            key[1],
            gate,
            flat=flat,
            stack_frames=ch_frames[key],
        )
        basis = _xor_basis(signatures)
        coord_sig = _enumerate_span(basis)
        sig2coord = {s: c for c, s in enumerate(coord_sig)}
        listed = tuple(sig2coord[s] for s in signatures)
        listed_param_ids = tuple(branch_pid for _ in listed)
        rank = len(basis)
        dem_lines = tuple(sig_to_line[s] for s in coord_sig[1:])
        channels.append(
            _Channel(
                listed_param_ids=listed_param_ids,
                gate_name=gate,
                listed_local_ids=listed,
                dem_indices=dem_lines,
                hadamard=_hadamard_matrix(rank),
                inverse_system=np.linalg.inv(_parity_system_matrix(rank)),
            )
        )
    return channels


def compile_circuit(
    circuit: stim.Circuit,
    *,
    param_sharing: ParamSharing = "elementary",
) -> dict:
    """编译静态结构（不可训练），供 ``GateNoiseToDEM`` 使用。

    ``param_sharing`` 控制可学习标量如何绑定到电路噪声：

    - ``elementary``：每个 (时刻, 门, qubit/对) 独立（默认）；
    - ``dep``：全电路共用一个噪声概率；
    - ``time_shared``：按 (门类型, 空间 qubit 或对) 共享，不同时间轮共用同一标量。
    """
    elem_key_to_id, elem_keys, flat_index_to_param_ids, init_elem_probs = _scan_elementary_gate_params(
        circuit
    )
    learn_keys, init_gate_probs, elem_to_learn = _build_learnable_layout(
        elem_keys, init_elem_probs, param_sharing
    )
    flat = list(circuit.flattened())
    tick = 0
    noise_ops: list[NoiseOp] = []
    for i, op in enumerate(flat):
        if op.name == "TICK":
            tick += 1
            continue
        if op.name not in NOISE_GATES:
            continue
        p = float(op.gate_args_copy()[0]) if op.gate_args_copy() else 0.0
        noise_ops.append((i, tick, op.name, [t.value for t in op.targets_copy()], p))

    channels = _compile_channels(
        circuit, elem_key_to_id, elem_to_learn, flat_index_to_param_ids, noise_ops
    )
    ref_dem_probs = _dem_probs_from_stim(circuit)
    return {
        "num_dem": len(ref_dem_probs),
        "num_slots": len(learn_keys),
        "num_elementary": len(elem_keys),
        "num_channels": len(channels),
        "param_sharing": param_sharing,
        "slot_keys": learn_keys,
        "elementary_keys": elem_keys,
        "elem_to_learn": elem_to_learn,
        "channels": channels,
        "init_gate_probs": init_gate_probs,
        "ref_dem_probs": ref_dem_probs,
    }


def _branch_weight(gate_name: str, p: torch.Tensor) -> torch.Tensor:
    if gate_name in {"X_ERROR", "Y_ERROR", "Z_ERROR"}:
        return p
    if gate_name == "DEPOLARIZE1":
        return p / 3.0
    if gate_name == "DEPOLARIZE2":
        return p / 15.0
    raise ValueError(f"unsupported gate {gate_name!r}")


def _channel_local_probs(channel: _Channel, gate_probs: torch.Tensor) -> torch.Tensor:
    n = int(channel.hadamard.shape[0])
    if n <= 1:
        return torch.zeros(0, dtype=gate_probs.dtype, device=gate_probs.device)

    h = torch.as_tensor(channel.hadamard, dtype=gate_probs.dtype, device=gate_probs.device)
    inv = torch.as_tensor(channel.inverse_system, dtype=gate_probs.dtype, device=gate_probs.device)

    weights = torch.stack(
        [_branch_weight(channel.gate_name, gate_probs[pid]) for pid in channel.listed_param_ids]
    )

    q = torch.zeros(n, dtype=gate_probs.dtype, device=gate_probs.device)
    q[0] = 1.0 - weights.sum()
    if channel.listed_local_ids:
        idx = torch.tensor(channel.listed_local_ids, dtype=torch.long, device=gate_probs.device)
        q.index_add_(0, idx, weights)

    z = inv @ torch.log((h @ q)[1:])
    return (1.0 - torch.exp(z)) / 2.0


def _merge_channels(
    gate_probs: torch.Tensor,
    channels: list[_Channel],
    num_dem: int,
) -> torch.Tensor:
    out = torch.zeros(num_dem, dtype=gate_probs.dtype, device=gate_probs.device)
    for ch in channels:
        if not ch.dem_indices:
            continue
        local = _channel_local_probs(ch, gate_probs)
        idx = torch.tensor(ch.dem_indices, dtype=torch.long, device=gate_probs.device)
        cur = out.index_select(0, idx)
        out.index_copy_(0, idx, cur + local - 2.0 * cur * local)
    return out


class GateNoiseToDEM(nn.Module):
    """可微分 Gate → DEM；可学习标量个数由 ``param_sharing`` 决定。"""

    def __init__(
        self,
        circuit: stim.Circuit,
        *,
        learnable: bool = True,
        init_from_circuit: bool = True,
        param_mode: Literal["logit", "raw"] = "logit",
        param_sharing: ParamSharing = "elementary",
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        meta = compile_circuit(circuit, param_sharing=param_sharing)
        self.num_dem: int = meta["num_dem"]
        self.num_slots: int = meta["num_slots"]
        self.num_elementary: int = meta["num_elementary"]
        self.num_channels: int = meta["num_channels"]
        self.param_sharing: ParamSharing = meta["param_sharing"]
        self.slot_keys: list[LearnableKey] = meta["slot_keys"]
        self._channels: list[_Channel] = meta["channels"]
        self.param_mode = param_mode

        self.register_buffer(
            "ref_dem_probs",
            torch.tensor(meta["ref_dem_probs"], dtype=dtype, device=device),
        )

        init_p = np.clip(meta["init_gate_probs"], 1e-12, 1.0 - 1e-12)
        if param_mode == "logit":
            init_param = np.log(init_p / (1.0 - init_p))
        elif param_mode == "raw":
            init_param = init_p.copy()
        else:
            raise ValueError("param_mode must be 'logit' or 'raw'")

        init_tensor = torch.tensor(init_param, dtype=dtype, device=device)
        if learnable:
            self.gate_param = nn.Parameter(init_tensor.clone())
        else:
            self.register_buffer("gate_param", init_tensor)
            if not init_from_circuit:
                raise ValueError("learnable=False requires init_from_circuit=True")

        if not init_from_circuit:
            with torch.no_grad():
                self.gate_param.zero_()

    def gate_probs(self, gate_param: torch.Tensor | None = None) -> torch.Tensor:
        p = gate_param if gate_param is not None else self.gate_param
        if self.param_mode == "logit":
            return torch.sigmoid(p)
        return p.clamp(1e-12, 1.0 - 1e-12)

    def forward(self, gate_param: torch.Tensor | None = None) -> torch.Tensor:
        return _merge_channels(self.gate_probs(gate_param), self._channels, self.num_dem)

    def dem_weights(self, dem_probs: torch.Tensor | None = None) -> torch.Tensor:
        p = (dem_probs if dem_probs is not None else self.forward()).clamp(1e-12, 1.0 - 1e-12)
        return torch.log((1.0 - p) / p)


def _validate_circuit(circuit: stim.Circuit, *, label: str, param_sharing: ParamSharing) -> None:
    module = GateNoiseToDEM(circuit, learnable=True, param_sharing=param_sharing, dtype=torch.float64)
    print(
        f"  [{param_sharing}] learnable={module.num_slots} elementary={module.num_elementary} "
        f"channels={module.num_channels} dem={module.num_dem}"
    )
    with torch.no_grad():
        pred = module()
        ref = module.ref_dem_probs
        max_abs = float((pred - ref).abs().max())
        if max_abs > 1e-12:
            worst = int(torch.argmax((pred - ref).abs()).item())
            raise AssertionError(
                f"{label} {param_sharing}: max_abs={max_abs}, line {worst}: "
                f"pred={float(pred[worst]):.6g} stim={float(ref[worst]):.6g}"
            )
    module.zero_grad()
    module().sum().backward()
    assert module.gate_param.grad is not None and torch.isfinite(module.gate_param.grad).all()


def _validate_d3r3_surface_code() -> None:
    print("=" * 72)
    print("GateNoiseToDEM：d=3, r=3 surface code (decompose_errors=False)")
    print("=" * 72)
    circuit = build_d3r3_surface_code_circuit(noise_rate=0.001)
    for sharing in ("elementary", "dep", "time_shared"):
        _validate_circuit(circuit, label="surface d3r3", param_sharing=sharing)
    print("验证完成。")


if __name__ == "__main__":
    _validate_d3r3_surface_code()
