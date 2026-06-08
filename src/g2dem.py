"""
可微分的门级噪声 → DEM 概率（Gate → DEM）。

对固定 ``stim.Circuit``，由 ``param_sharing`` 决定可学习标量个数；前向复现 Stim
``ErrorAnalyzer``（``decompose_errors=False``, ``flatten_loops=True``）的
``add_error`` 合并链，与 ``circuit.detector_error_model()`` 在 patched 门概率下一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ParamSharing = Literal[
    "elementary",
    "dep",
    "time_shared",
    "time_shared_dep2",
    "dep2",
    "time_shared_dep12",
    "time_shared_dep12_M_DD",
    "time_shared_dep12_M_DD_I",
    "time_shared_dep12_M_R_DD_I",
]

import numpy as np
import stim
import torch
import torch.nn as nn
import torch.nn.functional as F

NOISE_GATES = frozenset({"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR", "M", "MR"})
SINGLE_QUBIT_NOISE = frozenset({"DEPOLARIZE1", "X_ERROR", "Z_ERROR", "M", "MR"})
_MEASUREMENT_NOISE_GATES = frozenset({"M", "MR"})
_TWO_QUBIT_GATES = frozenset(
    {
        "CX",
        "CY",
        "CZ",
        "CNOT",
        "SWAP",
        "ISWAP",
        "SQRT_XX",
        "SQRT_YY",
        "SQRT_ZZ",
        "MXX",
        "MYY",
        "MZZ",
    }
)
_RESET_GATES = frozenset({"R", "RX", "RY", "RZ"})
_MEASURE_GATES = frozenset({"M", "MR", "MX", "MY", "MZ", "MPP"})
_IDLING_GATES = frozenset({"I"})
_ANNOTATION_GATES = frozenset({"QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"})

ParamKey = tuple
LearnableKey = tuple


def _is_learnable_noise_instruction(op: stim.CircuitInstruction) -> bool:
    if op.name not in NOISE_GATES:
        return False
    if op.name in _MEASUREMENT_NOISE_GATES:
        return bool(op.gate_args_copy())
    return True


def build_d3r3_surface_code_circuit(
    *,
    task: str = "surface_code:rotated_memory_z",
    distance: int = 3,
    rounds: int = 3,
    noise_rate: float = 0.001,
) -> stim.Circuit:
    return build_surface_code_circuit(
        distance=distance,
        rounds=rounds,
        error_prob=noise_rate,
        task=task,
        noise_model="depolarizing",
    )


def build_surface_code_circuit(
    *,
    distance: int,
    rounds: int,
    error_prob: float = 0.001,
    task: str = "surface_code:rotated_memory_z",
    noise_model: Literal["si1000", "depolarizing"] = "si1000",
) -> stim.Circuit:
    from gate.check import (
        build_surface_code_circuit_depolarizing,
        build_surface_code_circuit_si1000,
    )

    if noise_model == "depolarizing":
        return build_surface_code_circuit_depolarizing(
            distance=distance, rounds=rounds, error_prob=error_prob, task=task
        )
    if noise_model == "si1000":
        return build_surface_code_circuit_si1000(
            distance=distance, rounds=rounds, error_prob=error_prob, task=task
        )
    raise ValueError(f"unknown noise_model={noise_model!r}")


def build_repetition_circuit(
    *,
    distance: int,
    rounds: int,
    error_prob: float = 0.001,
    noise_model: Literal["si1000", "depolarizing"] = "si1000",
) -> stim.Circuit:
    from gate.pl_opt import (
        build_repetition_circuit_depolarizing,
        build_repetition_circuit_si1000,
    )

    if noise_model == "depolarizing":
        return build_repetition_circuit_depolarizing(
            distance=distance, rounds=rounds, error_prob=error_prob
        )
    if noise_model == "si1000":
        return build_repetition_circuit_si1000(
            distance=distance, rounds=rounds, error_prob=error_prob
        )
    raise ValueError(f"unknown noise_model={noise_model!r}")


def _dem_probs_from_stim(circuit: stim.Circuit) -> np.ndarray:
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    return np.array(
        [float(dem[i].args_copy()[0]) for i in range(dem.num_errors) if dem[i].type == "error"],
        dtype=np.float64,
    )


def _depolarize2_pairs(targets: list[int]) -> list[tuple[int, int]]:
    if len(targets) % 2 != 0:
        raise ValueError(f"DEPOLARIZE2 requires an even number of targets, got {targets}")
    return [(targets[i], targets[i + 1]) for i in range(0, len(targets), 2)]


def _elementary_param_key(flat_idx: int, tick: int, gate: str, targets: list[int]) -> list[ParamKey]:
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
) -> tuple[dict[ParamKey, int], list[ParamKey], np.ndarray]:
    flat = list(circuit.flattened())
    param_key_to_id: dict[ParamKey, int] = {}
    param_keys: list[ParamKey] = []
    init_probs: list[float] = []
    tick = 0

    for i, op in enumerate(flat):
        if op.name == "TICK":
            tick += 1
            continue
        if not _is_learnable_noise_instruction(op):
            continue
        p_inst = float(op.gate_args_copy()[0])
        targets = [t.value for t in op.targets_copy()]
        for pk in _elementary_param_key(i, tick, op.name, targets):
            if pk not in param_key_to_id:
                param_key_to_id[pk] = len(param_keys)
                param_keys.append(pk)
                init_probs.append(p_inst)

    return param_key_to_id, param_keys, np.array(init_probs, dtype=np.float64)


def _infer_noise_base_p(init_elem_probs: np.ndarray) -> float:
    """Infer noise-model reference ``p`` (e.g. si1000 ``error_prob``) from circuit rates."""
    rounded = np.round(init_elem_probs.astype(np.float64), 12)
    vals, counts = np.unique(rounded, return_counts=True)
    positive = vals[vals > 0]
    if positive.size == 0:
        raise ValueError("no positive elementary noise rates in circuit")
    if positive.size == 1:
        return float(positive[0])
    # Typical si1000: p, p/10, 2p, 5p — use the most frequent rate as base ``p``.
    pos_counts = counts[vals > 0]
    return float(positive[np.argmax(pos_counts)])


def _elementary_to_learnable_key(
    elem_key: ParamKey,
    sharing: ParamSharing,
    *,
    init_prob: float | None = None,
) -> LearnableKey:
    if sharing == "dep":
        return ("dep",)
    if sharing == "elementary":
        return elem_key
    if sharing == "time_shared":
        if init_prob is None:
            raise ValueError("time_shared requires init_prob to distinguish noise rates")
        return _time_shared_learnable_key(elem_key, init_prob=float(init_prob))
    raise ValueError(f"unknown param_sharing {sharing!r}")


def _time_shared_learnable_key(elem_key: ParamKey, *, init_prob: float) -> LearnableKey:
    rate = round(float(init_prob), 12)
    gate, kind = elem_key[2], elem_key[3]
    if kind == "q":
        return (gate, "q", elem_key[4], rate)
    if kind == "pair":
        return (gate, "pair", elem_key[4], elem_key[5], rate)
    raise ValueError(f"unknown elementary target kind {kind!r}")


def _dep2_after_two_qubit_flat_indices(circuit: stim.Circuit) -> frozenset[int]:
    """Flat indices of ``DEPOLARIZE2`` after a two-qubit gate in the same syndrome round.

    Hardware circuits (e.g. sycamore_old) may insert ``TICK`` / ``DEPOLARIZE1`` between
    ``CZ`` and ``DEPOLARIZE2``; those instructions do not clear the pending flag.
    """
    flat = list(circuit.flattened())
    pending_2q = False
    out: set[int] = set()
    no_reset = SINGLE_QUBIT_NOISE | frozenset({"TICK", "DEPOLARIZE2", "PAULI_ERROR", "I"})
    for i, op in enumerate(flat):
        if op.name in _TWO_QUBIT_GATES:
            pending_2q = True
            continue
        if op.name == "DEPOLARIZE2" and pending_2q:
            out.add(i)
            continue
        if op.name in no_reset:
            continue
        pending_2q = False
    return frozenset(out)


def _elementary_is_dep2_after_2q(elem_key: ParamKey, after_2q_flat: frozenset[int]) -> bool:
    return elem_key[2] == "DEPOLARIZE2" and elem_key[0] in after_2q_flat


def _qubits_in_instruction(op: stim.CircuitInstruction) -> list[int]:
    return [t.value for t in op.targets_copy() if t.is_qubit_target]


def _is_single_qubit_gate_instruction(op: stim.CircuitInstruction) -> bool:
    """Heuristic: unitary-like 1Q operation (not reset/measure/idle/noise/annotation)."""
    if op.name in NOISE_GATES | _RESET_GATES | _MEASURE_GATES | _IDLING_GATES | _ANNOTATION_GATES:
        return False
    targets = op.targets_copy()
    if not targets:
        return False
    qubit_targets = [t for t in targets if t.is_qubit_target]
    return len(qubit_targets) == len(targets)


def _dep1_after_single_qubit_flat_indices(circuit: stim.Circuit) -> frozenset[int]:
    """Flat indices of ``DEPOLARIZE1`` immediately after a unitary 1Q gate on each target qubit.

    Per qubit: the most recent non-passthrough instruction on that qubit must be a unitary
    single-qubit gate (H, X, ...). Explicit ``I`` idling on that qubit breaks the link.
    ``DEPOLARIZE1`` after idling (including post-syndrome DD blocks) is never included here.
    """
    flat = list(circuit.flattened())
    pending_1q: set[int] = set()
    out: set[int] = set()
    passthrough = frozenset({"TICK", "DEPOLARIZE1", "PAULI_ERROR"})
    for i, op in enumerate(flat):
        if _is_single_qubit_gate_instruction(op):
            pending_1q.update(_qubits_in_instruction(op))
            continue
        if op.name == "I":
            pending_1q.difference_update(_qubits_in_instruction(op))
            continue
        if op.name == "DEPOLARIZE1":
            if any(q in pending_1q for q in _qubits_in_instruction(op)):
                out.add(i)
            continue
        if op.name in passthrough:
            continue
        touched = _qubits_in_instruction(op)
        if touched:
            pending_1q.difference_update(touched)
    return frozenset(out)


def _elementary_is_dep1_after_1q(elem_key: ParamKey, after_1q_flat: frozenset[int]) -> bool:
    return elem_key[2] == "DEPOLARIZE1" and elem_key[0] in after_1q_flat


def _dep1_after_idle_flat_indices(circuit: stim.Circuit) -> frozenset[int]:
    """Flat indices of ``DEPOLARIZE1`` after explicit ``I`` idling on the same qubit(s)."""
    flat = list(circuit.flattened())
    pending_idle: set[int] = set()
    out: set[int] = set()
    passthrough = frozenset({"TICK", "DEPOLARIZE1", "PAULI_ERROR"})
    for i, op in enumerate(flat):
        if op.name == "I":
            pending_idle.update(_qubits_in_instruction(op))
            continue
        if _is_single_qubit_gate_instruction(op):
            pending_idle.difference_update(_qubits_in_instruction(op))
            continue
        if op.name == "DEPOLARIZE1":
            if any(q in pending_idle for q in _qubits_in_instruction(op)):
                out.add(i)
            continue
        if op.name in passthrough:
            continue
        touched = _qubits_in_instruction(op)
        if touched:
            pending_idle.difference_update(touched)
    return frozenset(out)


def _elementary_is_idle_dep1(elem_key: ParamKey, after_idle_flat: frozenset[int]) -> bool:
    return elem_key[2] == "DEPOLARIZE1" and elem_key[0] in after_idle_flat


def _x_error_before_measure_flat_indices(circuit: stim.Circuit) -> frozenset[int]:
    """Flat indices of ``X_ERROR`` that belong to the pre-measurement noise block."""
    flat = list(circuit.flattened())
    out: set[int] = set()
    passthrough = frozenset({"TICK", "I", "DEPOLARIZE1", "PAULI_ERROR"})
    for i, op in enumerate(flat):
        if op.name != "X_ERROR":
            continue
        for j in range(i + 1, len(flat)):
            name = flat[j].name
            if name in passthrough:
                continue
            if name in _MEASURE_GATES:
                out.add(i)
            break
    return frozenset(out)


def _qubit_coords(circuit: stim.Circuit) -> dict[int, tuple[float, float]]:
    coords: dict[int, tuple[float, float]] = {}
    for op in circuit.flattened():
        if op.name != "QUBIT_COORDS":
            continue
        targets = op.targets_copy()
        if not targets or not targets[0].is_qubit_target:
            continue
        args = op.gate_args_copy()
        if len(args) < 2:
            continue
        coords[targets[0].value] = (float(args[0]), float(args[1]))
    return coords


def _data_qubit_indices_from_coords(circuit: stim.Circuit) -> frozenset[int] | None:
    """Data qubits on a rotated surface layout: ``(x + y)`` even (from ``QUBIT_COORDS``)."""
    coords = _qubit_coords(circuit)
    if not coords:
        return None
    return frozenset(
        q
        for q, (x, y) in coords.items()
        if (int(round(x)) + int(round(y))) % 2 == 0
    )


def _ancilla_qubit_indices_from_coords(circuit: stim.Circuit) -> frozenset[int] | None:
    """Complement of data qubits when ``QUBIT_COORDS`` are present."""
    coords = _qubit_coords(circuit)
    data = _data_qubit_indices_from_coords(circuit)
    if not coords or data is None:
        return None
    return frozenset(set(coords) - set(data))


def _final_data_measure_flat_index(circuit: stim.Circuit) -> int | None:
    """Last ``M`` whose targets are all data qubits; falls back to last ``M`` if coords missing."""
    flat = list(circuit.flattened())
    data_qubits = _data_qubit_indices_from_coords(circuit)
    if data_qubits is None:
        for i in range(len(flat) - 1, -1, -1):
            if flat[i].name in _MEASURE_GATES:
                return i
        return None
    last_data_m: int | None = None
    for i, op in enumerate(flat):
        if op.name not in _MEASURE_GATES:
            continue
        qs = {t.value for t in op.targets_copy() if t.is_qubit_target}
        if qs and qs <= data_qubits:
            last_data_m = i
    return last_data_m


def _x_error_before_syndrome_measure_flat_indices(circuit: stim.Circuit) -> frozenset[int]:
    """Ancilla ``X_ERROR`` in the pre-syndrome-``M`` chain (not only the last hop before ``M``)."""
    flat = list(circuit.flattened())
    final_data_m_idx = _final_data_measure_flat_index(circuit)
    if final_data_m_idx is None:
        return frozenset()
    ancilla = _ancilla_qubit_indices_from_coords(circuit)
    out: set[int] = set()
    passthrough = frozenset({"TICK", "I", "DEPOLARIZE1", "PAULI_ERROR"})
    for i, op in enumerate(flat):
        if op.name != "X_ERROR":
            continue
        qs = {t.value for t in op.targets_copy() if t.is_qubit_target}
        if ancilla is not None and not qs <= ancilla:
            continue
        j = i + 1
        while j < len(flat):
            name = flat[j].name
            if name in passthrough:
                j += 1
                continue
            if name == "X_ERROR":
                j += 1
                continue
            break
        if flat[j].name in _MEASURE_GATES and j != final_data_m_idx:
            out.add(i)
    return frozenset(out)


def _x_error_after_reset_flat_indices(circuit: stim.Circuit) -> frozenset[int]:
    """Flat indices of ``X_ERROR`` that belong to the post-reset noise block."""
    flat = list(circuit.flattened())
    pending_reset = False
    out: set[int] = set()
    passthrough = frozenset({"TICK", "I", "DEPOLARIZE1", "PAULI_ERROR"})
    for i, op in enumerate(flat):
        if op.name in _RESET_GATES:
            pending_reset = True
            continue
        if op.name == "X_ERROR" and pending_reset:
            out.add(i)
            continue
        if op.name in passthrough:
            continue
        pending_reset = False
    return frozenset(out)


def _elementary_is_x_error_at(elem_key: ParamKey, flat_indices: frozenset[int]) -> bool:
    return elem_key[2] == "X_ERROR" and elem_key[0] in flat_indices


def _elementary_is_dep1_at(elem_key: ParamKey, flat_indices: frozenset[int]) -> bool:
    return elem_key[2] == "DEPOLARIZE1" and elem_key[0] in flat_indices


def _dep1_dd_after_syndrome_data_idle_flat_indices(circuit: stim.Circuit) -> frozenset[int]:
    """``DEPOLARIZE1`` on data qubits after ``I`` idling following a syndrome ``M`` round."""
    flat = list(circuit.flattened())
    final_data_m_idx = _final_data_measure_flat_index(circuit)
    data = _data_qubit_indices_from_coords(circuit)
    ancilla = _ancilla_qubit_indices_from_coords(circuit)
    if final_data_m_idx is None or data is None:
        return frozenset()
    out: set[int] = set()
    skip_before_idle = frozenset(
        {"TICK", "PAULI_ERROR", "DETECTOR", "QUBIT_COORDS", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"}
    )
    for mi, m_op in enumerate(flat):
        if m_op.name not in _MEASURE_GATES or mi == final_data_m_idx:
            continue
        m_qs = {t.value for t in m_op.targets_copy() if t.is_qubit_target}
        if ancilla is not None and not m_qs <= ancilla:
            continue
        j = mi + 1
        while j < len(flat) and flat[j].name in skip_before_idle:
            j += 1
        if j >= len(flat) or flat[j].name != "I":
            continue
        i_qs = {t.value for t in flat[j].targets_copy() if t.is_qubit_target}
        if not i_qs or not i_qs <= data:
            continue
        j += 1
        while j < len(flat) and flat[j].name == "TICK":
            j += 1
        while j < len(flat):
            op = flat[j]
            if op.name == "TICK":
                j += 1
                continue
            if op.name == "DEPOLARIZE1":
                qs = {t.value for t in op.targets_copy() if t.is_qubit_target}
                if qs and qs <= data:
                    out.add(j)
                j += 1
                continue
            break
    return frozenset(out)


def _dep1_reset_round_simultaneous_idle_flat_indices(circuit: stim.Circuit) -> frozenset[int]:
    """``DEPOLARIZE1`` after data ``I`` idling that immediately follows a round ``R`` reset.

    These are the per-round reset-synchronous data idle errors (often zero-rate placeholders
    in hardware circuits). Detected structurally via ``R -> I(data) -> ... -> DEPOLARIZE1``,
    not by inspecting noise rates.
    """
    flat = list(circuit.flattened())
    data = _data_qubit_indices_from_coords(circuit)
    if data is None:
        return frozenset()
    out: set[int] = set()
    passthrough = frozenset({"TICK", "X_ERROR", "PAULI_ERROR"})
    for ri, op in enumerate(flat):
        if op.name not in _RESET_GATES:
            continue
        j = ri + 1
        while j < len(flat) and flat[j].name == "TICK":
            j += 1
        if j >= len(flat) or flat[j].name != "I":
            continue
        idle_qs = {t.value for t in flat[j].targets_copy() if t.is_qubit_target}
        if not idle_qs or not idle_qs <= data:
            continue
        j += 1
        while j < len(flat) and flat[j].name in passthrough:
            j += 1
        if j >= len(flat) or flat[j].name != "DEPOLARIZE1":
            continue
        while j < len(flat) and flat[j].name == "DEPOLARIZE1":
            dep_qs = {t.value for t in flat[j].targets_copy() if t.is_qubit_target}
            if dep_qs and dep_qs <= idle_qs:
                out.add(j)
            j += 1
    return frozenset(out)


def _dep1_initial_round_idle_excluded_flat_indices(circuit: stim.Circuit) -> frozenset[int]:
    """Initial-round idle ``DEPOLARIZE1`` after the first ``CX`` (not learnable as ``DEPOLARIZE1_IDLE``).

    Hardware layout after the first ``CX``:
    ``I`` on ancilla qubits, then ``DEPOLARIZE1`` on all data qubits (d² idling during
    ancilla ``I``), then ``DEPOLARIZE1`` on those ancilla qubits.
    """
    flat = list(circuit.flattened())
    data = _data_qubit_indices_from_coords(circuit)
    anc = _ancilla_qubit_indices_from_coords(circuit)
    if data is None or anc is None:
        return frozenset()
    cx_idx = next((i for i, op in enumerate(flat) if op.name == "CX"), None)
    if cx_idx is None:
        return frozenset()
    j = cx_idx + 1
    while j < len(flat) and flat[j].name == "TICK":
        j += 1
    if j >= len(flat) or flat[j].name != "I":
        return frozenset()
    anc_idle_qs = {t.value for t in flat[j].targets_copy() if t.is_qubit_target}
    if not anc_idle_qs or not anc_idle_qs <= anc:
        return frozenset()
    j += 1
    while j < len(flat) and flat[j].name == "TICK":
        j += 1
    out: set[int] = set()
    data_seen: set[int] = set()
    data_start = j
    while j < len(flat) and flat[j].name == "DEPOLARIZE1":
        qs = {t.value for t in flat[j].targets_copy() if t.is_qubit_target}
        if not qs or not qs <= data:
            break
        data_seen.update(qs)
        j += 1
    if data_seen == data:
        out.update(range(data_start, j))
    while j < len(flat) and flat[j].name == "DEPOLARIZE1":
        qs = {t.value for t in flat[j].targets_copy() if t.is_qubit_target}
        if not qs or not qs <= anc_idle_qs:
            break
        out.add(j)
        j += 1
    return frozenset(out)


def _dep1_idle_learnable_flat_indices(
    circuit: stim.Circuit, *, exclude_initial_round: bool = False
) -> frozenset[int]:
    """``DEPOLARIZE1`` after ``I`` idling on the same qubit, excluding DD and reset-sync blocks."""
    after_idle = _dep1_after_idle_flat_indices(circuit)
    dd = _dep1_dd_after_syndrome_data_idle_flat_indices(circuit)
    reset_sync = _dep1_reset_round_simultaneous_idle_flat_indices(circuit)
    out = after_idle - dd - reset_sync
    if exclude_initial_round:
        out -= _dep1_initial_round_idle_excluded_flat_indices(circuit)
    return out


def _elementary_is_idle_i_dep1(elem_key: ParamKey, idle_i_flat: frozenset[int]) -> bool:
    return elem_key[2] == "DEPOLARIZE1" and elem_key[0] in idle_i_flat


def _dep2_family_learnable_key(
    elem_key: ParamKey,
    sharing: Literal["dep2", "time_shared_dep2"],
    *,
    init_prob: float,
) -> LearnableKey:
    if sharing == "dep2":
        return elem_key
    return _time_shared_learnable_key(elem_key, init_prob=init_prob)


def _build_dep2_family_layout(
    elem_keys: list[ParamKey],
    init_elem_probs: np.ndarray,
    sharing: Literal["dep2", "time_shared_dep2"],
    *,
    circuit: stim.Circuit,
) -> tuple[list[LearnableKey], np.ndarray, list[int], np.ndarray]:
    """Learnable post-2Q ``DEPOLARIZE2`` only; ``dep2`` = one param per site, ``time_shared_dep2`` groups by pair+rate."""
    after_2q_flat = _dep2_after_two_qubit_flat_indices(circuit)
    learn_key_to_id: dict[LearnableKey, int] = {}
    learn_keys: list[LearnableKey] = []
    init_sums: list[float] = []
    init_counts: list[int] = []
    elem_to_learn: list[int] = []

    for eid, ek in enumerate(elem_keys):
        if not _elementary_is_dep2_after_2q(ek, after_2q_flat):
            elem_to_learn.append(-1)
            continue
        lk = _dep2_family_learnable_key(
            ek, sharing, init_prob=float(init_elem_probs[eid])
        )
        if lk not in learn_key_to_id:
            learn_key_to_id[lk] = len(learn_keys)
            learn_keys.append(lk)
            init_sums.append(0.0)
            init_counts.append(0)
        lid = learn_key_to_id[lk]
        elem_to_learn.append(lid)
        init_sums[lid] += float(init_elem_probs[eid])
        init_counts[lid] += 1

    if not learn_keys:
        raise ValueError(
            f"{sharing}: no DEPOLARIZE2 after two-qubit gates found in circuit"
        )
    init_learn = np.array(
        [init_sums[i] / init_counts[i] for i in range(len(learn_keys))],
        dtype=np.float64,
    )
    multipliers = np.ones(len(elem_keys), dtype=np.float64)
    return learn_keys, init_learn, elem_to_learn, multipliers


def _build_dep12_time_shared_layout(
    elem_keys: list[ParamKey],
    init_elem_probs: np.ndarray,
    *,
    circuit: stim.Circuit,
) -> tuple[list[LearnableKey], np.ndarray, list[int], np.ndarray]:
    """Learnable set = dep1 after 1Q gate + dep2 after 2Q gate, with time-shared grouping."""
    after_1q_flat = _dep1_after_single_qubit_flat_indices(circuit)
    after_idle_flat = _dep1_after_idle_flat_indices(circuit)
    after_2q_flat = _dep2_after_two_qubit_flat_indices(circuit)

    learn_key_to_id: dict[LearnableKey, int] = {}
    learn_keys: list[LearnableKey] = []
    init_sums: list[float] = []
    init_counts: list[int] = []
    elem_to_learn: list[int] = []

    for eid, ek in enumerate(elem_keys):
        is_dep1_1q = _elementary_is_dep1_after_1q(ek, after_1q_flat) and not _elementary_is_idle_dep1(
            ek, after_idle_flat
        )
        is_dep2_2q = _elementary_is_dep2_after_2q(ek, after_2q_flat)
        is_learnable = is_dep1_1q or is_dep2_2q
        if not is_learnable:
            elem_to_learn.append(-1)
            continue
        if is_dep1_1q:
            # For dep1-after-1Q, share by qubit only (ignore gate kind/rate bucket).
            lk = ("DEPOLARIZE1", "q", ek[4])
        else:
            lk = _time_shared_learnable_key(ek, init_prob=float(init_elem_probs[eid]))
        if lk not in learn_key_to_id:
            learn_key_to_id[lk] = len(learn_keys)
            learn_keys.append(lk)
            init_sums.append(0.0)
            init_counts.append(0)
        lid = learn_key_to_id[lk]
        elem_to_learn.append(lid)
        init_sums[lid] += float(init_elem_probs[eid])
        init_counts[lid] += 1

    if not learn_keys:
        raise ValueError(
            "time_shared_dep12: no DEPOLARIZE1-after-1Q or DEPOLARIZE2-after-2Q sites found in circuit"
        )
    init_learn = np.array(
        [init_sums[i] / init_counts[i] for i in range(len(learn_keys))],
        dtype=np.float64,
    )
    multipliers = np.ones(len(elem_keys), dtype=np.float64)
    return learn_keys, init_learn, elem_to_learn, multipliers


def _build_dep12_extended_time_shared_layout(
    elem_keys: list[ParamKey],
    init_elem_probs: np.ndarray,
    *,
    circuit: stim.Circuit,
    include_idle: bool = False,
    include_m: bool = False,
    include_m_ancilla_only: bool = False,
    include_dd: bool = False,
    include_idle_i: bool = False,
    exclude_initial_idle_i: bool = False,
    include_r: bool = False,
) -> tuple[list[LearnableKey], np.ndarray, list[int], np.ndarray]:
    """Extended dep12 family with optional idling and/or measurement-reset noise."""
    after_1q_flat = _dep1_after_single_qubit_flat_indices(circuit)
    after_2q_flat = _dep2_after_two_qubit_flat_indices(circuit)
    after_idle_flat = _dep1_after_idle_flat_indices(circuit)
    dd_flat = _dep1_dd_after_syndrome_data_idle_flat_indices(circuit) if include_dd else frozenset()
    idle_i_flat = (
        _dep1_idle_learnable_flat_indices(circuit, exclude_initial_round=exclude_initial_idle_i)
        if include_idle_i
        else frozenset()
    )
    if include_m:
        if include_m_ancilla_only:
            before_m_x_flat = _x_error_before_syndrome_measure_flat_indices(circuit)
        else:
            before_m_x_flat = _x_error_before_measure_flat_indices(circuit)
    else:
        before_m_x_flat = frozenset()
    after_r_x_flat = _x_error_after_reset_flat_indices(circuit) if include_r else frozenset()

    learn_key_to_id: dict[LearnableKey, int] = {}
    learn_keys: list[LearnableKey] = []
    init_sums: list[float] = []
    init_counts: list[int] = []
    elem_to_learn: list[int] = []

    for eid, ek in enumerate(elem_keys):
        is_dep2_2q = _elementary_is_dep2_after_2q(ek, after_2q_flat)
        is_dd = _elementary_is_dep1_at(ek, dd_flat) if include_dd else False
        is_idle_i = _elementary_is_idle_i_dep1(ek, idle_i_flat) if include_idle_i else False
        is_idle_site = _elementary_is_idle_dep1(ek, after_idle_flat)
        is_idle = is_idle_site if include_idle else False
        is_x_before_m = _elementary_is_x_error_at(ek, before_m_x_flat) if include_m else False
        is_x_after_r = _elementary_is_x_error_at(ek, after_r_x_flat) if include_r else False
        is_dep1_1q = (
            _elementary_is_dep1_after_1q(ek, after_1q_flat) and not is_dd and not is_idle_site
        )
        is_learnable = (
            is_dep1_1q or is_dep2_2q or is_idle or is_idle_i or is_x_before_m or is_x_after_r or is_dd
        )
        if not is_learnable:
            elem_to_learn.append(-1)
            continue
        if is_dd:
            lk = ("DEPOLARIZE1_DD", "q", ek[4])
        elif is_idle_i:
            lk = ("DEPOLARIZE1_IDLE", "q", ek[4])
        elif is_x_before_m:
            lk = ("X_ERROR_M", "q", ek[4])
        elif is_x_after_r:
            lk = ("X_ERROR_R", "q", ek[4])
        elif is_dep1_1q:
            lk = ("DEPOLARIZE1", "q", ek[4])
        else:
            lk = _time_shared_learnable_key(ek, init_prob=float(init_elem_probs[eid]))
        if lk not in learn_key_to_id:
            learn_key_to_id[lk] = len(learn_keys)
            learn_keys.append(lk)
            init_sums.append(0.0)
            init_counts.append(0)
        lid = learn_key_to_id[lk]
        elem_to_learn.append(lid)
        init_sums[lid] += float(init_elem_probs[eid])
        init_counts[lid] += 1

    if not learn_keys:
        raise ValueError("no learnable sites found for requested dep12 extension")
    init_learn = np.array(
        [init_sums[i] / init_counts[i] for i in range(len(learn_keys))],
        dtype=np.float64,
    )
    multipliers = np.ones(len(elem_keys), dtype=np.float64)
    return learn_keys, init_learn, elem_to_learn, multipliers


def _build_learnable_layout(
    elem_keys: list[ParamKey],
    init_elem_probs: np.ndarray,
    sharing: ParamSharing,
    *,
    circuit: stim.Circuit | None = None,
) -> tuple[list[LearnableKey], np.ndarray, list[int], np.ndarray]:
    learn_key_to_id: dict[LearnableKey, int] = {}
    learn_keys: list[LearnableKey] = []
    init_sums: list[float] = []
    init_counts: list[int] = []
    elem_to_learn: list[int] = []

    if sharing == "dep":
        base_p = _infer_noise_base_p(init_elem_probs)
        multipliers = init_elem_probs.astype(np.float64) / base_p
        learn_keys = [("dep",)]
        init_learn = np.array([base_p], dtype=np.float64)
        elem_to_learn = [0] * len(elem_keys)
        return learn_keys, init_learn, elem_to_learn, multipliers

    if sharing in ("time_shared_dep2", "dep2"):
        if circuit is None:
            raise ValueError(f"{sharing} requires circuit to locate DEPOLARIZE2 after 2Q gates")
        return _build_dep2_family_layout(
            elem_keys, init_elem_probs, sharing, circuit=circuit
        )
    if sharing == "time_shared_dep12":
        if circuit is None:
            raise ValueError("time_shared_dep12 requires circuit to locate dep1/dep2 trainable sites")
        return _build_dep12_time_shared_layout(
            elem_keys, init_elem_probs, circuit=circuit
        )
    if sharing == "time_shared_dep12_M_DD":
        if circuit is None:
            raise ValueError(
                "time_shared_dep12_M_DD requires circuit to locate dep1/dep2/M/DD trainable sites"
            )
        return _build_dep12_extended_time_shared_layout(
            elem_keys,
            init_elem_probs,
            circuit=circuit,
            include_idle=False,
            include_m=True,
            include_m_ancilla_only=True,
            include_dd=True,
            include_r=False,
        )
    if sharing == "time_shared_dep12_M_DD_I":
        if circuit is None:
            raise ValueError(
                "time_shared_dep12_M_DD_I requires circuit to locate dep1/dep2/M/DD/idle trainable sites"
            )
        return _build_dep12_extended_time_shared_layout(
            elem_keys,
            init_elem_probs,
            circuit=circuit,
            include_idle=False,
            include_m=True,
            include_m_ancilla_only=True,
            include_dd=True,
            include_idle_i=True,
            exclude_initial_idle_i=True,
            include_r=False,
        )
    if sharing == "time_shared_dep12_M_R_DD_I":
        if circuit is None:
            raise ValueError(
                "time_shared_dep12_M_R_DD_I requires circuit to locate dep1/dep2/M/DD/idle/R trainable sites"
            )
        return _build_dep12_extended_time_shared_layout(
            elem_keys,
            init_elem_probs,
            circuit=circuit,
            include_idle=False,
            include_m=True,
            include_m_ancilla_only=True,
            include_dd=True,
            include_idle_i=True,
            include_r=True,
        )

    multipliers = np.ones(len(elem_keys), dtype=np.float64)
    for eid, ek in enumerate(elem_keys):
        lk = _elementary_to_learnable_key(ek, sharing, init_prob=float(init_elem_probs[eid]))
        if lk not in learn_key_to_id:
            learn_key_to_id[lk] = len(learn_keys)
            learn_keys.append(lk)
            init_sums.append(0.0)
            init_counts.append(0)
        lid = learn_key_to_id[lk]
        elem_to_learn.append(lid)
        init_sums[lid] += float(init_elem_probs[eid])
        init_counts[lid] += 1

    init_learn = np.array([s / c for s, c in zip(init_sums, init_counts)], dtype=np.float64)
    return learn_keys, init_learn, elem_to_learn, multipliers


def _expand_gate_probs(
    learn_probs: np.ndarray,
    elem_to_learn: list[int],
    multipliers: np.ndarray,
    init_elem_probs: np.ndarray,
) -> np.ndarray:
    """Per-elementary gate probabilities used by replay and circuit patching."""
    out = np.empty(len(elem_to_learn), dtype=np.float64)
    for eid, lid in enumerate(elem_to_learn):
        if lid < 0:
            out[eid] = float(init_elem_probs[eid])
        else:
            out[eid] = float(multipliers[eid]) * float(learn_probs[lid])
    return out


def _circuit_with_gate_probs(
    circuit: stim.Circuit,
    gate_probs: np.ndarray,
    *,
    param_key_to_id: dict[ParamKey, int],
) -> stim.Circuit:
    flat = list(circuit.flattened())
    out = stim.Circuit()
    tick = 0
    for i, op in enumerate(flat):
        if op.name == "TICK":
            out.append("TICK")
            tick += 1
            continue
        if not _is_learnable_noise_instruction(op):
            out.append(op)
            continue
        targets = [t.value for t in op.targets_copy()]
        keys = _elementary_param_key(i, tick, op.name, targets)
        if op.name == "DEPOLARIZE2":
            by_eid: dict[int, list[int]] = {}
            for pk in keys:
                eid = param_key_to_id[pk]
                lo, hi = pk[4], pk[5]
                by_eid.setdefault(eid, []).extend([lo, hi])
            for eid, qs in by_eid.items():
                out.append(op.name, qs, float(gate_probs[eid]))
        else:
            by_eid_q: dict[int, list[int]] = {}
            for pk in keys:
                if pk[3] != "q":
                    continue
                eid = param_key_to_id[pk]
                by_eid_q.setdefault(eid, []).append(pk[4])
            for eid, qs in by_eid_q.items():
                out.append(op.name, qs, float(gate_probs[eid]))
    return out


def _circuit_with_learn_probs(
    circuit: stim.Circuit,
    learn_probs: np.ndarray,
    *,
    param_key_to_id: dict[ParamKey, int],
    elem_to_learn: list[int],
    gate_prob_multipliers: np.ndarray,
    init_elem_probs: np.ndarray,
) -> stim.Circuit:
    gate_probs = _expand_gate_probs(
        learn_probs, elem_to_learn, gate_prob_multipliers, init_elem_probs
    )
    return _circuit_with_gate_probs(circuit, gate_probs, param_key_to_id=param_key_to_id)


def _probe_dem(
    circuit: stim.Circuit,
    learn_probs: np.ndarray,
    *,
    param_key_to_id: dict[ParamKey, int],
    elem_to_learn: list[int],
    gate_prob_multipliers: np.ndarray,
    init_elem_probs: np.ndarray,
) -> np.ndarray:
    patched = _circuit_with_learn_probs(
        circuit,
        learn_probs,
        param_key_to_id=param_key_to_id,
        elem_to_learn=elem_to_learn,
        gate_prob_multipliers=gate_prob_multipliers,
        init_elem_probs=init_elem_probs,
    )
    return _dem_probs_from_stim(patched)


def dem_from_gate_probs(
    gate_probs: torch.Tensor,
    *,
    merge_ops,
    stim_dem: stim.DetectorErrorModel,
    num_dem: int | None = None,
) -> torch.Tensor:
    from src.stim_error_analyzer import dem_probs_from_replay, replay_error_probs

    errors = replay_error_probs(merge_ops, gate_probs)
    out = dem_probs_from_replay(
        errors, stim_dem, dtype=gate_probs.dtype, device=gate_probs.device
    )
    if num_dem is not None and out.shape[0] != num_dem:
        raise ValueError(f"expected num_dem={num_dem}, got {out.shape[0]}")
    return out


def compile_circuit(
    circuit: stim.Circuit,
    *,
    param_sharing: ParamSharing = "time_shared_dep12",
) -> dict:
    from src.stim_error_analyzer import compile_stim_analyzer

    param_key_to_id, elem_keys, init_elem_probs = _scan_elementary_gate_params(circuit)
    learn_keys, init_learn_probs, elem_to_learn, gate_prob_multipliers = _build_learnable_layout(
        elem_keys, init_elem_probs, param_sharing, circuit=circuit
    )
    init_gate_probs = _expand_gate_probs(
        init_learn_probs, elem_to_learn, gate_prob_multipliers, init_elem_probs
    )
    ref_dem_probs = _probe_dem(
        circuit,
        init_learn_probs,
        param_key_to_id=param_key_to_id,
        elem_to_learn=elem_to_learn,
        gate_prob_multipliers=gate_prob_multipliers,
        init_elem_probs=init_elem_probs,
    )
    flat, merge_ops, stim_dem = compile_stim_analyzer(
        circuit,
        param_key_to_id=param_key_to_id,
        elem_to_learn=elem_to_learn,
        elem_keys=elem_keys,
        init_gate_probs=init_gate_probs,
        ref_dem_probs=ref_dem_probs,
        init_learn_probs=init_learn_probs,
        gate_prob_multipliers=gate_prob_multipliers,
        init_elem_probs=init_elem_probs,
    )
    stim_dem_probs = _dem_probs_from_stim(circuit)
    return {
        "num_dem": len(ref_dem_probs),
        "num_slots": len(learn_keys),
        "num_elementary": len(elem_keys),
        "param_sharing": param_sharing,
        "slot_keys": learn_keys,
        "elementary_keys": elem_keys,
        "elem_to_learn": elem_to_learn,
        "gate_prob_multipliers": gate_prob_multipliers,
        "init_learn_probs": init_learn_probs,
        "init_elem_probs": init_elem_probs,
        "param_key_to_id": param_key_to_id,
        "flat_circuit": flat,
        "merge_ops": merge_ops,
        "stim_dem": stim_dem,
        "init_gate_probs": init_gate_probs,
        "ref_dem_probs": ref_dem_probs,
        "stim_dem_probs": stim_dem_probs,
    }


class GateNoiseToDEM(nn.Module):
    """可微分 Gate → DEM；可学习标量个数由 ``param_sharing`` 决定。"""

    def __init__(
        self,
        circuit: stim.Circuit,
        *,
        learnable: bool = True,
        init_from_circuit: bool = True,
        param_mode: Literal["logit", "raw"] = "logit",
        param_sharing: ParamSharing = "time_shared_dep12",
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        meta = compile_circuit(circuit, param_sharing=param_sharing)
        self.num_dem: int = meta["num_dem"]
        self.num_slots: int = meta["num_slots"]
        self.num_elementary: int = meta["num_elementary"]
        self.param_sharing: ParamSharing = meta["param_sharing"]
        self.slot_keys: list[LearnableKey] = meta["slot_keys"]
        self._merge_ops = meta["merge_ops"]
        self._stim_dem = meta["stim_dem"]
        self._elem_to_learn = meta["elem_to_learn"]
        self.param_mode = param_mode

        self.register_buffer(
            "init_elementary_probs",
            torch.tensor(meta["init_elem_probs"], dtype=dtype, device=device),
        )
        self.register_buffer(
            "gate_prob_multipliers",
            torch.tensor(meta["gate_prob_multipliers"], dtype=dtype, device=device),
        )

        self.register_buffer(
            "ref_dem_probs",
            torch.tensor(meta["ref_dem_probs"], dtype=dtype, device=device),
        )
        self.register_buffer(
            "stim_ref_dem_probs",
            torch.tensor(meta["stim_dem_probs"], dtype=dtype, device=device),
        )

        init_p = np.clip(meta["init_learn_probs"], 1e-12, 1.0 - 1e-12)
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
            learn = torch.sigmoid(p)
        else:
            learn = p.clamp(1e-12, 1.0 - 1e-12)
        out = self.init_elementary_probs.clone()
        for eid, lid in enumerate(self._elem_to_learn):
            if lid >= 0:
                out[eid] = learn[lid] * self.gate_prob_multipliers[eid]
        return out

    def forward(self, gate_param: torch.Tensor | None = None) -> torch.Tensor:
        return dem_from_gate_probs(
            self.gate_probs(gate_param),
            merge_ops=self._merge_ops,
            stim_dem=self._stim_dem,
            num_dem=self.num_dem,
        )

    def dem_weights(self, dem_probs: torch.Tensor | None = None) -> torch.Tensor:
        p = (dem_probs if dem_probs is not None else self.forward()).clamp(1e-12, 1.0 - 1e-12)
        return torch.log((1.0 - p) / p)


def elementary_dep2_indices(elem_to_learn: list[int] | np.ndarray) -> np.ndarray:
    """Elementary gate indices that belong to learnable post-2Q DEPOLARIZE2 slots."""
    return np.flatnonzero(np.asarray(elem_to_learn, dtype=np.int64) >= 0)


def dep2_learn_probs_from_gate_probs(
    g2d: GateNoiseToDEM,
    gate_probs: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """Extract learnable-slot probabilities from a full elementary gate vector."""
    device = g2d.init_elementary_probs.device
    dtype = g2d.init_elementary_probs.dtype
    gate_probs = torch.as_tensor(gate_probs, dtype=dtype, device=device).reshape(-1)
    learn = torch.zeros(g2d.num_slots, dtype=dtype, device=device)
    for eid, lid in enumerate(g2d._elem_to_learn):
        if lid >= 0:
            learn[lid] = gate_probs[eid] / g2d.gate_prob_multipliers[eid]
    return learn.clamp(1e-12, 1.0 - 1e-12)


def gate_probs_from_fixed_non_dep2(
    g2d: GateNoiseToDEM,
    non_dep2_gate_probs: torch.Tensor,
    dep2_learn_probs: torch.Tensor,
) -> torch.Tensor:
    """Assemble elementary gate_probs with fixed non-dep2 sites and given dep2 learnable rates."""
    if non_dep2_gate_probs.shape[0] != g2d.num_elementary:
        raise ValueError(
            f"non_dep2_gate_probs length {non_dep2_gate_probs.shape[0]} "
            f"!= num_elementary {g2d.num_elementary}"
        )
    if dep2_learn_probs.shape[0] != g2d.num_slots:
        raise ValueError(
            f"dep2_learn_probs length {dep2_learn_probs.shape[0]} != num_slots {g2d.num_slots}"
        )

    out = non_dep2_gate_probs.to(
        dtype=g2d.init_elementary_probs.dtype,
        device=g2d.init_elementary_probs.device,
    ).clone()
    dep2_learn = dep2_learn_probs.clamp(1e-12, 1.0 - 1e-12)
    for eid, lid in enumerate(g2d._elem_to_learn):
        if lid >= 0:
            out[eid] = dep2_learn[lid] * g2d.gate_prob_multipliers[eid]
    return out.clamp(1e-12, 1.0 - 1e-12)


@dataclass(frozen=True)
class Dep2InverseSolveResult:
    dep2_learn_probs: torch.Tensor
    gate_probs: torch.Tensor
    dep2_elementary_probs: torch.Tensor
    fitted_dem: torch.Tensor
    dem_max_abs_error: float


def solve_dep2_gate_probs_from_dem(
    g2d: GateNoiseToDEM,
    target_dem: torch.Tensor | np.ndarray,
    non_dep2_gate_probs: torch.Tensor | np.ndarray,
    *,
    init_dep2_learn_probs: torch.Tensor | np.ndarray | None = None,
    steps: int = 1500,
    lr: float = 0.08,
    tol: float = 1e-11,
) -> Dep2InverseSolveResult:
    """Fix non-dep2 elementary gate errors and solve dep2 rates to match a target DEM.

    Requires ``param_sharing`` in ``(\"dep2\", \"time_shared_dep2\")``.
    """
    if g2d.param_sharing not in ("dep2", "time_shared_dep2"):
        raise ValueError(
            f"solve_dep2_gate_probs_from_dem requires dep2/time_shared_dep2 sharing, "
            f"got {g2d.param_sharing!r}"
        )

    device = g2d.init_elementary_probs.device
    dtype = g2d.init_elementary_probs.dtype

    target = torch.as_tensor(target_dem, dtype=dtype, device=device).reshape(-1)
    if target.shape[0] != g2d.num_dem:
        raise ValueError(f"target_dem length {target.shape[0]} != num_dem {g2d.num_dem}")

    non_dep2 = torch.as_tensor(non_dep2_gate_probs, dtype=dtype, device=device).reshape(-1)
    if non_dep2.shape[0] != g2d.num_elementary:
        raise ValueError(
            f"non_dep2_gate_probs length {non_dep2.shape[0]} != num_elementary {g2d.num_elementary}"
        )

    if init_dep2_learn_probs is None:
        init_learn = dep2_learn_probs_from_gate_probs(g2d, non_dep2)
    else:
        init_learn = torch.as_tensor(init_dep2_learn_probs, dtype=dtype, device=device).reshape(-1)
        if init_learn.shape[0] != g2d.num_slots:
            raise ValueError(
                f"init_dep2_learn_probs length {init_learn.shape[0]} != num_slots {g2d.num_slots}"
            )

    logit = torch.logit(init_learn.clamp(1e-6, 1.0 - 1e-6)).detach().requires_grad_(True)
    opt = torch.optim.Adam([logit], lr=lr)

    best_loss = float("inf")
    best_logit = logit.detach().clone()

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        dep2_learn = torch.sigmoid(logit)
        gate_p = gate_probs_from_fixed_non_dep2(g2d, non_dep2, dep2_learn)
        pred = dem_from_gate_probs(
            gate_p,
            merge_ops=g2d._merge_ops,
            stim_dem=g2d._stim_dem,
            num_dem=g2d.num_dem,
        )
        loss = F.mse_loss(pred, target)
        loss.backward()
        opt.step()

        lv = float(loss.detach().cpu())
        if lv < best_loss:
            best_loss = lv
            best_logit = logit.detach().clone()
        if lv < tol:
            break

    with torch.no_grad():
        dep2_learn = torch.sigmoid(best_logit)
        gate_p = gate_probs_from_fixed_non_dep2(g2d, non_dep2, dep2_learn)
        fitted_dem = dem_from_gate_probs(
            gate_p,
            merge_ops=g2d._merge_ops,
            stim_dem=g2d._stim_dem,
            num_dem=g2d.num_dem,
        )
        dem_err = float((fitted_dem - target).abs().max().cpu())
        dep2_eids = elementary_dep2_indices(g2d._elem_to_learn)

    return Dep2InverseSolveResult(
        dep2_learn_probs=dep2_learn.detach(),
        gate_probs=gate_p.detach(),
        dep2_elementary_probs=gate_p.detach()[dep2_eids],
        fitted_dem=fitted_dem.detach(),
        dem_max_abs_error=dem_err,
    )


def _validate_circuit(
    circuit: stim.Circuit,
    *,
    label: str,
    param_sharing: ParamSharing,
    max_abs_tol: float = 1e-12,
) -> None:
    module = GateNoiseToDEM(circuit, learnable=True, param_sharing=param_sharing, dtype=torch.float64)
    print(
        f"  [{param_sharing}] learnable={module.num_slots} elementary={module.num_elementary} "
        f"dem={module.num_dem}"
    )
    with torch.no_grad():
        pred = module()
        ref = module.ref_dem_probs
        max_abs = float((pred - ref).abs().max())
        if max_abs > max_abs_tol:
            worst = int(torch.argmax((pred - ref).abs()).item())
            raise AssertionError(
                f"{label} {param_sharing}: max_abs={max_abs}, line {worst}: "
                f"pred={float(pred[worst]):.6g} stim={float(ref[worst]):.6g}"
            )
        print(f"    max|g2d-stim|={max_abs:.3e}")
    module.zero_grad()
    module().sum().backward()
    assert module.gate_param.grad is not None and torch.isfinite(module.gate_param.grad).all()


def _validate_phase_b() -> None:
    print("=" * 72)
    print("GateNoiseToDEM Phase B (decompose_errors=False, flatten_loops=True)")
    print("=" * 72)
    cases: list[tuple[str, stim.Circuit, float]] = [
        (
            "rep si1000 d3r3",
            build_repetition_circuit(distance=3, rounds=3, noise_model="si1000"),
            1e-12,
        ),
        (
            "rep dep d3r3",
            build_repetition_circuit(distance=3, rounds=3, noise_model="depolarizing"),
            1e-12,
        ),
        (
            "rep si1000 d5r7",
            build_repetition_circuit(distance=5, rounds=7, noise_model="si1000"),
            1e-12,
        ),
        (
            "surface si1000 d3r3",
            build_surface_code_circuit(distance=3, rounds=3, noise_model="si1000"),
            1e-12,
        ),
        (
            "surface dep d3r3",
            build_surface_code_circuit(distance=3, rounds=3, noise_model="depolarizing"),
            1e-12,
        ),
        (
            "surface si1000 d5r7",
            build_surface_code_circuit(distance=5, rounds=7, noise_model="si1000"),
            1e-12,
        ),
    ]
    for label, circuit, tol in cases:
        for sharing in ("time_shared", "dep"):
            _validate_circuit(circuit, label=label, param_sharing=sharing, max_abs_tol=tol)
    print("验证完成。")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    _validate_phase_b()