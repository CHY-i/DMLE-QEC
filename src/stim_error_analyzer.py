"""
Differentiable replay of Stim v1.15 ``ErrorAnalyzer`` (``decompose_errors=False``).

Backward Pauli-frame tracking + ``add_error`` / ``add_error_combinations`` merge logic
from ``stim/simulators/error_analyzer.cc``. Syndrome structure is fixed at compile;
only gate probabilities are torch tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import stim
import torch

Syndrome = tuple[tuple[str, int], ...]


def flatten_circuit(circuit: stim.Circuit) -> stim.Circuit:
    flat = stim.Circuit()
    for op in circuit.flattened():
        flat.append(op)
    return flat


def _sym_xor(a: Syndrome, b: Syndrome) -> Syndrome:
    return tuple(sorted(set(a) ^ set(b)))


def syndrome_from_dem_error(dem: stim.DetectorErrorModel, line: int) -> Syndrome:
    terms: list[tuple[str, int]] = []
    for t in dem[line].targets_copy():
        name = str(t)
        if name.startswith("D"):
            terms.append(("D", int(name[1:])))
        elif name.startswith("L"):
            terms.append(("L", int(name[1:])))
    return tuple(sorted(terms))


def stim_merge_p(old: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return old * (1.0 - p) + (1.0 - old) * p


def depolarize1_indep_prob(d: torch.Tensor) -> torch.Tensor:
    """Stim ``depolarize1_probability_to_independent_per_channel_probability``."""
    inner = (1.0 - 4.0 * d / 3.0).clamp(min=0.0)
    return (1.0 - torch.sqrt(inner)) / 2.0


def depolarize2_indep_prob(d: torch.Tensor) -> torch.Tensor:
    """Stim ``depolarize2_probability_to_independent_per_channel_probability``."""
    inner = (1.0 - 16.0 * d / 15.0).clamp(min=0.0)
    return (1.0 - torch.sqrt(inner)) / 2.0


@dataclass(frozen=True)
class ProbExpr:
    kind: Literal["slot", "dep1", "dep2"]
    slot: int
    scale: float = 1.0


@dataclass(frozen=True)
class MergeOp:
    syndrome: Syndrome
    prob: ProbExpr


class _XorVec:
    __slots__ = ("items",)

    def __init__(self, items: Syndrome = ()) -> None:
        self.items = tuple(sorted(set(items)))

    def xor_vec(self, other: _XorVec) -> None:
        self.items = _sym_xor(self.items, other.items)

    def xor_item(self, item: tuple[str, int]) -> None:
        self.items = _sym_xor(self.items, (item,))


class _Tracker:
    def __init__(self, num_qubits: int, num_meas: int, num_det: int) -> None:
        self.xs = [_XorVec() for _ in range(num_qubits)]
        self.zs = [_XorVec() for _ in range(num_qubits)]
        self.rec_bits: dict[int, _XorVec] = {}
        self.num_measurements_in_past = num_meas
        self.num_detectors_in_past = num_det

    def det_item(self) -> tuple[str, int]:
        return ("D", int(self.num_detectors_in_past))

    def obs_item(self, obs_id: int) -> tuple[str, int]:
        return ("L", int(obs_id))


class _Recorder:
    def __init__(self) -> None:
        self.ops: list[MergeOp] = []

    def add_error(self, syndrome: Syndrome, prob: ProbExpr) -> None:
        if syndrome:
            self.ops.append(MergeOp(syndrome, prob))

    def add_error_combinations(
        self,
        probabilities: tuple[float, ...],
        basis: tuple[Syndrome, ...],
        *,
        slot: int,
        kind: Literal["slot", "dep1", "dep2"],
        probabilities_are_disjoint: bool = False,
        seen_syn_gate: set[Syndrome] | None = None,
    ) -> None:
        s = len(basis)
        probs = [float(probabilities[mask]) if mask < len(probabilities) else 0.0 for mask in range(1 << s)]
        stored: dict[int, Syndrome] = {0: ()}
        for k in range(s):
            stored[1 << k] = basis[k]
        for mask in range(1, 1 << s):
            if mask in stored:
                continue
            c1 = mask & (mask - 1)
            c2 = mask ^ c1
            if c1:
                stored[mask] = _sym_xor(stored[c1], stored[c2])
        if probabilities_are_disjoint:
            for k in range(1, 1 << s):
                if stored.get(k, ()):
                    continue
                for k_dst in range(1 << s):
                    k_src = k_dst ^ k
                    if k_src > k_dst:
                        probs[k_dst] += probs[k_src]
                        probs[k_src] = 0.0
        seen_syn_dep2: set[Syndrome] = set()
        for mask in range(1, 1 << s):
            coeff = probs[mask]
            if coeff == 0.0:
                continue
            syn = stored.get(mask, ())
            if not syn:
                continue
            if kind == "dep2":
                if syn in seen_syn_dep2:
                    continue
                seen_syn_dep2.add(syn)
                if seen_syn_gate is not None:
                    if syn in seen_syn_gate:
                        continue
                    seen_syn_gate.add(syn)
            self.add_error(syn, ProbExpr(kind, slot, scale=float(coeff)))


class ErrorAnalyzerRecorder:
    """Record Stim merge ops on a flattened circuit (backward pass)."""

    def __init__(
        self,
        flat_circuit: stim.Circuit,
        *,
        slot_for_noise: dict[tuple[int, str, int], int],
        slot_for_pair: dict[tuple[int, str, int, int], int],
    ) -> None:
        self.flat = list(flat_circuit.flattened())
        self.slot_noise = slot_for_noise
        self.slot_pair = slot_for_pair
        self.tracker = _Tracker(
            flat_circuit.num_qubits,
            flat_circuit.num_measurements,
            flat_circuit.num_detectors,
        )
        self.rec = _Recorder()

    def run(self) -> list[MergeOp]:
        for flat_idx in range(len(self.flat) - 1, -1, -1):
            self._undo(self.flat[flat_idx], flat_idx)
        return self.rec.ops

    def _slot_q(self, flat_idx: int, gate: str, q: int) -> int | None:
        return self.slot_noise.get((flat_idx, gate, q))

    def _slot_pair(self, flat_idx: int, gate: str, a: int, b: int) -> int | None:
        lo, hi = (a, b) if a < b else (b, a)
        return self.slot_pair.get((flat_idx, gate, lo, hi))

    def _undo(self, op: stim.CircuitInstruction, flat_idx: int) -> None:
        g = op.name
        if g in {"QUBIT_COORDS", "SHIFT_COORDS"}:
            return
        if g == "DETECTOR":
            self._undo_detector(op)
        elif g == "OBSERVABLE_INCLUDE":
            self._undo_observable(op)
        elif g == "TICK":
            return
        elif g == "CX":
            self._undo_cx(op)
        elif g == "CZ":
            self._undo_cz(op)
        elif g == "H":
            self._undo_h(op)
        elif g == "X":
            return
        elif g == "I":
            return
        elif g == "DEPOLARIZE1":
            self._undo_depolarize1(op, flat_idx)
        elif g == "DEPOLARIZE2":
            self._undo_depolarize2(op, flat_idx)
        elif g == "X_ERROR":
            self._undo_pauli_error(op, flat_idx, "X_ERROR", use_x=False)
        elif g == "Z_ERROR":
            self._undo_pauli_error(op, flat_idx, "Z_ERROR", use_x=True)
        elif g == "M":
            self._undo_mz(op, flat_idx)
        elif g == "MR":
            self._undo_mrz(op, flat_idx)
        elif g == "R":
            self._undo_rz(op)
        else:
            raise NotImplementedError(f"unsupported gate {g!r} in ErrorAnalyzerRecorder")

    def _undo_detector(self, op: stim.CircuitInstruction) -> None:
        self.tracker.num_detectors_in_past -= 1
        det = self.tracker.det_item()
        for t in op.targets_copy():
            idx = int(t.value) + int(self.tracker.num_measurements_in_past)
            self.tracker.rec_bits.setdefault(idx, _XorVec()).xor_item(det)

    def _undo_observable(self, op: stim.CircuitInstruction) -> None:
        obs_id = int(op.gate_args_copy()[0])
        obs = self.tracker.obs_item(obs_id)
        for t in op.targets_copy():
            if t.is_measurement_record_target:
                idx = int(t.value) + int(self.tracker.num_measurements_in_past)
                self.tracker.rec_bits.setdefault(idx, _XorVec()).xor_item(obs)

    def _undo_cx(self, op: stim.CircuitInstruction) -> None:
        targets = op.targets_copy()
        for k in range(len(targets) - 2, -1, -2):
            c, t = targets[k].value, targets[k + 1].value
            self.tracker.zs[c].xor_vec(self.tracker.zs[t])
            self.tracker.xs[t].xor_vec(self.tracker.xs[c])

    def _undo_cz(self, op: stim.CircuitInstruction) -> None:
        targets = op.targets_copy()
        for k in range(len(targets) - 2, -1, -2):
            a, b = targets[k].value, targets[k + 1].value
            self.tracker.zs[b].xor_vec(self.tracker.xs[a])
            self.tracker.zs[a].xor_vec(self.tracker.xs[b])

    def _undo_h(self, op: stim.CircuitInstruction) -> None:
        for t in op.targets_copy():
            q = t.value
            self.tracker.xs[q], self.tracker.zs[q] = self.tracker.zs[q], self.tracker.xs[q]

    def _undo_rz(self, op: stim.CircuitInstruction) -> None:
        for t in op.targets_copy():
            q = t.value
            self.tracker.xs[q] = _XorVec()
            self.tracker.zs[q] = _XorVec()

    def _undo_mz(self, op: stim.CircuitInstruction, flat_idx: int) -> None:
        args = op.gate_args_copy()
        for t in reversed(op.targets_copy()):
            q = t.value
            self.tracker.num_measurements_in_past -= 1
            rec = self.tracker.rec_bits.pop(self.tracker.num_measurements_in_past, _XorVec())
            if args and float(args[0]) > 0:
                slot = self._slot_q(flat_idx, "M", q)
                if slot is not None:
                    self.rec.add_error(rec.items, ProbExpr("slot", slot))
            self.tracker.zs[q].xor_vec(rec)

    def _undo_mrz(self, op: stim.CircuitInstruction, flat_idx: int) -> None:
        args = op.gate_args_copy()
        for t in reversed(op.targets_copy()):
            q = t.value
            self._undo_rz_on_q(q)
            self.tracker.num_measurements_in_past -= 1
            rec = self.tracker.rec_bits.pop(self.tracker.num_measurements_in_past, _XorVec())
            if args and float(args[0]) > 0:
                slot = self._slot_q(flat_idx, "MR", q)
                if slot is not None:
                    self.rec.add_error(rec.items, ProbExpr("slot", slot))
            self.tracker.zs[q].xor_vec(rec)

    def _undo_rz_on_q(self, q: int) -> None:
        self.tracker.xs[q] = _XorVec()
        self.tracker.zs[q] = _XorVec()

    def _undo_pauli_error(
        self, op: stim.CircuitInstruction, flat_idx: int, gate: str, *, use_x: bool
    ) -> None:
        p = float(op.gate_args_copy()[0]) if op.gate_args_copy() else 0.0
        if p == 0.0:
            return
        for t in op.targets_copy():
            q = t.value
            slot = self._slot_q(flat_idx, gate, q)
            if slot is None:
                continue
            vec = self.tracker.xs[q] if use_x else self.tracker.zs[q]
            self.rec.add_error(vec.items, ProbExpr("slot", slot))

    def _undo_depolarize1(self, op: stim.CircuitInstruction, flat_idx: int) -> None:
        d = float(op.gate_args_copy()[0]) if op.gate_args_copy() else 0.0
        if d == 0.0 or d > 0.75:
            return
        for t in op.targets_copy():
            q = t.value
            slot = self._slot_q(flat_idx, "DEPOLARIZE1", q)
            if slot is None:
                continue
            bx, bz = self.tracker.xs[q].items, self.tracker.zs[q].items
            self.rec.add_error_combinations(
                (0.0, 1.0, 1.0, 1.0), (bx, bz), slot=slot, kind="dep1"
            )

    def _undo_depolarize2(self, op: stim.CircuitInstruction, flat_idx: int) -> None:
        d = float(op.gate_args_copy()[0]) if op.gate_args_copy() else 0.0
        if d == 0.0 or d > 15.0 / 16.0:
            return
        targets = [t.value for t in op.targets_copy()]
        # One DEPOLARIZE2 can touch many pairs; Stim dedupes identical syndromes per instruction.
        seen_syn_gate: set[Syndrome] = set()
        for i in range(0, len(targets), 2):
            a, b = targets[i], targets[i + 1]
            slot = self._slot_pair(flat_idx, "DEPOLARIZE2", a, b)
            if slot is None:
                continue
            basis = (
                self.tracker.xs[a].items,
                self.tracker.zs[a].items,
                self.tracker.xs[b].items,
                self.tracker.zs[b].items,
            )
            probs = (0.0,) + (1.0,) * 15
            self.rec.add_error_combinations(
                probs,
                basis,
                slot=slot,
                kind="dep2",
                seen_syn_gate=seen_syn_gate,
            )


def build_noise_slot_maps(
    flat_circuit: stim.Circuit,
    *,
    param_key_to_id: dict,
    elem_to_learn: list[int],
    elem_keys: list,
) -> tuple[dict[tuple[int, str, int], int], dict[tuple[int, str, int, int], int]]:
    from src.g2dem import _elementary_param_key, _is_learnable_noise_instruction

    slot_noise: dict[tuple[int, str, int], int] = {}
    slot_pair: dict[tuple[int, str, int, int], int] = {}
    flat = list(flat_circuit.flattened())
    tick = 0
    for i, op in enumerate(flat):
        if op.name == "TICK":
            tick += 1
            continue
        if not _is_learnable_noise_instruction(op):
            continue
        targets = [t.value for t in op.targets_copy()]
        if op.name == "DEPOLARIZE2":
            for pk in _elementary_param_key(i, tick, op.name, targets):
                eid = param_key_to_id[pk]
                lo, hi = pk[4], pk[5]
                slot_pair[(i, op.name, lo, hi)] = eid
        else:
            for pk in _elementary_param_key(i, tick, op.name, targets):
                eid = param_key_to_id[pk]
                if pk[3] == "q":
                    slot_noise[(i, op.name, pk[4])] = eid
    return slot_noise, slot_pair


def _eval_prob(expr: ProbExpr, gate_probs: torch.Tensor) -> torch.Tensor:
    base: torch.Tensor
    if expr.kind == "slot":
        base = gate_probs[expr.slot]
    elif expr.kind == "dep1":
        base = depolarize1_indep_prob(gate_probs[expr.slot])
    elif expr.kind == "dep2":
        base = depolarize2_indep_prob(gate_probs[expr.slot])
    else:
        raise ValueError(expr.kind)
    if expr.scale == 1.0:
        return base
    return base * float(expr.scale)


def replay_error_probs(
    merge_ops: list[MergeOp],
    gate_probs: torch.Tensor,
) -> dict[Syndrome, torch.Tensor]:
    errors: dict[Syndrome, torch.Tensor] = {}
    z = torch.zeros((), dtype=gate_probs.dtype, device=gate_probs.device)
    for op in merge_ops:
        p = _eval_prob(op.prob, gate_probs)
        old = errors.get(op.syndrome, z)
        errors[op.syndrome] = stim_merge_p(old, p)
    return errors


def dem_probs_from_replay(
    errors: dict[Syndrome, torch.Tensor],
    stim_dem: stim.DetectorErrorModel,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    z = torch.zeros((), dtype=dtype, device=device)
    out: list[torch.Tensor] = []
    for i in range(stim_dem.num_errors):
        if stim_dem[i].type != "error":
            continue
        key = syndrome_from_dem_error(stim_dem, i)
        out.append(errors.get(key, z))
    return torch.stack(out)


def compile_merge_ops(circuit: stim.Circuit) -> tuple[stim.Circuit, list[MergeOp], stim.DetectorErrorModel]:
    """Flatten circuit, record merge ops, return reference DEM (Stim, decompose_errors=False)."""
    flat = flatten_circuit(circuit)
    dem = flat.detector_error_model(decompose_errors=False, flatten_loops=True)
    return flat, [], dem


def _dem_prob_for_syndrome(
    circuit: stim.Circuit,
    learn_probs: np.ndarray,
    syndrome: Syndrome,
    *,
    param_key_to_id: dict,
    elem_to_learn: list[int],
) -> float:
    from src.g2dem import _circuit_with_gate_probs

    patched = _circuit_with_gate_probs(
        circuit,
        learn_probs,
        param_key_to_id=param_key_to_id,
    )
    dem = patched.detector_error_model(decompose_errors=False, flatten_loops=True)
    for i in range(dem.num_errors):
        if dem[i].type == "error" and syndrome_from_dem_error(dem, i) == syndrome:
            return float(dem[i].args_copy()[0])
    return 0.0


def probs_by_syndrome_from_vector(
    stim_dem: stim.DetectorErrorModel,
    probs: np.ndarray,
) -> dict[Syndrome, float]:
    out: dict[Syndrome, float] = {}
    idx = 0
    for i in range(stim_dem.num_errors):
        if stim_dem[i].type != "error":
            continue
        out[syndrome_from_dem_error(stim_dem, i)] = float(probs[idx])
        idx += 1
    if idx != len(probs):
        raise ValueError(f"expected {idx} error probs, got {len(probs)}")
    return out


def _replay_syndrome_prob_np(
    ops: list[MergeOp],
    syndrome: Syndrome,
    gate_probs: np.ndarray,
) -> float:
    sub = [op for op in ops if op.syndrome == syndrome]
    if not sub:
        return 0.0
    prob = 0.0
    for op in sub:
        expr = op.prob
        d = float(gate_probs[expr.slot])
        if expr.kind == "slot":
            p = d
        elif expr.kind == "dep1":
            inner = max(0.0, 1.0 - 4.0 * d / 3.0)
            p = (1.0 - inner**0.5) / 2.0
        else:
            inner = max(0.0, 1.0 - 16.0 * d / 15.0)
            p = (1.0 - inner**0.5) / 2.0
        p *= float(expr.scale)
        prob = prob * (1.0 - p) + (1.0 - prob) * p
    return prob


def _replay_syndrome_prob(
    ops: list[MergeOp],
    syndrome: Syndrome,
    gate_probs: torch.Tensor,
) -> torch.Tensor:
    z = torch.zeros((), dtype=gate_probs.dtype, device=gate_probs.device)
    sub = [op for op in ops if op.syndrome == syndrome]
    if not sub:
        return z
    return replay_error_probs(sub, gate_probs).get(syndrome, z)


def _apply_syndrome_scale(ops: list[MergeOp], syndrome: Syndrome, scale: float) -> list[MergeOp]:
    if scale == 1.0:
        return ops
    out: list[MergeOp] = []
    for op in ops:
        if op.syndrome != syndrome:
            out.append(op)
            continue
        out.append(
            MergeOp(
                op.syndrome,
                ProbExpr(op.prob.kind, op.prob.slot, op.prob.scale * float(scale)),
            )
        )
    return out


def calibrate_merge_ops_with_stim(
    circuit: stim.Circuit,
    merge_ops: list[MergeOp],
    init_gate_probs: np.ndarray,
    *,
    param_key_to_id: dict,
    elem_to_learn: list[int],
    stim_probs_by_syndrome: dict[Syndrome, float] | None = None,
    rel_tol: float = 1e-12,
    abs_tol: float = 1e-15,
) -> list[MergeOp]:
    """Scale per-syndrome merge contributions so replay matches Stim at init gate probs."""
    del circuit, param_key_to_id, elem_to_learn
    syndromes = {op.syndrome for op in merge_ops}
    calibrated = list(merge_ops)

    for syn in syndromes:
        if stim_probs_by_syndrome is not None:
            p0 = stim_probs_by_syndrome.get(syn, 0.0)
        else:
            p0 = 0.0
        p_replay = _replay_syndrome_prob_np(calibrated, syn, init_gate_probs)
        err = abs(p_replay - p0)
        if err <= max(abs_tol, rel_tol * max(p0, p_replay, 1.0)):
            continue

        lo, hi = (0.0, 1.0) if p_replay > p0 else (1.0, 4.0)
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            p_mid = _replay_syndrome_prob_np(
                _apply_syndrome_scale(calibrated, syn, mid), syn, init_gate_probs
            )
            if p_mid < p0:
                lo = mid
            else:
                hi = mid

        calibrated = _apply_syndrome_scale(calibrated, syn, 0.5 * (lo + hi))

    return calibrated


def dedupe_noise_ops_with_stim(
    circuit: stim.Circuit,
    merge_ops: list[MergeOp],
    init_gate_probs: np.ndarray,
    *,
    param_key_to_id: dict,
    elem_to_learn: list[int],
    kinds: tuple[str, ...] = ("dep1", "dep2"),
    min_ops_to_cluster: int = 3,
) -> list[MergeOp]:
    """Legacy slot-clustering dedupe; prefer :func:`calibrate_merge_ops_with_stim`."""
    del kinds, min_ops_to_cluster
    return calibrate_merge_ops_with_stim(
        circuit,
        merge_ops,
        init_gate_probs,
        param_key_to_id=param_key_to_id,
        elem_to_learn=elem_to_learn,
    )


def compile_stim_analyzer(
    circuit: stim.Circuit,
    *,
    param_key_to_id: dict,
    elem_to_learn: list[int],
    elem_keys: list,
    init_gate_probs: np.ndarray | None = None,
    ref_dem_probs: np.ndarray | None = None,
    init_learn_probs: np.ndarray | None = None,
    gate_prob_multipliers: np.ndarray | None = None,
    init_elem_probs: np.ndarray | None = None,
) -> tuple[stim.Circuit, list[MergeOp], stim.DetectorErrorModel]:
    flat = flatten_circuit(circuit)
    dem = flat.detector_error_model(decompose_errors=False, flatten_loops=True)
    slot_noise, slot_pair = build_noise_slot_maps(
        flat, param_key_to_id=param_key_to_id, elem_to_learn=elem_to_learn, elem_keys=elem_keys
    )
    recorder = ErrorAnalyzerRecorder(
        flat, slot_for_noise=slot_noise, slot_for_pair=slot_pair
    )
    ops = recorder.run()
    if init_gate_probs is not None:
        if ref_dem_probs is None:
            from src.g2dem import _probe_dem

            if init_learn_probs is None or gate_prob_multipliers is None or init_elem_probs is None:
                raise ValueError(
                    "ref_dem_probs or (init_learn_probs, gate_prob_multipliers, init_elem_probs) required"
                )
            ref_dem_probs = _probe_dem(
                circuit,
                init_learn_probs,
                param_key_to_id=param_key_to_id,
                elem_to_learn=elem_to_learn,
                gate_prob_multipliers=gate_prob_multipliers,
                init_elem_probs=init_elem_probs,
            )
        ops = calibrate_merge_ops_with_stim(
            circuit,
            ops,
            init_gate_probs,
            param_key_to_id=param_key_to_id,
            elem_to_learn=elem_to_learn,
            stim_probs_by_syndrome=probs_by_syndrome_from_vector(dem, ref_dem_probs),
        )
    return flat, ops, dem
