"""
Differentiable replay of Stim v1.15 ``ErrorAnalyzer`` (``decompose_errors=False``).

Backward Pauli-frame tracking and exact independent-fault merging. Depolarizing
channels are factored according to the rank of their detector/logical signature
image. Allowed syndrome structure, including zero-rate sites, is fixed at
compile; only gate probabilities are torch tensors.
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


def _depolarize_indep_prob(d: torch.Tensor, *, rank: int, pauli_bits: int) -> torch.Tensor:
    """Independent rate on each nonzero image of a depolarizing fault.

    The nontrivial Fourier moments of a uniform Pauli channel are
    ``1 - 2**pauli_bits * d / (2**pauli_bits - 1)``. A rank-r signature
    image has 2**(r-1) independent mechanisms odd under each such moment.
    This gives the exponent below, including when some Pauli faults have
    identical or invisible detector/logical effects.
    """
    if not 1 <= rank <= pauli_bits:
        raise ValueError(f"expected signature rank in [1, {pauli_bits}], got {rank}")
    factor = (1 << pauli_bits) / ((1 << pauli_bits) - 1)
    if bool(torch.any(~torch.isfinite(d) | (d < 0) | (d > 1.0 / factor))):
        raise ValueError(f"depolarizing rate must be finite and in [0, {1.0 / factor}]")
    if rank == 1:
        return d * (factor / 2.0)
    # log1p/expm1 retain precision and the derivative at a zero initial rate.
    return -0.5 * torch.expm1(torch.log1p(-factor * d) / (1 << (rank - 1)))


def depolarize1_indep_prob(d: torch.Tensor, *, rank: int = 2) -> torch.Tensor:
    """Independent DEPOLARIZE1 rate for a signature image of the given rank."""
    return _depolarize_indep_prob(d, rank=rank, pauli_bits=2)


def depolarize2_indep_prob(d: torch.Tensor, *, rank: int = 4) -> torch.Tensor:
    """Independent DEPOLARIZE2 rate; full rank uses an eighth root."""
    return _depolarize_indep_prob(d, rank=rank, pauli_bits=4)


@dataclass(frozen=True)
class ProbExpr:
    kind: Literal["slot", "dep1", "dep2"]
    slot: int
    scale: float = 1.0
    rank: int | None = None


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

    def add_depolarizing_error(
        self, basis: tuple[Syndrome, ...], *, slot: int, kind: Literal["dep1", "dep2"]
    ) -> None:
        image: set[Syndrome] = {()}
        for vector in basis:
            image |= {_sym_xor(syndrome, vector) for syndrome in image}
        rank = len(image).bit_length() - 1
        for syndrome in sorted(image - {()}):
            self.add_error(syndrome, ProbExpr(kind, slot, rank=rank))


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
            if args:
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
            if args:
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
        for t in op.targets_copy():
            q = t.value
            slot = self._slot_q(flat_idx, gate, q)
            if slot is None:
                continue
            vec = self.tracker.xs[q] if use_x else self.tracker.zs[q]
            self.rec.add_error(vec.items, ProbExpr("slot", slot))

    def _undo_depolarize1(self, op: stim.CircuitInstruction, flat_idx: int) -> None:
        for t in op.targets_copy():
            q = t.value
            slot = self._slot_q(flat_idx, "DEPOLARIZE1", q)
            if slot is None:
                continue
            bx, bz = self.tracker.xs[q].items, self.tracker.zs[q].items
            self.rec.add_depolarizing_error((bx, bz), slot=slot, kind="dep1")

    def _undo_depolarize2(self, op: stim.CircuitInstruction, flat_idx: int) -> None:
        targets = [t.value for t in op.targets_copy()]
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
            # Pairs are independent fault sites, even within one instruction.
            self.rec.add_depolarizing_error(basis, slot=slot, kind="dep2")


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
        base = depolarize1_indep_prob(gate_probs[expr.slot], rank=expr.rank or 2)
    elif expr.kind == "dep2":
        base = depolarize2_indep_prob(gate_probs[expr.slot], rank=expr.rank or 4)
    else:
        raise ValueError(expr.kind)
    if expr.scale == 1.0:
        return base
    return base * float(expr.scale)


def replay_error_probs(
    merge_ops: list[MergeOp],
    gate_probs: torch.Tensor,
) -> dict[Syndrome, torch.Tensor]:
    if gate_probs.ndim != 1:
        raise ValueError("gate_probs must be a one-dimensional elementary probability vector")
    if bool(torch.any(~torch.isfinite(gate_probs) | (gate_probs < 0) | (gate_probs > 1))):
        raise ValueError("gate probabilities must be finite and in [0, 1]")
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
    for i in range(len(stim_dem)):
        if stim_dem[i].type != "error":
            continue
        key = syndrome_from_dem_error(stim_dem, i)
        out.append(errors.get(key, z))
    return torch.stack(out) if out else torch.empty(0, dtype=dtype, device=device)


def compile_merge_ops(circuit: stim.Circuit) -> tuple[stim.Circuit, list[MergeOp], stim.DetectorErrorModel]:
    """Compile elementary fault support, including sites initialized at zero."""
    from src.g2dem import _scan_elementary_gate_params

    key_to_id, keys, _ = _scan_elementary_gate_params(circuit)
    return compile_stim_analyzer(
        circuit, param_key_to_id=key_to_id, elem_to_learn=list(range(len(keys))), elem_keys=keys
    )


def probs_by_syndrome_from_vector(
    stim_dem: stim.DetectorErrorModel,
    probs: np.ndarray,
) -> dict[Syndrome, float]:
    out: dict[Syndrome, float] = {}
    idx = 0
    for i in range(len(stim_dem)):
        if stim_dem[i].type != "error":
            continue
        out[syndrome_from_dem_error(stim_dem, i)] = float(probs[idx])
        idx += 1
    if idx != len(probs):
        raise ValueError(f"expected {idx} error probs, got {len(probs)}")
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
    """Compatibility validator; no longer rescales fault probabilities.

    A mismatch is an algebra or circuit-support error. Fitting a constant to
    hide it would corrupt off-reference probabilities and control gradients.
    """
    del elem_to_learn
    if stim_probs_by_syndrome is None:
        from src.g2dem import _circuit_with_gate_probs

        patched = _circuit_with_gate_probs(circuit, init_gate_probs, param_key_to_id=param_key_to_id)
        dem = patched.detector_error_model(decompose_errors=False, flatten_loops=True)
        stim_probs_by_syndrome = {
            syndrome_from_dem_error(dem, i): float(dem[i].args_copy()[0])
            for i in range(len(dem)) if dem[i].type == "error"
        }
    replay = replay_error_probs(merge_ops, torch.as_tensor(init_gate_probs, dtype=torch.float64))
    for syndrome in replay.keys() | stim_probs_by_syndrome.keys():
        actual = float(replay.get(syndrome, 0.0))
        expected = stim_probs_by_syndrome.get(syndrome, 0.0)
        if not np.isclose(actual, expected, atol=abs_tol, rtol=rel_tol):
            raise ValueError(
                f"exact DEM replay disagrees with Stim for {syndrome}: {actual} != {expected}; "
                "reference-point calibration is not supported"
            )
    return list(merge_ops)


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
    """Compatibility alias for exact replay validation; does not deduplicate sites."""
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
    """Compile the supported noise instructions without reference-rate fitting.

    The exact probability expressions depend only on the circuit's Clifford
    structure. When ``init_gate_probs`` is given, the replay is checked against
    Stim at those rates and at a second point with every rate moved on its own;
    the other initialization keywords are retained for API compatibility.
    The returned DEM additionally contains allowed zero-rate error columns.
    """
    flat = flatten_circuit(circuit)
    dem = flat.detector_error_model(decompose_errors=False, flatten_loops=True)
    slot_noise, slot_pair = build_noise_slot_maps(
        flat, param_key_to_id=param_key_to_id, elem_to_learn=elem_to_learn, elem_keys=elem_keys
    )
    recorder = ErrorAnalyzerRecorder(
        flat, slot_for_noise=slot_noise, slot_for_pair=slot_pair
    )
    ops = recorder.run()
    # The allowed fault topology is independent of its initial probabilities.
    # Stim omits zero-rate mechanisms, so append those columns explicitly while
    # preserving existing error order and all detector/observable declarations.
    present = {
        syndrome_from_dem_error(dem, i)
        for i in range(len(dem)) if dem[i].type == "error"
    }
    missing = {op.syndrome for op in ops} - present
    if missing:
        complete = stim.DetectorErrorModel()
        for instruction in dem:
            if instruction.type == "error":
                complete.append(instruction)
        for syndrome in sorted(missing):
            targets = [
                stim.target_relative_detector_id(index) if kind == "D"
                else stim.target_logical_observable_id(index)
                for kind, index in syndrome
            ]
            complete.append("error", 0.0, targets)
        for instruction in dem:
            if instruction.type != "error":
                complete.append(instruction)
        dem = complete
    if init_gate_probs is not None:
        # Agreement with Stim at the compiled rates alone does not test how the
        # replay varies with them. Compare at a second point as well, with every
        # elementary rate rescaled on its own and zero-rate sites switched on.
        init = np.asarray(init_gate_probs, dtype=np.float64)
        positive = init[init > 0]
        floor = float(positive.min()) if positive.size else 1e-3
        scales = np.random.default_rng(0).uniform(0.5, 1.0, size=init.shape)
        moved = np.where(init > 0, init, floor) * scales
        for tag, gate_probs in (("compiled", init), ("rescaled", moved)):
            try:
                # The replay and Stim round differently (by up to ~1e-15), which
                # leaves no margin under the validator's default absolute tolerance.
                calibrate_merge_ops_with_stim(
                    circuit,
                    ops,
                    gate_probs,
                    param_key_to_id=param_key_to_id,
                    elem_to_learn=elem_to_learn,
                    abs_tol=1e-12,
                )
            except ValueError as e:
                raise ValueError(f"at the {tag} gate rates: {e}") from e
    return flat, ops, dem
