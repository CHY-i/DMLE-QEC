"""
Broadcast learned repetition-code DEMs (lab ``build_stim_circuit``) to larger ``round``.

This module **does not** use :func:`src.utils.broadcast_dem` (surface-code-oriented time
tiling). Instead it encodes the **repetition-memory** layer structure inferred from the
native Stim DEM for ``phase flip`` + ``reset=False``:

- ``num_detectors = (d-1) * (r+1)`` — one time slice of ``d-1`` measure-like detectors
  per round index ``t = 0 … r`` (initial slice, ``r-1`` bulk slices, final slice).
- Flattened DEM errors fall into classes:

  * same-layer pairs ``(t, t)`` with count ``d1-1`` where ``d1 = d-1``;
  * adjacent-layer pairs ``(t, t+1)`` with count ``2*d1-1`` for ``t < r``;
  * final same-layer ``(r, r)`` with count ``d1-1``;
  * two single-detector mechanisms per time layer ``t``;
  * ``D + L0`` correlated errors on the last spatial index each round (special at ``t=0``).

Extending ``r_src → r_tgt`` copies **learned** probabilities from the shorter DEM by:

* **Prefix** ``t ≤ r_src``: identical detector indices as in the ``r_src`` DEM → use the
  learned instruction directly.
* **New bulk** ``r_src < t < r_tgt``: same local pattern as time ``r_src-1`` (cross-layer
  ``(r_src-1, r_src)`` and singles / ``D+L`` there).
* **Final** ``t = r_tgt``: same local pattern as ``t = r_src`` in the source DEM.

These rules were checked against native DEMs for several ``(r_src, r_tgt, d)`` pairs so
that every target mechanism maps to an existing source key.

Run from repo root::

    python logical/broadcast.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import stim
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import update_dem  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_rep_circuits", REPO_ROOT / "logical" / "repetition_code_exp_data" / "circuits.py"
)
_rep_circuits = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_rep_circuits)
build_stim_circuit = _rep_circuits.build_stim_circuit


def _lab_circuit(*, distance: int, rounds: int, circuit_type: str, reset: bool) -> stim.Circuit:
    qubits = np.arange(2 * distance - 1)
    ini_state = np.zeros(distance, dtype=int)
    ct = circuit_type.replace("_", " ")
    return build_stim_circuit(
        qubits=qubits,
        ini_state=ini_state,
        cycle=rounds,
        circuit_type=ct,
        reset=reset,
        add_noise=True,
    )


def assert_repetition_dem_invariants(dem: stim.DetectorErrorModel, *, distance: int, rounds: int) -> None:
    d1 = distance - 1
    expected_det = d1 * (rounds + 1)
    if dem.num_detectors != expected_det:
        raise ValueError(
            f"expected repetition-code num_detectors={expected_det} "
            f"((d-1)*(r+1)) for d={distance}, r={rounds}, got {dem.num_detectors}"
        )


def _expected_num_errors_phase_flip_reset0(*, distance: int, rounds: int) -> int:
    """Empirical closed form for this family (matches native DEM for tested d)."""
    d1 = distance - 1
    return d1 * (3 * rounds + 1) + 1


def _map_detector_index(di: int, *, r_src: int, r_tgt: int, d1: int) -> int:
    """Map a detector index in the ``r_tgt`` layout to a detector index in the ``r_src`` DEM."""
    t, s = divmod(di, d1)
    if t <= r_src:
        return di
    if t == r_tgt:
        return r_src * d1 + s
    return (r_src - 1) * d1 + s


def _map_pair_indices(a: int, b: int, *, r_src: int, r_tgt: int, d1: int) -> tuple[int, int]:
    """Map an undirected detector pair in ``r_tgt`` to a pair of indices in the ``r_src`` DEM."""
    if a > b:
        a, b = b, a
    ta, tb = a // d1, b // d1
    sa, sb = a % d1, b % d1
    if tb not in (ta, ta + 1):
        raise ValueError(f"non-graphlike pair in repetition DEM: D{a} D{b}")
    if tb == ta:
        if tb <= r_src:
            return (a, b)
        return (r_src * d1 + sa, r_src * d1 + sb)
    if tb <= r_src:
        return (a, b)
    if ta >= r_src:
        return ((r_src - 1) * d1 + sa, r_src * d1 + sb)
    return (a, b)


def _index_learned_dem(dem: stim.DetectorErrorModel) -> tuple[dict, dict, dict]:
    """Index probabilities from a compact DEM (no explicit detector lines)."""
    pair: dict[tuple[int, int], float] = {}
    single: dict[int, float] = {}
    det_obs: dict[tuple[int, int], float] = {}  # (detector_id, logical_obs_id)

    for i in range(dem.num_errors):
        inst = dem[i]
        p = float(inst.args_copy()[0])
        targets = inst.targets_copy()
        dets = [t.val for t in targets if t.is_relative_detector_id()]
        obs_ids = [t.val for t in targets if t.is_logical_observable_id()]
        if len(dets) == 2 and not obs_ids:
            a, b = sorted(dets)
            pair[(a, b)] = p
        elif len(dets) == 1 and not obs_ids:
            single[dets[0]] = p
        elif len(dets) == 1 and len(obs_ids) == 1:
            det_obs[(dets[0], obs_ids[0])] = p
        else:
            raise ValueError(f"unsupported DEM error instruction: {inst}")
    return pair, single, det_obs


def broadcast_repetition_dem_learned(
    error_rates: np.ndarray,
    *,
    distance: int,
    r_src: int,
    r_tgt: int,
    circuit_type: str = "phase_flip",
    reset: bool = False,
) -> stim.DetectorErrorModel:
    """
    Return a DEM for ``r_tgt`` rounds whose **structure** matches
    ``build_stim_circuit(..., cycle=r_tgt, ...)`` but error probabilities are taken from
    the learned ``r_src`` DEM using the repetition tiling rules above.

    Currently implemented for ``circuit_type == "phase_flip"`` and ``reset is False``.
    """
    if circuit_type.replace("_", " ") != "phase flip":
        raise NotImplementedError("only phase_flip is implemented for repetition broadcast")
    if reset:
        raise NotImplementedError("only reset=False is implemented for repetition broadcast")

    d1 = distance - 1
    dem_src = _lab_circuit(distance=distance, rounds=r_src, circuit_type="phase_flip", reset=False).detector_error_model()
    if error_rates.shape[0] != dem_src.num_errors:
        raise ValueError(
            f"error_rates length {error_rates.shape[0]} != r={r_src} DEM num_errors {dem_src.num_errors}"
        )
    dem_src = update_dem(dem_src, error_rates)
    pair, single, det_obs = _index_learned_dem(dem_src)

    dem_tgt = _lab_circuit(distance=distance, rounds=r_tgt, circuit_type="phase_flip", reset=False).detector_error_model()
    if dem_tgt.num_errors != _expected_num_errors_phase_flip_reset0(distance=distance, rounds=r_tgt):
        raise RuntimeError(
            "unexpected num_errors for native r_tgt DEM; update _expected_num_errors_phase_flip_reset0"
        )

    out = stim.DetectorErrorModel()
    for i in range(dem_tgt.num_errors):
        inst = dem_tgt[i]
        targets = inst.targets_copy()
        dets = [t.val for t in targets if t.is_relative_detector_id()]
        obs_ids = [t.val for t in targets if t.is_logical_observable_id()]

        if len(dets) == 2 and not obs_ids:
            a, b = dets
            ka, kb = _map_pair_indices(a, b, r_src=r_src, r_tgt=r_tgt, d1=d1)
            if ka > kb:
                ka, kb = kb, ka
            p = pair[(ka, kb)]
        elif len(dets) == 1 and not obs_ids:
            di = dets[0]
            k = _map_detector_index(di, r_src=r_src, r_tgt=r_tgt, d1=d1)
            p = single[k]
        elif len(dets) == 1 and len(obs_ids) == 1:
            di = dets[0]
            k = _map_detector_index(di, r_src=r_src, r_tgt=r_tgt, d1=d1)
            p = det_obs[(k, obs_ids[0])]
        else:
            raise ValueError(f"unsupported target DEM error instruction: {inst}")

        out.append(stim.DemInstruction("error", [p], targets))

    assert_repetition_dem_invariants(out, distance=distance, rounds=r_tgt)
    if out.num_errors != dem_tgt.num_errors:
        raise RuntimeError("broadcast output num_errors mismatch (should not happen)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Broadcast learned r=4 repetition DEM (lab) to larger rounds.")
    ap.add_argument(
        "--pt",
        type=Path,
        default=REPO_ROOT / "data" / "logical_qubit" / "dem" / "dmle_r04_phase_flip_reset0.pt",
        help="Torch file with key 'error_rates' (same ordering as r=4 lab DEM errors).",
    )
    ap.add_argument("--distance", type=int, default=9, help="repetition-code distance (default 9).")
    ap.add_argument("--source-round", type=int, default=4, help="round count of the source DEM / .pt (default 4).")
    ap.add_argument(
        "--target-rounds",
        type=int,
        nargs="+",
        default=[7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40],
        help="Round counts to broadcast to (each saved as its own .txt).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "logical_qubit" / "dem" / "phase_flip_reset0",
        help="Output directory (same naming style as training logs).",
    )
    ap.add_argument("--circuit-type", type=str, default="phase_flip", choices=["phase_flip"])
    ap.add_argument("--reset", type=int, default=0, choices=[0])
    args = ap.parse_args()

    pt_path = args.pt.resolve()
    if not pt_path.is_file():
        raise FileNotFoundError(pt_path)

    loaded = torch.load(pt_path, map_location="cpu")
    if "error_rates" not in loaded:
        raise KeyError(f"{pt_path} must contain 'error_rates' tensor.")
    er = np.asarray(loaded["error_rates"], dtype=np.float64).reshape(-1)

    d = int(args.distance)
    r0 = int(args.source_round)
    circuit_type = args.circuit_type
    reset = bool(args.reset)

    print(
        f"[info] Repetition-code DEM from logical.repetition_code_exp_data.circuits.build_stim_circuit "
        f"(same as rep_pl_opt / data_prcess): d={d}, r_src={r0}, {circuit_type}, reset={int(reset)}. "
        f"Using repetition-specific broadcast (not src.utils.broadcast_dem).",
        flush=True,
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for r1 in args.target_rounds:
        r1 = int(r1)
        if r1 <= r0:
            print(f"[skip] target round {r1} <= source {r0}", flush=True)
            continue
        out_dem = broadcast_repetition_dem_learned(er, distance=d, r_src=r0, r_tgt=r1, circuit_type=circuit_type, reset=reset)
        stem = f"dmle_r{r1:02d}_{circuit_type}_reset{int(reset)}"
        out_path = out_dir / f"{stem}.txt"
        out_path.write_text(str(out_dem), encoding="utf-8")
        print(f"[write] {out_path}  (num_errors={out_dem.num_errors})", flush=True)


if __name__ == "__main__":
    main()
