"""
Repetition-code experiment I/O.

The upstream ``repetition_code_exp_data-main/src`` tree is vendored under
:mod:`logical.repetition_code_exp_data`.  This module re-exports that API and
adds DMLE-specific helpers (``.npy`` export, :func:`measurements_to_detection_and_observables`).
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import stim
import tqdm
from numpy.typing import NDArray

from logical.repetition_code_exp_data.circuits import build_stim_circuit
from logical.repetition_code_exp_data.data_prcess import (
    convert_ini_state_str,
    detection_event_fraction,
    fit_cycle_raise,
    fit_Lambda,
    subsampling_logical_error,
    subsampling_logical_error_per_cycle,
)
from logical.repetition_code_exp_data.raw_data_reader import (
    DataLab,
    GridData,
    default_raw_data_dir,
    filename_decode,
    labrad_urldecode,
    load_hdf5,
)

__all__ = [
    "DataLab",
    "GridData",
    "build_stim_circuit",
    "default_raw_data_dir",
    "filename_decode",
    "labrad_urldecode",
    "load_hdf5",
    "measurements_to_detection_and_observables",
    "export_repetition_hdf5_to_decoding_npy",
    "subsampling_logical_error",
    "subsampling_logical_error_per_cycle",
    "detection_event_fraction",
    "fit_cycle_raise",
    "fit_Lambda",
    "convert_ini_state_str",
    "load_dets_obs_for_cycle_from_raw",
]


def measurements_to_detection_and_observables(
    measurements: NDArray,
    *,
    qubits: Iterable[int],
    ini_state: Iterable[int],
    cycle: int,
    circuit_type: str,
    reset: bool,
    add_noise: bool = True,
    **kwargs,
) -> Tuple[NDArray, NDArray, stim.Circuit]:
    """Convert raw measurement bits to Stim detection events and logical observable flips."""
    circuit = build_stim_circuit(
        qubits=qubits,
        ini_state=ini_state,
        cycle=cycle,
        circuit_type=circuit_type,
        reset=reset,
        add_noise=add_noise,
        **kwargs,
    )
    converter = circuit.compile_m2d_converter()
    detection_events, observable_flips = converter.convert(
        measurements=np.asarray(measurements, dtype=np.bool_),
        separate_observables=True,
    )
    return detection_events, observable_flips, circuit


def _dataset_number_from_hdf5_path(filepath: Path) -> int:
    stem = filepath.stem
    num_part = ""
    for char in stem:
        if char.isdigit():
            num_part += char
        else:
            break
    if not num_part:
        raise ValueError(f"No dataset number in filename: {filepath}")
    return int(num_part)


def _collect_dets_obs_from_hdf5(
    filepath: Path, data_dir: Path
) -> Tuple[int, str, bool, np.ndarray, np.ndarray]:
    dl = DataLab(data_dir=data_dir)
    dl.load_dataset(_dataset_number_from_hdf5_path(filepath), noisy=False)
    if "shape" not in dl.parameters:
        dl.parameters["shape"] = [
            round(dl.data.shape[0] / dl.parameters["stats"]),
            dl.parameters["stats"],
        ]
    cycle = int(dl.parameters["cycle"])
    circuit_type = str(dl.parameters["circuit_type"])
    reset = bool(dl.parameters["reset_after_measure"])
    ini_state_idx, _, state = dl.get_data(dep_index=np.arange(dl.data.shape[1] - 2))
    qubits = np.arange(len(dl.parameters["qubits"]))
    ini_states_param = dl.parameters["ini_state"]

    dets_parts: List[np.ndarray] = []
    obs_parts: List[np.ndarray] = []
    for i in range(len(ini_state_idx)):
        idx = int(round(ini_state_idx[i]))
        ini_state = ini_states_param[idx]
        measurements = state[i]
        dets, obs, _ = measurements_to_detection_and_observables(
            measurements,
            qubits=qubits,
            ini_state=ini_state,
            cycle=cycle,
            circuit_type=circuit_type,
            reset=reset,
            add_noise=False,
        )
        dets_parts.append(np.asarray(dets, dtype=np.uint8))
        obs_parts.append(np.asarray(obs, dtype=np.uint8).reshape(-1))

    dets_all = np.vstack(dets_parts)
    obs_all = np.concatenate(obs_parts)
    return cycle, circuit_type, reset, dets_all, obs_all


def load_dets_obs_for_cycle_from_raw(
    *,
    cycle: int,
    circuit_type: str,
    reset: int,
    raw_dir: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load experimental ``(dets, obs)`` from raw HDF5, using the **same** conversion as
    :func:`export_repetition_hdf5_to_decoding_npy` and the PyMatching / ``MWPM_dem``
    subsampling pipeline: :func:`build_stim_circuit` + :func:`measurements_to_detection_and_observables`
    with ``add_noise=False`` for each ini-state slice, then merge all HDF5 files that share
    the same ``(cycle, circuit_type, reset)``.

    Parameters
    ----------
    cycle
        Experiment ``cycle`` (same as Stim ``round`` / CLI ``--round``).
    circuit_type
        ``\"phase_flip\"`` or ``\"phase flip\"`` (normalized internally).
    reset
        ``0`` / ``1`` or bool; compared to HDF5 ``reset_after_measure``.
    raw_dir
        Directory of ``*.hdf5``. Default: :func:`default_raw_data_dir` / ``REPETITION_CODE_RAW_DIR``.
    """
    raw = Path(raw_dir) if raw_dir is not None else default_raw_data_dir()
    ct_target = circuit_type.replace("_", " ")
    reset_b = bool(reset)

    parts: List[Tuple[np.ndarray, np.ndarray]] = []
    for fp in sorted(raw.glob("*.hdf5")):
        try:
            c, ct, r, dets, obs = _collect_dets_obs_from_hdf5(fp, raw)
        except Exception:
            continue
        if int(c) != int(cycle):
            continue
        if str(ct) != ct_target:
            continue
        if bool(r) != reset_b:
            continue
        parts.append((dets, obs))

    if not parts:
        raise FileNotFoundError(
            f"No HDF5 under {raw} matched cycle={cycle}, circuit_type={ct_target!r}, reset={reset_b}. "
            "Set REPETITION_CODE_RAW_DIR or pass raw_dir=..."
        )

    dets_merged = np.vstack([p[0] for p in parts])
    obs_merged = np.concatenate([p[1] for p in parts])
    return dets_merged, obs_merged


def export_repetition_hdf5_to_decoding_npy(
    raw_dir: Optional[Path] = None,
    out_root: Optional[Path] = None,
) -> None:
    raw = Path(raw_dir) if raw_dir is not None else default_raw_data_dir()
    repo_root = Path(__file__).resolve().parent.parent
    out = Path(out_root) if out_root is not None else repo_root / "data" / "repetition_decoding"
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(raw.glob("*.hdf5"))
    buckets: Dict[Tuple[int, str, bool], List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    errors: List[Tuple[str, str]] = []

    for fp in tqdm.tqdm(files, desc="HDF5 → dets/obs"):
        try:
            c, ct, r, dets, obs = _collect_dets_obs_from_hdf5(fp, raw)
            buckets[(c, ct, r)].append((dets, obs))
        except Exception as e:
            errors.append((fp.name, repr(e)))

    for (c, ct, r), parts in buckets.items():
        ct_slug = ct.replace(" ", "_")
        name = f"r{c:02d}_{ct_slug}_reset{int(r)}"
        dets_merged = np.vstack([p[0] for p in parts])
        obs_merged = np.concatenate([p[1] for p in parts])
        np.save(out / f"{name}_dets.npy", dets_merged)
        np.save(out / f"{name}_obs.npy", obs_merged)
        print(f"Wrote {name}: dets {dets_merged.shape}, obs {obs_merged.shape} ({len(parts)} files)")

    if errors:
        print(f"Errors ({len(errors)}):")
        for fn, err in errors[:30]:
            print(f"  {fn}: {err}")
        if len(errors) > 30:
            print(f"  ... {len(errors) - 30} more")


if __name__ == "__main__":
    export_repetition_hdf5_to_decoding_npy()
