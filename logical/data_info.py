"""
Scan repetition-code HDF5 raw datasets: unique distances, cycles, file counts, row counts.

Lightweight: reads DataVault attrs + ``f0`` shape only (does not load full measurement tables).

For each HDF5 file, after the same ``shape`` convention as ``process_data`` (grid
``[n_rows // stats, stats]``), each slice ``state[i]`` passed to
``measurements_to_detection_and_observables`` has ``shape[0] == stats``, i.e.
``detection_events.shape[0] == stats`` for that slice. Total experimental rows
per file is ``n_rows == f0.shape[0]`` (all ini-state slices combined).

Usage (from repo root)::

    conda run -n qec python logical/data_info.py

Environment:

- ``REPETITION_CODE_RAW_DIR`` — directory of ``*.hdf5`` files (default: same as ``process_data``).
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import h5py

from logical.process_data import default_raw_data_dir, labrad_urldecode


def _parameters_from_hdf5(hdf5_file: h5py.File) -> Dict[str, Any]:
    datavault = hdf5_file["DataVault"]
    attrs = dict(datavault.attrs)
    if "parameters" in hdf5_file:
        keys = hdf5_file["parameters"]["keys"]
        values = hdf5_file["parameters"]["values"]
        parameters_from_dataset = [
            (k.decode("utf-8"), labrad_urldecode(v.decode("utf-8")))
            for k, v in zip(keys, values)
        ]
    else:
        parameters_from_dataset = []

    parameters: List[Tuple[str, Any]] = []
    for key, value in attrs.items():
        if key.startswith("Param."):
            parameters.append((key[len("Param.") :], labrad_urldecode(value)))
    parameters.extend(parameters_from_dataset)
    return dict(parameters)


def peek_dataset_meta(path: Path) -> Tuple[int, int, int, int, int, str]:
    """
    Returns
    -------
    distance
        ``round((len(qubits) + 1) / 2)``, same convention as ``data_prcess._logical_error``.
    cycle
        From dataset parameters.
    n_rows
        ``DataVault`` row count (``f0.shape[0]``).
    stats
        Parameter ``stats``; equals ``detection_events.shape[0]`` for one ini-state
        slice (same as ``state[i]`` first dimension in ``detection_event_fraction``).
    n_ini_slices
        ``n_rows // stats`` (number of ini-state / grid slices in the file).
    title
        Dataset title string.

    Raises
    ------
    ValueError
        If ``n_rows`` is not divisible by ``stats`` (unexpected layout).
    """
    with h5py.File(path, "r", swmr=True) as f:
        dv = f["DataVault"]
        title = str(dv.attrs.get("Title", ""))
        p = _parameters_from_hdf5(f)
        cycle = int(p["cycle"])
        qubits = p["qubits"]
        n_q = len(qubits)
        distance = round((n_q + 1) / 2)
        n_rows = int(dv["f0"].shape[0])
        stats = int(p["stats"])
        if stats <= 0:
            raise ValueError(f"invalid stats={stats}")
        if n_rows % stats != 0:
            raise ValueError(f"n_rows={n_rows} not divisible by stats={stats}")
        n_ini_slices = n_rows // stats
    return distance, cycle, n_rows, stats, n_ini_slices, title


def main() -> None:
    raw_dir = Path(os.environ.get("REPETITION_CODE_RAW_DIR", str(default_raw_data_dir())))
    files = sorted(raw_dir.glob("*.hdf5"))
    print(f"RAW dir: {raw_dir}")
    print(f"HDF5 files: {len(files)}")

    distances: List[int] = []
    cycles: List[int] = []
    rows_per_file: List[int] = []
    stats_per_file: List[int] = []
    ini_slices_per_file: List[int] = []
    errors: List[Tuple[str, str]] = []

    for fp in files:
        try:
            d, c, n, stats, n_ini, _title = peek_dataset_meta(fp)
            distances.append(d)
            cycles.append(c)
            rows_per_file.append(n)
            stats_per_file.append(stats)
            ini_slices_per_file.append(n_ini)
        except Exception as e:
            errors.append((fp.name, repr(e)))

    print()
    print("Unique distances:", sorted(set(distances)))
    print("Distance histogram:", dict(sorted(Counter(distances).items())))
    print()
    print("Unique cycles:", sorted(set(cycles)))
    print("Cycle histogram:", dict(sorted(Counter(cycles).items())))
    print()
    print("Successfully read files:", len(rows_per_file))
    print("Total rows (sum of DataVault lengths):", sum(rows_per_file))
    print()
    print(
        "Samples per ini-state slice (== parameter 'stats' == "
        "detection_events.shape[0] for one convert):"
    )
    print("  Unique values:", sorted(set(stats_per_file)))
    print("  Histogram:", dict(sorted(Counter(stats_per_file).items())))
    print()
    print(
        "Ini-state / grid slices per file (n_rows // stats, "
        "len(ini_state_idx) in detection_event_fraction):"
    )
    print("  Unique values:", sorted(set(ini_slices_per_file)))
    print("  Histogram:", dict(sorted(Counter(ini_slices_per_file).items())))
    if errors:
        print()
        print("Failed files:", len(errors))
        for name, err in errors[:20]:
            print(" ", name, err)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


if __name__ == "__main__":
    main()
