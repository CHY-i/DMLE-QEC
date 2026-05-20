"""
Load repetition-code HDF5 data and convert to Stim detection_events + observables.

Run from repository root with the ``qec`` environment::

    conda run -n qec python logical/test_repetition_decode_data.py

Optional environment variables:

- ``REPETITION_CODE_RAW_DIR`` — directory of ``*.hdf5`` files (default: public dataset path).
- ``REPETITION_CODE_TEST_DATASET`` — integer id in filename (default: 3378, from upstream notebook).
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from pathlib import Path

from logical.process_data import DataLab, measurements_to_detection_and_observables


def main() -> None:
    raw_dir = Path(
        os.environ.get(
            "REPETITION_CODE_RAW_DIR",
            "/data/public/logical/repetition_code_exp_data-main/data/raw",
        )
    )
    dataset = int(os.environ.get("REPETITION_CODE_TEST_DATASET", "3378"))

    dl = DataLab(data_dir=raw_dir)
    dl.load_dataset(dataset, noisy=True)

    if "shape" not in dl.parameters:
        dl.parameters["shape"] = [
            round(dl.data.shape[0] / dl.parameters["stats"]),
            dl.parameters["stats"],
        ]

    cycle = dl.parameters["cycle"]
    circuit_type = dl.parameters["circuit_type"]
    reset = dl.parameters["reset_after_measure"]
    ini_state_all = dl.parameters["ini_state"]

    ini_state_idx, _, state = dl.get_data(dep_index=np.arange(dl.data.shape[1] - 2))

    dep_idx = 0
    idx = int(round(ini_state_idx[dep_idx]))
    measurements = state[dep_idx]
    ini_state = ini_state_all[idx]

    qubits = np.arange(len(dl.parameters["qubits"]))

    detection_events, observable_flips, circuit = measurements_to_detection_and_observables(
        measurements,
        qubits=qubits,
        ini_state=ini_state,
        cycle=cycle,
        circuit_type=circuit_type,
        reset=reset,
        add_noise=False,
    )

    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)

    print("dataset", dataset)
    print("circuit_type", circuit_type, "cycle", cycle, "reset", reset)
    print("measurements shape", measurements.shape)
    print("detection_events shape", detection_events.shape, "dtype", detection_events.dtype)
    print("dem.num_detectors", dem.num_detectors, "match", dem.num_detectors == detection_events.shape[1])
    print("observable_flips shape", observable_flips.shape)
    print("first shot first 24 detectors:", np.asarray(detection_events[0, :24]).astype(int).tolist())
    print("first 12 logical obs bits:", np.asarray(observable_flips[:12]).astype(int).tolist())


if __name__ == "__main__":
    main()
