"""Prepare a short Google dataset whose DEM is a canonical broadcast source.

This differs from a simple detector-layer crop.  For a d7 r10 memory DEM with
Google-style broadcast structure, the one-round bulk template used by
``broadcast_dem`` is selected by ``max_t == bulk_layer``.  Therefore the short
source DEM keeps:

  - base mechanisms with max_t <= bulk_layer
  - tail mechanisms with max_t >= source_rounds - 1

The tail mechanisms may reference the last bulk layer before the tail; those
detectors are mapped onto the short source bulk layer.  This matches the overlap
rule used by DMLE-QEC/src/utils.py::broadcast_dem.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import stim

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
DEFAULT_WILLOW_ROOT = Path(
    os.environ.get("PATCHDMLE_WILLOW_DATA", Path(REPO_ROOT) / "dataset/willow_surface")
) / "d7_at_q6_7"
from src import (
    build_broadcast_source_coordinates,
    crop_detection_events,
    detector_layer_counts,
    extract_broadcast_source_dem,
)


def _round_count(round_tag):
    text = str(round_tag)
    if text.startswith("r"):
        return int(text[1:])
    return int(text)


def _detector_time(detector_id: int, rounds: int) -> int:
    if detector_id < 24:
        return 0
    middle_end = 24 + 48 * (rounds - 1)
    if detector_id < middle_end:
        return 1 + (detector_id - 24) // 48
    return rounds


def _default_output_root(bulk_layer):
    return f"{REPO_ROOT}/dataset/google_broadcast_source/d7_at_q6_7_bulk{int(bulk_layer)}"


def _read_dem(source_root: Path, dem_path: str | None):
    if dem_path is not None:
        return stim.DetectorErrorModel.from_file(str(dem_path)), str(dem_path)
    noisy_circuit_path = source_root / "circuit_noisy_si1000.stim"
    noisy_circuit = stim.Circuit.from_file(str(noisy_circuit_path))
    return (
        noisy_circuit.detector_error_model(decompose_errors=False, flatten_loops=True),
        str(noisy_circuit_path) + " decompose_errors=False",
    )


def main(
    data_root=str(DEFAULT_WILLOW_ROOT),
    basis="X",
    source_round_tag="r10",
    target_round_tag="r04",
    bulk_layer=2,
    output_root=None,
    dem_path=None,
):
    basis = basis.upper()
    source_root = Path(data_root) / basis / source_round_tag
    target_root = Path(output_root or _default_output_root(bulk_layer)) / basis / target_round_tag
    target_root.mkdir(parents=True, exist_ok=True)

    ideal_circuit_path = source_root / "circuit_ideal.stim"
    det_path = source_root / "detection_events.b8"
    obs_path = source_root / "obs_flips_actual.b8"
    metadata_path = source_root / "metadata.json"
    required = [ideal_circuit_path, det_path, obs_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source Google files: " + ", ".join(missing))

    source_rounds = _round_count(source_round_tag)
    target_rounds = _round_count(target_round_tag)
    if source_rounds != 10 or target_rounds != 4:
        raise ValueError("This script currently implements the d7 r10 -> d7 r4 broadcast source.")
    if int(bulk_layer) < 2 or int(bulk_layer) > source_rounds - 2:
        raise ValueError("bulk_layer must be in the strict repeated bulk range 2..8 for r10.")

    keep_layers = [0, 1, int(bulk_layer), source_rounds - 1, source_rounds]
    ideal_circuit = stim.Circuit.from_file(str(ideal_circuit_path))
    detector_coordinates = ideal_circuit.get_detector_coordinates()
    source_dem, dem_source = _read_dem(source_root, dem_path)
    source_coords = build_broadcast_source_coordinates(detector_coordinates, keep_layers)
    source_dem_result = extract_broadcast_source_dem(source_dem, source_rounds, int(bulk_layer))
    short_dem = source_dem_result["dem"]

    dets = stim.read_shot_data_file(
        path=str(det_path),
        format="b8",
        num_detectors=source_dem.num_detectors,
        bit_packed=False,
    )
    obs = stim.read_shot_data_file(
        path=str(obs_path),
        format="b8",
        num_detectors=1,
        bit_packed=False,
    )
    cropped_dets = np.asarray(
        crop_detection_events(dets, source_coords["kept_old_ids"]),
        dtype=np.bool_,
    )
    cropped_obs = np.asarray(obs, dtype=np.bool_)

    stim.write_shot_data_file(
        path=str(target_root / "detection_events.b8"),
        data=cropped_dets,
        format="b8",
        num_detectors=cropped_dets.shape[1],
    )
    stim.write_shot_data_file(
        path=str(target_root / "obs_flips_actual.b8"),
        data=cropped_obs,
        format="b8",
        num_observables=1,
    )
    (target_root / "initial_dem_non_decomposed.dem").write_text(str(short_dem))
    with open(target_root / "detector_coordinates.json", "w") as f:
        json.dump(
            {str(k): list(v) for k, v in sorted(source_coords["new_detector_coordinates"].items())},
            f,
            indent=2,
        )

    source_metadata = {}
    if metadata_path.exists():
        source_metadata = json.loads(metadata_path.read_text())

    target_metadata = {
        "basis": basis,
        "source_root": str(source_root),
        "source_round_tag": source_round_tag,
        "target_round_tag": target_round_tag,
        "source_rounds": source_rounds,
        "target_rounds": target_rounds,
        "bulk_layer": int(bulk_layer),
        "keep_layers_for_data": keep_layers,
        "dem_source": dem_source,
        "dem_extraction": "broadcast_source",
        "dem_rule": "keep errors with max_t <= bulk_layer and max_t >= source_rounds - 1",
        "time_map_for_data": source_coords["time_map"],
        "shots": int(cropped_dets.shape[0]),
        "source_detector_count": int(source_dem.num_detectors),
        "target_detector_count": int(short_dem.num_detectors),
        "source_dem_error_count": int(source_dem.num_errors),
        "target_dem_error_count": int(short_dem.num_errors),
        "kept_error_count": int(source_dem_result["kept_error_count"]),
        "skipped_error_count": int(source_dem_result["skipped_error_count"]),
        "kept_errors_by_source_max_t": source_dem_result["kept_by_max_t"],
        "source_layer_counts": detector_layer_counts(detector_coordinates),
        "target_layer_counts": detector_layer_counts(source_coords["new_detector_coordinates"]),
        "source_metadata": source_metadata,
    }
    with open(target_root / "metadata.json", "w") as f:
        json.dump(target_metadata, f, indent=2)

    print("# Google broadcast-source dataset prepared")
    print(f"source_root: {source_root}")
    print(f"target_root: {target_root}")
    print(f"dem_source: {dem_source}")
    print(f"keep_layers_for_data: {keep_layers}")
    print(f"shots: {cropped_dets.shape[0]}")
    print(f"detectors: {source_dem.num_detectors} -> {short_dem.num_detectors}")
    print(f"errors: {source_dem.num_errors} -> {short_dem.num_errors}")
    print(f"skipped_errors: {source_dem_result['skipped_error_count']}")
    print(f"kept_errors_by_source_max_t: {source_dem_result['kept_by_max_t']}")
    print(f"target layer counts: {detector_layer_counts(source_coords['new_detector_coordinates'])}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
