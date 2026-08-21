from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import stim


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def setup_logger(log_path: Path, name: str = "patchdmle_nn") -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def round_tag_to_int(round_tag: str) -> int:
    text = str(round_tag)
    if text.startswith("r"):
        return int(text[1:])
    return int(text)


def infer_epoch_from_path(path: str | Path) -> int | None:
    m = re.search(r"(?:^|[_-])epoch[_-]?(\d+)(?:[_\.-]|$)", Path(path).name)
    return int(m.group(1)) if m else None


def _detector_coordinates_by_id(dem: stim.DetectorErrorModel) -> dict[int, tuple[float, ...]]:
    coords = dem.get_detector_coordinates()
    return {int(k): _canonical_detector_xyz(v) for k, v in coords.items()}


def _coord_triples(coord) -> list[list[float]]:
    values = [float(x) for x in coord]
    if len(values) < 3:
        return []
    end = len(values) - (len(values) % 3)
    return [values[i : i + 3] for i in range(0, end, 3)]


def _canonical_detector_xyz(coord) -> tuple[float, float, float]:
    triples = _coord_triples(coord)
    if not triples:
        raise ValueError(f"Detector coordinate has no complete triples: {coord}")
    first_x, first_y, first_t = triples[0]
    last_x, last_y, _last_t = triples[-1]
    return (
        float(round(last_x)),
        float(round(last_y)),
        float(round(first_t)),
    )


def surface_layout_from_dem(dem: stim.DetectorErrorModel):
    """Build a dense time x spatial-slot layout from DEM detector coordinates.

    The layout is intentionally surface-code specific. It supports Google-style
    memory DEMs with half detector layers at the initial/final boundaries and
    full detector layers in the bulk.
    """
    detector_coords = _detector_coordinates_by_id(dem)
    if len(detector_coords) != dem.num_detectors:
        raise ValueError(
            f"DEM coordinate count {len(detector_coords)} != num_detectors {dem.num_detectors}."
        )

    det_ids = np.arange(dem.num_detectors, dtype=np.int64)
    times = np.array([int(round(detector_coords[int(i)][2])) for i in det_ids], dtype=np.int64)
    unique_times = np.array(sorted(set(times.tolist())), dtype=np.int64)
    time_to_index = {int(t): i for i, t in enumerate(unique_times)}
    cycles = len(unique_times)

    ids_by_time = {int(t): det_ids[times == t] for t in unique_times}
    counts = {t: len(ids) for t, ids in ids_by_time.items()}
    all_xy = np.array(
        [
            [detector_coords[int(i)][0], detector_coords[int(i)][1]]
            for i in det_ids
        ],
        dtype=np.float32,
    )
    slot_xy = np.unique(all_xy, axis=0)
    # Preserve the y-major/x-minor ordering used by the original d5 training.
    # This keeps existing pretrained checkpoints valid while remaining
    # independent of code distance and native DEM detector order.
    slot_xy = slot_xy[np.lexsort((slot_xy[:, 0], slot_xy[:, 1]))]
    num_slots = len(slot_xy)
    max_layer_size = max(counts.values())
    if num_slots != max_layer_size:
        raise ValueError(
            f"Found {num_slots} unique spatial slots but the largest detector "
            f"layer has {max_layer_size} detectors."
        )
    distance = math.isqrt(num_slots + 1)
    if distance * distance - 1 != num_slots:
        distance = None
    layer_counts = np.array(
        [counts[int(t)] for t in unique_times],
        dtype=np.int64,
    )

    mins = slot_xy.min(axis=0)
    spans = slot_xy.max(axis=0) - mins
    norm_coords = np.zeros_like(slot_xy, dtype=np.float32)
    for axis in range(2):
        if spans[axis] > 0:
            norm_coords[:, axis] = (slot_xy[:, axis] - mins[axis]) / spans[axis] - 0.5

    xy_to_slot = {
        (float(x), float(y)): int(slot)
        for slot, (x, y) in enumerate(slot_xy)
    }

    time_indices = np.empty(dem.num_detectors, dtype=np.int64)
    slot_indices = np.empty(dem.num_detectors, dtype=np.int64)
    for det_id in det_ids:
        coord = detector_coords[int(det_id)]
        xy = (float(coord[0]), float(coord[1]))
        time_indices[int(det_id)] = time_to_index[int(round(coord[2]))]
        if xy in xy_to_slot:
            slot_indices[int(det_id)] = xy_to_slot[xy]
        else:
            diff = slot_xy - np.asarray(xy, dtype=np.float32)
            slot_indices[int(det_id)] = int(np.argmin(np.sum(diff * diff, axis=1)))

    layout_pairs = list(zip(time_indices.tolist(), slot_indices.tolist()))
    if len(set(layout_pairs)) != dem.num_detectors:
        duplicates = sorted(
            pair for pair in set(layout_pairs) if layout_pairs.count(pair) > 1
        )
        raise ValueError(
            "Detector-to-layout mapping has collisions; multiple detectors would "
            f"overwrite the same time/slot entries: {duplicates[:10]}"
        )

    first_time = int(unique_times[0])
    first_slots = set(slot_indices[ids_by_time[first_time]].tolist())
    det_types = np.ones(num_slots, dtype=np.int64)
    for slot in first_slots:
        det_types[slot] = 0

    return {
        "coords": norm_coords.astype(np.float32),
        "det_types": det_types.astype(np.int64),
        "time_indices": time_indices,
        "slot_indices": slot_indices,
        "cycles": cycles,
        "num_slots": num_slots,
        "distance": distance,
        "layer_counts": layer_counts,
        "unique_times": unique_times,
    }


def point_cloud_from_dem(dets: np.ndarray, dem: stim.DetectorErrorModel, layout: dict | None = None):
    if layout is None:
        layout = surface_layout_from_dem(dem)
    dets = np.asarray(dets)
    if dets.ndim != 2:
        raise ValueError(f"dets must have shape [shots, detectors], got {dets.shape}")
    if dets.shape[1] != len(layout["time_indices"]):
        raise ValueError(
            f"dets has {dets.shape[1]} columns but layout expects {len(layout['time_indices'])}."
        )
    point_cloud = np.zeros(
        (dets.shape[0], int(layout["cycles"]), int(layout["num_slots"])),
        dtype=dets.dtype,
    )
    point_cloud[:, layout["time_indices"], layout["slot_indices"]] = dets
    return point_cloud


def sample_dem_batch(sampler, shots: int):
    sample = sampler.sample(shots=shots)
    if len(sample) == 2:
        dets, obs = sample
    else:
        dets, obs = sample[0], sample[1]
    return np.asarray(dets, dtype=np.int8), np.asarray(obs, dtype=np.int8).reshape(-1)


def read_experimental_round(root: str | Path, dem: stim.DetectorErrorModel):
    root = Path(root)
    det_path = root / "detection_events.b8"
    obs_path = root / "obs_flips_actual.b8"
    if not det_path.exists() or not obs_path.exists():
        raise FileNotFoundError(f"Expected detection_events.b8 and obs_flips_actual.b8 under {root}")
    dets = stim.read_shot_data_file(
        path=str(det_path),
        format="b8",
        num_detectors=dem.num_detectors,
        bit_packed=False,
    ).astype(np.int8)
    obs = stim.read_shot_data_file(
        path=str(obs_path),
        format="b8",
        num_detectors=1,
        bit_packed=False,
    ).reshape(-1).astype(np.int8)
    if len(dets) != len(obs):
        raise ValueError(
            f"Experimental detector shots ({len(dets)}) != observable shots ({len(obs)})."
        )
    return dets, obs


def save_json(path: str | Path, data: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
