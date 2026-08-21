import numpy as np
import torch
import stim
from collections import Counter


def PCM(dem):

    pcm = np.zeros([dem.num_detectors, dem.num_errors])
    l = np.zeros([dem.num_observables, dem.num_errors])

    error_idx = 0
    for e in dem.flattened():
        if e.type != "error":
            continue
        Dec = e.targets_copy()

        for j in range(len(Dec)):
            D = str(Dec[j])
            if D.startswith('D'):
                idx = int(D[1:])
                pcm[idx, error_idx] = 1.#e.args_copy()[0]

            elif D.startswith('L'):
                idx = int(D[1:])
                l[idx, error_idx] = 1
        error_idx += 1

    if error_idx != dem.num_errors:
        raise ValueError(f"Parsed {error_idx} errors, expected {dem.num_errors}")

    non_zero_rows = np.where(pcm.sum(axis=1) != 0)[0]
    pcm = pcm[non_zero_rows, :]
    return  pcm, l


def update_dem(dem, ers):
    new_dem = stim.DetectorErrorModel()
    error_idx = 0
    for instruction in dem.flattened():
    # print(instruction.type)
        if instruction.type == "error":
            args = instruction.args_copy()
            targets = instruction.targets_copy()
            new_p = float(ers[error_idx])
            new_dem.append(stim.DemInstruction(
                "error",
                args=[new_p],
                targets=targets
            ))
            error_idx += 1
        else:
            new_dem.append(instruction)
    if error_idx != len(ers):
        raise ValueError(f"Updated {error_idx} DEM errors, got {len(ers)} error rates")
    return new_dem


def _coord_triples(coord):
    values = list(coord)
    if len(values) < 3:
        return []
    end = len(values) - (len(values) % 3)
    return [values[i:i + 3] for i in range(0, end, 3)]


def _canonical_detector_xyz(coord):
    """Reduce a detector coordinate list to one canonical (x, y, t).

    For Google-style memory detectors with multiple triples, use:
      - time from the first triple (the current detector layer),
      - spatial position from the last triple.

    This matches the intended simplification used for downstream subsampling.
    """
    triples = _coord_triples(coord)
    if not triples:
        raise ValueError(f"Detector coordinate has no complete triples: {coord}")
    first_x, first_y, first_t = triples[0]
    last_x, last_y, _ = triples[-1]
    return (
        int(round(last_x)),
        int(round(last_y)),
        int(round(first_t)),
    )


def _first_layer_xy_set(detector_coordinates):
    first_layer = set()
    for coord in detector_coordinates.values():
        x, y, t = _canonical_detector_xyz(coord)
        if t == 0:
            first_layer.add((x, y))
    return first_layer


def _reference_first_layer_xy_set(d):
    circuit = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_z",
        distance=int(d),
        rounds=1,
        after_clifford_depolarization=0.001,
        before_measure_flip_probability=0.001,
        after_reset_flip_probability=0.001,
    )
    return _first_layer_xy_set(circuit.get_detector_coordinates())


def _reference_d7r1_first_layer_xy_set():
    return _reference_first_layer_xy_set(7)


def _find_xy_transform_to_reference(detector_coordinates, reference_d=7):
    source = sorted(_first_layer_xy_set(detector_coordinates))
    target = _reference_first_layer_xy_set(reference_d)
    if len(source) != len(target) or not source:
        return lambda x, y: (int(round(x)), int(round(y)))

    if set(source) == target:
        return lambda x, y: (int(round(x)), int(round(y)))

    for a in range(-2, 3):
        for b in range(-2, 3):
            for c in range(-2, 3):
                for d in range(-2, 3):
                    if a * d - b * c == 0:
                        continue
                    transformed = [(a * x + b * y, c * x + d * y) for x, y in source]
                    for tx, ty in target:
                        ox = tx - transformed[0][0]
                        oy = ty - transformed[0][1]
                        mapped = {(u + ox, v + oy) for u, v in transformed}
                        if mapped == target:
                            return (
                                lambda x, y, a=a, b=b, c=c, d=d, ox=ox, oy=oy:
                                (int(round(a * x + b * y + ox)), int(round(c * x + d * y + oy)))
                            )

    return lambda x, y: (int(round(x)), int(round(y)))


def _matching_detector_rows(detector_coordinates, target_xy, xy_transform):
    rows = []
    for d_idx, coord in detector_coordinates.items():
        x, y, _ = _canonical_detector_xyz(coord)
        if xy_transform(x, y) in target_xy:
            rows.append(d_idx)
    return sorted(set(rows))


class DetectorCoordinateProxy:
    """Minimal circuit-like wrapper exposing detector coordinates."""

    def __init__(self, detector_coordinates):
        self._detector_coordinates = {
            int(k): tuple(float(x) for x in v)
            for k, v in detector_coordinates.items()
        }

    def get_detector_coordinates(self):
        return dict(self._detector_coordinates)


def detector_layer_counts(detector_coordinates):
    counts = {}
    for coord in detector_coordinates.values():
        _, _, t = _canonical_detector_xyz(coord)
        counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items()))


def crop_detector_coordinates(detector_coordinates, keep_layers):
    keep_layers = [int(layer) for layer in keep_layers]
    time_map = {old_t: new_t for new_t, old_t in enumerate(keep_layers)}
    kept_old_ids = []
    new_detector_coordinates = {}
    old_to_new = {}

    for old_id in sorted(detector_coordinates):
        x, y, old_t = _canonical_detector_xyz(detector_coordinates[old_id])
        if old_t not in time_map:
            continue
        new_id = len(kept_old_ids)
        kept_old_ids.append(old_id)
        old_to_new[old_id] = new_id
        new_detector_coordinates[new_id] = (
            float(x),
            float(y),
            float(time_map[old_t]),
        )

    return {
        "keep_layers": keep_layers,
        "time_map": time_map,
        "kept_old_ids": kept_old_ids,
        "old_to_new": old_to_new,
        "new_detector_coordinates": new_detector_coordinates,
    }


def crop_detection_events(detection_events, kept_old_ids):
    kept_old_ids = np.asarray(kept_old_ids, dtype=np.int64)
    if isinstance(detection_events, torch.Tensor):
        return detection_events.index_select(
            dim=1, index=torch.as_tensor(kept_old_ids, device=detection_events.device)
        )
    return detection_events[:, kept_old_ids]


def crop_dem_by_detector_coordinates(dem, detector_coordinates, keep_layers):
    crop = crop_detector_coordinates(detector_coordinates, keep_layers)
    keep_set = set(crop["kept_old_ids"])
    old_to_new = crop["old_to_new"]

    new_dem = stim.DetectorErrorModel()
    kept_error_count = 0
    dropped_mixed_error_count = 0
    dropped_outside_error_count = 0

    for inst in dem:
        if inst.type != "error":
            continue

        det_targets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        kept_flags = [det in keep_set for det in det_targets]

        if det_targets:
            if all(kept_flags):
                new_targets = []
                for target in inst.targets_copy():
                    if target.is_relative_detector_id():
                        new_targets.append(
                            stim.DemTarget.relative_detector_id(old_to_new[target.val])
                        )
                    else:
                        new_targets.append(target)
                new_dem.append(
                    stim.DemInstruction("error", inst.args_copy(), new_targets)
                )
                kept_error_count += 1
            elif any(kept_flags):
                dropped_mixed_error_count += 1
            else:
                dropped_outside_error_count += 1
        else:
            new_dem.append(inst)
            kept_error_count += 1

    for new_id in sorted(crop["new_detector_coordinates"]):
        new_dem.append(
            stim.DemInstruction(
                "detector",
                list(crop["new_detector_coordinates"][new_id]),
                [stim.DemTarget.relative_detector_id(new_id)],
            )
        )

    crop.update(
        {
            "dem": new_dem,
            "kept_error_count": kept_error_count,
            "dropped_mixed_error_count": dropped_mixed_error_count,
            "dropped_outside_error_count": dropped_outside_error_count,
        }
    )
    return crop


def parse_dem_coordinates_and_errors(dem):
    detector_coords = {}
    errors = []
    for inst in dem.flattened():
        if inst.type == "detector":
            args = inst.args_copy()
            if len(args) < 3:
                continue
            det_id = None
            for target in inst.targets_copy():
                if target.is_relative_detector_id():
                    det_id = target.val
                    break
            if det_id is not None:
                detector_coords[det_id] = _canonical_detector_xyz(args)
        elif inst.type == "error":
            errors.append(inst)
    return detector_coords, errors


def build_broadcast_source_coordinates(detector_coordinates, keep_layers):
    """Build canonical coordinates for a short broadcast-source dataset.

    This is used for the experimental detection-event columns. It keeps the
    requested old detector layers and rewrites their times to consecutive
    short-source layers.
    """
    return crop_detector_coordinates(detector_coordinates, keep_layers)


def extract_broadcast_source_dem(dem, source_rounds, bulk_layer):
    """Extract a short DEM that is a source for `broadcast_dem`.

    This is not a simple detector-layer projection. For Google d7 memory DEMs
    with one-round middle bulk repetition, the short source keeps:

    - base/pattern mechanisms with max detector time <= `bulk_layer`
    - tail mechanisms with max detector time >= `source_rounds - 1`

    Tail mechanisms can reference the last bulk layer before the tail. Those
    references are folded onto the short source bulk layer, matching the overlap
    convention used by DMLE-QEC's forward `broadcast_dem`.
    """
    detector_coords, errors = parse_dem_coordinates_and_errors(dem)
    source_tail = int(source_rounds) - 1
    source_final = int(source_rounds)
    source_last_bulk = int(source_rounds) - 2
    bulk_layer = int(bulk_layer)

    short_layer_map = {
        0: 0,
        1: 1,
        bulk_layer: 2,
        source_last_bulk: 2,
        source_tail: 3,
        source_final: 4,
    }

    new_detectors = []
    coord_to_new_id = {}

    def add_detector(x, y, t):
        key = (int(x), int(y), int(t))
        if key in coord_to_new_id:
            return coord_to_new_id[key]
        new_id = len(new_detectors)
        coord_to_new_id[key] = new_id
        new_detectors.append((key[0], key[1], key[2], new_id))
        return new_id

    for _old_id, (x, y, old_t) in sorted(detector_coords.items()):
        if old_t in (0, 1, bulk_layer, source_tail, source_final):
            add_detector(x, y, short_layer_map[old_t])

    def map_detector(old_id):
        x, y, old_t = detector_coords[old_id]
        if old_t not in short_layer_map:
            return None
        return coord_to_new_id.get((x, y, short_layer_map[old_t]))

    out = stim.DetectorErrorModel()
    kept_error_count = 0
    skipped_error_count = 0
    kept_by_max_t = Counter()

    for inst in errors:
        det_targets = [
            t.val for t in inst.targets_copy() if t.is_relative_detector_id()
        ]
        if not det_targets:
            out.append(inst)
            kept_error_count += 1
            continue

        times = [detector_coords[det][2] for det in det_targets]
        max_t = max(times)
        is_base = max_t <= bulk_layer
        is_tail = max_t >= source_tail
        if not (is_base or is_tail):
            skipped_error_count += 1
            continue

        new_targets = []
        ok = True
        for target in inst.targets_copy():
            if target.is_relative_detector_id():
                new_id = map_detector(target.val)
                if new_id is None:
                    ok = False
                    break
                new_targets.append(stim.DemTarget.relative_detector_id(new_id))
            else:
                new_targets.append(target)

        if not ok:
            skipped_error_count += 1
            continue

        out.append(stim.DemInstruction("error", inst.args_copy(), new_targets))
        kept_error_count += 1
        kept_by_max_t[max_t] += 1

    for x, y, t, new_id in sorted(new_detectors, key=lambda item: item[3]):
        out.append(
            stim.DemInstruction(
                "detector",
                [x, y, t],
                [stim.DemTarget.relative_detector_id(new_id)],
            )
        )

    return {
        "dem": out,
        "kept_error_count": kept_error_count,
        "skipped_error_count": skipped_error_count,
        "kept_by_max_t": dict(sorted(kept_by_max_t.items())),
        "short_layer_map": dict(sorted(short_layer_map.items())),
    }


def broadcast_dem(origin_dem, broadcast_time_layer, repeat_chunk):
    """Broadcast a short DEM by repeating its final bulk chunk.

    This mirrors the broadcast rule used by DMLE-QEC for Google memory
    experiments. For an origin DEM with max detector time `T`, the final two
    layers `T-1,T` are treated as the tail, and the chunk immediately before
    the tail is repeated until the target max time is reached.
    """
    dem_flat = origin_dem.flattened()
    detector_coords = {}
    original_detectors = []
    original_errors = []
    coords_to_old_id = {}
    max_time = 0

    for instruction in dem_flat:
        if instruction.type == "detector":
            args = instruction.args_copy()
            if len(args) < 3:
                continue
            x, y, t = int(args[0]), int(args[1]), int(args[2])
            max_time = max(max_time, t)
            det_id = None
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    det_id = target.val
                    break
            if det_id is None:
                continue
            detector_coords[det_id] = (x, y, t)
            coords_to_old_id[(x, y, t)] = det_id
            original_detectors.append({"x": x, "y": y, "t": t, "id": det_id})
        elif instruction.type == "error":
            original_errors.append(instruction)

    broadcast_time_layer = int(broadcast_time_layer)
    repeat_chunk = int(repeat_chunk)
    boundary_layer = max_time - 1
    pattern_start_layer = boundary_layer - repeat_chunk
    base_limit = boundary_layer - 1
    total_shift = broadcast_time_layer - max_time

    if repeat_chunk <= 0:
        raise ValueError("repeat_chunk must be positive")
    if total_shift <= 0:
        raise ValueError(
            f"Target time {broadcast_time_layer} must be greater than origin max time {max_time}"
        )
    if total_shift % repeat_chunk != 0:
        raise ValueError(
            f"Total shift {total_shift} is not a multiple of repeat_chunk {repeat_chunk}"
        )

    num_repeats = total_shift // repeat_chunk
    sorted_detectors = sorted(original_detectors, key=lambda item: (item["t"], item["id"]))
    new_detectors = []
    id_map = {}

    def add_detector(x, y, t):
        new_id = len(new_detectors)
        new_detectors.append({"x": x, "y": y, "t": t, "id": new_id})
        return new_id

    def add_overlap_mapping(det, copy_key, new_id):
        if det["t"] != pattern_start_layer + repeat_chunk:
            return
        start_key = (det["x"], det["y"], pattern_start_layer)
        if start_key in coords_to_old_id:
            id_map[(coords_to_old_id[start_key], copy_key)] = new_id

    for det in sorted_detectors:
        if det["t"] <= base_limit:
            new_id = add_detector(det["x"], det["y"], det["t"])
            id_map[(det["id"], 0)] = new_id
            add_overlap_mapping(det, 1, new_id)

    for k in range(1, num_repeats + 1):
        for det in sorted_detectors:
            if pattern_start_layer <= det["t"] <= base_limit:
                new_id = add_detector(det["x"], det["y"], det["t"] + k * repeat_chunk)
                id_map[(det["id"], k)] = new_id
                add_overlap_mapping(det, k + 1, new_id)

    for det in sorted_detectors:
        if det["t"] >= boundary_layer:
            new_id = add_detector(det["x"], det["y"], det["t"] + total_shift)
            id_map[(det["id"], "end")] = new_id

    def get_mapped_id(old_id, copy_key):
        if (old_id, copy_key) in id_map:
            return id_map[(old_id, copy_key)]

        x, y, t = detector_coords[old_id]
        equiv_key = (x, y, t + repeat_chunk)
        if equiv_key not in coords_to_old_id:
            return None

        equiv_id = coords_to_old_id[equiv_key]
        if copy_key == "end":
            return get_mapped_id(old_id, num_repeats)
        if isinstance(copy_key, int) and copy_key > 0:
            return get_mapped_id(equiv_id, copy_key - 1)
        return None

    new_errors = []

    def add_error_using_map(targets, copy_key, instruction_args):
        new_targets = []
        has_detector_target = False
        for target in targets:
            if target.is_relative_detector_id():
                current_key = copy_key
                if copy_key == "end" and (target.val, "end") not in id_map:
                    current_key = num_repeats
                new_id = get_mapped_id(target.val, current_key)
                if new_id is None:
                    return
                new_targets.append(stim.DemTarget.relative_detector_id(new_id))
                has_detector_target = True
            else:
                new_targets.append(target)
        if has_detector_target:
            new_errors.append(stim.DemInstruction("error", instruction_args, new_targets))

    for instruction in original_errors:
        targets = instruction.targets_copy()
        current_times = [
            detector_coords[target.val][2]
            for target in targets
            if target.is_relative_detector_id()
        ]
        if not current_times:
            continue
        max_t = max(current_times)

        if max_t <= base_limit:
            add_error_using_map(targets, 0, instruction.args_copy())
        if pattern_start_layer <= max_t <= base_limit:
            for k in range(1, num_repeats + 1):
                add_error_using_map(targets, k, instruction.args_copy())
        if max_t >= boundary_layer:
            add_error_using_map(targets, "end", instruction.args_copy())

    output_dem = stim.DetectorErrorModel()
    for err in new_errors:
        output_dem.append(err)
    for det in sorted(new_detectors, key=lambda item: item["id"]):
        output_dem.append(
            stim.DemInstruction(
                "detector",
                [det["x"], det["y"], det["t"]],
                [stim.DemTarget.relative_detector_id(det["id"])],
            )
        )
    return output_dem


def _full_edge_mask_for_rows(pcm, rows, columns):
    """Return a mask selecting subgraph error columns whose full support is inside rows."""
    if len(columns) == 0:
        return np.zeros(0, dtype=bool)
    rows = np.asarray(rows, dtype=int)
    columns = np.asarray(columns, dtype=int)
    visible_support = np.asarray(pcm[rows, :][:, columns].sum(axis=0)).reshape(-1)
    full_support = np.asarray(pcm[:, columns].sum(axis=0)).reshape(-1)
    return visible_support == full_support


def subsample_d5_pcms(d, r, print_info=False, return_edge_masks=False):
    """Decompose a d=7 surface code into 5 overlapping d=5 sub-codes.

    Uses the d=5 template coordinate matching strategy (same as subsample_d3_pcms):
    - d=5 standalone surface code has 24 spatial coordinates (z=1 layer)
    - template center is at (5, 5)
    - 5 windows (4 corners + 1 center) cover the full d=7 grid
    - each sub-PCM has exactly 120 detectors (= standard d=5 r=5 structure)

    Args:
        d: distance of the original surface code (should be 7)
        r: number of rounds
        print_info: if True, print detailed decomposition info

    Returns:
        tuple: (sub_pcms, sub_dets, sub_errors)
            - sub_pcms: list of 5 PCM sub-matrices
            - sub_dets: list of detector index arrays in original d=7
            - sub_errors: list of error index arrays in original d=7
    """
    if d != 7:
        raise ValueError(f"subsample_d5_pcms only supports d=7, got d={d}")

    # d=5 template coordinates (z=1 layer of standalone d=5 surface code)
    d5_template_coords = np.array([
        [0, 4], [0, 8], [2, 0], [2, 2], [2, 4], [2, 6], [2, 8],
        [4, 2], [4, 4], [4, 6], [4, 8], [4, 10],
        [6, 0], [6, 2], [6, 4], [6, 6], [6, 8],
        [8, 2], [8, 4], [8, 6], [8, 8], [8, 10],
        [10, 2], [10, 6]
    ])
    d5_center = np.array([5, 5])

    # 5 window centers (4 corners + 1 center), spacing = 4
    centers = [(5, 5), (9, 5), (5, 9), (9, 9), (7, 7)]

    # Standard d=5 detector counts per time layer
    # z=0: 12, z=1..r: 24 each, z=r+1: 12 → total = 12 + 24*(r) + 12 = 24*(r+1)? No, for r rounds there are r+1 intervals
    # Actually: z=0 (init, 12), z=1..r (bulk, 24 each), z=r+1 (final, 12) but dem has z=0..r
    # Standalone d=5 r=5: z=0:12, z=1:24, z=2:24, z=3:24, z=4:24, z=5:12 → total 120
    expected_total = 12 + 24 * (r - 1) + 12  # = 12 + 24*(r-1) + 12 = 24*(r-1) + 24

    sc_circuit = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_z",
        distance=d,
        rounds=r,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
    )
    dem = sc_circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    pcm_d, _ = PCM(dem)
    coors_d = sc_circuit.get_detector_coordinates()

    sub_pcms = []
    sub_errors = []
    sub_dets = []
    sub_full_masks = []
    total_edges = set()

    for i, (cx, cy) in enumerate(centers):
        offset = np.array([cx, cy]) - d5_center
        det_coors = d5_template_coords + offset

        rows = []
        for idx in range(len(det_coors)):
            dx, dy = det_coors[idx]
            for d_idx, coord in coors_d.items():
                cx_d, cy_d = int(coord[0]), int(coord[1])
                if cx_d == dx and cy_d == dy:
                    rows.append(d_idx)
        rows = sorted(set(rows))

        non_zero_columns = np.nonzero(pcm_d[rows].sum(0))[0]
        sub_pcm = pcm_d[rows, :][:, non_zero_columns]
        full_mask = _full_edge_mask_for_rows(pcm_d, rows, non_zero_columns)

        sub_dets.append(rows)
        sub_pcms.append(sub_pcm)
        sub_errors.append(non_zero_columns)
        sub_full_masks.append(full_mask)
        total_edges = total_edges | set(non_zero_columns)

        if print_info:
            nte = len(non_zero_columns)
            des = np.count_nonzero((pcm_d[rows].sum(0) % 2))
            print(f'sub[{i}] center=({cx},{cy}): dets={len(rows)}, pcm={sub_pcm.shape}, '
                  f'errs={nte}, dangling={des}, internal={nte-des}')

    if print_info:
        print(f'\nOriginal d={d} PCM shape: {pcm_d.shape}')
        print(f'Total unique errors covered: {len(total_edges)} / {pcm_d.shape[1]}')
        print(f'Uncovered errors: {pcm_d.shape[1] - len(total_edges)}')
        print(f'Number of d=5 sub-codes: {len(sub_pcms)}')

    if return_edge_masks:
        return sub_pcms, sub_dets, sub_errors, sub_full_masks
    return sub_pcms, sub_dets, sub_errors


def _d3_template_coords():
    return np.array([
        [2, 0],
        [2, 2],
        [4, 2],
        [6, 2],
        [0, 4],
        [2, 4],
        [4, 4],
        [4, 6],
    ])


def _d3_window_centers(d):
    if d < 3 or d % 2 == 0:
        raise ValueError(f"d3 subsampling requires an odd distance >= 3, got d={d}")

    layers = (d - 1) // 2
    centers = []
    for layer in range(layers, 0, -1):
        layer_index = layers - layer
        first_center = np.array([3 + 2 * layer_index, 3 + 2 * layer_index])
        for i in range(layer ** 2):
            center = first_center + np.array([4 * (i % layer), 4 * (i // layer)])
            center_tuple = (int(center[0]), int(center[1]))
            if center_tuple not in centers:
                centers.append(center_tuple)
    return centers


def subsample_d3_pcms(d, r, print_info=False, return_edge_masks=False):
    """Decompose a standard simulated surface code into overlapping d=3 sub-codes.

    For d=5 this produces 5 d=3 sub-PCMs: four corner windows plus one center
    window. The return format matches `subsample_d5_pcms`.
    """
    sc_circuit = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_z",
        distance=d,
        rounds=r,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
    )
    dem = sc_circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    return subsample_d3_pcms_from_circuit(
        sc_circuit,
        dem,
        d=d,
        print_info=print_info,
        return_edge_masks=return_edge_masks,
    )


def subsample_d3_pcms_from_circuit(
    circuit,
    dem,
    d,
    print_info=False,
    check_coverage=True,
    return_edge_masks=False,
):
    """Build overlapping d=3 sub-PCMs from a circuit/DEM pair.

    This is the d=5 -> d=3 analogue of `subsample_d5_pcms_from_circuit`.
    It also works for larger odd d, using the same concentric d=3 window
    placement rule inherited from the original DMLE-QEC implementation.
    """
    d3_template_coords = _d3_template_coords()
    d3_center = np.array([3, 3])
    centers = _d3_window_centers(int(d))

    pcm_d, _ = PCM(dem)
    coors_d = circuit.get_detector_coordinates()
    xy_transform = _find_xy_transform_to_reference(coors_d, reference_d=d)

    sub_pcms = []
    sub_errors = []
    sub_dets = []
    sub_full_masks = []
    total_edges = set()

    for i, (cx, cy) in enumerate(centers):
        offset = np.array([cx, cy]) - d3_center
        det_coors = d3_template_coords + offset
        target_xy = {tuple(xy) for xy in det_coors.tolist()}
        rows = _matching_detector_rows(coors_d, target_xy, xy_transform)

        non_zero_columns = np.nonzero(pcm_d[rows].sum(0))[0]
        sub_pcm = pcm_d[rows, :][:, non_zero_columns]
        full_mask = _full_edge_mask_for_rows(pcm_d, rows, non_zero_columns)

        sub_dets.append(rows)
        sub_pcms.append(sub_pcm)
        sub_errors.append(non_zero_columns)
        sub_full_masks.append(full_mask)
        total_edges = total_edges | set(non_zero_columns)

        if print_info:
            nte = len(non_zero_columns)
            des = np.count_nonzero((pcm_d[rows].sum(0) % 2))
            print(f'sub[{i}] center=({cx},{cy}): dets={len(rows)}, pcm={sub_pcm.shape}, '
                  f'errs={nte}, dangling={des}, internal={nte-des}')

    if check_coverage and len(total_edges) != pcm_d.shape[1]:
        raise ValueError(
            f"Subsampled d=3 PCMs cover {len(total_edges)} / {pcm_d.shape[1]} error terms. "
            "Check detector coordinate mapping before training."
        )

    if print_info:
        print(f'\nOriginal PCM shape: {pcm_d.shape}')
        print(f'Original dangling edges: {np.count_nonzero((pcm_d.sum(0) % 2))}')
        print(f'Total unique errors covered: {len(total_edges)} / {pcm_d.shape[1]}')
        print(f'Uncovered errors: {pcm_d.shape[1] - len(total_edges)}')
        print(f'Number of d=3 sub-codes: {len(sub_pcms)}')

    if return_edge_masks:
        return sub_pcms, sub_dets, sub_errors, sub_full_masks
    return sub_pcms, sub_dets, sub_errors


def subsample_d5_pcms_from_circuit(circuit, dem, print_info=False, return_edge_masks=False):
    """Build the same 5 overlapping d=5 sub-PCMs from an arbitrary d=7 circuit/DEM pair."""
    d5_template_coords = np.array([
        [0, 4], [0, 8], [2, 0], [2, 2], [2, 4], [2, 6], [2, 8],
        [4, 2], [4, 4], [4, 6], [4, 8], [4, 10],
        [6, 0], [6, 2], [6, 4], [6, 6], [6, 8],
        [8, 2], [8, 4], [8, 6], [8, 8], [8, 10],
        [10, 2], [10, 6]
    ])
    d5_center = np.array([5, 5])
    centers = [(5, 5), (9, 5), (5, 9), (9, 9), (7, 7)]

    pcm_d, _ = PCM(dem)
    coors_d = circuit.get_detector_coordinates()
    xy_transform = _find_xy_transform_to_reference(coors_d, reference_d=7)

    sub_pcms = []
    sub_errors = []
    sub_dets = []
    sub_full_masks = []
    total_edges = set()

    for i, (cx, cy) in enumerate(centers):
        offset = np.array([cx, cy]) - d5_center
        det_coors = d5_template_coords + offset
        target_xy = {tuple(xy) for xy in det_coors.tolist()}
        rows = _matching_detector_rows(coors_d, target_xy, xy_transform)

        non_zero_columns = np.nonzero(pcm_d[rows].sum(0))[0]
        sub_pcm = pcm_d[rows, :][:, non_zero_columns]
        full_mask = _full_edge_mask_for_rows(pcm_d, rows, non_zero_columns)

        sub_dets.append(rows)
        sub_pcms.append(sub_pcm)
        sub_errors.append(non_zero_columns)
        sub_full_masks.append(full_mask)
        total_edges = total_edges | set(non_zero_columns)

        if print_info:
            nte = len(non_zero_columns)
            des = np.count_nonzero((pcm_d[rows].sum(0) % 2))
            print(f'sub[{i}] center=({cx},{cy}): dets={len(rows)}, pcm={sub_pcm.shape}, '
                  f'errs={nte}, dangling={des}, internal={nte-des}')

    if len(total_edges) != pcm_d.shape[1]:
        raise ValueError(
            f"Subsampled d=5 PCMs cover {len(total_edges)} / {pcm_d.shape[1]} error terms. "
            "Check detector coordinate mapping before training."
        )

    if print_info:
        print(f'\nOriginal PCM shape: {pcm_d.shape}')
        print(f'Total unique errors covered: {len(total_edges)} / {pcm_d.shape[1]}')
        print(f'Uncovered errors: {pcm_d.shape[1] - len(total_edges)}')
        print(f'Number of d=5 sub-codes: {len(sub_pcms)}')

    if return_edge_masks:
        return sub_pcms, sub_dets, sub_errors, sub_full_masks
    return sub_pcms, sub_dets, sub_errors


def get_error_rates(dem):
    er = []
    for instruction in dem.flattened():
        if instruction.type == "error":
            er.append(instruction.args_copy()[0])
    return np.array(er)

def get_weights(dem):
    er = []
    for instruction in dem.flattened():
        if instruction.type == "error":
            er.append(instruction.args_copy()[0])
    return np.array(np.log((1-np.array(er))/np.array(er)))

def contract(tree, tensors: list[torch.Tensor]) -> torch.Tensor:
    """Contract tensors according to the optimized tree."""
    return _contract_recursive(tree, tensors)


def _contract_recursive(tree_dict: dict, tensors: list[torch.Tensor]) -> torch.Tensor:
    # Check for leaf node: has tensorindex but no args
    if "tensorindex" in tree_dict and "args" not in tree_dict:
        return tensors[tree_dict["tensorindex"]-1]
    args = [_contract_recursive(arg, tensors) for arg in tree_dict["args"]]
    ixs_raw = tree_dict["eins"]["ixs"]
    iy_raw = tree_dict["eins"]["iy"]

    # Julia NestedEinsum can have "through labels" — labels that appear in ixs
    # but not as dimensions of the child tensor (they were contracted earlier).
    # We need to expand child tensors with missing dimensions.
    expanded_args = []
    for ix, t in zip(ixs_raw, args):
        if len(ix) > t.ndim:
            # Find which labels are missing from tensor
            # The contracted labels were removed from the tensor but still in ixs
            # We need to add back singleton dims for each missing label
            # Simple heuristic: add dims at the end for each extra label
            n_extra = len(ix) - t.ndim
            for _ in range(n_extra):
                t = t.unsqueeze(-1)
        expanded_args.append(t)
    return _einsum_int(ixs_raw, iy_raw, expanded_args)


def _einsum_int(ixs: list[list[int]], iy: list[int], tensors: list[torch.Tensor]) -> torch.Tensor:
    """Execute einsum with integer index labels."""
    allow_ascii = list(range(65, 90)) + list(range(97, 122))  # A-Z, a-z
    uniquelabels = list(set(sum(ixs, start=[]) + iy))
    label_map = {l: chr(allow_ascii[i]) for i, l in enumerate(uniquelabels)}
    inputs = ",".join("".join(label_map[l] for l in ix) for ix in ixs)
    output = "".join(label_map[l] for l in iy)
    return torch.einsum(f"{inputs}->{output}", *tensors)


if __name__ == "__main__":
    None
