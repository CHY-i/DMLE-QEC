from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import stim
import tqec
from tqec.utils.noise_model import NoiseModel

from surface_utils import read_experimental_round, surface_layout_from_dem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply TQEC SI1000 noise to an exact ideal circuit and calibrate its "
            "global p to experimental mean detector-event density."
        )
    )
    parser.add_argument("--ideal-circuit", required=True)
    parser.add_argument("--eval-data-root", required=True)
    parser.add_argument("--output-dem", required=True)
    parser.add_argument("--output-circuit", default=None)
    parser.add_argument("--reference-dem", default=None)
    parser.add_argument("--expected-shots", type=int, default=75_000)
    parser.add_argument("--low-p", type=float, default=1e-8)
    parser.add_argument("--high-p", type=float, default=0.02)
    parser.add_argument("--density-tolerance", type=float, default=1e-9)
    parser.add_argument("--max-iterations", type=int, default=40)
    return parser.parse_args()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_detector_density(dem: stim.DetectorErrorModel) -> tuple[float, np.ndarray]:
    """Return exact detector marginals for independent DEM error mechanisms."""
    parity_factors = np.ones(dem.num_detectors, dtype=np.float64)
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        probability = float(instruction.args_copy()[0])
        detectors = {
            int(target.val)
            for target in instruction.targets_copy()
            if target.is_relative_detector_id()
        }
        for detector in detectors:
            parity_factors[detector] *= 1.0 - 2.0 * probability
    marginals = 0.5 * (1.0 - parity_factors)
    return float(np.mean(marginals)), marginals


def normalize_detector_coordinates(
    dem: stim.DetectorErrorModel,
    ideal_circuit: stim.Circuit,
) -> stim.DetectorErrorModel:
    """Replace repeated/shifted TQEC coordinate annotations with ideal coordinates."""
    ideal_coords = ideal_circuit.get_detector_coordinates()
    if len(ideal_coords) != ideal_circuit.num_detectors:
        raise ValueError(
            f"Ideal circuit has coordinates for {len(ideal_coords)} of "
            f"{ideal_circuit.num_detectors} detectors."
        )

    normalized = stim.DetectorErrorModel()
    seen_detectors = set()
    for instruction in dem.flattened():
        if instruction.type != "detector":
            normalized.append(instruction)
            continue

        targets = instruction.targets_copy()
        detector_ids = [
            int(target.val)
            for target in targets
            if target.is_relative_detector_id()
        ]
        if len(detector_ids) != 1:
            raise ValueError(f"Unexpected detector annotation: {instruction}")
        detector_id = detector_ids[0]
        coords = [float(value) for value in ideal_coords[detector_id]]
        normalized.append(stim.DemInstruction("detector", coords, targets))
        seen_detectors.add(detector_id)

    if len(seen_detectors) != ideal_circuit.num_detectors:
        raise ValueError(
            f"Normalized coordinates cover {len(seen_detectors)} of "
            f"{ideal_circuit.num_detectors} detectors."
        )
    return normalized


def build_si1000(
    ideal_circuit: stim.Circuit,
    p: float,
) -> tuple[stim.Circuit, stim.DetectorErrorModel]:
    noisy_circuit = NoiseModel.si1000(p).noisy_circuit(ideal_circuit)
    raw_dem = noisy_circuit.detector_error_model(
        decompose_errors=False,
        flatten_loops=True,
    )
    dem = normalize_detector_coordinates(raw_dem, ideal_circuit)
    return noisy_circuit, dem


def target_signature_set(dem: stim.DetectorErrorModel) -> set[tuple[str, ...]]:
    return {
        tuple(str(target) for target in instruction.targets_copy())
        for instruction in dem.flattened()
        if instruction.type == "error"
    }


def main() -> None:
    args = parse_args()
    ideal_path = Path(args.ideal_circuit)
    eval_root = Path(args.eval_data_root)
    output_dem = Path(args.output_dem)
    output_circuit = (
        Path(args.output_circuit)
        if args.output_circuit
        else output_dem.with_name("circuit_noisy_si1000.stim")
    )

    ideal_circuit = stim.Circuit.from_file(str(ideal_path))
    if ideal_circuit.num_observables != 1:
        raise ValueError(
            f"Expected one logical observable, got {ideal_circuit.num_observables}."
        )

    # The DEM argument is used only to determine detector count when reading b8.
    _, probe_dem = build_si1000(ideal_circuit, args.low_p)
    experimental_dets, experimental_obs = read_experimental_round(
        eval_root, probe_dem
    )
    if args.expected_shots and len(experimental_dets) != args.expected_shots:
        raise ValueError(
            f"Expected {args.expected_shots} experimental shots, "
            f"loaded {len(experimental_dets)}."
        )
    target_density = float(np.mean(experimental_dets))

    trace = []

    def evaluate_p(p: float):
        noisy, dem = build_si1000(ideal_circuit, p)
        density, marginals = expected_detector_density(dem)
        trace.append({"p": float(p), "density": float(density)})
        return density, marginals, noisy, dem

    low_density, _, _, _ = evaluate_p(args.low_p)
    high_density, _, _, _ = evaluate_p(args.high_p)
    if not low_density <= target_density <= high_density:
        raise ValueError(
            "Target density is not bracketed: "
            f"low p={args.low_p} -> {low_density}, "
            f"target={target_density}, "
            f"high p={args.high_p} -> {high_density}."
        )

    low_p = float(args.low_p)
    high_p = float(args.high_p)
    final = None
    for _ in range(args.max_iterations):
        mid_p = 0.5 * (low_p + high_p)
        density, marginals, noisy, dem = evaluate_p(mid_p)
        final = (mid_p, density, marginals, noisy, dem)
        if abs(density - target_density) <= args.density_tolerance:
            break
        if density < target_density:
            low_p = mid_p
        else:
            high_p = mid_p

    if final is None:
        raise RuntimeError("SI1000 calibration did not evaluate a midpoint.")
    calibrated_p, calibrated_density, marginals, noisy_circuit, dem = final

    if dem.num_detectors != ideal_circuit.num_detectors:
        raise ValueError(
            f"DEM detectors {dem.num_detectors} != ideal circuit detectors "
            f"{ideal_circuit.num_detectors}."
        )
    if dem.num_observables != ideal_circuit.num_observables:
        raise ValueError(
            f"DEM observables {dem.num_observables} != ideal circuit observables "
            f"{ideal_circuit.num_observables}."
        )
    layout = surface_layout_from_dem(dem)

    reference_comparison = None
    if args.reference_dem:
        reference_path = Path(args.reference_dem)
        reference = stim.DetectorErrorModel.from_file(str(reference_path))
        generated_signatures = target_signature_set(dem)
        reference_signatures = target_signature_set(reference)
        reference_comparison = {
            "path": str(reference_path),
            "sha256": file_sha256(reference_path),
            "num_detectors": int(reference.num_detectors),
            "num_errors": int(reference.num_errors),
            "same_detector_count": bool(
                reference.num_detectors == dem.num_detectors
            ),
            "same_error_target_set": bool(
                reference_signatures == generated_signatures
            ),
            "generated_unique_targets": len(generated_signatures),
            "reference_unique_targets": len(reference_signatures),
            "common_unique_targets": len(
                generated_signatures & reference_signatures
            ),
        }

    output_dem.parent.mkdir(parents=True, exist_ok=True)
    output_circuit.parent.mkdir(parents=True, exist_ok=True)
    output_dem.write_text(str(dem), encoding="utf-8")
    output_circuit.write_text(str(noisy_circuit), encoding="utf-8")

    metadata = {
        "method": "TQEC NoiseModel.si1000",
        "tqec_version": getattr(tqec, "__version__", "unknown"),
        "ideal_circuit": str(ideal_path),
        "ideal_circuit_sha256": file_sha256(ideal_path),
        "eval_data_root": str(eval_root),
        "detection_events_sha256": file_sha256(
            eval_root / "detection_events.b8"
        ),
        "obs_flips_sha256": file_sha256(eval_root / "obs_flips_actual.b8"),
        "experimental_shots": int(len(experimental_dets)),
        "experimental_observable_flip_rate": float(np.mean(experimental_obs)),
        "density_statistic": "mean of all detector-event bits",
        "experimental_detector_event_density": target_density,
        "calibrated_p": float(calibrated_p),
        "calibrated_expected_detector_event_density": float(
            calibrated_density
        ),
        "absolute_density_error": float(
            abs(calibrated_density - target_density)
        ),
        "per_detector_expected_density": {
            "min": float(np.min(marginals)),
            "mean": float(np.mean(marginals)),
            "max": float(np.max(marginals)),
            "std": float(np.std(marginals)),
        },
        "calibration_trace": trace,
        "output_dem": str(output_dem),
        "output_dem_sha256": file_sha256(output_dem),
        "output_circuit": str(output_circuit),
        "output_circuit_sha256": file_sha256(output_circuit),
        "decompose_errors": False,
        "coordinates_replaced_from_ideal_circuit": True,
        "num_detectors": int(dem.num_detectors),
        "num_errors": int(dem.num_errors),
        "num_observables": int(dem.num_observables),
        "layout_cycles": int(layout["cycles"]),
        "layout_slots": int(layout["num_slots"]),
        "reference_comparison": reference_comparison,
    }
    metadata_path = output_dem.with_suffix(output_dem.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"calibrated_p={calibrated_p:.12g}")
    print(f"experimental_density={target_density:.12g}")
    print(f"expected_density={calibrated_density:.12g}")
    print(
        f"detectors/errors/observables="
        f"{dem.num_detectors}/{dem.num_errors}/{dem.num_observables}"
    )
    print(f"layout cycles/slots={layout['cycles']}/{layout['num_slots']}")
    print(f"wrote {output_dem}")
    print(f"wrote {output_circuit}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
