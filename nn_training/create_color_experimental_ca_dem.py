#!/usr/bin/env python3
"""Fit a correlation-analysis DEM to packed experimental detector events."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import stim


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CA_ROOT = PROJECT_ROOT / "src" / "ca_ler_compare_ref"
sys.path.insert(0, str(CA_ROOT))

from correlation_analysis import cal_multi_body_correlations  # noqa: E402
from notebook_common import (  # noqa: E402
    create_dem_from_analysis,
    extract_hyperedges_from_dem,
    flatten_prob_dicts,
)


DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "PATCHDMLE_COLOR_DATA",
        PROJECT_ROOT / "dataset/color_code",
    )
) / "d5X/r05"
DEFAULT_REFERENCE_DEM = (
    DEFAULT_DATA_ROOT
    / "decoding_results/chromobius_decoder_with_si1000_prior/error_model.dem"
)
DEFAULT_OUTPUT = (
  PROJECT_ROOT / "nn_training/dems/color/d5_r5/ca.dem"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_error_signatures(dem: stim.DetectorErrorModel) -> list[tuple[str, ...]]:
    return [
        tuple(str(target) for target in instruction.targets_copy())
        for instruction in dem.flattened()
        if instruction.type == "error"
    ]


def sanitize_probabilities(
    estimated: dict[frozenset[int], float],
    reference: dict[frozenset[int], float],
) -> tuple[dict[frozenset[int], float], int]:
    result = {}
    fallback_count = 0
    for edge, reference_probability in reference.items():
        probability = float(estimated.get(edge, reference_probability))
        if not np.isfinite(probability) or probability < 0.0 or probability >= 0.5:
            probability = float(reference_probability)
            fallback_count += 1
        result[edge] = probability
    return result, fallback_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--reference-dem", type=Path, default=DEFAULT_REFERENCE_DEM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-shots", type=int, default=1000)
    parser.add_argument("--max-shots", type=int)
    parser.add_argument(
        "--no-correct-in-step",
        action="store_false",
        dest="correct_in_step",
        help="Disable fallback to SI1000 while solving high-to-low correlations.",
    )
    parser.set_defaults(correct_in_step=True)
    args = parser.parse_args()

    detection_events_path = args.data_root / "detection_events.b8"
    reference_dem = stim.DetectorErrorModel.from_file(str(args.reference_dem)).flattened()
    detectors = stim.read_shot_data_file(
        path=str(detection_events_path),
        format="b8",
        num_detectors=reference_dem.num_detectors,
        bit_packed=False,
    ).astype(np.float32, copy=False)
    if args.max_shots is not None:
        detectors = detectors[: args.max_shots]
    if not len(detectors):
        raise ValueError("No experimental shots were loaded.")

    reference_probs, hyperedge_list = extract_hyperedges_from_dem(reference_dem)
    if len(reference_probs) != reference_dem.num_errors:
        raise ValueError(
            "Each error mechanism must have a unique non-empty detector signature: "
            f"found {len(reference_probs)} signatures for {reference_dem.num_errors} errors."
        )

    started = time.perf_counter()
    ca_diagnostics = {}
    probability_groups, _ = cal_multi_body_correlations(
        detectors,
        hyperedge_list=hyperedge_list,
        device=args.device,
        correct_in_step=args.correct_in_step,
        batch_shots=args.batch_shots,
        reference_values=reference_probs,
        diagnostics=ca_diagnostics,
    )
    estimated_probs, fallback_count = sanitize_probabilities(
        flatten_prob_dicts(probability_groups), reference_probs
    )
    ca_dem = create_dem_from_analysis(reference_dem, estimated_probs)

    if ordered_error_signatures(ca_dem) != ordered_error_signatures(reference_dem):
        raise RuntimeError("CA reconstruction changed the ordered DEM error topology.")
    if ca_dem.num_detectors != reference_dem.num_detectors:
        raise RuntimeError("CA reconstruction changed the detector count.")
    if ca_dem.num_observables != reference_dem.num_observables:
        raise RuntimeError("CA reconstruction changed the observable count.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ca_dem.to_file(str(args.output))
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata = {
        "method": "correlation_analysis",
        "source": "Google Color-code experimental detection events",
        "data_root": str(args.data_root.resolve()),
        "detection_events": str(detection_events_path.resolve()),
        "detection_events_sha256": sha256(detection_events_path),
        "reference_dem": str(args.reference_dem.resolve()),
        "reference_dem_sha256": sha256(args.reference_dem),
        "output_dem": str(args.output.resolve()),
        "output_dem_sha256": sha256(args.output),
        "shots": int(len(detectors)),
        "num_detectors": int(ca_dem.num_detectors),
        "num_observables": int(ca_dem.num_observables),
        "num_errors": int(ca_dem.num_errors),
        "maximum_hyperedge_order": int(len(hyperedge_list)),
        "hyperedge_order_counts": [int(len(group)) for group in hyperedge_list],
        "correct_in_step": bool(args.correct_in_step),
        "in_step_reference_fallbacks": ca_diagnostics,
        "post_analysis_invalid_probability_fallbacks": int(fallback_count),
        "device": args.device,
        "batch_shots": int(args.batch_shots),
        "runtime_seconds": time.perf_counter() - started,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Loaded {len(detectors)} experimental shots with {detectors.shape[1]} detectors.")
    print(f"Fitted {len(estimated_probs)} CA error probabilities in {metadata['runtime_seconds']:.2f} s.")
    print(f"In-step SI1000 reference fallbacks: {ca_diagnostics}.")
    print(f"Post-analysis SI1000 reference fallbacks: {fallback_count}.")
    print(f"Wrote {args.output}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
