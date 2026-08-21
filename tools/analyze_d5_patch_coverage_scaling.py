#!/usr/bin/env python3
"""Check complete DEM-error coverage by tiled d=5 surface-code patches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import stim


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import PCM


D5_TEMPLATE = np.array(
    [
        [0, 4],
        [0, 8],
        [2, 0],
        [2, 2],
        [2, 4],
        [2, 6],
        [2, 8],
        [4, 2],
        [4, 4],
        [4, 6],
        [4, 8],
        [4, 10],
        [6, 0],
        [6, 2],
        [6, 4],
        [6, 6],
        [6, 8],
        [8, 2],
        [8, 4],
        [8, 6],
        [8, 8],
        [8, 10],
        [10, 2],
        [10, 6],
    ],
    dtype=int,
)
D5_CENTER = np.array([5, 5])
FIXED_D7_CENTERS = [(5, 5), (9, 5), (5, 9), (9, 9), (7, 7)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "analysis_outputs/d5_patch_coverage_scaling.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=REPO_ROOT / "d5_patch_coverage_scaling.md",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Test d=5..17, r=1/4/9, X/Z, decomposed/non-decomposed.",
    )
    return parser.parse_args()


def regular_d5_centers(distance: int) -> list[tuple[int, int]]:
    """Tile d5 windows on the stabilizer grid with four-coordinate spacing."""
    if distance < 5 or distance % 2 == 0:
        raise ValueError(
            f"Expected an odd surface-code distance >= 5, got {distance}"
        )
    width = (distance - 3) // 2
    return [
        (5 + 4 * x_index, 5 + 4 * y_index)
        for y_index in range(width)
        for x_index in range(width)
    ]


def build_circuit_and_dem(
    distance: int,
    rounds: int,
    basis: str,
    decompose_errors: bool,
) -> tuple[stim.Circuit, stim.DetectorErrorModel]:
    circuit = stim.Circuit.generated(
        code_task=f"surface_code:rotated_memory_{basis.lower()}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.001,
        before_measure_flip_probability=0.001,
        after_reset_flip_probability=0.001,
    )
    dem = circuit.detector_error_model(
        decompose_errors=decompose_errors,
        flatten_loops=True,
    )
    return circuit, dem


def logical_error_mask(dem: stim.DetectorErrorModel) -> np.ndarray:
    values = []
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        values.append(
            any(
                target.is_logical_observable_id()
                for target in instruction.targets_copy()
            )
        )
    return np.asarray(values, dtype=bool)


def patch_detector_rows(
    circuit: stim.Circuit,
    centers: list[tuple[int, int]],
) -> list[set[int]]:
    rows_by_xy: dict[tuple[int, int], list[int]] = {}
    for detector, coordinate in circuit.get_detector_coordinates().items():
        xy = (int(round(coordinate[0])), int(round(coordinate[1])))
        rows_by_xy.setdefault(xy, []).append(int(detector))

    result = []
    for center in centers:
        target_coordinates = D5_TEMPLATE + np.asarray(center) - D5_CENTER
        rows = {
            detector
            for xy in map(tuple, target_coordinates.tolist())
            for detector in rows_by_xy.get(xy, [])
        }
        result.append(rows)
    return result


def analyze_configuration(
    distance: int,
    rounds: int,
    basis: str,
    decompose_errors: bool,
    centers: list[tuple[int, int]],
    layout: str,
) -> dict:
    circuit, dem = build_circuit_and_dem(
        distance=distance,
        rounds=rounds,
        basis=basis,
        decompose_errors=decompose_errors,
    )
    pcm, _ = PCM(dem)
    detector_coordinates = circuit.get_detector_coordinates()
    detector_xy = {
        detector: (
            int(round(coordinate[0])),
            int(round(coordinate[1])),
        )
        for detector, coordinate in detector_coordinates.items()
    }
    patch_rows = patch_detector_rows(circuit, centers)
    detector_union = set().union(*patch_rows)
    logical_mask = logical_error_mask(dem)

    categories = {
        "full_only": 0,
        "full_and_partial": 0,
        "partial_only": 0,
        "uncovered": 0,
        "no_detector": 0,
    }
    logical_categories = dict.fromkeys(categories, 0)
    full_occurrences = 0
    partial_occurrences = 0
    max_spatial_span = [0, 0]

    for error_index in range(pcm.shape[1]):
        support = set(np.flatnonzero(pcm[:, error_index]).tolist())
        if not support:
            category = "no_detector"
        else:
            support_xy = np.asarray(
                [detector_xy[detector] for detector in support]
            )
            span = support_xy.max(axis=0) - support_xy.min(axis=0)
            max_spatial_span[0] = max(
                max_spatial_span[0], int(span[0])
            )
            max_spatial_span[1] = max(
                max_spatial_span[1], int(span[1])
            )

            full_count = sum(support <= rows for rows in patch_rows)
            partial_count = sum(
                bool(support & rows) and not support <= rows
                for rows in patch_rows
            )
            full_occurrences += full_count
            partial_occurrences += partial_count

            if full_count and partial_count:
                category = "full_and_partial"
            elif full_count:
                category = "full_only"
            elif partial_count:
                category = "partial_only"
            else:
                category = "uncovered"

        categories[category] += 1
        if logical_mask[error_index]:
            logical_categories[category] += 1

    return {
        "layout": layout,
        "distance": distance,
        "rounds": rounds,
        "basis": basis.upper(),
        "decompose_errors": decompose_errors,
        "patch_count": len(centers),
        "centers": centers,
        "detectors": int(dem.num_detectors),
        "detectors_covered": len(detector_union),
        "errors": int(dem.num_errors),
        "logical_errors": int(logical_mask.sum()),
        "categories": categories,
        "logical_categories": logical_categories,
        "full_occurrences": full_occurrences,
        "partial_occurrences": partial_occurrences,
        "max_spatial_span": max_spatial_span,
        "complete_coverage": bool(
            len(detector_union) == dem.num_detectors
            and categories["partial_only"] == 0
            and categories["uncovered"] == 0
            and categories["no_detector"] == 0
        ),
    }


def result_row(result: dict) -> str:
    categories = result["categories"]
    logical = result["logical_categories"]
    return (
        f"| {result['distance']} | {result['basis']} | "
        f"{result['rounds']} | {result['decompose_errors']} | "
        f"{result['patch_count']} | "
        f"{result['detectors_covered']}/{result['detectors']} | "
        f"{result['errors']} | {categories['partial_only']} | "
        f"{categories['uncovered']} | {logical['partial_only']} | "
        f"{logical['uncovered']} | {result['complete_coverage']} |"
    )


def write_markdown(path: Path, results: dict) -> None:
    primary = results["primary"]
    fixed = results["fixed_five_comparison"]
    stress = results["stress"]
    lines = [
        "# d5 Patch Coverage Scaling",
        "",
        "## Definition",
        "",
        "An error is completely covered when all detector targets of that",
        "full-DEM error mechanism occur in at least one d5 patch.",
        "",
        "The scalable layout uses a regular grid of d5 windows with centers",
        "",
        "```text",
        "(5 + 4 i, 5 + 4 j),",
        "i,j = 0,...,(d-5)/2.",
        "```",
        "",
        "The number of patches is therefore `((d-3)/2)^2`.",
        "",
        "## Primary Results",
        "",
        "| d | basis | rounds | decomposed | patches | detectors | errors | partial-only | uncovered | logical partial-only | logical uncovered | complete |",
        "|---:|:---:|---:|:---:|---:|:---:|---:|---:|---:|---:|---:|:---:|",
    ]
    lines.extend(result_row(result) for result in primary)
    lines.extend(
        [
            "",
            "## Reusing The Fixed Five d7 Patches",
            "",
            "| d | basis | rounds | decomposed | patches | detectors | errors | partial-only | uncovered | logical partial-only | logical uncovered | complete |",
            "|---:|:---:|---:|:---:|---:|:---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    lines.extend(result_row(result) for result in fixed)
    lines.extend(
        [
            "",
            "The fixed five-patch layout is valid for d7 but does not scale",
            "unchanged to d9 or d11.",
            "",
            "## Stress Test",
            "",
            f"- Configurations tested: {stress['configurations']}",
            f"- Complete configurations: {stress['complete']}",
            f"- Failed configurations: {stress['failed']}",
            f"- Maximum observed spatial error span: {stress['max_spatial_span']}",
            "",
            "The stress test covers odd distances 5 through 17, X/Z memory,",
            "round counts 1/4/9, and decomposed/non-decomposed DEMs.",
            "",
            "## Scope",
            "",
            "This establishes complete coverage for standard Stim rotated",
            "surface-code circuits with the local circuit-level noise used here.",
            "It is not a guarantee for arbitrary DEMs containing nonlocal or",
            "user-injected correlated error mechanisms whose detector support",
            "is wider than a d5 window.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    primary = []
    for distance in (7, 9, 11, 13):
        for basis in ("X", "Z"):
            primary.append(
                analyze_configuration(
                    distance=distance,
                    rounds=4,
                    basis=basis,
                    decompose_errors=False,
                    centers=regular_d5_centers(distance),
                    layout="regular",
                )
            )

    fixed = [
        analyze_configuration(
            distance=distance,
            rounds=4,
            basis="X",
            decompose_errors=False,
            centers=FIXED_D7_CENTERS,
            layout="fixed_d7_five",
        )
        for distance in (7, 9, 11)
    ]

    stress_results = []
    if args.stress:
        for distance in range(5, 18, 2):
            for rounds in (1, 4, 9):
                for basis in ("X", "Z"):
                    for decompose_errors in (False, True):
                        stress_results.append(
                            analyze_configuration(
                                distance=distance,
                                rounds=rounds,
                                basis=basis,
                                decompose_errors=decompose_errors,
                                centers=regular_d5_centers(distance),
                                layout="regular",
                            )
                        )
    else:
        stress_results = primary

    max_span = [
        max(result["max_spatial_span"][axis] for result in stress_results)
        for axis in (0, 1)
    ]
    failed = [
        {
            "distance": result["distance"],
            "rounds": result["rounds"],
            "basis": result["basis"],
            "decompose_errors": result["decompose_errors"],
            "categories": result["categories"],
        }
        for result in stress_results
        if not result["complete_coverage"]
    ]
    output = {
        "primary": primary,
        "fixed_five_comparison": fixed,
        "stress": {
            "configurations": len(stress_results),
            "complete": len(stress_results) - len(failed),
            "failed": len(failed),
            "failures": failed,
            "max_spatial_span": max_span,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    write_markdown(args.output_markdown, output)
    print(json.dumps(output["stress"], indent=2))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_markdown}")


if __name__ == "__main__":
    main()
