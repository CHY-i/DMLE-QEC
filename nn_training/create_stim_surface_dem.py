from __future__ import annotations

import argparse
from pathlib import Path

import stim

from surface_utils import save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Create a Stim-generated surface-code DEM for NN pretraining.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="surface_code:rotated_memory_x")
    parser.add_argument("--distance", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--physical-error-rate", type=float, default=5e-4)
    parser.add_argument("--decompose-errors", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    circuit = stim.Circuit.generated(
        code_task=args.task,
        distance=args.distance,
        rounds=args.rounds,
        after_clifford_depolarization=args.physical_error_rate,
        after_reset_flip_probability=args.physical_error_rate,
        before_measure_flip_probability=args.physical_error_rate,
        before_round_data_depolarization=args.physical_error_rate,
    )
    dem = circuit.detector_error_model(
        decompose_errors=args.decompose_errors,
        flatten_loops=True,
    )
    output.write_text(str(dem))

    save_json(
        output.with_suffix(output.suffix + ".metadata.json"),
        {
            "task": args.task,
            "distance": args.distance,
            "rounds": args.rounds,
            "physical_error_rate": args.physical_error_rate,
            "decompose_errors": bool(args.decompose_errors),
            "num_detectors": dem.num_detectors,
            "num_errors": dem.num_errors,
            "num_observables": dem.num_observables,
        },
    )
    print(f"wrote {output}")
    print(f"detectors/errors/observables: {dem.num_detectors}/{dem.num_errors}/{dem.num_observables}")


if __name__ == "__main__":
    main()
