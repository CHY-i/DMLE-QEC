from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import stim
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the error probabilities in a dMLE checkpoint as a DEM."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-dem", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--list-index",
        type=int,
        default=-1,
        help="Entry selected from list/tuple checkpoints; -1 selects the last epoch.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_error_targets(dem: stim.DetectorErrorModel) -> list[tuple[str, ...]]:
    return [
        tuple(str(target) for target in instruction.targets_copy())
        for instruction in dem.flattened()
        if instruction.type == "error"
    ]


def replace_probabilities(
    dem: stim.DetectorErrorModel, probabilities: np.ndarray
) -> stim.DetectorErrorModel:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (dem.num_errors,):
        raise ValueError(
            f"Checkpoint has {len(probabilities)} probabilities, "
            f"but the base DEM has {dem.num_errors} errors."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Checkpoint contains non-finite probabilities.")
    if np.any((probabilities <= 0) | (probabilities >= 1)):
        raise ValueError("Checkpoint probabilities must be strictly between 0 and 1.")

    output = stim.DetectorErrorModel()
    error_index = 0
    for instruction in dem.flattened():
        if instruction.type == "error":
            output.append(
                stim.DemInstruction(
                    "error",
                    [float(probabilities[error_index])],
                    instruction.targets_copy(),
                )
            )
            error_index += 1
        else:
            output.append(instruction)
    if ordered_error_targets(output) != ordered_error_targets(dem):
        raise RuntimeError("Error-target ordering changed while exporting the DEM.")
    return output


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    base_dem_path = Path(args.base_dem).resolve()
    output_path = Path(args.output).resolve()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_format = type(checkpoint).__name__
    checkpoint_entries = None
    checkpoint_index = None
    checkpoint_epoch = None
    checkpoint_nll = None
    checkpoint_ler = None
    if isinstance(checkpoint, (list, tuple)):
        if not checkpoint:
            raise ValueError(f"Checkpoint contains no epochs: {checkpoint_path}")
        probabilities = checkpoint[args.list_index]
        checkpoint_entries = len(checkpoint)
        checkpoint_index = args.list_index % len(checkpoint)
        checkpoint_epoch = checkpoint_index + 1
    elif isinstance(checkpoint, torch.Tensor):
        probabilities = checkpoint
        checkpoint_epoch = 1
    elif isinstance(checkpoint, dict):
        if "oer" not in checkpoint:
            raise KeyError(
                f"Checkpoint has no 'oer' probability vector: {checkpoint_path}"
            )
        probabilities = checkpoint["oer"]
        checkpoint_epoch = int(checkpoint["epoch"])
        checkpoint_nll = float(checkpoint["nll"])
        checkpoint_ler = (
            float(checkpoint["eval_ler"])
            if checkpoint.get("eval_ler") is not None
            else None
        )
    else:
        raise TypeError(f"Unsupported checkpoint type: {checkpoint_format}")
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()

    base_dem = stim.DetectorErrorModel.from_file(str(base_dem_path))
    output_dem = replace_probabilities(base_dem, probabilities)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(str(output_dem), encoding="utf-8")
    os.replace(temporary_path, output_path)

    metadata = {
        "method": "full_tensor_network_dMLE",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_format": checkpoint_format,
        "checkpoint_entries": checkpoint_entries,
        "checkpoint_list_index": checkpoint_index,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_nll": checkpoint_nll,
        "checkpoint_ler": checkpoint_ler,
        "base_dem": str(base_dem_path),
        "base_dem_sha256": sha256_file(base_dem_path),
        "output_dem": str(output_path),
        "output_dem_sha256": sha256_file(output_path),
        "num_detectors": output_dem.num_detectors,
        "num_errors": output_dem.num_errors,
        "num_observables": output_dem.num_observables,
        "ordered_error_targets_unchanged": True,
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
