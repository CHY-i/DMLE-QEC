"""
Generate contraction trees for d=7 -> d=5 subsampled surface code TNs.

Workflow:
  1. Export each sub-PCM's einsum equation as .txt to tn_eqns/
  2. Call Julia (run_tn.jl) with OMEinsumContractionOrders TreeSA to find contraction trees
  3. Save tree JSONs to path/d7r{r}/

Usage:
    # Generate trees for r=3:
    python script/generate_d7_subsample_trees.py --r 3

    # Generate trees for r=5:
    python script/generate_d7_subsample_trees.py --r 5 --julia-threads 8

    # Generate trees for Google experimental X/r10:
    python script/generate_d7_subsample_trees.py --source google --basis X --round-tag r10 --julia-threads 192
"""

import sys
import os
import json
import subprocess
import shutil
import time
from pathlib import Path
import numpy as np
import stim
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
DEFAULT_WILLOW_ROOT = Path(
    os.environ.get("PATCHDMLE_WILLOW_DATA", Path(REPO_ROOT) / "dataset/willow_surface")
) / "d7_at_q6_7"
from src import DetectorCoordinateProxy, subsample_d5_pcms, subsample_d5_pcms_from_circuit, TensorNetwork


def export_einsum_equation(tn, out_path):
    """Export TensorNetwork's einsum equation string to a .txt file.

    Format is the opt_einsum equation string (e.g. "ab,bc,...->...").
    This can be read by Julia's qec_from_eqns().
    """
    with open(out_path, 'w') as f:
        f.write(tn.eq_str)
    print(f"  Equation exported to {out_path} ({len(tn.eq_str)} chars)")


def call_julia_find_tree(
    eq_file,
    out_file,
    julia_bin="julia",
    julia_script=None,
    seed=8,
    ntrials=192,
    niters=20,
    sc_target=33.0,
    sc_weight=0.5,
    rw_weight=1024.0,
    beta_start=1.0,
    beta_stop=10.0,
    use_slicer=True,
    init_tree_file="",
    beta_steps=2,
    tc_weight=1.0,
    initializer="",
    decomposition="tree",
):
    """Call Julia run_tn.jl to find an optimal contraction tree."""
    if julia_script is None:
        julia_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_tn.jl")

    cmd = [
        julia_bin,
        julia_script,
        eq_file,
        out_file,
        str(seed),
        str(ntrials),
        str(niters),
        str(sc_target),
        str(sc_weight),
        str(rw_weight),
        str(beta_start),
        str(beta_stop),
        "slicer" if use_slicer else "noslicer",
        init_tree_file,
        str(beta_steps),
        str(tc_weight),
        initializer,
        decomposition,
    ]
    print(f"  Calling Julia: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True, text=True, timeout=7200,
        env={**os.environ, "JULIA_NUM_THREADS": str(os.environ.get("JULIA_NUM_THREADS", "4"))},
    )
    elapsed = time.time() - t0

    # Print Julia output
    for line in result.stdout.strip().split('\n'):
        print(f"  [Julia] {line}")

    if result.returncode != 0:
        print(f"  [Julia ERROR] {result.stderr[:500]}")
        return False

    # Verify output file
    if os.path.exists(out_file):
        size_kb = os.path.getsize(out_file) / 1024
        print(f"  Tree saved to {out_file} ({size_kb:.1f} KB, {elapsed:.1f}s)")

        # Quick format check
        with open(out_file) as f:
            tree_data = json.load(f)
        if 'tree' in tree_data and 'eins' in tree_data['tree']:
            print(f"  Tree format: correct (nested einsum tree)")
        else:
            print(f"  WARNING: unexpected tree format! Keys: {list(tree_data.keys())}")
        return True
    else:
        print(f"  ERROR: Output file not created!")
        return False


def _round_count(round_tag):
    if str(round_tag).startswith("r"):
        return int(str(round_tag)[1:])
    return int(round_tag)


def _google_sub_pcms(data_root, basis, round_tag, decompose_errors):
    root = Path(data_root) / basis.upper() / round_tag
    ideal_circuit_path = root / "circuit_ideal.stim"
    noisy_circuit_path = root / "circuit_noisy_si1000.stim"
    cropped_dem_path = root / "initial_dem_non_decomposed.dem"
    detector_coords_path = root / "detector_coordinates.json"

    if ideal_circuit_path.exists() and noisy_circuit_path.exists():
        ideal_circuit = stim.Circuit.from_file(str(ideal_circuit_path))
        noisy_circuit = stim.Circuit.from_file(str(noisy_circuit_path))
        dem = noisy_circuit.detector_error_model(
            decompose_errors=decompose_errors,
            flatten_loops=True,
        )
        dem_source = "circuit_noisy_si1000.stim"
    elif cropped_dem_path.exists() and detector_coords_path.exists():
        with open(detector_coords_path) as f:
            detector_coordinates = {
                int(k): tuple(v)
                for k, v in json.load(f).items()
            }
        ideal_circuit = DetectorCoordinateProxy(detector_coordinates)
        dem = stim.DetectorErrorModel.from_file(str(cropped_dem_path))
        dem_source = "initial_dem_non_decomposed.dem"
    else:
        raise FileNotFoundError(
            f"Missing Google circuit files or cropped DEM+coordinates under {root}"
        )

    print(f"\nExtracting sub-PCMs from Google experimental data...")
    print(f"  root: {root}")
    print(f"  initial DEM: {dem_source}")
    print(f"  decompose_errors: {decompose_errors}")
    print(f"  detectors: {dem.num_detectors}")
    print(f"  errors: {dem.num_errors}")
    return subsample_d5_pcms_from_circuit(ideal_circuit, dem, print_info=True)


def _parse_sub_indices(sub_indices):
    if sub_indices in (None, "", "all"):
        return None
    if isinstance(sub_indices, int):
        return {sub_indices}
    if isinstance(sub_indices, (list, tuple)):
        return {int(x) for x in sub_indices}
    return {int(x.strip()) for x in str(sub_indices).split(",") if x.strip()}


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def main(r=3, julia_bin="julia", julia_threads=None, source="simulation",
         data_root=str(DEFAULT_WILLOW_ROOT),
         basis="X", round_tag=None, decompose_errors=False,
         stage="export_search",
         seed=8, ntrials=192, niters=20,
         sc_target=33.0, sc_weight=0.5, rw_weight=1024.0,
         beta_start=1.0, beta_stop=10.0, beta_steps=2,
         tc_weight=1.0, use_slicer=True, initializer="",
         decomposition="tree", init_tree_dir=None,
         out_name=None, candidate_tag="", init_candidate_tag="",
         sub_indices="all"):
    d = 7
    source = source.lower()
    decompose_errors = _parse_bool(decompose_errors)
    use_slicer = _parse_bool(use_slicer)
    if round_tag is None:
        round_tag = f"r{int(r):02d}"
    r = _round_count(round_tag) if source in {"google", "experimental"} else int(r)

    print(f"=== Generating contraction trees for d={d}, r={r} using Julia TreeSA ===")
    print(f"Source: {source}")
    print(f"Stage: {stage}")

    # Set Julia threads
    if julia_threads is not None:
        os.environ["JULIA_NUM_THREADS"] = str(julia_threads)
    print(f"Julia threads: {os.environ.get('JULIA_NUM_THREADS', 'auto')}")

    # 1. Get sub-PCMs
    if source in {"google", "experimental"}:
        basis = basis.upper()
        default_out_name = f'd{d}r{r}_google{basis}_subsample'
        sub_pcms, sub_dets, sub_errors = _google_sub_pcms(
            data_root=data_root,
            basis=basis,
            round_tag=round_tag,
            decompose_errors=decompose_errors,
        )
    elif source in {"simulation", "sim"}:
        default_out_name = f'd{d}r{r}'
        print(f"\nExtracting sub-PCMs from d={d} simulated surface code...")
        sub_pcms, sub_dets, sub_errors = subsample_d5_pcms(d=d, r=r, print_info=True)
    else:
        raise ValueError("source must be 'simulation' or 'google'")

    if out_name is None:
        out_name = default_out_name

    # 2. Setup output directories
    out_dir = os.path.join(REPO_ROOT, f'path/{out_name}')
    eq_dir = os.path.join(REPO_ROOT, f'tn_eqns/{out_name}')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(eq_dir, exist_ok=True)

    selected_subs = _parse_sub_indices(sub_indices)

    # 3. Export all einsum equations first so the full workload is visible up front.
    exported = []
    for i, (pcm, dets, errs) in enumerate(zip(sub_pcms, sub_dets, sub_errors)):
        if selected_subs is not None and i not in selected_subs:
            continue
        eq_file = os.path.join(eq_dir, f'eq_{i}.txt')

        if os.path.exists(eq_file):
            print(f"\nsub[{i}]: equation already exists, skipping export")
            exported.append((i, pcm))
            continue

        print(f"\n{'='*60}")
        print(f"sub[{i}]: exporting equation, PCM shape={pcm.shape}")
        print(f"{'='*60}")

        tn = TensorNetwork(pcm=pcm, priors_logits=torch.zeros(pcm.shape[1], dtype=torch.float64),
                           dev='cpu', dtype=torch.float64)
        export_einsum_equation(tn, eq_file)
        exported.append((i, pcm))

    candidate_tag = str(candidate_tag).strip()
    tag_prefix = f"_{candidate_tag}" if candidate_tag else ""
    init_tree_dir = init_tree_dir or out_dir
    init_candidate_tag = str(init_candidate_tag).strip()
    init_tag_prefix = f"_{init_candidate_tag}" if init_candidate_tag else ""

    # 4. For each sub-PCM, find a contraction tree.
    for i, pcm in exported:
        out_basename = f'subsample_tree_{i}{tag_prefix}.json'
        out_file = os.path.join(out_dir, out_basename)
        eq_file = os.path.join(eq_dir, f'eq_{i}.txt')

        if os.path.exists(out_file):
            print(f"\nsub[{i}]: tree already exists, skipping search")
            continue

        if stage != "refine":
            dup = False
            for j in range(i):
                if selected_subs is not None and j not in selected_subs:
                    continue
                if pcm.shape == sub_pcms[j].shape and np.array_equal(pcm, sub_pcms[j]):
                    src = os.path.join(out_dir, f'subsample_tree_{j}{tag_prefix}.json')
                    if os.path.exists(src):
                        shutil.copy(src, out_file)
                        print(f"\nsub[{i}]: identical to sub[{j}], copied tree")
                        dup = True
                        break
            if dup:
                continue

        print(f"\n{'='*60}")
        print(f"sub[{i}]: searching tree from {os.path.basename(eq_file)}")
        print(f"{'='*60}")

        init_tree_file = ""
        stage_initializer = initializer
        stage_use_slicer = use_slicer
        if stage == "refine":
            init_tree_file = os.path.join(init_tree_dir, f'subsample_tree_{i}{init_tag_prefix}.json')
            if not os.path.exists(init_tree_file):
                print(f"  Missing init tree for sub[{i}]: {init_tree_file}")
                continue
            if not stage_initializer:
                stage_initializer = "specified"

        success = call_julia_find_tree(
            eq_file,
            out_file,
            julia_bin=julia_bin,
            seed=seed,
            ntrials=ntrials,
            niters=niters,
            sc_target=sc_target,
            sc_weight=sc_weight,
            rw_weight=rw_weight,
            beta_start=beta_start,
            beta_stop=beta_stop,
            use_slicer=stage_use_slicer,
            init_tree_file=init_tree_file,
            beta_steps=beta_steps,
            tc_weight=tc_weight,
            initializer=stage_initializer,
            decomposition=decomposition,
        )
        if not success:
            print(f"  Failed to generate tree for sub[{i}], continuing...")

    # 5. Summary
    print(f"\n=== Done! Files in {out_dir}:")
    if os.path.exists(out_dir):
        for f in sorted(os.listdir(out_dir)):
            print(f"  {f}")

    print(f"\nEquation files in {eq_dir}:")
    if os.path.exists(eq_dir):
        for f in sorted(os.listdir(eq_dir)):
            print(f"  {f}")


if __name__ == '__main__':
    import fire
    fire.Fire(main)
