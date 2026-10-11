"""
Test ``TensorNetwork.find_contraction_tree`` (contraction order search with artensor) on the
distance-3 rotated surface code.

1. Small DEM (one round, 18 errors): syndrome probabilities from the artensor tree against
   brute-force enumeration of all error configurations.
2. Circuit-level DEM (three rounds): NLL and gradients from the artensor tree against the
   default opt_einsum contraction, complexity of the tree, save / load round trip.
3. A short optimization of the priors with the artensor tree.

Run from repo root:

    python script/test_tn_order_artensor.py
"""

from __future__ import annotations

import itertools
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import opt_einsum as oe
import stim
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import PCM, TensorNetwork, get_error_rates  # noqa: E402
from src.utils import order_to_tree  # noqa: E402

DTYPE = torch.float64


def surface_code_dem(rounds: int, **noise) -> stim.DetectorErrorModel:
    circuit = stim.Circuit.generated(code_task="surface_code:rotated_memory_z", distance=3, rounds=rounds, **noise)
    return circuit.detector_error_model(decompose_errors=False, flatten_loops=True)


def perturbed(er: torch.Tensor) -> torch.Tensor:
    signs = 2 * torch.bernoulli(torch.ones(len(er), dtype=DTYPE) / 2) - 1.0
    return er + signs * er * torch.rand_like(er)


def tree_complexity(tree: dict, bond_dims: dict) -> tuple[float, float]:
    """log10 FLOPs and log2 size of the largest tensor of a tree, counted from its einsums."""
    log2_size = lambda labels: sum(math.log2(bond_dims[label]) for label in set(labels))
    flops, sc = 0.0, max(log2_size(ix) for ix in tree["inputs"])
    stack = [tree["tree"]]
    while stack:
        node = stack.pop()
        if node["isleaf"]:
            continue
        stack.extend(node["args"])
        flops += 2.0 ** log2_size(sum(node["eins"]["ixs"], []))
        sc = max(sc, log2_size(node["eins"]["iy"]))
    return math.log10(flops), sc


def path_to_order(path, n: int) -> list[tuple[int, int]]:
    """opt_einsum path (operands are popped, the result appended) as an artensor-style order."""
    ids, order = list(range(n)), []
    for step in path:
        first, *rest = [ids[k] for k in step]
        for k in sorted(step, reverse=True):
            ids.pop(k)
        order += [(first, other) for other in rest]
        ids.append(first)
    return order


def test_exact_probabilities() -> None:
    dem = surface_code_dem(rounds=1, before_round_data_depolarization=0.05, before_measure_flip_probability=0.03)
    pcm, _ = PCM(dem)
    er = get_error_rates(dem)
    n_check, n_bit = pcm.shape

    # p(s) = sum of the probabilities of all error configurations with syndrome s.
    errors = np.array(list(itertools.product([0, 1], repeat=n_bit)))
    probs = np.prod(np.where(errors == 1, er, 1 - er), axis=1)
    syndrome_ids = ((errors @ pcm.T) % 2).astype(np.int64) @ (1 << np.arange(n_check))
    p_exact = np.bincount(syndrome_ids, weights=probs, minlength=2**n_check)

    tn = TensorNetwork(pcm=pcm, priors_logits=torch.logit(torch.from_numpy(er)), dtype=DTYPE)
    tn.tree = tn.find_contraction_tree(batch_size=1, trials=2, iters=10)
    assert tn.tree is not None

    syndromes = torch.tensor([[(s >> i) & 1 for i in range(n_check)] for s in range(2**n_check)])
    with torch.no_grad():
        p_tn = np.array([torch.exp(-tn(s)).item() for s in syndromes])

    print(f"[exact] {n_check} detectors, {n_bit} errors: sum p = {p_tn.sum():.12f}, "
          f"max |p_tn - p_exact| = {np.abs(p_tn - p_exact).max():.2e}")
    assert np.allclose(p_tn, p_exact, rtol=1e-9, atol=1e-14)


def test_circuit_level(batch_size: int = 1000) -> None:
    error_prob = 0.001
    dem = surface_code_dem(rounds=3, after_clifford_depolarization=error_prob,
                           before_measure_flip_probability=error_prob, after_reset_flip_probability=error_prob)
    pcm, _ = PCM(dem)
    er = torch.from_numpy(get_error_rates(dem)).to(DTYPE)
    priors_logits = torch.logit(perturbed(er))
    dets, _, _ = dem.compile_sampler(seed=75328).sample(shots=batch_size)
    dets = torch.from_numpy(dets * 1.0)

    tn = TensorNetwork(pcm=pcm, priors_logits=priors_logits, dtype=DTYPE)
    tn_ref = TensorNetwork(pcm=pcm, priors_logits=priors_logits, dtype=DTYPE)

    t0 = time.perf_counter()
    tree = tn.find_contraction_tree(batch_size=batch_size)
    search_time = time.perf_counter() - t0
    assert tree is not None

    # The tree that gets executed must have the complexity artensor reports for its order.
    import artensor
    inputs, bond_dims, output = tn.einsum_structure(batch_size)
    assert tree["inputs"] == inputs and tree["output"] == output and len(inputs) == len(tn.xor_list) + sum(pcm.shape)
    order, slices, ctree = artensor.load_ome_order(tree, bond_dims)
    tc, sc, _ = ctree.tree_complexity()
    tc_tree, sc_tree = tree_complexity(tree, bond_dims)
    assert slices == [] and abs(tc - tc_tree) < 1e-9 and abs(sc - sc_tree) < 1e-9

    # Reference: the default opt_einsum path, measured with the same complexity count.
    shapes = [tuple(bond_dims[label] for label in ix) for ix in inputs]
    tn_ref.path, _ = oe.contract_path(tn.eq_str, *shapes, shapes=True, optimize='auto')
    tc_ref, sc_ref = tree_complexity(order_to_tree(inputs, output, path_to_order(tn_ref.path, len(inputs))), bond_dims)
    print(f"[d3r3] {len(inputs)} tensors, batch {batch_size}, search time {search_time:.1f}s")
    print(f"[d3r3] artensor  : log10 FLOPs {tc:.2f}, log2 size {sc:.2f}")
    print(f"[d3r3] opt_einsum: log10 FLOPs {tc_ref:.2f}, log2 size {sc_ref:.2f}")

    with tempfile.TemporaryDirectory() as tmp:
        tree_file = f"{tmp}/d3r3.json"
        tn.save_tree(tree, filename=tree_file)
        tn.load_tree(tree_file)
    assert tn.tree == tree

    times = {}
    for name, model in (("artensor", tn), ("opt_einsum", tn_ref)):
        t0 = time.perf_counter()
        loss = model(dets)
        loss.backward()
        times[name] = time.perf_counter() - t0
    nll, nll_ref = tn(dets).item(), tn_ref(dets).item()
    grad, grad_ref = tn.priors_logits.grad, tn_ref.priors_logits.grad
    grad_err = ((grad - grad_ref).abs().max() / grad_ref.abs().max()).item()
    print(f"[d3r3] NLL artensor {nll:.12f} | opt_einsum {nll_ref:.12f} | relative grad error {grad_err:.2e}")
    print(f"[d3r3] forward + backward: artensor {times['artensor']:.2f}s | opt_einsum {times['opt_einsum']:.2f}s")
    assert abs(nll - nll_ref) < 1e-10 * abs(nll_ref)
    assert grad_err < 1e-8


def test_optimization(num_shots: int = 200000, batch_size: int = 10000, epochs: int = 20, lr: float = 0.02) -> None:
    error_prob = 0.005
    dem = surface_code_dem(rounds=3, after_clifford_depolarization=error_prob,
                           before_measure_flip_probability=error_prob, after_reset_flip_probability=error_prob)
    pcm, _ = PCM(dem)
    er = torch.from_numpy(get_error_rates(dem)).to(DTYPE)
    init_er = perturbed(er)
    dets, _, _ = dem.compile_sampler(seed=75328).sample(shots=num_shots)
    dataloader = DataLoader(TensorDataset(torch.from_numpy(dets * 1.0)), batch_size=batch_size, shuffle=True)

    tn = TensorNetwork(pcm=pcm, priors_logits=torch.logit(init_er), dtype=DTYPE)
    tn.tree = tn.find_contraction_tree(batch_size=batch_size)
    assert tn.tree is not None

    optimizer = torch.optim.Adam(tn.parameters(), lr=lr)
    mre = lambda: (abs(torch.sigmoid(tn.priors_logits.detach()) - er) / er).mean().item()
    mre_init, loss_list = mre(), []
    for epoch in range(1, epochs + 1):
        losses = []
        for syndrome_data, in dataloader:
            loss = tn(syndrome_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_list.append(np.mean(losses))
        if epoch % 10 == 0 or epoch == 1:
            print(f"[train] epoch {epoch:3d} loss {loss_list[-1]:.6f} mre {mre():.4f}")

    print(f"[train] mre {mre_init:.4f} -> {mre():.4f}, loss {loss_list[0]:.6f} -> {loss_list[-1]:.6f}")
    assert loss_list[-1] < loss_list[0]
    assert mre() < 0.5 * mre_init


def main() -> None:
    torch.manual_seed(0)
    test_exact_probabilities()
    test_circuit_level()
    test_optimization()
    print("OK")


if __name__ == "__main__":
    main()
