# PatchDMLE research code

Run all commands below from this directory.

## Layout

- `src/`: tensor-network likelihoods, DEM utilities, and decoders.
- `script/`: DEM alignment, temporal processing, and contraction-tree entry points.
- `nn_training/`: Transformer decoder training and the DEMs used in the paper.
- `path/`: production contraction trees used by the likelihood calculations.
- `tn_eqns/`: tensor equations corresponding to the contraction trees.

Raw experimental data, generated samples, results, logs, and training
checkpoints are not stored in Git.

## Environment

Python 3.11 is recommended. The reference environment uses PyTorch 2.9.1 with
CUDA 13.0. On a compatible Linux GPU machine, install PyTorch from its CUDA
wheel index and then install the remaining dependencies:

```bash
python -m pip install torch==2.9.1+cu130 torchvision==0.24.1+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

Use the corresponding official PyTorch wheel index if the target machine uses
a different CUDA version, and adjust the two PyTorch pins in
`requirements.txt` accordingly.

Generating new contraction trees additionally requires Julia and
`OMEinsumContractionOrders.jl`. The production trees needed for the reported
experiments are already included in `path/`.

## Alignment scripts

| File | Purpose |
| --- | --- |
| `generate_d7_subsample_trees.py` | Export five patch equations and optimize their contraction trees. |
| `run_tn.jl` | Julia/TreeSA helper called by the tree generator. |
| `prepare_google_broadcast_source_round.py` | Construct the representative distance-7, `r=4` experimental instance. |
| `train_d3_google_dmle.py` | Full-likelihood alignment for the distance-3 Sycamore data. |
| `train_d7_google_subsample.py` | Shared Google-data loader and direct patchwise alignment routine. |
| `train_d7_google_broadcast_tesseract.py` | Main distance-7 experimental alignment, broadcast, and Tesseract evaluation. |
| `train_d7_subsample.py` | Base simulated five-patch alignment routine. |
| `train_d7_fixed_pool_sampling.py` | Fixed-data experiments used to study training-data volume. |
| `train_d7_online_sampling.py` | Online-sampling recovery from a perturbed known DEM. |
| `train_d7r10_tied_fixed_pool.py` | Full-`r=10` temporal-tying validation with a fixed sample pool. |

The old decoder comparisons, plotting diagnostics, and test scripts are not
needed to run the reported workflow and are not included.

Run a lightweight source check with:

```bash
python -m compileall -q src script nn_training
```

## Experimental data

The public datasets are not duplicated in this repository:

- Sycamore surface-code data: [Zenodo 11403595](https://doi.org/10.5281/zenodo.11403595).
- Willow surface-code data: [Zenodo 13273331](https://doi.org/10.5281/zenodo.13273331).
- Willow color-code data: [Zenodo 14238944](https://doi.org/10.5281/zenodo.14238944).

Place the extracted files under `dataset/` or pass their locations explicitly.
The main scripts also recognize:

- `PATCHDMLE_SYCAMORE_DATA`: root of the Sycamore distance-3/5 datasets.
- `PATCHDMLE_WILLOW_DATA`: root containing `d7_at_q6_7/X/r10`.
- `PATCHDMLE_COLOR_DATA`: root containing the color-code `d3X` and `d5X` data.

## Quickstart: Google distance-7 data

The reported scalable workflow starts from the Google distance-7, `r=10`
experiment, retains a representative `r=4` instance for likelihood alignment,
and broadcasts the optimized bulk parameters back to `r=10`.

First point the repository to the extracted Willow dataset and verify the
expected layout:

```bash
export PATCHDMLE_WILLOW_DATA=/absolute/path/to/google_willow
test -f "$PATCHDMLE_WILLOW_DATA/d7_at_q6_7/X/r10/detection_events.b8"
```

Construct the representative instance from the `r=10` records:

```bash
python script/prepare_google_broadcast_source_round.py \
  --data_root="$PATCHDMLE_WILLOW_DATA/d7_at_q6_7" \
  --basis=X \
  --source_round_tag=r10 \
  --target_round_tag=r04 \
  --bulk_layer=2
```

Then optimize the mean five-patch likelihood and broadcast the fitted DEM back
to `r=10`:

```bash
python script/train_d7_google_broadcast_tesseract.py \
  --train_round_root dataset/google_broadcast_source/d7_at_q6_7_bulk2/X/r04 \
  --eval_round_root "$PATCHDMLE_WILLOW_DATA/d7_at_q6_7/X/r10" \
  --path_dir path/d7r4_googleX_subsample \
  --dev cuda:0 \
  --epochs 250 \
  --batch_size 1024 \
  --mini_batch 256 \
  --decode_interval 10
```

Reduce `--mini_batch` if the contraction exceeds available GPU memory; this
does not change the optimizer batch size. The run is written under
`data/google/d7r4_googleX_broadcast_tesseract/<timestamp>/`. The final
`*_final_epoch*.pt` file contains the optimized DEM parameters and complete
training state.

The five required contraction trees are already provided in
`path/d7r4_googleX_subsample/`. No Julia installation is needed unless they are
regenerated.

The full-length `r=10` patchwise objective can also be optimized directly:

```bash
python script/train_d7_google_subsample.py \
  --data_root="$PATCHDMLE_WILLOW_DATA/d7_at_q6_7" \
  --basis=X \
  --round_tag=r10 \
  --path_dir=path/d7r10_googleX_subsample \
  --dev=cuda:0
```

This path uses the five contraction trees in
`path/d7r10_googleX_subsample/` and has higher memory and computation costs
than the representative-`r=4` workflow.

### Temporal parameter tying

In `train_d7r10_tied_fixed_pool.py`, `tied` means that error mechanisms
related by translation across repeated bulk rounds share one trainable
parameter. The five-patch objective is still evaluated on the complete `r=10`
detector sequence, but its 8,137 error probabilities are gathered from the
2,809 parameters of the representative `r=4` DEM. Boundary parameters remain
distinct.

Thus, the three distance-7 configurations differ as follows:

- **Representative `r=4` and broadcast:** optimize the reduced `r=4`
  likelihood, then broadcast the fitted bulk parameters to `r=10`.
- **Direct full `r=10`:** optimize the full-length five-patch objective with
  independently trainable full-DEM parameters.
- **Tied full `r=10`:** optimize the same full-length objective while sharing
  parameters among time-translated bulk mechanisms.

The current tied script performs this validation on a fixed synthetic sample
pool; `fixed_pool` refers to reuse of that finite training dataset and is
independent of parameter tying.

The fixed-pool and online-sampling recovery experiments are implemented in
`script/train_d7_fixed_pool_sampling.py` and
`script/train_d7_online_sampling.py`. Neural-decoder commands and model settings
are documented in `nn_training/README.md`.
