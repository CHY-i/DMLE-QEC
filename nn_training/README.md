# Neural-decoder training

This directory contains the Transformer decoder used in the paper and the
four DEM conditions compared in each code setting.

## Files

| File | Purpose |
| --- | --- |
| `model.py` | Recurrent Transformer, 2D RoPE attention, and logical readout. |
| `surface_utils.py` | DEM sampling, experimental-data loading, and detector layouts. |
| `train_surface_dem.py` | Shared online-training entry point for all four DEM conditions. |
| `evaluate_surface_model.py` | Evaluate a saved neural-decoder checkpoint. |
| `create_calibrated_si1000_dem.py` | Calibrate an SI1000 DEM to detector-event density. |
| `create_color_experimental_ca_dem.py` | Reconstruct a color-code CA DEM from experimental detections. |
| `create_stim_surface_dem.py` | Generate a generic surface-code DEM with Stim. |
| `export_dmle_checkpoint_dem.py` | Export a full-likelihood alignment checkpoint as a DEM. |
| `export_patchdmle_checkpoint_dem.py` | Export and optionally broadcast a patchwise distance-7 checkpoint. |
| `run_scripts/` | Reproducible launch commands for the reported code settings. |

The former `tests/`, bootstrap-comparison, and X/Z-dual diagnostic scripts are
not part of the paper workflow and are not included in the public package.

## DEMs

The public DEMs follow one layout throughout:

```text
dems/
  surface/{d3_r5,d5_r5,d7_r10}/{si1000,ca,rl,aligned}.dem
  color/{d3_r5,d5_r5}/{si1000,ca,rl,aligned}.dem
```

Each name identifies the DEM used to sample neural-decoder training data.
`aligned.dem` is the likelihood-aligned model studied in this work.

## Training protocol

- X basis.
- Seed `71534`.
- Online sampling: every optimization step uses a newly sampled batch.
- Global batch size `1024` for the reported runs.
- `d_model=256`, `nhead=8`, `num_layers=3`, and `dropout=0.1`.
- AdamW with learning rate `1e-4`, minimum learning rate `1e-6`, weight decay
  `0.01`, and gradient clipping at `1.0`.

Within each code setting, all four runs use the same network initialization and
training hyperparameters. Dataset locations are supplied through
`PATCHDMLE_SYCAMORE_DATA`, `PATCHDMLE_WILLOW_DATA`, or
`PATCHDMLE_COLOR_DATA`.

The reported surface-code runs use a common warm-start checkpoint within each
code setting. Set `PRETRAINED=/path/to/checkpoint` when invoking a launch
script to reproduce that initialization; omitting it starts all four conditions
from the same random initialization. The color-code runs use random
initialization. Set `STEPS` or `BATCH_SIZE` explicitly for shorter smoke runs.

For example, run the four distance-5 surface-code conditions with:

```bash
bash nn_training/run_scripts/run_surface_d5_r5.sh si1000 0
bash nn_training/run_scripts/run_surface_d5_r5.sh ca 1
bash nn_training/run_scripts/run_surface_d5_r5.sh rl 2
bash nn_training/run_scripts/run_surface_d5_r5.sh aligned 3
```
