# Contraction trees

Each JSON file stores a precomputed tensor-network contraction tree. The five
`subsample_tree_*.json` files correspond to the five overlapping distance-5
patches.

## Main workflow

| Directory | Use |
| --- | --- |
| `d7r4/` | Simulated distance-7, `r=4` DEM-recovery experiments. |
| `d7r4_googleX_subsample/` | Representative Google X-basis `r=4` instance used for the reported patchwise alignment. |
| `d7r10_new/` | Full-`r=10` temporally tied fixed-pool validation. |
| `d7r10_googleX_subsample/` | Direct patchwise contraction trees for the Google X-basis `r=10` model. |
| `d7r10_googleX_rl_prior/` | Alternative `r=10` tree set generated from the RL-prior topology. |

## Omitted historical distance-7, r=4 tree searches

The following contraction-order experiments are not included in the public
package because no current training entry point uses them:

- `full` contracts the unpatched model.
- `nll_ca_grouped*` and `nll_ca_split*` use CA-derived tensor equations with
  grouped or split error constructions.
- `nll_google_rl*` uses tensor equations derived from the Google RL prior.
- `default` and `refined` suffixes denote alternative tree-search outputs for
  the same equations; hashes confirm that their equation files are identical.

Their names do not record every generation flag, so they should not be treated
as separately documented methods.
