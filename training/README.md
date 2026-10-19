# Training and reproducibility

The current model is one shared 16-dimensional residue encoder with four output
heads, trained as a three-seed ensemble. The release package contains inference
code and weights; multi-GB training inputs live in separate archives.

## Retrain the model

This workflow starts from precomputed training inputs rather than rebuilding
them from PDB structures and simulations.

1. Install this checkout and PyTorch, as in [installation](../docs/INSTALL.md).
2. Obtain `shared_corrected_full800_inputs.zip`, listed in
   [DATA_ACCESS.json](DATA_ACCESS.json).
   **Public deposition is pending:** there is no download link or DOI yet.
3. Extract `full800.pt` and `full800.json`, then run:

```bash
python training/train_shared.py --cache /path/to/full800.pt --check-only
python training/train_shared.py --cache /path/to/full800.pt --output runs/shared
```

The script checks the training data and trains the three-network ensemble on
the designated 800 proteins using the saved protocol. It does not evaluate
held-out performance or replace the packaged weights. Training defaults to CPU;
the final refits use a GPU. With CUDA-enabled PyTorch, add `--device cuda` to
match that setup. Results may differ slightly across devices and PyTorch versions.

For an implementation smoke test, use `--seeds 17 --smoke-steps 1` with a
new output directory. Such checkpoints are explicitly marked **smoke_test_only**,
not release weights. An existing output directory is never overwritten.

[shared_protocol.json](shared_protocol.json) records normalization, loss,
optimizer, fit IDs and preselected schedules. [development_split.json](development_split.json)
records the historical 500-fit / 100-tune / 200-audit development split.
Final refitting uses all 800 original training proteins with fixed schedules;
normalization is recomputed on those 800. The separate 100 validation and 100
test proteins are excluded from both fitting and this cache.

The label corpus contains 1,000 protein chains, each with three independently
seeded restrained explicit-water replicas.

Fdewet fitting uses nonterminal residues with mean hydration count >7 and
replica-mean Fdewet in [3.8, 8.7]. PC fitting uses the hydration/terminal mask
without a Fdewet filter, with fit-only 2nd/98th-percentile winsorization.
Inference values are not numerically clipped.

## Evaluation and coordinate sensitivity

The original validation and test cohorts contain 100 proteins each. Scores
below use raw three-replica mean labels within the fitting eligibility masks.

| Target | Validation RMSE | Validation ρ | Test RMSE | Test ρ |
| --- | ---: | ---: | ---: | ---: |
| Fdewet | 0.450 | 0.811 | 0.429 | 0.821 |
| PC1 | 3.470 | 0.856 | 3.215 | 0.864 |
| PC2 | 2.561 | 0.795 | 2.471 | 0.809 |
| PC3 | 0.871 | 0.843 | 0.822 | 0.849 |

On a fixed 30-protein training panel, 0.001 Å Gaussian coordinate jitter plus
PDB rounding changed Fdewet by 0.045 on average and 0.146 at the 95th percentile.
Mean within-protein rank correlation was 0.992. Discrete probe sampling means
predictions are not smooth functions of coordinates.

## Previous model

The original per-target MPNN scripts and data remain available. See
[legacy training](LEGACY.md); those are not the current shared-model recipe.
