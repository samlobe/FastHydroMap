# FastHydroMap Training

This directory contains the minimal training pipeline used to build the
FastHydroMap direct MPNN weights from residue-level `Fdewet`, `PC1`, `PC2`, or
`PC3` targets.

The large graph tensors and checkpoints are generated artifacts and are not
stored in Git. They can be rebuilt from the scripts, CSV metadata, and source
PDB structures.

## Files

- `data/all_residue_results.csv`: residue-level target and feature table.
- `data/splits.yaml`: PDB-level train/validation/test split.
- `01_build_sasa.py`: recomputes residue SASA features and SASA normalization
  statistics from the source PDB structures.
- `02_build_mpnn_graphs.py`: converts the CSV and PDB structures into cached
  PyTorch Geometric graph tensors.
- `03_train_mpnn.py`: trains validation-stage or production MPNN weights.
- `train_mpnn_common.py`: shared dataset, model, optimizer, and evaluation
  helpers.
- `residue_keys.py`: stable residue identifiers for chains and insertion codes.

## Required Data

Download the source PDB files to `training/data/rcsb_pdbs/`.

## Production Configuration

- `k_nn=12`
- `n_rbf=3`
- `rbf_min=2.0`
- `rbf_max=14.0`
- `rbf_sigma=4.0`
- `hidden=24`
- `depth=2`
- `head_hidden=20`
- trust mask: `avg_n_waters > 7.0` and `3.8 <= Fdewet_pred <= 8.7`

## Reproduce Fdewet Training

From the repository root:

```bash
python training/02_build_mpnn_graphs.py --k 12 --n-rbf 3 --rbf-min 2.0 --rbf-max 14.0 --rbf-sigma 4.0
python training/03_train_mpnn.py --stage val --seed 48 --report-test
python training/03_train_mpnn.py --stage prod --seed 48 --copy-to-package
```

If you want to skip the validation-stage JSON lookup, pass the production epoch
count explicitly:

```bash
python training/03_train_mpnn.py --stage prod --seed 48 --epochs 22 --copy-to-package
```

## PC Target Training

`data/all_residue_results.csv` includes `PC1`, `PC2`, and `PC3`. Use
`--target` to train the same architecture against a PC target:

```bash
python training/03_train_mpnn.py --stage val --target PC1 --report-test
python training/03_train_mpnn.py --stage prod --target PC1 --copy-to-package
```

For PC targets, `--mask-source auto` uses the CSV `trusted` column. The released
PC weights were trained with target winsorization at the 2nd and 98th
percentiles of the trusted fitting split:

```bash
python training/03_train_mpnn.py --stage val --target PC1 --winsor-lower 0.02 --winsor-upper 0.98 --report-test
python training/03_train_mpnn.py --stage prod --target PC1 --winsor-lower 0.02 --winsor-upper 0.98 --copy-to-package
```

Repeat with `--target PC2` or `--target PC3` for the other PC regressors. The
production stage writes a target-specific local weight under `training/models/`.
When `--copy-to-package` is supplied, the package weight is updated at:

```bash
src/FastHydroMap/weights/mpnn_latest.pt      # Fdewet_pred
src/FastHydroMap/weights/mpnn_pc1_latest.pt  # PC1
src/FastHydroMap/weights/mpnn_pc2_latest.pt  # PC2
src/FastHydroMap/weights/mpnn_pc3_latest.pt  # PC3
```
