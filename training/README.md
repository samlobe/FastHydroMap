# FastHydroMap Training

This directory contains the minimal training pipeline used to build the
FastHydroMap direct MPNN weights from residue-level Fdewet targets. The same
graph/training path can also be used for PC target experiments (`PC1`, `PC2`,
`PC3`) from the metadata CSV.

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
- `03_train_mpnn_val.py`: trains on the training split and early-stops on the
  validation split.
- `04_train_mpnn_prod.py`: retrains on train+validation for production weights.
- `05_visualize_pc_targets.py`: writes a lightweight SVG/CSV report for
  inspecting PC distributions and trusted/untrusted residues.
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

## Reproduce Training

From the repository root:

```bash
python training/02_build_mpnn_graphs.py --k 12 --n-rbf 3 --rbf-min 2.0 --rbf-max 14.0 --rbf-sigma 4.0
python training/03_train_mpnn_val.py --seed 48 --report-test
python training/04_train_mpnn_prod.py --seed 48
```

If you want to skip the validation-stage JSON lookup, pass the production epoch
count explicitly:

```bash
python training/04_train_mpnn_prod.py --seed 48 --epochs 22
```

## PC Target Experiments

`data/all_residue_results.csv` includes `PC1`, `PC2`, and `PC3`. To inspect the
targets and the trusted/untrusted split:

```bash
python training/05_visualize_pc_targets.py
```

This writes local generated files under `training/figures/`.

To train the same direct MPNN architecture against `PC1`:

```bash
python training/03_train_mpnn_val.py --target PC1 --report-test
python training/04_train_mpnn_prod.py --target PC1
```

For PC targets, `--mask-source auto` uses the CSV `trusted` column. The other
available modes are `--mask-source fdewet` for the original Fdewet rule and
`--mask-source all` for every residue with a finite target value. PC experiment
weights stay under `training/models/`; the production script only copies the
default Fdewet model into the package weights directory.
