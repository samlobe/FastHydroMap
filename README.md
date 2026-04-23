# FastHydroMap

FastHydroMap predicts per-residue dewetting free energies (`Fdewet`) from protein structures and trajectories.

## Quick Start

Start from a fresh Python environment. FastHydroMap currently supports Python `3.11`, `3.12`, and `3.13`.

```bash
pip install fasthydromap
fasthydromap install-torch
fasthydromap predict your_structure.pdb -o outputs/your_structure_fdewet
```

`fasthydromap install-torch` defaults to the CPU build, which is usually the right choice for current FastHydroMap workloads because SASA preprocessing dominates runtime.

Advanced installation options, Docker usage, GPU Torch variants, and release/developer workflows are documented in [docs/INSTALL.md](/home/sam/Research/FastHydroMap_development/FastHydroMap/docs/INSTALL.md) and [docs/PYPI_RELEASE.md](/home/sam/Research/FastHydroMap_development/FastHydroMap/docs/PYPI_RELEASE.md).

## Inputs

FastHydroMap supports:

- Single protein structures in `PDB` format
- Protein trajectories in `DCD` or `XTC` format together with a matching topology `PDB`

Typical usage:

```bash
# Single structure
fasthydromap predict examples/1A1U.pdb -o outputs/1A1U_fdewet

# Trajectory
fasthydromap predict-trajectory examples/proteinG.pdb examples/proteinG_short.dcd -o outputs/proteinG_fdewet
```

## Outputs

For a single structure, FastHydroMap writes:

- `*.csv`: per-residue `Fdewet` predictions
- `*.pdb`: a copy of the input structure with predicted `Fdewet` written to B-factors

For a trajectory, FastHydroMap writes CSV files containing per-frame and per-residue predictions.

Residues are tracked using chain-aware residue labels, including insertion codes when present.

## Model Scope

FastHydroMap was trained on structured single-chain proteins and the 20 canonical amino-acid chemistries.
Predictions for PTMs and other non-canonical chemistries should be treated cautiously.

## Test your installation
```bash
fasthydromap predict examples/1A1U.pdb -o outputs/1A1U_fdewet
# Trajectory run (DCD or XTC)
fasthydromap predict-trajectory examples/proteinG.pdb examples/proteinG_short.dcd
```

## Acknowledgements

[Shell Lab](https://theshelllab.org/) and [Shea Group](https://labs.chem.ucsb.edu/shea/joan-emma/)
