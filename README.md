# FastHydroMap

[![PyPI version](https://img.shields.io/pypi/v/fasthydromap)](https://pypi.org/project/fasthydromap/)
[![Python versions](https://img.shields.io/pypi/pyversions/fasthydromap)](https://pypi.org/project/fasthydromap/)
[![DOI](https://zenodo.org/badge/1023802589.svg)](https://doi.org/10.5281/zenodo.19744335)

FastHydroMap predicts per-residue dewetting free energies (`Fdewet`) from protein structures and trajectories.

<p align="center">
  <img
    src="https://raw.githubusercontent.com/samlobe/FastHydroMap/main/images/FastHydroMap_image.png"
    alt="FastHydroMap overview"
    width="720"
  />
</p>

## Quick Start

Use a fresh Python environment. Python `3.11` to `3.14` are supported.

```bash
pip install fasthydromap
fasthydromap install-torch
fasthydromap predict your_structure.pdb -o outputs/your_structure_fdewet
```

`fasthydromap install-torch` defaults to the CPU build, which is usually the right choice for current FastHydroMap workloads because SASA preprocessing dominates runtime.

Advanced installation options, Docker usage, GPU Torch variants, and release workflows are documented in [INSTALL.md](https://github.com/samlobe/FastHydroMap/blob/main/docs/INSTALL.md) and [PYPI_RELEASE.md](https://github.com/samlobe/FastHydroMap/blob/main/docs/PYPI_RELEASE.md).

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

- `*.csv`: one row per residue with `Fdewet`; with `--parts`, intrinsic and context columns are included
- `*.pdb`: a copy of the input structure with predicted `Fdewet` written to B-factors

For a trajectory, FastHydroMap writes wide CSV files containing one row per frame and one column per residue.
Use `--parts` to also write intrinsic, context, and per-frame summary CSVs.

## Model Scope

FastHydroMap was trained on structured single-chain proteins and the 20 canonical amino-acid chemistries.
Predictions for PTMs and other non-canonical chemistries should be treated cautiously.

## Visualization

FastHydroMap writes `Fdewet` values to the B-factor column of output PDBs, so you can color structures directly in molecular viewers.

ChimeraX:

```bash
color bfactor range 4,6.5 palette ^lipophilicity
```

PyMOL:

```bash
spectrum b, red_white_blue, minimum=4, maximum=6.5
```

For dynamic hydrophobicity visualization in a MD trajectory, see the teaching-oriented example script
[`scripts/chimerax_fdewet_trajectory_example.py`](scripts/chimerax_fdewet_trajectory_example.py) with a ChimeraX implementation you can adjust.

## Citation

If you use FastHydroMap in your research, please cite the software release:

[![DOI](https://zenodo.org/badge/1023802589.svg)](https://doi.org/10.5281/zenodo.19744335)

Lobo, S. FastHydroMap (Version 0.1.2) [Computer software]. Zenodo.
https://doi.org/10.5281/zenodo.19744335

When the manuscript becomes available, please cite that as well.

## Acknowledgements

[Shell Lab](https://theshelllab.org/) and [Shea Group](https://labs.chem.ucsb.edu/shea/joan-emma/)
