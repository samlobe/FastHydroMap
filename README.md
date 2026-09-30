# FastHydroMap

FastHydroMap predicts residue-level hydrophobicity (`Fdewet`) and water-structure
maps (`PC1`, `PC2`, `PC3`) from protein structures and trajectories.

This branch prepares **FastHydroMap v2** for release: improved training labels,
explicit histidine states, and more detailed local surface features. Models are
included; prediction needs no MD simulation, network access, or research files.

## Quick start

Install this checkout in a fresh Python 3.11–3.14 environment:

```bash
pip install -e .
fasthydromap install-torch
fasthydromap predict protein.pdb --model v2 --prepare -o results/protein
```

`--prepare` adds missing heavy atoms and saves a prepared PDB for inspection.
Omit it for already prepared structures. CPU inference is the default.
During this preview, `--model v2` selects the new models; omitting it retains
published v1 behavior. **This is not yet a published v2 release.**

## Water-structure maps

```bash
# One map
fasthydromap predict protein.pdb --model v2 --prepare --quantity pc1 -o results/pc1

# All four maps, sharing the structural calculations
fasthydromap predict protein.pdb --model v2 --prepare --quantity all -o results/maps
```

Outputs include a CSV, PDBs with predictions in the B-factor column, and a JSON
record identifying models and histidine assignments. Use `--parts` to include
the model's intrinsic and context contributions.

Histidines default to neutral HIE unless specified in the PDB or with
`--histidine-states states.csv`. This is not a pKa prediction.
See [model and input notes](docs/V2.md) for examples and limitations.

## Trajectories

Use a prepared topology whose atoms match the trajectory:

```bash
fasthydromap predict-trajectory protein.pdb trajectory.dcd --model v2 -o results/trajectory
```

DCD and XTC are supported. Output is a frame-by-residue CSV. Do not use
`--prepare` with a trajectory. Make molecules whole first; periodic-boundary
wrapping is not repaired automatically.

## Visualization

Open an output PDB in ChimeraX and color its B-factors:

```text
color bfactor range 4,6.5 palette ^lipophilicity
```

The existing [ChimeraX bundle](docs/ChimeraX_tutorial.md) supports the published
release. Its integration with this v2 preview has not yet been verified.

## Scope

The models were trained on standard single-chain proteins. Assemblies are
exploratory; predictions for buried or terminal residues require caution.
Some predictions remain sensitive to small coordinate changes. Experimental
smoothing models are not included.

See [v2 model notes](docs/V2.md), [installation](docs/INSTALL.md), and the
[release checklist](docs/V2_RELEASE.md).

## Citation

Lobo, S.; Najafi, S.; Shea, J.-E.; Shell, M. S. *Context-Aware Hydrophobicity
Modeling: HydroMap and FastHydroMap*. bioRxiv (2026).
[doi:10.64898/2026.06.07.730647](https://doi.org/10.64898/2026.06.07.730647).

[Shell Lab](https://theshelllab.org/) · [Shea Group](https://labs.chem.ucsb.edu/shea/joan-emma/)
