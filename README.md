# FastHydroMap

FastHydroMap predicts residue-level hydrophobicity (`Fdewet`) and water-structure
maps (`PC1`, `PC2`, `PC3`) from protein structures. It runs on a
CPU without explicit-water simulations. Lower Fdewet indicates a more hydrophobic
local surface.

![FastHydroMap overview](https://raw.githubusercontent.com/samlobe/FastHydroMap/main/images/FastHydroMap_image.png)

## Quick start

In a fresh Python environment (Python 3.11–3.14):

```bash
python -m pip install fasthydromap
fasthydromap install-torch
fasthydromap predict protein.pdb -o results/protein
```

This writes one CSV with all four scores, four PDBs with the scores in their
B-factor columns, and an NPZ file of [residue embeddings](https://github.com/samlobe/FastHydroMap/blob/main/docs/MODEL.md#residue-embeddings)
for downstream analysis.

For missing heavy atoms, add `--prepare` and inspect the saved prepared structure.

## ChimeraX

In ChimeraX, open **Tools → More Tools…**, find **FastHydroMap** in the
[Toolshed](https://cxtoolshed.rbvi.ucsf.edu/), and click **Install**.
Use bundle 0.2.1 or newer and restart ChimeraX after upgrading an existing bundle.
Run `fasthydromap install` once to set up its Python dependencies, then predict
and color a structure:

```text
fasthydromap #1
fasthydromap #1 quantity pc1
```

See the [ChimeraX tutorial](https://github.com/samlobe/FastHydroMap/blob/main/docs/ChimeraX_tutorial.md) for installation,
coloring output PDBs, and interpreting the maps.

## Trajectories

```bash
fasthydromap predict-trajectory protein.pdb trajectory.dcd -o results/trajectory
```

DCD and XTC trajectory files are supported. This writes `trajectory.npz` with all four scores
per frame and residue, plus residue IDs and frame/time information.
Add `--csv` for a wide `_total.csv`, or `--embeddings` to include per-frame
embeddings in the NPZ.
Make sure you have a complete topology with matching atom order, and make molecules whole
before prediction.
See [how to load the results](https://github.com/samlobe/FastHydroMap/blob/main/docs/MODEL.md#loading-trajectory-results).

## Inputs and scope

Use standard protein heavy atoms. Ordinary HIS defaults to neutral HIE;
HID/HIE/HIP names or `--histidine-states states.csv` specify an assignment.
FastHydroMap does not estimate pKa. See [input details](https://github.com/samlobe/FastHydroMap/blob/main/docs/MODEL.md).

Training used restrained, single-chain protein simulations. Buried residues,
termini, assemblies, and unsupported chemistry need caution.

## Documentation and citation

- [Installation](https://github.com/samlobe/FastHydroMap/blob/main/docs/INSTALL.md)
- [Python API, inputs, embeddings, and model](https://github.com/samlobe/FastHydroMap/blob/main/docs/MODEL.md)
- [Training and data](https://github.com/samlobe/FastHydroMap/blob/main/training/README.md)
- [Protein list and dataset splits](https://github.com/samlobe/FastHydroMap/blob/main/training/proteins.csv)
- [HydroMap labels, structures and training inputs](https://doi.org/10.5281/zenodo.23543197)

Lobo, S.; Najafi, S.; Shea, J.-E.; Shell, M. S. *Context-Aware Hydrophobicity
Modeling: HydroMap and FastHydroMap*. bioRxiv (2026).
[doi:10.64898/2026.06.07.730647](https://doi.org/10.64898/2026.06.07.730647).

[Shell Lab](https://theshelllab.org/) · [Shea Group](https://labs.chem.ucsb.edu/shea/joan-emma/)
