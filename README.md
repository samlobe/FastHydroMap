# FastHydroMap

FastHydroMap predicts per-residue dewetting free energies (`Fdewet`) from protein structures and trajectories.

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

# Installation

## Recommended pip install

FastHydroMap is packaged as a normal Python project and can be installed from a wheel, source tree, or future PyPI release.

Create a fresh environment:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Once the package is published, install FastHydroMap itself with:
```bash
pip install FastHydroMap
```

If you are developing from a local checkout before publishing, use:
```bash
git clone https://github.com/samlobe/FastHydroMap.git
cd FastHydroMap
pip install .
```

Install PyTorch. CPU is the recommended default:
```bash
fasthydromap install-torch
```

Optional NVIDIA GPU install:
```bash
fasthydromap install-torch --variant cu121
```

Optional older CUDA runtime:
```bash
fasthydromap install-torch --variant cu118
```

For current FastHydroMap inference workloads, CPU is often similar in end-to-end speed because preprocessing dominates runtime. `torch_geometric` and `torch_scatter` are no longer required for inference.

## Conda / mamba install

If you prefer a fully managed scientific stack, the conda environments are still supported.

Recommended Python 3.11 environment:
```bash
mamba env create -f environment.py311.yml
conda activate FastHydroMap311
```

Legacy Python 3.10 environment:
```bash
mamba env create -f environment.yml
conda activate FastHydroMap
```

Then install Torch:
```bash
./scripts/install_torch.sh cpu
```

And install FastHydroMap from your checkout:
```bash
pip install -e . --no-deps
```

Using `--no-deps` avoids pip replacing conda-managed packages such as `openmm`.

## Build release artifacts

To build a source distribution and wheel locally:
```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

This creates files such as:
- `dist/fasthydromap-0.1.0.tar.gz`
- `dist/fasthydromap-0.1.0-py3-none-any.whl`

## Test your installation
```bash
mkdir -p outputs
fasthydromap predict examples/1A1U.pdb -o outputs/1A1U_fdewet
./scripts/smoke_test_install.sh
```

## Trajectory sanity checks

```bash
# Trajectory run (DCD or XTC)
fasthydromap predict-trajectory examples/proteinG.pdb examples/proteinG_short.dcd
```

## Docker

```bash
# CPU image (recommended)
docker build -t fasthydromap:cpu .

# Single-structure prediction
docker run --rm -u "$(id -u):$(id -g)" -v "$(pwd)/docker_out:/out" fasthydromap:cpu \
  predict /opt/FastHydroMap/examples/1A1U.pdb -o /out/1A1U_fdewet

# Trajectory prediction (total + intrinsic + context outputs)
docker run --rm -u "$(id -u):$(id -g)" -v "$(pwd)/docker_out:/out" fasthydromap:cpu \
  predict-trajectory /opt/FastHydroMap/examples/proteinG.pdb /opt/FastHydroMap/examples/proteinG_short.dcd -o /out/proteinG_fdewet_traj

# Optional CUDA 12.1 image build
docker build --build-arg TORCH_VARIANT=cu121 -t fasthydromap:cu121 .

# Optional CUDA run (requires NVIDIA container runtime)
docker run --rm --gpus all -u "$(id -u):$(id -g)" -v "$(pwd)/docker_out:/out" fasthydromap:cu121 \
  predict /opt/FastHydroMap/examples/1A1U.pdb -o /out/1A1U_fdewet
```

Docker complements the pip and conda installs; it remains the most reproducible option when you want a fully pinned runtime.

## Acknowledgements

[Shell Lab](https://theshelllab.org/) and [Shea Group](https://labs.chem.ucsb.edu/shea/joan-emma/)
