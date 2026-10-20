# Installation

Python 3.12 is recommended. CI covers Python 3.11–3.14 on Linux and macOS,
plus the test suite on Windows with Python 3.12.

## Recommended pip install

Start from a fresh Python environment, then run:

```bash
pip install fasthydromap
fasthydromap install-torch
```

`fasthydromap install-torch` defaults to CPU PyTorch. Structural feature calculation
dominates inference time. Four CPU workers are the default; a GPU is not required.

Run a first prediction:

```bash
fasthydromap predict your_structure.pdb -o outputs/your_structure
```

If you want to try a bundled repo example from a local checkout:

```bash
fasthydromap predict examples/1A1U.pdb -o outputs/1A1U
```

The latest model exports all four maps together: one CSV, four quantity-suffixed
B-factor PDBs, one residue-embedding NPZ, and a provenance JSON.

The bundled predictor runs on CPU even when CUDA PyTorch is installed. GPU
training is supported by the [training recipe](../training/README.md), but CUDA
is not needed for prediction.

## Conda / mamba install

If you prefer a conda-managed scientific stack:

```bash
mamba create -n fasthydromap python=3.12 pip
conda activate fasthydromap
pip install fasthydromap
fasthydromap install-torch
```

For local development from this repository:

```bash
git clone https://github.com/samlobe/FastHydroMap.git
cd FastHydroMap
pip install -e .
fasthydromap install-torch
```

Use `--no-deps` only if all dependencies listed in `pyproject.toml` are already
installed; it does not create a complete environment on its own.

For development and Docker reproducibility, the repository also includes
`environment.dev.yml`. That file is not the recommended user install path.

## Docker

CPU image:

```bash
docker build -t fasthydromap:cpu .
docker run --rm -u "$(id -u):$(id -g)" -v "$(pwd)/docker_out:/out" fasthydromap:cpu \
  predict /opt/FastHydroMap/examples/1A1U.pdb -o /out/1A1U
```

Trajectory prediction:

```bash
docker run --rm -u "$(id -u):$(id -g)" -v "$(pwd)/docker_out:/out" fasthydromap:cpu \
  predict-trajectory /opt/FastHydroMap/examples/proteinG.pdb /opt/FastHydroMap/examples/proteinG_short.dcd -o /out/proteinG_maps_traj
```

## Local packaging checks

To build release artifacts locally:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

You can also run the smoke-test helper from a local checkout:

```bash
./scripts/smoke_test_install.sh
```
