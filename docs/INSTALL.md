# Installation

FastHydroMap supports Python `3.11`, `3.12`, `3.13`, and `3.14`. Python `3.11` is the recommended default.

## Recommended pip install

Start from a fresh Python environment, then run:

```bash
pip install fasthydromap
fasthydromap install-torch
```

`fasthydromap install-torch` defaults to the CPU PyTorch build, which is usually the right default because SASA preprocessing dominates runtime for current FastHydroMap workloads.

Run a first prediction:

```bash
fasthydromap predict your_structure.pdb -o outputs/your_structure_fdewet
```

If you want to try a bundled repo example from a local checkout:

```bash
fasthydromap predict examples/1A1U.pdb -o outputs/1A1U_fdewet
```

## Optional GPU Torch setup

Optional NVIDIA GPU install:

```bash
fasthydromap install-torch --variant cu121
```

Optional older CUDA runtime:

```bash
fasthydromap install-torch --variant cu118
```

## Conda / mamba install

If you prefer a conda-managed scientific stack:

```bash
mamba create -n fasthydromap python=3.11 pip
conda activate fasthydromap
pip install fasthydromap
fasthydromap install-torch
```

For local development from this repository:

```bash
git clone https://github.com/samlobe/FastHydroMap.git
cd FastHydroMap
pip install -e . --no-deps
fasthydromap install-torch
```

Using `--no-deps` helps avoid replacing conda-managed scientific packages.

For development and Docker reproducibility, the repository also includes
`environment.dev.yml`. That file is not the recommended user install path.

## Docker

CPU image:

```bash
docker build -t fasthydromap:cpu .
docker run --rm -u "$(id -u):$(id -g)" -v "$(pwd)/docker_out:/out" fasthydromap:cpu \
  predict /opt/FastHydroMap/examples/1A1U.pdb -o /out/1A1U_fdewet
```

Trajectory prediction:

```bash
docker run --rm -u "$(id -u):$(id -g)" -v "$(pwd)/docker_out:/out" fasthydromap:cpu \
  predict-trajectory /opt/FastHydroMap/examples/proteinG.pdb /opt/FastHydroMap/examples/proteinG_short.dcd -o /out/proteinG_fdewet_traj
```

Optional CUDA 12.1 image:

```bash
docker build --build-arg TORCH_VARIANT=cu121 -t fasthydromap:cu121 .
docker run --rm --gpus all -u "$(id -u):$(id -g)" -v "$(pwd)/docker_out:/out" fasthydromap:cu121 \
  predict /opt/FastHydroMap/examples/1A1U.pdb -o /out/1A1U_fdewet
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
