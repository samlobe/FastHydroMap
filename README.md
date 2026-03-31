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
### 1. Create the base conda environment
```bash
conda env create -f environment.yml
conda activate FastHydroMap
```
### 2. Install PyTorch + PyG wheels
Recommended (CPU):
```bash
./scripts/install_torch_pyg.sh cpu
```
Optional (NVIDIA GPU):
```bash
# CUDA 12.1-compatible wheel set
./scripts/install_torch_pyg.sh cu121

# CUDA 11.8-compatible wheel set
./scripts/install_torch_pyg.sh cu118
```
Pick the CUDA variant that is compatible with your driver/runtime (`nvidia-smi`).
For current FastHydroMap inference workloads, CPU is typically similar in end-to-end speed because preprocessing dominates runtime.

### 3. Install FastHydroMap
```bash
git clone https://github.com/samlobe/FastHydroMap.git
cd FastHydroMap
pip install -e .
```

### 4. Test your installation
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

## Acknowledgements

[Shell Lab](https://theshelllab.org/) and [Shea Group](https://labs.chem.ucsb.edu/shea/joan-emma/)
