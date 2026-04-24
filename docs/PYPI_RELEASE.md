# Releasing FastHydroMap

This project now builds standard Python release artifacts:

- `dist/fasthydromap-<version>.tar.gz`
- `dist/fasthydromap-<version>-py3-none-any.whl`

## 1. Run tests

From repo root:

```bash
conda run -n FastHydroMap311 python -m pytest -q tests
```

## 2. Build release artifacts

```bash
./scripts/build_release.sh
```

This runs:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

## 3. Smoke-test the wheel in a fresh environment

```bash
./scripts/test_wheel_install.sh
```

That script builds the project, creates a fresh venv, installs the wheel, and verifies:

- `fasthydromap --help`
- `fasthydromap install-torch --dry-run`
- `fasthydromap predict examples/1A1U.pdb -o ...`

For a manual end-to-end check in the same environment:

```bash
source .wheel-test/bin/activate
fasthydromap install-torch --variant cpu
fasthydromap predict examples/1A1U.pdb -o /tmp/1A1U_fdewet
```

## 4. Create a Git tag

Example:

```bash
VERSION=0.1.1
git tag "v${VERSION}"
git push origin "v${VERSION}"
```

## 5. Publish a GitHub Release

Create a GitHub Release for the tag and upload:

- `dist/fasthydromap-${VERSION}.tar.gz`
- `dist/fasthydromap-${VERSION}-py3-none-any.whl`

These artifacts can be installed directly with pip:

```bash
pip install "https://github.com/samlobe/FastHydroMap/releases/download/v${VERSION}/fasthydromap-${VERSION}-py3-none-any.whl"
```

## 6. Publish to PyPI

Once you are ready for public package installation:

```bash
python -m twine upload "dist/fasthydromap-${VERSION}"*
```

Then users can install with:

```bash
pip install fasthydromap
fasthydromap install-torch
```

## Notes on Torch

FastHydroMap keeps Torch out of the mandatory package dependencies so the public install path stays lightweight and avoids surprise CUDA-heavy Torch installs on Linux. The recommended user flow is:

```bash
pip install fasthydromap
fasthydromap install-torch
```

CPU is the recommended default because SASA preprocessing usually dominates runtime. GPU remains an advanced manual option:

```bash
fasthydromap install-torch --variant cu121
```
