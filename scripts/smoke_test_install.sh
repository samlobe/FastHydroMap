#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
python -m pip check
python -m pytest -q
SMOKE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/fasthydromap-smoke.XXXXXXXX")"
fasthydromap --help >/dev/null
# Install CPU Torch beforehand: fasthydromap install-torch --variant cpu
fasthydromap predict tests/data/v2_chain.pdb -o "${SMOKE_ROOT}/all"
fasthydromap predict examples/1A1U.pdb --model v0 -o "${SMOKE_ROOT}/legacy"
echo "Smoke outputs: ${SMOKE_ROOT}"
