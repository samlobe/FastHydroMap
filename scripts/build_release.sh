#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [empty-output-directory]" >&2
  exit 2
fi
BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/fasthydromap-build.XXXXXXXX")"
BUILD_PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${1:-${BUILD_ROOT}/dist}"
"${BUILD_PYTHON}" - "${OUTPUT_DIR}" <<'PY'
from pathlib import Path
import sys

directory = Path(sys.argv[1])
if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
    raise SystemExit(f"Output directory must be empty: {directory}")
directory.mkdir(parents=True, exist_ok=True)
PY
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

# Build tools stay out of the caller's Python environment.
"${BUILD_PYTHON}" -m venv "${BUILD_ROOT}/tools"
BUILD_PYTHON="${BUILD_ROOT}/tools/bin/python"
"${BUILD_PYTHON}" -m pip install --upgrade pip build twine
"${BUILD_PYTHON}" -m build "${ROOT_DIR}" --outdir "${OUTPUT_DIR}"
"${BUILD_PYTHON}" -m twine check "${OUTPUT_DIR}"/*.whl "${OUTPUT_DIR}"/*.tar.gz
echo "Release artifacts (not uploaded): ${OUTPUT_DIR}"
