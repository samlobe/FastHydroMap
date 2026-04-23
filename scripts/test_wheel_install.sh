#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"
TEST_ENV="${TEST_ENV:-${ROOT_DIR}/.wheel-test}"

python -m pip install --upgrade build
python -m build

WHEEL_PATH="$(ls -1 "${DIST_DIR}"/fasthydromap-*.whl | tail -n 1)"

rm -rf "${TEST_ENV}"
python -m venv "${TEST_ENV}"
source "${TEST_ENV}/bin/activate"
python -m pip install --upgrade pip
python -m pip install "${WHEEL_PATH}"

fasthydromap --help
fasthydromap install-torch --dry-run
fasthydromap install-torch --variant cpu
OUTROOT="${TEST_ENV}/smoke_1A1U"
fasthydromap predict "${ROOT_DIR}/examples/1A1U.pdb" -o "${OUTROOT}"

test -f "${OUTROOT}.csv"
test -f "${OUTROOT}.pdb"

echo
echo "Wheel install smoke test passed in ${TEST_ENV}"
