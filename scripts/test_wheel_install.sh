#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [wheel-file]" >&2
  exit 2
fi
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/fasthydromap-wheel.XXXXXXXX")"
echo "Wheel checks and logs: ${TEST_ROOT}"
TEST_PYTHON="${PYTHON:-python}"

if [ "$#" -eq 1 ]; then
  WHEEL="$("${TEST_PYTHON}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$1")"
else
  PYTHON="${TEST_PYTHON}" "${ROOT_DIR}/scripts/build_release.sh" "${TEST_ROOT}/dist"
  WHEELS=("${TEST_ROOT}"/dist/*.whl)
  test "${#WHEELS[@]}" -eq 1
  WHEEL="${WHEELS[0]}"
fi
"${TEST_PYTHON}" - "${WHEEL}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import zipfile

wheel = Path(sys.argv[1])
if wheel.suffix != ".whl":
    raise SystemExit(f"Expected a wheel file: {wheel}")
with zipfile.ZipFile(wheel) as archive:
    names = set(archive.namelist())
    root = "FastHydroMap/weights/"
    manifest = json.loads(archive.read(root + "v2/manifest.json"))
    expected = {root + "v2/" + member["file"] for member in manifest["members"]}
    assert len(expected) == 3, "Expected three shared-model checkpoints"
    assert {name for name in names if name.startswith(root + "v2/") and name.endswith(".pt")} == expected
    assert len([name for name in names if name.startswith(root) and name.endswith(".pt")]) == 7
    assert "FastHydroMap/v2/chemistry.json" in names
    assert root + "sasa_feature_stats.npz" in names
    for member in manifest["members"]:
        digest = hashlib.sha256(archive.read(root + "v2/" + member["file"])).hexdigest()
        assert digest == member["sha256"], f"Checksum mismatch: {member['file']}"
print(f"Testing wheel SHA256: {hashlib.sha256(wheel.read_bytes()).hexdigest()}")
PY
"${TEST_PYTHON}" -m venv "${TEST_ROOT}/env"
source "${TEST_ROOT}/env/bin/activate"
python -m pip install --upgrade pip
python -m pip install "${WHEEL}"

# Run outside the source tree to catch missing package data/imports.
cd "${TEST_ROOT}"
unset PYTHONPATH
fasthydromap --help
fasthydromap install-torch --dry-run
fasthydromap install-torch --variant cpu
python -m pip check
PDB="${ROOT_DIR}/tests/data/v2_chain.pdb"
fasthydromap predict "${PDB}" -o "${TEST_ROOT}/all"
fasthydromap predict "${PDB}" --model v0 -o "${TEST_ROOT}/legacy"
fasthydromap predict-trajectory "${ROOT_DIR}/examples/proteinG.pdb" \
  "${ROOT_DIR}/examples/proteinG_short.dcd" -o "${TEST_ROOT}/trajectory"

test -f all.csv
test -f all.json
test -f all_embeddings.npz
test -f all_fdewet.pdb
test -f all_pc1.pdb
test -f all_pc2.pdb
test -f all_pc3.pdb
test -f legacy.csv
test -f legacy.pdb
test -f trajectory.npz
test -f trajectory.json
test ! -e trajectory_total.csv
test ! -e trajectory_fdewet.pdb
test ! -e trajectory_pc1.pdb
test ! -e trajectory_pc2.pdb
test ! -e trajectory_pc3.pdb

python -c 'import json, numpy as np; from FastHydroMap import selftest; selftest(); m=json.load(open("all.json")); assert m["model"]=="latest"; z=np.load("all_embeddings.npz", allow_pickle=False); assert z["embedding"].shape[1]==16'
python -c 'import numpy as np; z=np.load("trajectory.npz", allow_pickle=False); assert z["scores"].shape[2]==4; assert "embedding" not in z.files'
echo "Wheel smoke test passed. Artifacts retained in ${TEST_ROOT}"
