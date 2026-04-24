#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*

echo
echo "Release artifacts are ready in dist/"
