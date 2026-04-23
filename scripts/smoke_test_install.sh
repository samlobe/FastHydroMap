#!/usr/bin/env bash
set -euo pipefail

pytest -q \
  tests/test_atom_names.py \
  tests/test_sasa.py \
  tests/test_graph_builder.py \
  tests/test_predictor_regression.py \
  tests/test_install_torch.py

OUTDIR="${OUTDIR:-./outputs/smoke_test_install}"
mkdir -p "${OUTDIR}"
fasthydromap --help >/dev/null
fasthydromap install-torch --variant cpu
fasthydromap predict examples/1A1U.pdb -o "${OUTDIR}/1A1U_fdewet"
