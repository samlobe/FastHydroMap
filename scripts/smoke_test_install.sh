#!/usr/bin/env bash
set -euo pipefail

pytest -q \
  tests/test_atom_names.py \
  tests/test_sasa.py \
  tests/test_graph_builder.py \
  tests/test_predictor_regression.py
