#!/usr/bin/env bash
set -euo pipefail

echo "[install] install_torch_pyg.sh is deprecated; FastHydroMap inference no longer requires PyG."
"$(dirname "$0")/install_torch.sh" "$@"
