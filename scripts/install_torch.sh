#!/usr/bin/env bash
set -euo pipefail

variant="${1:-cpu}"          # cpu | cu118 | cu121
torch_version="${TORCH_VERSION:-2.2.2}"
torchvision_version="${TORCHVISION_VERSION:-0.17.2}"
torchaudio_version="${TORCHAUDIO_VERSION:-2.2.2}"

case "${variant}" in
  cpu)
    torch_index_url="https://download.pytorch.org/whl/cpu"
    torch_tag="torch-${torch_version}+cpu"
    torch_pkg="torch==${torch_version}"
    ;;
  cu118|cu121)
    torch_index_url="https://download.pytorch.org/whl/${variant}"
    torch_tag="torch-${torch_version}+${variant}"
    torch_pkg="torch==${torch_version}+${variant}"
    ;;
  *)
    echo "Unknown variant '${variant}'. Expected: cpu, cu118, or cu121." >&2
    exit 2
    ;;
esac

echo "[install] PyTorch ${torch_pkg} from ${torch_index_url}"
python -m pip install --upgrade pip
python -m pip install \
  "${torch_pkg}" \
  "torchvision==${torchvision_version}" \
  "torchaudio==${torchaudio_version}" \
  --index-url "${torch_index_url}"

echo "[install] installed torch runtime for ${torch_tag}"
