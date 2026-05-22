#!/usr/bin/env bash
# Окружение для инференса OpenVLA + эксперимент со сдвигом объекта
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${ROOT}/.venv"
PYTHON="${PYTHON:-/usr/local/bin/python3.11}"

echo "==> Python: ${PYTHON}"
"${PYTHON}" --version

if [[ ! -d "${VENV}" ]]; then
  "${PYTHON}" -m venv "${VENV}"
fi
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
pip install --upgrade pip setuptools wheel packaging ninja

echo "==> PyTorch 2.2 + CUDA 12.1"
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121

pip install -r "${ROOT}/requirements.txt"

echo "==> flash-attn (опционально, ~5–15 мин)"
pip install "flash-attn==2.5.5" --no-build-isolation || {
  echo "WARN: flash-attn не собрался; в ноутбуке используйте attn_implementation='sdpa'"
}

python -c "
import torch
print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"
echo "Готово: source ${VENV}/bin/activate"
