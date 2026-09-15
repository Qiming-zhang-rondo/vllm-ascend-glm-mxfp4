#!/usr/bin/env bash
set -euo pipefail

# Set these before Python starts: .pth hooks run before Python module imports.
export FLA_NPU_DISABLE_PTH=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# Preserve an explicit setting. Default matches the user's verified container.
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-1}"
exec "${PYTHON:-python3}" "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_qbmm.py" "$@"
