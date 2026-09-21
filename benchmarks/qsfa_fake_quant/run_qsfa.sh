#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

task_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_repo=$(cd -- "$task_dir/../.." && pwd)
task_python=python3
if [[ ${1:-} == --python ]]; then
    [[ $# -ge 2 ]] || { echo 'Missing --python executable' >&2; exit 2; }
    task_python=$2
    shift 2
fi

# Disable the known FLA .pth injection before Python starts. Reuse CANN and
# custom OPP paths already active in this container; never source/install/build.
export FLA_NPU_DISABLE_PTH=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_LAUNCH_BLOCKING=1

task_output="$task_repo/.qsfa-fake-quant/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$task_output/plog"
export ASCEND_PROCESS_LOG_PATH="$task_output/plog"
echo "QSFA single-operator precision screening; existing Python: $task_python"
echo "No installs, dependency downloads, builds, or framework patches. Logs: $task_output"
"$task_python" -I -c '
import runpy, sys
sys.path.insert(0, sys.argv.pop(1))
runpy.run_module("qsfa_fake_quant.run", run_name="__main__")
' "$task_repo/benchmarks" --output "$task_output/results.json" "$@" 2>&1 | tee "$task_output/run.log"
