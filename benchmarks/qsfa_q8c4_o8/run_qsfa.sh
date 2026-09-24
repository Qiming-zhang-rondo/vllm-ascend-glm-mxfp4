#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export FLA_NPU_DISABLE_PTH=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_LAUNCH_BLOCKING=1
task_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_repo=$(cd -- "$task_dir/../.." && pwd)
task_python=${PYTHON:-python3}
task_build_only=0
task_jobs=4
task_runner=benchmarks.qsfa_q8c4_o8.compare
task_run_args=()
task_implementation=tiled
while (($#)); do
    case "$1" in
        --build-only) task_build_only=1; shift ;;
        --candidate-only) task_runner=benchmarks.qsfa_q8c4_o8.run; shift ;;
        --implementation)
            (($# >= 2)) || { echo '--implementation requires a value' >&2; exit 2; }
            task_implementation=$2; task_run_args+=("$1" "$2"); shift 2 ;;
        --implementation=*)
            task_implementation=${1#*=}; task_run_args+=("$1"); shift ;;
        --python)
            (($# >= 2)) || { echo '--python requires an executable' >&2; exit 2; }
            task_python=$2; shift 2 ;;
        --jobs)
            (($# >= 2)) || { echo '--jobs requires a value' >&2; exit 2; }
            task_jobs=$2; shift 2 ;;
        *) task_run_args+=("$1"); shift ;;
    esac
done
task_build_args=(--jobs "$task_jobs")
if [[ $task_implementation == official ]]; then
    task_build_args+=(--official)
    if [[ $task_runner == benchmarks.qsfa_q8c4_o8.run && $task_build_only == 0 ]]; then
        echo 'The official path requires its source-control case first; omit --candidate-only.' >&2
        exit 2
    fi
fi
[[ $(uname -s) == Linux ]] || { echo 'Run inside the existing A5 Linux container.' >&2; exit 2; }
task_run_dir="$task_dir/.runs/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$task_run_dir/plog"
export ASCEND_PROCESS_LOG_PATH="$task_run_dir/plog"
exec > >(tee "$task_run_dir/run.log") 2>&1
echo "Run log: $task_run_dir/run.log"
echo 'Reusing container Python/CANN/torch; no pip, downloads, OPP install or VA changes.'
cd "$task_repo"
git rev-parse HEAD
# -I excludes the script directory/PYTHONPATH. Add only this repository root;
# importing via modules also avoids the repository tools/bisect shadowing stdlib.
task_bootstrap='import pathlib,runpy,sys,types
root,module,*args=sys.argv[1:]
sys.path.insert(0,root)
# A site-package named benchmarks must not supersede our namespace directory.
package=types.ModuleType("benchmarks")
package.__path__=[str(pathlib.Path(root)/"benchmarks")]
sys.modules["benchmarks"]=package
sys.argv=[module,*args]
runpy.run_module(module,run_name="__main__")'
"$task_python" -I -c "$task_bootstrap" "$task_repo" benchmarks.qsfa_q8c4_o8.build \
    --result-file "$task_run_dir/build.json" "${task_build_args[@]}"
task_library=$("$task_python" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["library"])' "$task_run_dir/build.json")
if ((task_build_only)); then
    echo "Build complete; library: $task_library"
    exit 0
fi
"$task_python" -I -c "$task_bootstrap" "$task_repo" "$task_runner" \
    --library "$task_library" --output "$task_run_dir/results.json" "${task_run_args[@]}"
