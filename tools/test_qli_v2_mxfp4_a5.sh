#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

# Also works when downloaded alone as test_qli.sh. Fetch test source only;
# never install Python packages or download build dependencies.
# Preserve the old VA_* overrides for existing users of this launcher.
task_repo_url=${VA_REPO:-https://github.com/Qiming-zhang-rondo/vllm-ascend-glm-mxfp4.git}
task_ref=${VA_REF:-main}
task_workdir=${VA_WORKDIR:-/workspace/vllm-ascend-glm-mxfp4-optest}
task_python=python3
task_update=0
task_build_op=0
task_installed_only=0
task_check_only=0
task_cann_root=
task_opapi_lib=
task_soc=${SOC_VERSION:-ascend950dt_9582}
task_json_include=
task_jobs=
task_used_cache=0
task_args=()

usage() {
    cat <<'EOF'
Usage: bash test_qli_v2_mxfp4_a5.sh [launcher options] [benchmark options]

Run inside the existing A5 container; do not create a container or install VA.
First execute a real mode-5 C4 compute probe. If the operator is unavailable,
build only QLI V2 and metadata using existing local CANN build dependencies.
Missing build dependencies cause a diagnostic error, never a download.

Launcher options:
  --installed-only       Probe/test existing operators; never build an operator
  --build-op             Build the bundled C4 operator before testing
  --check-only           Only probe actual C4 compute support
  --update               Fast-forward this test checkout from its configured repo
  --workdir PATH         Checkout to reuse when this script was downloaded alone
  --python PATH          Container Python interpreter (default: python3)
  --cann-root PATH       Existing CANN toolkit root
  --opapi-lib PATH       Use this exact existing/custom op-api shared library
  --soc NAME             Build target (default: SOC_VERSION or ascend950dt_9582)
  --json-include PATH    Existing include directory containing nlohmann/json.hpp
  --jobs N               Operator build parallelism
  --help                Show this message

Benchmark options are forwarded to the Python runner, for example:
  --query-tokens 1 --key-tokens 8192 --warmup 5 --iterations 20 --output result.json
  --prefill-tokens 57344 --chunk-size 8192 --reference-rows 16 --output prefill_56k.json

Prefill runs the full prompt in causal Q chunks with growing K prefixes.
Numeric reference checks sample rows; QLI timing excludes preparation and metadata.
For a downloaded launcher, disable FLA's Python startup hook before invocation:
  FLA_NPU_DISABLE_PTH=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 bash test_qli.sh --update ...

Within a checkout no code is fetched unless --update is supplied.
Existing checkouts and local edits are never moved or deleted.
EOF
}

while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --update) task_update=1; shift ;;
        --build-op) task_build_op=1; shift ;;
        --installed-only) task_installed_only=1; shift ;;
        --check-only) task_check_only=1; shift ;;
        --workdir|--python|--cann-root|--opapi-lib|--soc|--json-include|--jobs)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --workdir) task_workdir=$2 ;;
                --python) task_python=$2 ;;
                --cann-root) task_cann_root=$2 ;;
                --opapi-lib) task_opapi_lib=$2; task_installed_only=1 ;;
                --soc) task_soc=$2 ;;
                --json-include) task_json_include=$2 ;;
                --jobs) task_jobs=$2 ;;
            esac
            shift 2 ;;
        *) task_args+=("$1"); shift ;;
    esac
done

if ((task_build_op && task_installed_only)); then
    echo '--build-op cannot be combined with --installed-only or --opapi-lib.' >&2
    exit 2
fi
[[ $(uname -s) == Linux ]] || { echo 'Run this script in the existing A5 Linux container.' >&2; exit 2; }
command -v "$task_python" >/dev/null || { echo "Python is unavailable: $task_python" >&2; exit 2; }

task_script_dir=
if [[ -n ${BASH_SOURCE[0]:-} ]]; then
    task_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fi
if [[ -n "$task_script_dir" && -f "$task_script_dir/test_qli_v2_mxfp4_a5.py" ]]; then
    task_checkout=$(cd -- "$task_script_dir/.." && pwd)
else
    task_checkout=$task_workdir
    if [[ ! -e "$task_checkout" ]]; then
        command -v git >/dev/null || { echo 'git is required to fetch the test source.' >&2; exit 2; }
        git clone --depth 1 --branch "$task_ref" "$task_repo_url" "$task_checkout"
        task_update=0
    elif [[ ! -e "$task_checkout/.git" ]]; then
        echo "Existing path is not a Git checkout: $task_checkout; choose --workdir." >&2
        exit 2
    fi
fi

if ((task_update)); then
    if [[ -n $(git -C "$task_checkout" status --porcelain --untracked-files=no) ]]; then
        echo 'Checkout has local changes; leaving them intact. Update or use a separate --workdir.' >&2
        exit 2
    fi
    git -C "$task_checkout" pull --ff-only "$task_repo_url" "$task_ref"
fi
if [[ ! -f "$task_checkout/tools/qli_container_backend.py" ]]; then
    echo "Old installer-based test checkout: $task_checkout. Rerun with --update." >&2
    exit 2
fi
cd -- "$task_checkout"
mkdir -p .qli-op-build
task_log="qli_a5_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee "$task_log") 2>&1
echo "Using existing container Python: $task_python"
echo "Test source: $task_checkout"
if [[ -e .git ]]; then git rev-parse HEAD; fi
echo 'No pip install, dependency downloads, or container/image operations are performed.'

if [[ -z "$task_cann_root" && -z ${ASCEND_HOME_PATH:-} ]]; then
    for task_env in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/cann/set_env.sh; do
        if [[ -f "$task_env" ]]; then
            set +u
            # shellcheck disable=SC1090
            source "$task_env"
            set -u
            break
        fi
    done
fi
task_backend_args=()
if [[ -n "$task_cann_root" ]]; then task_backend_args+=(--cann-root "$task_cann_root"); fi
if [[ -n "$task_opapi_lib" ]]; then task_backend_args+=(--opapi-lib "$task_opapi_lib"); fi
task_build_args=(--python "$task_python" --soc "$task_soc")
if [[ -n "$task_cann_root" ]]; then task_build_args+=(--cann-root "$task_cann_root"); fi
if [[ -n "$task_json_include" ]]; then task_build_args+=(--json-include "$task_json_include"); fi
if [[ -n "$task_jobs" ]]; then task_build_args+=(--jobs "$task_jobs"); fi

use_operator_manifest() {
    task_opapi_lib=$("$task_python" -c 'import json; print(json.load(open(".qli-op-build/install.json"))["opapi_lib"])')
    task_opp_root=$("$task_python" -c 'import json; print(json.load(open(".qli-op-build/install.json"))["opp_root"])')
    [[ -f "$task_opapi_lib" && -d "$task_opp_root" ]] || { echo 'Build manifest points to missing outputs.' >&2; return 2; }
    export ASCEND_CUSTOM_OPP_PATH="$task_opp_root${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
    export LD_LIBRARY_PATH="$(dirname -- "$task_opapi_lib")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    task_backend_args+=(--opapi-lib "$task_opapi_lib")
}

build_operator() {
    bash tools/build_qli_v2_mxfp4_a5.sh "${task_build_args[@]}"
    use_operator_manifest
}

if ((task_build_op)); then build_operator; fi
if ((!task_build_op && !task_installed_only)) && [[ -f .qli-op-build/install.json ]]; then
    if bash tools/build_qli_v2_mxfp4_a5.sh "${task_build_args[@]}" --reuse-only > .qli-op-build/reuse.log 2>&1; then
        use_operator_manifest
        task_used_cache=1
        echo "Reuse matching private QLI build: $task_opapi_lib"
    else
        echo 'Previous private build is not reusable; probing the container. Details: .qli-op-build/reuse.log'
    fi
fi
task_probe_status=0
"$task_python" tools/test_qli_v2_mxfp4_a5.py ${task_backend_args[@]+"${task_backend_args[@]}"} ${task_args[@]+"${task_args[@]}"} \
    --check-only --output .qli-op-build/probe-installed.json || task_probe_status=$?
if ((task_probe_status == 78 && !task_installed_only && !task_build_op && !task_used_cache)); then
    echo 'Existing CANN did not pass the C4 compute probe; building the bundled operator offline.'
    build_operator
    task_probe_status=0
    "$task_python" tools/test_qli_v2_mxfp4_a5.py ${task_backend_args[@]+"${task_backend_args[@]}"} ${task_args[@]+"${task_args[@]}"} \
        --check-only --output .qli-op-build/probe-custom.json || task_probe_status=$?
fi
if ((task_probe_status)); then
    echo "C4 compute probe failed (exit $task_probe_status). See $task_log and .qli-op-build/probe-*.json."
    exit "$task_probe_status"
fi
if ((task_check_only)); then exit 0; fi
"$task_python" tools/test_qli_v2_mxfp4_a5.py ${task_backend_args[@]+"${task_backend_args[@]}"} ${task_args[@]+"${task_args[@]}"}
