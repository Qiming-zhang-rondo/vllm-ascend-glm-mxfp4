#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Source this in the service shell before starting Python/vLLM. Never rebuilds.
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    echo 'Use: source tools/activate_qli_mxfp4_a5.sh [--manifest PATH | --opapi-lib PATH]' >&2
    exit 2
fi

_qli_activate_existing_runtime() {
    local qli_script_dir qli_exports
    qli_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || return 1
    # -I -S avoids Python site injection and tools/bisect stdlib shadowing.
    qli_exports=$(python3 -I -S "$qli_script_dir/qli_runtime_env.py" resolve "$@") || return 1
    # Check in a new process with the proposed loader paths, without changing
    # the caller's environment if the library or one of its exports is missing.
    (
        eval "$qli_exports"
        python3 -I -S "$qli_script_dir/qli_runtime_env.py" check "$@"
    ) || return 1
    eval "$qli_exports"
    echo 'QLI runtime environment is ready for a NEW service process; no compilation or installation performed.'
}

if _qli_activate_existing_runtime "$@"; then
    unset -f _qli_activate_existing_runtime
else
    unset -f _qli_activate_existing_runtime
    return 1
fi
