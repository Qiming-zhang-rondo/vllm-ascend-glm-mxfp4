#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
python=python3
args=("$@")
for ((index = 0; index < ${#args[@]}; index++)); do
    if [[ "${args[index]}" == --python ]]; then
        if ((index + 1 >= ${#args[@]})); then
            echo '--python requires an executable' >&2
            exit 2
        fi
        python="${args[index + 1]}"
    elif [[ "${args[index]}" == --python=* ]]; then
        python="${args[index]#--python=}"
    fi
done
exec "$python" "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/build_qli_v2_mxfp4_a5.py" "$@"
