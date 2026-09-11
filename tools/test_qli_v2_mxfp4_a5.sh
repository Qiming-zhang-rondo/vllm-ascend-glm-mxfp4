#!/usr/bin/env bash
set -euo pipefail

# Build this branch in an existing Ascend A5 container and execute only the
# QuantLightningIndexerV2 MXFP4 bring-up test. No model weights or pytest are
# required. Optional overrides: VA_REPO, VA_REF, VA_WORKDIR, SOC_VERSION.

VA_REPO="${VA_REPO:-https://github.com/Qiming-zhang-rondo/vllm-ascend-glm-mxfp4.git}"
VA_REF="${VA_REF:-main}"
VA_WORKDIR="${VA_WORKDIR:-/workspace/vllm-ascend-glm-mxfp4-optest}"
export SOC_VERSION="${SOC_VERSION:-ascend950dt_9582}"

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

case "$SOC_VERSION" in
    *950*) ;;
    *) echo "SOC_VERSION=$SOC_VERSION does not identify an Ascend 950/A5 target" >&2; exit 2 ;;
esac

if [ -e "$VA_WORKDIR" ]; then
    backup_path="${VA_WORKDIR}.backup.$(date +%Y%m%d%H%M%S)"
    echo "Moving the previous checkout to $backup_path"
    mv "$VA_WORKDIR" "$backup_path"
fi

git clone --depth 1 --branch "$VA_REF" "$VA_REPO" "$VA_WORKDIR"
git -C "$VA_WORKDIR" diff --check
echo "Testing commit $(git -C "$VA_WORKDIR" rev-parse HEAD)"

# Rebuild _C_ascend so the mode-5 logical dtype adapter under test comes from
# this checkout instead of the package baked into the container image.
python3 -m pip install --no-deps -e "$VA_WORKDIR"
python3 "$VA_WORKDIR/tools/test_qli_v2_mxfp4_a5.py"
