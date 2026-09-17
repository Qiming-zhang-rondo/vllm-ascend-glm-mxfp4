#!/usr/bin/env bash
set -euo pipefail

# Install the GLM SFA QLI V2 MXFP4 framework integration without modifying the
# vLLM-Ascend source tree baked into the container. The script checks out the
# exact deployment baseline in a separate directory, verifies the patch in full,
# applies it atomically, and installs that checkout in editable mode.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${QLI_PATCH_FILE:-$SCRIPT_DIR/../patches/v0.26.0-8bfdcf2fe931/qli-indexer-mxfp4.patch}"
VA_UPSTREAM_REPO="${VA_UPSTREAM_REPO:-https://github.com/vllm-project/vllm-ascend.git}"
VA_BASE_REF="${VA_BASE_REF:-v0.26.0 deployment baseline}"
VA_BASE_COMMIT="${VA_BASE_COMMIT:-8bfdcf2fe931f7d535e0e67a4e4eba233bccb598}"
VA_WORKDIR="${VA_WORKDIR:-/workspace/vllm-ascend-qli-mxfp4-v0.26.0}"
export SOC_VERSION="${SOC_VERSION:-ascend950dt_9582}"
# Deployment images may provide the AV `ascend_vllm` platform plugin in
# addition to the community `ascend` plugin. Prefer AV when present because it
# carries deployment-specific features; select exactly one to satisfy vLLM.
if [[ -z "${VLLM_PLUGINS:-}" ]]; then
    if python3 - <<'PY' >/dev/null 2>&1
from importlib.metadata import entry_points

plugins = entry_points(group="vllm.platform_plugins")
raise SystemExit(0 if any(ep.name == "ascend_vllm" for ep in plugins) else 1)
PY
    then
        export VLLM_PLUGINS=ascend_vllm
    else
        export VLLM_PLUGINS=ascend
    fi
fi
echo "Using vLLM platform plugin: $VLLM_PLUGINS"

update=0
check_only=0
while (($#)); do
    case "$1" in
        --update) update=1 ;;
        --check-only) check_only=1 ;;
        -h|--help)
            cat <<EOF
Usage: bash tools/install_glm_mxfp4_a5.sh [--update] [--check-only]

  --update      move an existing work directory aside and recreate it
  --check-only  verify the exact base and patch applicability; do not install

Environment overrides: VA_UPSTREAM_REPO, VA_BASE_REF, VA_BASE_COMMIT,
VA_WORKDIR, QLI_PATCH_FILE, SOC_VERSION.
EOF
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
test -s "$PATCH_FILE" || { echo "QLI patch not found: $PATCH_FILE" >&2; exit 1; }

case "$SOC_VERSION" in
    *950*) ;;
    *) echo "SOC_VERSION=$SOC_VERSION does not identify an Ascend 950/A5 target" >&2; exit 2 ;;
esac

if [[ -e "$VA_WORKDIR" && "$update" -eq 1 ]]; then
    backup_path="${VA_WORKDIR}.backup.$(date +%Y%m%d%H%M%S)"
    echo "Moving previous patched checkout to $backup_path"
    mv "$VA_WORKDIR" "$backup_path"
fi

if [[ ! -d "$VA_WORKDIR/.git" ]]; then
    [[ ! -e "$VA_WORKDIR" ]] || {
        echo "$VA_WORKDIR exists but is not a Git checkout; use another VA_WORKDIR or move it aside" >&2
        exit 1
    }
    mkdir -p "$VA_WORKDIR"
    git -C "$VA_WORKDIR" init -q
    git -C "$VA_WORKDIR" remote add origin "$VA_UPSTREAM_REPO"
    echo "Fetching vLLM-Ascend $VA_BASE_REF at $VA_BASE_COMMIT"
    git -C "$VA_WORKDIR" fetch --depth 1 origin "$VA_BASE_COMMIT"
    git -C "$VA_WORKDIR" checkout -q --detach FETCH_HEAD
fi

actual_commit="$(git -C "$VA_WORKDIR" rev-parse HEAD)"
if [[ "$actual_commit" != "$VA_BASE_COMMIT" ]]; then
    echo "Refusing to patch unexpected vLLM-Ascend base." >&2
    echo "Expected: $VA_BASE_COMMIT ($VA_BASE_REF)" >&2
    echo "Actual:   $actual_commit" >&2
    echo "Use --update to recreate the isolated checkout." >&2
    exit 3
fi

if git -C "$VA_WORKDIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
    echo "QLI MXFP4 patch is already applied."
else
    if [[ -n "$(git -C "$VA_WORKDIR" status --porcelain)" ]]; then
        echo "Refusing to patch a modified worktree: $VA_WORKDIR" >&2
        git -C "$VA_WORKDIR" status --short >&2
        exit 3
    fi
    git -C "$VA_WORKDIR" apply --check "$PATCH_FILE"
    if [[ "$check_only" -eq 1 ]]; then
        echo "Patch preflight passed for $VA_BASE_COMMIT; no files changed."
        exit 0
    fi
    git -C "$VA_WORKDIR" apply "$PATCH_FILE"
    git -C "$VA_WORKDIR" diff --check
    echo "Applied QLI MXFP4 patch to isolated vLLM-Ascend checkout."
fi

if [[ "$check_only" -eq 1 ]]; then
    echo "Patched checkout verified; installation skipped."
    exit 0
fi

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    set +u
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
fi
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    set +u
    # shellcheck disable=SC1091
    source /usr/local/Ascend/nnal/atb/set_env.sh
    set -u
fi

echo "Installing patched vLLM-Ascend from $VA_WORKDIR"
python3 -m pip install --no-deps --no-build-isolation -e "$VA_WORKDIR"

python3 - <<'PY'
import torch
import torch_npu
import vllm_ascend

from vllm_ascend.utils import enable_custom_op

enable_custom_op()
required = {
    "npu_quant_lightning_indexer_v2": hasattr(
        torch.ops._C_ascend, "npu_quant_lightning_indexer_v2"
    ),
    "npu_quant_lightning_indexer_v2_metadata": hasattr(
        torch.ops._C_ascend, "npu_quant_lightning_indexer_v2_metadata"
    ),
    "npu_dynamic_mx_quant": hasattr(torch_npu, "npu_dynamic_mx_quant"),
}
print("vllm_ascend:", vllm_ascend.__file__)
for name, available in required.items():
    print(f"{name}: {available}")
if not all(required.values()):
    missing = ", ".join(name for name, available in required.items() if not available)
    raise RuntimeError(f"A5 runtime is missing required MXFP4 indexer APIs: {missing}")
PY

cat <<'EOF'
QLI V2/MXFP4 framework patch and capability smoke check passed.

Keep the platform selected above in the GLM-5.2/5.3 service process. On an AV
deployment image this is normally:
  export VLLM_PLUGINS=ascend_vllm

Add this to the GLM-5.2/5.3 serve command:
  --additional-config '{"enable_sparse_li_c8":true,"sfa_indexer_quant_mode":"mxfp4"}'
EOF
