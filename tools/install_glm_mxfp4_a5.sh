#!/usr/bin/env bash
set -euo pipefail

# Clone this patched vLLM-Ascend tree, rebuild its extension in the current A5
# container, and run the focused QLI V2/MXFP4 contract tests.
#
# Optional overrides:
#   VA_REPO, VA_REF, VA_WORKDIR, SOC_VERSION

VA_REPO="${VA_REPO:-https://github.com/Qiming-zhang-rondo/vllm-ascend-glm-mxfp4.git}"
VA_REF="${VA_REF:-main}"
VA_WORKDIR="${VA_WORKDIR:-/workspace/vllm-ascend-glm-mxfp4}"
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

# The editable install rebuilds _C_ascend and makes the active interpreter use
# this checkout instead of the vllm-ascend copy baked into the container.
python3 -m pip install --no-deps -e "$VA_WORKDIR"

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
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
for name, available in required.items():
    print(f"{name}: {available}")
if not all(required.values()):
    missing = ", ".join(name for name, available in required.items() if not available)
    raise RuntimeError(f"A5 runtime is missing required MXFP4 indexer operators: {missing}")
PY

cd "$VA_WORKDIR"
python3 -m pytest -q tests/ut/attention/test_sfa_indexer.py
python3 -m pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_sfa_indexer_qli_v2.py

cat <<'EOF'
Installation and focused A5 QLI V2/MXFP4 checks passed.

Add this to the GLM-5.2/5.3 serve command:
  --additional-config '{"enable_sparse_li_c8":true,"sfa_indexer_quant_mode":"mxfp4"}'
EOF
