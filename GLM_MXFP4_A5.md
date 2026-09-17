# GLM-5.2/5.3 SFA Indexer MXFP4 on Ascend A5

The deployment patch is based on the vLLM-Ascend v0.26.0 deployment commit
`8bfdcf2fe931f7d535e0e67a4e4eba233bccb598`. It backports the
`QuantLightningIndexerV2` compute and metadata bindings, migrates the GLM SFA
indexer consumer to V2, and adds the A5 MXFP4 Q/K and cache producer contract.

First run the standalone QLI V2 MXFP4 compute-op test in the **existing A5
container**. It reuses its Python, torch, torch_npu and CANN, requires no model
weights, and does not install VA or download build dependencies:

```bash
curl -fsSL https://raw.githubusercontent.com/Qiming-zhang-rondo/vllm-ascend-glm-mxfp4/main/tools/test_qli_v2_mxfp4_a5.sh -o test_qli.sh && bash test_qli.sh --update
```

The test executes a real mode-5 C4 compute probe; Metadata/API presence alone
is insufficient. If necessary it selectively builds the bundled QLI V2 and
Metadata operators using existing CANN development dependencies, or reports
exactly which local prerequisites are missing. Matching private builds are
reused. It does not replace the system CANN or rebuild the VA package.

It validates C4 and FP8 accuracy against decoded-payload and original-input
references, checks strided cache addressing, and reports synchronous wall
P50/P90 latency. This includes ACLNN preparation and synchronization, not just
kernel execution. Logs and a JSON report are saved automatically.

See [the container test guide](QLI_SINGLE_OP_A5.md) for options, operator
capability checks, fixed startup/dependency bugs, and verification limits.
CANN compilation and A5 execution have not been validated locally.

The model installation below is a separate deployment workflow and is **not
invoked by the single-operator test**.

After the standalone operator passes, clone this delivery repository and run
the patch installer:

Run this inside the existing A5 vLLM-Ascend container:

```bash
git clone https://github.com/Qiming-zhang-rondo/vllm-ascend-glm-mxfp4.git
cd vllm-ascend-glm-mxfp4
bash tools/install_glm_mxfp4_a5.sh --update
```

The installer leaves the vLLM-Ascend copy baked into the container untouched.
It checks out the exact upstream commit in
`/workspace/vllm-ascend-qli-mxfp4-v0.26.0`, verifies the complete patch with
`git apply --check`, applies it, rebuilds `_C_ascend` through an editable
install, and checks that the required PyTorch NPU APIs are registered. It
refuses to apply the patch to another framework commit.

Start GLM-5.2 or GLM-5.3 with these additional settings:

```bash
--additional-config '{"enable_sparse_li_c8":true,"sfa_indexer_quant_mode":"mxfp4"}'
```

`enable_sparse_li_c8` allocates the quantized indexer cache. The separate
`sfa_indexer_quant_mode` option selects the QLI V2 MXFP4 producer/consumer
contract. The configuration rejects MXFP4 on non-A5 devices and when LI C8 is
disabled.

To verify patch applicability without compiling or installing anything:

```bash
bash tools/install_glm_mxfp4_a5.sh --check-only
```
