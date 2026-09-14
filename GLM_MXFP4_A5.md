# GLM-5.2/5.3 SFA Indexer MXFP4 on Ascend A5

This branch is based on vLLM-Ascend `main` at
`fd815467c221ee600137f6bdd53fe354d5e7c999`. It migrates the GLM SFA indexer
consumer to `QuantLightningIndexerV2` and adds the A5 MXFP4 Q/K and cache
producer contract.

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

After the standalone operator passes, install the branch and run the broader
focused tests:

Run this inside the existing A5 vLLM-Ascend container:

```bash
curl -fsSL https://raw.githubusercontent.com/Qiming-zhang-rondo/vllm-ascend-glm-mxfp4/main/tools/install_glm_mxfp4_a5.sh | bash
```

The installer keeps an existing checkout as a timestamped backup, clones this
repository to `/workspace/vllm-ascend-glm-mxfp4`, rebuilds `_C_ascend` through
an editable install, checks that the CANN/PyTorch NPU operators exist, and runs
the focused unit and on-device QLI V2 tests.

Start GLM-5.2 or GLM-5.3 with these additional settings:

```bash
--additional-config '{"enable_sparse_li_c8":true,"sfa_indexer_quant_mode":"mxfp4"}'
```

`enable_sparse_li_c8` allocates the quantized indexer cache. The separate
`sfa_indexer_quant_mode` option selects the QLI V2 MXFP4 producer/consumer
contract. The configuration rejects MXFP4 on non-A5 devices and when LI C8 is
disabled.

If the container uses another Ascend 950 target string, override it explicitly:

```bash
curl -fsSL https://raw.githubusercontent.com/Qiming-zhang-rondo/vllm-ascend-glm-mxfp4/main/tools/install_glm_mxfp4_a5.sh | env SOC_VERSION=<actual-ascend-950-soc> bash
```
