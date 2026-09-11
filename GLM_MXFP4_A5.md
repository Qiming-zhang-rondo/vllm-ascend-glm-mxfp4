# GLM-5.2/5.3 SFA Indexer MXFP4 on Ascend A5

This branch is based on vLLM-Ascend `main` at
`fd815467c221ee600137f6bdd53fe354d5e7c999`. It migrates the GLM SFA indexer
consumer to `QuantLightningIndexerV2` and adds the A5 MXFP4 Q/K and cache
producer contract.

First run the standalone QLI V2 MXFP4 compute-op bring-up. It needs no model
weights and does not depend on `pytest`:

```bash
curl -fsSL https://raw.githubusercontent.com/Qiming-zhang-rondo/vllm-ascend-glm-mxfp4/main/tools/test_qli_v2_mxfp4_a5.sh | bash
```

This invokes the required Metadata op and then executes only
`QuantLightningIndexerV2` with `quant_mode=5`. It covers both dense and
axis-0-strided PA caches and validates the packed E2M1 payload, E8M0 scales,
causal output counts, bounds, and uniqueness. It also checks Top-2048 recall
and score error against an FP32 reference, then reports synchronized P50/P90
compute latency for MXFP4 mode 5 and FP8 mode 1 on the same shape. The default
shape is one decode query, 64 index heads, D128, and an 8192-token K sequence.

The accuracy thresholds and benchmark size can be overridden without editing
the script: `QLI_MIN_TOPK_RECALL`, `QLI_MIN_SCORE_COSINE`,
`QLI_MAX_SCORE_NMAE`, `QLI_KEY_TOKENS`, `QLI_QUERY_TOKENS`, `QLI_WARMUP`, and
`QLI_ITERS`. Set `QLI_MAX_MXFP4_P50_MS` only when the target machine has an
agreed absolute latency gate.

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
