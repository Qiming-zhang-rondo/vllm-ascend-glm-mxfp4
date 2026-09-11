# GLM-5.2/5.3 SFA Indexer MXFP4 on Ascend A5

This branch is based on vLLM-Ascend `main` at
`fd815467c221ee600137f6bdd53fe354d5e7c999`. It migrates the GLM SFA indexer
consumer to `QuantLightningIndexerV2` and adds the A5 MXFP4 Q/K and cache
producer contract.

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
