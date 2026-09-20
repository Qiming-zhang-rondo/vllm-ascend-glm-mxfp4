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
bash tools/install_glm_mxfp4_a5.sh
```

The installer leaves the vLLM-Ascend copy baked into the container untouched.
It checks out the exact upstream commit in
`/workspace/vllm-ascend-qli-mxfp4-v0.26.0`, verifies the complete patch with
`git apply --check`, applies it, rebuilds `_C_ascend` through an editable
install when necessary, and checks the PyTorch wrappers and underlying ACLNN
symbols separately. A matching editable install is reused. It refuses to apply
the patch to another framework commit. `--update` deliberately recreates the
isolated checkout; it is not needed just to load an already-built QLI library.

Before starting the service, activate the existing QLI installation **in the
same shell**, after sourcing the container's CANN environment:

```bash
source /workspace/vllm-ascend-glm-mxfp4/tools/activate_qli_mxfp4_a5.sh
```

This first uses the delivery checkout's `.qli-op-build/install.json`, then the
standalone checkout's manifest at
`/workspace/vllm-ascend-glm-mxfp4-optest/.qli-op-build/install.json` (or that
checkout alongside the delivery repository). An installation elsewhere can be
selected with `--manifest /path/to/install.json` or
`--opapi-lib /path/to/vendor/op_api/lib/libcust_opapi.so`. Without a manifest,
it checks the configured CANN vendor libraries and `libopapi.so`.

The script prepends the selected vendor root to `ASCEND_CUSTOM_OPP_PATH`, its
API library directory to `LD_LIBRARY_PATH`, and disables FLA startup injection.
It checks both compute/metadata ACLNN APIs and their `GetWorkspaceSize` exports
in a fresh process and prints their library ownership. It performs no build,
installation, dependency download, or device computation. A failed activation
does not change the caller's environment. Explicit private installations must
contain all four APIs; incomplete installations do not silently use another
vendor's metadata.

The standalone test and the installer run in child shells, so their environment
exports do not persist into a separately launched service. The VA C++ adapter
snapshots vendor paths when loaded and caches symbol pointers at first use;
after activation, start **new workers**. A `torch.ops` registration alone cannot
detect missing CANN symbols. A symbol preflight does not prove MXFP4 compute,
model accuracy, or tiling/device-kernel ownership, and later code that changes
vendor priority can affect the service's actual library selection.

Start GLM-5.2 or GLM-5.3 with these additional settings:

```bash
--additional-config '{"enable_sparse_li_c8":true,"sfa_indexer_quant_mode":"mxfp4"}'
```

`enable_sparse_li_c8` allocates the quantized indexer cache. The separate
`sfa_indexer_quant_mode` option selects the QLI V2 MXFP4 producer/consumer
contract. The configuration rejects MXFP4 on non-A5 devices and when LI C8 is
disabled.

If profiling still shows the old LightningIndexer after restarting, install
temporary worker diagnostics in the **already patched deployment checkout**:

```bash
python3 -I /workspace/vllm-ascend-glm-mxfp4/tools/trace_qli_dispatch.py
```

Then restart using the same service command, including the QLI environment
activation above. The tool changes only Python diagnostic hooks and does not
rebuild/install VA or CANN, change quantization, or replace AV's methods. It
preserves local changes such as adding `FP8_DYNAMIC` to the layer filter.
Use `--va-root PATH` for another deployment checkout; `--check-only` validates
the expected methods without modifying files.

Look for `[QLI-DISPATCH]` in the worker output. The metadata-builder snapshot
inspects the actual Attention implementations in `static_forward_context`,
including `layer.impl` and `layer.mla_attn.impl`. It reports the effective
per-layer switches, indexer mode/cache prefix, layer-filter settings and the
file/line of the bound compute methods and their `DeviceOperator`. Methods
outside the community VA package also include a bounded source excerpt, when
available, to help identify AV overrides. Instances with identical settings
and methods are grouped; representative names and total counts are reported.

`device_entry` and `compute_entry` records identify entry into the instrumented
Python methods; they do not prove successful ACLNN/device execution. These
entry hooks are skipped inside `torch.compile` to avoid introducing diagnostic
code into the graph. Their absence alone does not prove a bypass. A snapshot
of a bound method likewise does not prove it was called. Tensor diagnostics
contain metadata only and do not read NPU tensor values.

Remove the temporary hooks once the route is understood, and restart before
collecting performance measurements:

```bash
python3 -I /workspace/vllm-ascend-glm-mxfp4/tools/trace_qli_dispatch.py --remove
```

Removal strips only the marked diagnostic blocks, preserving other edits;
the dormant helper remains on disk. Original modified files are also backed
up with the suffix `.before-qli-dispatch-trace`.

To verify patch applicability without compiling or installing anything:

```bash
bash tools/install_glm_mxfp4_a5.sh --check-only
```
