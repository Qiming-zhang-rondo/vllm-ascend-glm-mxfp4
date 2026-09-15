# QLI V2 MXFP4: reuse the existing A5 container

The benchmark first checks whether the container can execute the **compute**
operator with MXFP4. A metadata API symbol alone does not establish support.
If that capability is absent, `build_qli_v2_mxfp4_a5.sh` can build this checkout's
QLI V2 compute and metadata operators into a private directory.

For an existing downloaded `test_qli.sh`, update and run with:

```bash
bash test_qli.sh --update
```

The default test prepares both C4 and FP8 on CPU from the same finite FP16
inputs. C4 uses packed E2M1 bytes and D32 E8M0 scales; FP8 uses E4M3FN and
per-row FP32 scales. The A5 executes the real QLI V2 metadata and compute
operators. NPU DynamicMxQuant/DynamicQuant are not invoked. CPU packing does
not replace the A5 compute with dequantized arithmetic.

The producer rules follow CANN `ops-nn@2a77283db46e6648ff47bc8277442cf9c721e3c2`,
`quant/dynamic_mx_quant/tests/assets/golden.py` and
`quant/dynamic_quant/tests/assets/golden.py`. The C4 compute reference follows
`ops-transformer@632dddba712a4e6cace3f8b44f198aef8a82ce3e`,
`attention/quant_lightning_indexer_v2/tests/pytest/quant_lightning_indexer_v2_golden.py`
(`reduce_mxfp4_weighted_qk`). Error against the original FP16 inputs is also
reported using a separate FP32 reference. This tests QLI with synthetic inputs,
not the runtime quantizer or GLM model accuracy.

Default output includes C4/FP8 accuracy and synchronous call latency; descriptor
and workspace preparation, dispatch and synchronization are included, while
input preparation and metadata are excluded. Accuracy thresholds are evaluated
after both modes have produced their timing results. Optional
`--check-cache-layout` additionally exercises padded, offset cache views;
the default uses contiguous paged cache. `--check-only` runs a small real C4
compute smoke test without the full benchmark.

This helper does not install vLLM, vLLM-Ascend, PyTorch, torch_npu, CANN, or a
container image. It does not invoke a package manager or download source archives.
It uses the container's existing Python, CANN development toolkit, compilers,
JSON headers, and AICPU static libraries.

```bash
# Optional preflight only; neither builds nor installs anything.
bash tools/build_qli_v2_mxfp4_a5.sh --check-only

# Build only the two requested operators and their existing source dependencies.
bash tools/build_qli_v2_mxfp4_a5.sh
```

The top-level benchmark launcher calls this helper when needed. For a toolkit or
header directory already mounted elsewhere, pass `--cann-root /existing/toolkit`
or `--json-include /existing/include`. The latter must contain
`nlohmann/json.hpp`. `--python /existing/python`, `--jobs 8`, and
`--soc ascend950dt_9582` are also accepted. A5 variants normalize to CMake's
`ascend950` family.

JSON discovery checks the installed torch and VA locations without importing VA,
the CANN include trees, this checkout's existing third-party directory, and
system include directories. If needed, discovery searches for existing JSON
headers within the selected CANN tree. A runtime-only image can still lack
`op_build`, `bisheng`, the AICPU cross compiler, or AICPU static libraries. In that
case the helper exits **78**, lists missing paths, and downloads nothing. The
existing matching development toolkit must be available for compilation.

The dedicated `QLI_STANDALONE_OFFLINE` CMake mode:

- Selects `quant_lightning_indexer_v2` and `quant_lightning_indexer_v2_metadata`,
  explicitly declaring `lightning_indexer_v2` as a common source dependency so
  its shared vector headers are copied into the generated kernel source tree.
- Reuses existing JSON headers and skips the normal JSON download fallback.
- Omits unrelated ONNX/protobuf/absl build targets, FIA device tiling, and
  makeself `.run` packaging. The metadata kernel still links CANN's **installed**
  `libbase_ascend_protobuf.a` and `libaicpu_context.a`.
- Disables the hidden `prepare.sh` sub-build, which otherwise loses offline
  options and re-enables downloads. The helper explicitly runs configure,
  `prepare_build`, configure again, build, and `cmake --install`.
- Leaves the ordinary VA build behavior unchanged when this option is off.

Each build uses a fresh directory below `.qli-op-build`, identified by source
and CANN fingerprints plus an attempt timestamp. It does not clear another VA
build directory or overwrite a previous successful install. A success manifest
is written atomically to `.qli-op-build/install.json` only after the API library,
QLI V2 device binaries, metadata AICPU library, and registration JSON files are
present. It contains
`opapi_lib`, `opp_root`, the Git ref, fingerprints, and `device_tested: false`.
`--reuse-only` returns success only when the manifest matches the current source,
CANN, and SoC and those artifacts still exist. It never builds.

The benchmark launcher starts a new Python process with the private `opp_root`
in `ASCEND_CUSTOM_OPP_PATH`, adds `opapi_lib`'s directory to `LD_LIBRARY_PATH`, and
passes `--opapi-lib` to the backend. This is needed to load the **kernel and its
metadata producer together**, rather than merely exposing a new API symbol.

The build's link stubs stay in its private `build/stubs` directory and are never
installed or added to the test's library search path. Execution requires the
container's real `libopapi_math.so` and `libnnopbase.so`; preflight checks these
libraries. The selected existing Python is retained through the kernel compiler
phase as well as the initial CMake configuration.

Local validation covers command sequencing, no-package-manager execution,
dependency failures, incomplete installs, stale-manifest rejection, and the
offline CMake gates. This change has not been compiled with CANN or executed on
an A5 in the development environment. A successful build must still pass the
benchmark's real MXFP4 smoke, accuracy, and performance checks.
