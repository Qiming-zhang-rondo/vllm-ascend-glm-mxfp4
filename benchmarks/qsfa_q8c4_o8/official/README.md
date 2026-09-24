# Official QSFA source derivative

This directory builds two specializations of the **same vendored official QSFA
fused kernel**. `baseline.asc` keeps the FP8-cache/BF16-query/BF16-output path;
`candidate.asc` enables the local Q8/C4/O8 modifications. It does not call the
earlier three-kernel `tiled` implementation. The two translation units use
separate `BaseApi` namespaces to prevent template instantiation interposition.

The source is `cann/ops-transformer` at
`55498d91634277d4eec912499c027818a8c167fb`. `UPSTREAM.json` records every copied
file's original SHA256, including the license. Files under `vendor/` remain
under the CANN Open Software License Agreement Version 2.0 in
`vendor/LICENSE`; retain their copyright notices. Local edits are visible
against that pinned source, not silently described as upstream behavior.

## Invocation contract

`launch.h` declares the two host functions. Both launch one AIC and two AIVs
for a single query, one sequence, one KV head, PA page size 256, and selected
token count divisible by 128 (up to 8192). Candidate heads are 8/16/32;
control additionally supports 64. Q1 gives one official outer work item when
G <= 64; this is not evidence that the container's installed CANN binary uses
the same tiling or core count.

The caller supplies NPU `int32` sparse indices, a PA block table, Q cumulative
length `[1]`, and KV length `[cacheRows]`. The caller validates their values
and storage before invocation. Sparse indices are token indices, not byte
offsets. The kernel generates the official physical-offset table in
`selected * 8` bytes of user workspace. There is no split-G intermediate
cache or split-KV reduction in this restricted case.

| Storage | FP8 control | Q8/C4/O8 candidate |
| --- | --- | --- |
| Q row | 576 BF16 elements | 576 E4M3 bytes + 18 E8M0 D32 scales + 14 padding bytes |
| KV row | 512 E4M3 bytes + 128 BF16 RoPE bytes + 16 FP32 D128 scale bytes | 256 packed E2M1 bytes + 128 BF16 RoPE bytes + 16 E8M0 D32 scales + 16 padding bytes |
| Output row | 512 BF16 elements | 512 E4M3 bytes + 16 E8M0 D32 scales + 16 padding bytes |

`entry.h` calls the vendored `KvQuantSparseFlashAttentionMla::Init/Process`
directly. `tiling.h` constructs the official fields in local kernel memory
from scalar arguments. `qsfa_tiling_data.h` retains the
field names and C++ types from the copied host schema. **These structs are
not a serialized ACLNN tiling ABI**: no host-generated blob is reinterpreted,
and `GET_TILING_DATA`/`GetUserWorkspace` are deliberately absent. The AIV
still sends `CVSharedParams` through the official SSBUF/flag-15 handshake;
the AIC receives `nullptr` for the unused local tiling pointer, as upstream.

Template options are FD=false, PA=true, TND query, PA_BSND KV, template mode
1 (CFA), split-G=false, and vectorized physical addresses=true. The original
S2 tile is 128; the candidate uses 64 to fit its separate low-bit QK and
BF16-PV local buffers. This difference must remain visible in results.

## Build integration

Compile `baseline.asc` and `candidate.asc` in the already working standalone
QSFA `.so`, using the container CANN and existing runtime libraries. Add
these include roots, relative to this directory:

- `.`
- `vendor/attention/kv_quant_sparse_flash_attention/op_kernel`
- `vendor/common/include/op_kernel`

The second path resolves upstream `../../common/op_kernel/...` includes
without rearranging the vendored source tree. The third supplies `util.h`.
SDK includes (`kernel_operator.h`, `lib/matmul_intf.h`, etc.) come from the
installed CANN, not downloaded dependencies.

For candidate H <= 32, the planned simultaneous TPipe UB allocations total
195,328 bytes. Launch dynamic UB is 216 KiB; even conservatively adding
8 KiB of toolkit scratch keeps the allocation below that limit. The SIMT
memory contract separately reserves 8 KiB and at least 32 KiB Data Cache
from the device's 256 KiB UB. Do not raise dynamic UB to 224 KiB. Sparse
address preparation has a separate temporary phase; upstream calls
`TPipe::Reset()` before initializing the attention UB buffers. There are
no static `__ubuf__` arrays in our wrapper.

The control is SIMD-only and retains the original TPipe buffer sizes
(243,456 bytes simultaneously allocated). Its native launch explicitly
reserves 248 KiB dynamic UB, leaving 8 KiB for the toolkit. Unlike the
candidate TU, it has no SIMT VF or SIMT Data Cache requirement. The candidate's modified
buffer-lifetime and synchronization contracts require an A5 compile and run;
host schema/packing tests cannot establish device correctness or speedup.

Address-table preparation is also bounded separately: page-table bytes rounded
up to512, plus `12 * selected` bytes and8KiB scratch must fit the chosen
dynamic UB reservation. Oversized shapes are rejected before a device launch.

Local validation on 2026-09-24: host codec, layout, Cube-call contract, libtorch
Meta/binding and launcher tests pass. A5 results for the first compiled version
and the subsequent, not-yet-device-verified optimization are recorded below.
Ruff and Bash syntax checks pass. Repository `format.sh ci` cannot run because
pre-commit is absent locally; no dependencies were installed for this check.

The local schema uses the unique header `qsfa_tiling_data.h`. It must not use
CANN's generic `kernel_tiling/kernel_tiling.h` path: ASC may prioritize its
generated/SDK tiling directory, and SDK matmul code needs its own types from
that header. QSFA's schema and the SDK header now coexist instead of relying
on include search order. The schema's fields and compute logic are unchanged.

Native ASC also parses `BufferInfo::GetConsPipe()` in compilation passes that
do not expose the `PIPE_FIX` enumerator. The vendored `attn_buffer.h` selects
`PIPE_FIX` only under `__DAV_C310_CUBE__`, matching the official mixed entry's
Cube guard. Other passes use an unused `PIPE_M` placeholder: only the Cube
implementation consumes L0C. The original FIX events/cross-core synchronization
are unchanged. Host launch arguments use ordinary byte pointers; the kernel
parameters retain their `GM_ADDR` qualifiers.

The initial `__ASC_NPU_HOST__`-only guard was incomplete: CANN also uses
`__NPU_HOST__` and `__ASCC_HOST__`, and the A5 report still entered its `else`.
The report alone does not identify that pass as host or AIV. Compile regression
fixtures now omit `PIPE_FIX` for all non-Cube cases, including a macro-less
host case; only the Cube fixture supplies it and checks the real FIX selection.

A5 feedback for `fe292cbce` confirms library compilation and registration.
The first FP8 control call then stopped before launch on the binding's
ND-only check (`got format 0`). The shared raw-storage check now permits
contiguous, zero-offset NCHW(0) as well as ND(2), including internally allocated
BF16 output; both are linear base formats. Capacity is checked and opaque
NZ/FRACTAL formats remain rejected. No format conversion is added to timing.
The subsequent `f459142c4` A5 run reached compute and timing as recorded below.

## A5 evidence and next optimization

User-supplied profiling for `f459142c4`, Q `[1,8,576]`, KV `[8192,576]`,
selected 2048, reports the following device tasks. These are single captured
invocations, not the 20-sample synchronized wall-time medians.

| Path | Attention task duration | Block Num |
| --- | ---: | ---: |
| Q8/C4/O8 candidate | 933.904 us | 1 |
| Same-source FP8 control | 41.464 us | 1 |
| Installed native QSFA | 69.825 us | 32 |

Candidate task latency is 22.52 times the same-source control and 13.37 times
the installed native task. The two custom paths additionally execute a
ZerosLike task (1.290 us candidate, 1.127 us control). The listed wait time is
not part of the task duration and is not added to this table. Different native
tiling remains visible; task timing alone does not identify which internal
Cube/Vector stage dominates. This result establishes device-side regression,
not a Python-only overhead problem.

The candidate passed decoded-payload correctness (relative RMSE 0.007328),
but failed the existing original-BF16 quantization screen (relative RMSE
0.115499 > 0.10). Incremental relative RMSE against the C4/BF16 reference was
0.042892. These are synthetic operator output errors, not model accuracy.

The next revision removes three unnecessary costs without changing cache,
scale, query, output or accuracy-gate contracts:

1. `lowbit_vector.h::DecodeCache` processes packed pairs and builds BF16 V
   bits directly from E2M1/E8M0. It avoids FP8-to-FP32 software decode and
   multiplication, halves duplicate input loads and uses aligned B32/B16
   stores. Host tests enumerate every packed byte and every E8M0 code,
   including signed zero, subnormal, overflow and NaN cases.
2. `CopyOutKvUb2L1` coalesces nine transfers into three with identical bytes
   and the same single-buffer completion fence. No synchronization is removed.
3. `IterateBmm2QSFA` keeps PV's output tile N=128 independently of S2=64.
   This reduces candidate PV MMADs from eight to four per S2 tile. Reduction
   K, FP32 accumulation, P/V layouts, and existing L0 capacities remain the same.

QK still uses MXFP8 operands with FP32 accumulation; PV still uses BF16.
The candidate still has SIMT producers, repeated per-chunk synchronization,
and S2=64. This is a bounded optimization of the passing path, not a claim
that the measured regression is resolved. Its A5 compile, correctness and
task duration need a new run using the same `--implementation official --profile`
command. No dependencies or toolkit are installed by that command.
