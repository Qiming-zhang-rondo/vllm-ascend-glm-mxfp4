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
Meta/binding and launcher tests pass; CANN compilation and A5 execution pending.
Ruff and Bash syntax checks pass. Repository `format.sh ci` cannot run because
pre-commit is absent locally; no dependencies were installed for this check.

The local schema uses the unique header `qsfa_tiling_data.h`. It must not use
CANN's generic `kernel_tiling/kernel_tiling.h` path: ASC may prioritize its
generated/SDK tiling directory, and SDK matmul code needs its own types from
that header. QSFA's schema and the SDK header now coexist instead of relying
on include search order. The schema's fields and compute logic are unchanged.
