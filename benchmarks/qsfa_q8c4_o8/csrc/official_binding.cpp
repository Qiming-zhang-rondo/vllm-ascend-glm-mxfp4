// SPDX-License-Identifier: Apache-2.0
// Private single-operator entry; no global CANN or framework registration.
#include "../official/launch.h"
#include "storage_format.h"
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <initializer_list>
#include <tuple>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace {
using OfficialResult = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

template <bool Candidate>
void check_official(const at::Tensor& q, const at::Tensor& cache, const at::Tensor& idx,
                    const at::Tensor& table, const at::Tensor& cuq, const at::Tensor& kvlen, double scale)
{
    TORCH_CHECK(q.dim() == (Candidate ? 2 : 3), "official query rank mismatch");
    const auto h = q.size(Candidate ? 0 : 1);
    TORCH_CHECK(h == 8 || h == 16 || h == 32 || (!Candidate && h == 64),
                "official candidate supports H8/16/32; control also supports H64");
    if constexpr (Candidate) {
        TORCH_CHECK(q.size(1) == 608 && q.scalar_type() == at::kByte, "Q must be uint8[H,608]");
    } else {
        TORCH_CHECK(q.size(0) == 1 && q.size(2) == 576 && q.scalar_type() == at::kBFloat16,
                    "control Q must be BF16[1,H,576]");
    }
    TORCH_CHECK(cache.dim() == 4 && cache.size(1) == 256 && cache.size(2) == 1 &&
                cache.size(3) == (Candidate ? 416 : 656) && cache.scalar_type() == at::kByte,
                "combined PA cache must be uint8[blocks,256,1,416 or 656]");
    const auto blocks = cache.size(0), rows = blocks * 256;
    TORCH_CHECK(blocks > 0 && rows <= std::numeric_limits<int32_t>::max(), "invalid PA cache capacity");
    TORCH_CHECK(idx.dim() == 3 && idx.size(0) == 1 && idx.size(1) == 1 && idx.scalar_type() == at::kInt,
                "indices must be int32[1,1,S]");
    const auto selected = idx.size(2);
    TORCH_CHECK(selected >= 128 && selected <= 8192 && selected % 128 == 0 && selected <= rows,
                "S must be a multiple of128 in [128,8192] and <=cache capacity");
    const auto address_ub = ((blocks * 4 + 511) / 512) * 512 + selected * 12;
    TORCH_CHECK(address_ub + 8192 <= (Candidate ? 216 : 248) * 1024,
                "PA address preparation exceeds the fixed UB budget");
    TORCH_CHECK(table.sizes() == at::IntArrayRef({1, blocks}) && table.scalar_type() == at::kInt,
                "table must be int32[1,blocks]");
    TORCH_CHECK(cuq.sizes() == at::IntArrayRef({1}) && kvlen.sizes() == at::IntArrayRef({1}) &&
                cuq.scalar_type() == at::kInt && kvlen.scalar_type() == at::kInt,
                "cuq/kvlen must be int32[1]");
    TORCH_CHECK(std::isfinite(scale) && scale > 0 && scale <= std::numeric_limits<float>::max() &&
                static_cast<float>(scale) > 0, "scale must be positive finite FP32");
    for (const auto* tensor : {&q, &cache, &idx, &table, &cuq, &kvlen}) {
        TORCH_CHECK(tensor->device() == q.device() && tensor->is_contiguous() && tensor->storage_offset() == 0,
                    "inputs must be contiguous zero-offset tensors on the same device");
    }
    // Values (indices, table permutation, finite payload, cuq=1, kvlen<=capacity)
    // are checked before H2D by official_inputs.py. No per-call device readbacks.
}

template <bool Candidate>
OfficialResult official_meta(const at::Tensor& q, const at::Tensor& cache, const at::Tensor& idx,
                             const at::Tensor& table, const at::Tensor& cuq, const at::Tensor& kvlen, double scale)
{
    check_official<Candidate>(q, cache, idx, table, cuq, kvlen, scale);
    const auto h = q.size(Candidate ? 0 : 1);
    const auto bytes = q.options().dtype(at::kByte);
    if constexpr (Candidate) {
        auto packed = at::empty({h, 544}, bytes);
        return {packed.narrow(1, 0, 512), packed.narrow(1, 512, 16), at::empty({1}, bytes.dtype(at::kInt))};
    }
    return {at::empty({1,h,512}, q.options()), at::empty({0}, bytes), at::empty({1}, bytes.dtype(at::kInt))};
}

template <bool Candidate>
OfficialResult official_forward(const at::Tensor& q, const at::Tensor& cache, const at::Tensor& idx,
                                const at::Tensor& table, const at::Tensor& cuq, const at::Tensor& kvlen, double scale)
{
    check_official<Candidate>(q, cache, idx, table, cuq, kvlen, scale);
    const c10_npu::NPUGuard guard(q.device());
    const char* soc = aclrtGetSocName();
    TORCH_CHECK(soc && std::strncmp(soc,"Ascend950",9)==0, "official experiment requires A5/Ascend950");
    for (const auto* tensor : {&q, &cache, &idx, &table, &cuq, &kvlen})
        qsfa_storage::require_nd_format(*tensor);
    const auto h = q.size(Candidate ? 0 : 1);
    const auto bytes = q.options().dtype(at::kByte);
    auto workspace = at::empty({idx.numel() * 8}, bytes);
    auto packed = Candidate ? at::empty({h,544}, bytes) : at::empty({1,h,512}, q.options());
    auto empty = at::empty({0}, bytes);
    auto status = at::zeros({1}, bytes.dtype(at::kInt));
    for (const auto* tensor : {&workspace, &packed, &status}) qsfa_storage::require_nd_format(*tensor);
    const auto npu_stream = c10_npu::getCurrentNPUStream();
    const auto stream = npu_stream.stream(true);
    for (const auto* tensor : std::initializer_list<const at::Tensor*>{
            &q,&cache,&idx,&table,&cuq,&kvlen,&workspace,&packed,&status})
        c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), npu_stream);
    const auto launch = Candidate ? qsfa_official_q8c4_launch : qsfa_official_fp8_launch;
    launch(stream,q.data_ptr(),cache.data_ptr(),idx.data_ptr(),table.data_ptr(),cuq.data_ptr(),kvlen.data_ptr(),
           packed.data_ptr(),workspace.data_ptr(),h,cache.size(0)*256,idx.numel(),static_cast<float>(scale));
    if constexpr (Candidate) return {packed.narrow(1,0,512),packed.narrow(1,512,16),status};
    return {packed,empty,status};
}
TORCH_LIBRARY_FRAGMENT(qsfa_q8c4_o8, m) {
    m.def("forward_official(Tensor q, Tensor cache, Tensor idx, Tensor table, Tensor cuq, Tensor kvlen, float scale) "
          "-> (Tensor output, Tensor output_scale, Tensor status)");
    m.def("forward_official_fp8(Tensor q, Tensor cache, Tensor idx, Tensor table, Tensor cuq, Tensor kvlen, float scale) "
          "-> (Tensor output, Tensor output_scale, Tensor status)");
}
TORCH_LIBRARY_IMPL(qsfa_q8c4_o8, PrivateUse1, m) {
    m.impl("forward_official", TORCH_FN(official_forward<true>));
    m.impl("forward_official_fp8", TORCH_FN(official_forward<false>));
}
TORCH_LIBRARY_IMPL(qsfa_q8c4_o8, Meta, m) {
    m.impl("forward_official", TORCH_FN(official_meta<true>));
    m.impl("forward_official_fp8", TORCH_FN(official_meta<false>));
}
} // namespace
