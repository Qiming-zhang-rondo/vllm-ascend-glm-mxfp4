// SPDX-License-Identifier: Apache-2.0
// Standalone eager prototype. No VA registration or global operator replacement.
#include "launch.h"
#include "storage_format.h"

#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <tuple>

#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace {
using Result = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

void check_contract(const at::Tensor& q, const at::Tensor& qs,
                    const at::Tensor& kv, const at::Tensor& ks,
                    const at::Tensor& rope, const at::Tensor& indices, double scale)
{
    TORCH_CHECK(q.dim() == 2 && q.size(1) == 576, "q must be uint8 [H,576]");
    const auto h = q.size(0);
    TORCH_CHECK(h == 8 || h == 16 || h == 32 || h == 64, "H must be 8,16,32,64");
    TORCH_CHECK(kv.dim() == 2 && kv.size(1) == 256, "kv must be packed uint8 [K,256]");
    const auto k = kv.size(0);
    TORCH_CHECK(indices.dim() == 1, "indices must be int32 [S]");
    const auto s = indices.size(0);
    TORCH_CHECK(s >= 128 && s <= 8192 && s % 128 == 0, "S must be a multiple of 128 in [128,8192]");
    TORCH_CHECK(k >= s && k <= std::numeric_limits<int32_t>::max(), "K must be >=S and fit int32");
    TORCH_CHECK(qs.sizes() == at::IntArrayRef({h, 18}), "qs must be uint8 [H,18]");
    TORCH_CHECK(ks.sizes() == at::IntArrayRef({k, 16}), "ks must be uint8 [K,16]");
    TORCH_CHECK(rope.sizes() == at::IntArrayRef({k, 64}), "rope must be BF16 [K,64]");
    TORCH_CHECK(q.scalar_type() == at::kByte && qs.scalar_type() == at::kByte &&
                kv.scalar_type() == at::kByte && ks.scalar_type() == at::kByte,
                "q/kv payloads and E8M0 scales must use explicit uint8 byte storage");
    TORCH_CHECK(rope.scalar_type() == at::kBFloat16 && indices.scalar_type() == at::kInt,
                "rope must be BF16 and indices must be int32");
    TORCH_CHECK(std::isfinite(scale) && scale > 0 && scale <= std::numeric_limits<float>::max() &&
                static_cast<float>(scale) > 0,
                "scale must be positive finite FP32");
    for (const auto* tensor : {&q, &qs, &kv, &ks, &rope, &indices}) {
        TORCH_CHECK(tensor->device() == q.device(), "all inputs must be on the same device");
        TORCH_CHECK(tensor->is_contiguous() && tensor->storage_offset() == 0,
                    "all inputs must be contiguous with zero storage offset");
    }
}

void require_nd(const at::Tensor& tensor)
{
    TORCH_CHECK(tensor.is_privateuseone(), "all runtime inputs and allocations must be on NPU");
    qsfa_storage::require_nd_format(tensor);
}

Result meta(const at::Tensor& q, const at::Tensor& qs, const at::Tensor& kv,
            const at::Tensor& ks, const at::Tensor& rope, const at::Tensor& indices,
            double scale)
{
    check_contract(q, qs, kv, ks, rope, indices, scale);
    return {at::empty({q.size(0), 512}, q.options()),
            at::empty({q.size(0), 16}, q.options()),
            at::empty({1}, q.options().dtype(at::kInt))};
}

Result forward(const at::Tensor& q, const at::Tensor& qs, const at::Tensor& kv,
               const at::Tensor& ks, const at::Tensor& rope, const at::Tensor& indices,
               double scale)
{
    check_contract(q, qs, kv, ks, rope, indices, scale);
    for (const auto* tensor : {&q, &qs, &kv, &ks, &rope, &indices}) require_nd(*tensor);
    const c10_npu::NPUGuard guard(q.device());
    const char* soc = aclrtGetSocName();
    TORCH_CHECK(soc != nullptr && std::strncmp(soc, "Ascend950", 9) == 0,
                "This dav-3510 prototype requires an Ascend950/A5 device");
    const uint32_t h = q.size(0), m = ((h + 15) / 16) * 16;
    const uint32_t k = kv.size(0), s = indices.size(0);
    const auto bytes = q.options();
    const auto bf16 = bytes.dtype(at::kBFloat16), fp32 = bytes.dtype(at::kFloat);
    auto qn = at::empty({m, 512}, bytes), qns = at::empty({m, 16}, bytes);
    auto qr = at::empty({m, 64}, bf16);
    auto kn = at::empty({s, 512}, bytes), kns = at::empty({s, 16}, bytes);
    auto kr = at::empty({s, 64}, bf16), vt = at::empty({512, s}, bf16);
    auto scores = at::empty({m, s}, fp32), p = at::empty({m, s}, bf16);
    auto acc = at::empty({m, 512}, fp32);
    auto output = at::empty({h, 512}, bytes), output_scale = at::empty({h, 16}, bytes);
    auto status = at::empty({1}, bytes.dtype(at::kInt));
    for (const auto* tensor : {&qn, &qns, &qr, &kn, &kns, &kr, &vt, &scores, &p,
                              &acc, &output, &output_scale, &status}) require_nd(*tensor);

    // Flush torch_npu's host task queue after allocations, then launch all stages
    // onto that same stream. This does not synchronize the device.
    const auto npu_stream = c10_npu::getCurrentNPUStream();
    const auto stream = npu_stream.stream(true);
    // Raw launch bypasses OpCommand's lifetime bookkeeping. Record every input,
    // output and temporary allocation so cross-stream deallocation is deferred.
    // As with ordinary torch ops, the caller must establish producer-stream waits.
    for (const at::Tensor* tensor : std::initializer_list<const at::Tensor*>{&q, &qs, &kv, &ks, &rope, &indices,
                              &qn, &qns, &qr, &kn, &kns, &kr, &vt, &scores, &p,
                              &acc, &output, &output_scale, &status}) {
        c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), npu_stream);
    }
    qsfa_gather_launch(stream, q.data_ptr(), qs.data_ptr(), kv.data_ptr(), ks.data_ptr(),
                       rope.data_ptr(), indices.data_ptr(), qn.data_ptr(), qns.data_ptr(),
                       qr.data_ptr(), kn.data_ptr(), kns.data_ptr(), kr.data_ptr(),
                       vt.data_ptr(), status.data_ptr(), h, m, k, s);
    qsfa_qk_launch(stream, qn.data_ptr(), kn.data_ptr(), qns.data_ptr(), kns.data_ptr(),
                   qr.data_ptr(), kr.data_ptr(), scores.data_ptr(), m, s);
    qsfa_softmax_launch(stream, scores.data_ptr(), p.data_ptr(), m, s, static_cast<float>(scale));
    qsfa_pv_launch(stream, p.data_ptr(), vt.data_ptr(), acc.data_ptr(), m, s);
    qsfa_output_launch(stream, acc.data_ptr(), output.data_ptr(), output_scale.data_ptr(), h);
    // Caller synchronizes and rejects nonzero status before consuming results.
    return {output, output_scale, status};
}

TORCH_LIBRARY_FRAGMENT(qsfa_q8c4_o8, m)
{
    m.def("forward(Tensor q, Tensor qs, Tensor kv, Tensor ks, Tensor rope, Tensor indices, float scale) "
          "-> (Tensor output, Tensor output_scale, Tensor status)");
}
TORCH_LIBRARY_IMPL(qsfa_q8c4_o8, PrivateUse1, m) { m.impl("forward", TORCH_FN(forward)); }
TORCH_LIBRARY_IMPL(qsfa_q8c4_o8, Meta, m) { m.impl("forward", TORCH_FN(meta)); }
}  // namespace
