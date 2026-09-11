// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

namespace vllm_ascend {

inline void CheckQLIV2MxFp4Inputs(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &query_scale, const at::Tensor &key_scale,
    c10::string_view layout_q, c10::string_view layout_k)
{
    TORCH_CHECK(layout_q == "TND" && layout_k == "PA_BBND",
                "MXFP4 QLI V2 wrapper supports TND query and PA_BBND key");
    TORCH_CHECK(query.dim() == 3 && key.dim() == 4,
                "MXFP4 QLI V2 requires query [T,N,64], key [blocks,block_size,1,64]");
    TORCH_CHECK(query.size(2) == 64 && key.size(3) == 64 && key.size(2) == 1,
                "MXFP4 QLI V2 requires packed D64 bytes (logical D128), with one K head");
    TORCH_CHECK(query.size(1) >= 1 && query.size(1) <= 64,
                "MXFP4 QLI V2 query head count must be in [1,64]");
    for (const auto &tensor : {query, key}) {
        TORCH_CHECK(tensor.scalar_type() == at::kByte ||
                    tensor.scalar_type() == at::kFloat4_e2m1fn_x2,
                    "MXFP4 QLI V2 Q/K must use packed uint8 or float4_e2m1fn_x2 storage");
    }
    TORCH_CHECK(query_scale.dim() == 4 && query_scale.size(0) == query.size(0) &&
                    query_scale.size(1) == query.size(1) && query_scale.size(2) == 2 && query_scale.size(3) == 2,
                "MXFP4 QLI V2 query scale must be [T,N,2,2]");
    TORCH_CHECK(key_scale.dim() == 5 && key_scale.size(0) == key.size(0) &&
                    key_scale.size(1) == key.size(1) && key_scale.size(2) == 1 &&
                    key_scale.size(3) == 2 && key_scale.size(4) == 2,
                "MXFP4 QLI V2 key scale must be [blocks,block_size,1,2,2]");
    for (const auto &tensor : {query_scale, key_scale}) {
        TORCH_CHECK(tensor.scalar_type() == at::kByte ||
                    tensor.scalar_type() == at::kFloat8_e8m0fnu,
                    "MXFP4 QLI V2 scales must use uint8 E8M0 bits or float8_e8m0fnu");
        TORCH_CHECK(tensor.storage_offset() % 2 == 0,
                    "MXFP4 QLI V2 E8M0 storage offsets must be aligned to scale pairs");
    }
    TORCH_CHECK(query.is_contiguous() && query_scale.is_contiguous(),
                "MXFP4 QLI V2 query and query scale must be contiguous");
    for (const auto &tensor : {key, key_scale}) {
        int64_t stride = 1;
        for (int64_t axis = tensor.dim() - 1; axis >= 1; --axis) {
            TORCH_CHECK(tensor.size(axis) == 1 || tensor.stride(axis) == stride,
                        "MXFP4 QLI V2 caches may be noncontiguous only on axis 0");
            stride *= tensor.size(axis);
        }
        TORCH_CHECK(tensor.stride(0) >= stride && tensor.stride(0) % 2 == 0,
                    "MXFP4 QLI V2 cache blocks must be disjoint and pair-aligned");
    }
    TORCH_CHECK(weights.scalar_type() == at::kFloat && weights.is_contiguous() &&
                weights.dim() == 2 && weights.size(0) == query.size(0) && weights.size(1) == query.size(1),
                "MXFP4 QLI V2 weights must be contiguous FP32 [T,N]");
}

}  // namespace vllm_ascend
