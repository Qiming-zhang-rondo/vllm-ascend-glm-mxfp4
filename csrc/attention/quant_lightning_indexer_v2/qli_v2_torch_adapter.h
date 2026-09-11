// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

// Include after aclnn_torch_adapter/op_api_common.h. ADL selects this scoped
// overload in ConvertTypes; all existing operators keep their current adapter.
#include "qli_v2_packed_layout.h"

namespace vllm_ascend {

struct QLIV2TensorWrapper {
    const at::Tensor &tensor;
    aclDataType logical_dtype;
};

inline aclTensor *ConvertType(const QLIV2TensorWrapper &wrapped)
{
    static const auto create_tensor = GET_OP_API_FUNC(aclCreateTensor);
    TORCH_CHECK(create_tensor != nullptr, "aclCreateTensor is unavailable");
    const auto &tensor = wrapped.tensor;
    TORCH_CHECK(IsOpInputBaseFormat(tensor) && tensor.element_size() == 1,
                "MXFP4 QLI V2 requires base-format, byte-addressed tensor storage");
    std::vector<int64_t> shape(tensor.sizes().begin(), tensor.sizes().end());
    std::vector<int64_t> strides(tensor.strides().begin(), tensor.strides().end());
    int64_t offset = tensor.storage_offset();
    int64_t storage_size = tensor.storage().nbytes();
    if (wrapped.logical_dtype == ACL_FLOAT4_E2M1) {
        auto layout = MakeQLIV2PackedLayout(shape, strides, offset, storage_size);
        shape = std::move(layout.shape);
        strides = std::move(layout.strides);
        offset = layout.storage_offset;
        storage_size = layout.storage_size;
    } else {
        TORCH_CHECK(wrapped.logical_dtype == ACL_FLOAT8_E8M0,
                    "QLI V2 byte wrapper only supports E2M1 or E8M0");
    }
    return create_tensor(shape.data(), shape.size(), wrapped.logical_dtype,
                         strides.data(), offset, ACL_FORMAT_ND, &storage_size, 1,
                         const_cast<void *>(tensor.storage().data()));
}

}  // namespace vllm_ascend
