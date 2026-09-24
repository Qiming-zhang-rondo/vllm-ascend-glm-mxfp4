// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <cstdint>

namespace qsfa_storage {
inline void require_linear_base_format(const at::Tensor& tensor)
{
    TORCH_CHECK(tensor.is_contiguous() && tensor.storage_offset() == 0,
        "raw kernel storage must be contiguous with zero storage offset");
    TORCH_CHECK(tensor.storage().nbytes() >= tensor.nbytes(),
        "raw kernel storage capacity is smaller than its logical payload");
    // Ascend/pytorch v2.10.0 npu_native_functions.yaml registers this schema.
    // NPUBridge has no export annotation: do not depend on its private C++ ABI.
    static const auto get_format = c10::Dispatcher::singleton()
        .findSchemaOrThrow("npu::get_npu_format", "")
        .typed<int64_t(const at::Tensor&)>();
    constexpr int64_t ND_FORMAT = 2; // ACL_FORMAT_ND
    constexpr int64_t NCHW_FORMAT = 0; // ACL_FORMAT_NCHW
    const auto format = get_format.call(tensor);
    // torch_npu can assign NCHW to normal contiguous inputs/allocations.
    // Like ND, this is a base format with linear, unblocked storage. See
    // csrc/aclnn_torch_adapter/op_api_common.h::IsOpInputBaseFormat.
    // Do not accept opaque layouts merely because is_contiguous() is true.
    TORCH_CHECK(format == ND_FORMAT || format == NCHW_FORMAT,
        "raw kernel storage must be contiguous ND (2) or NCHW (0); opaque/other formats are unsupported; got format ",
        format, "; shape ", tensor.sizes());
}
} // namespace qsfa_storage
