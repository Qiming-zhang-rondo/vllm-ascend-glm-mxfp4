// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <cstdint>

namespace qsfa_storage {
inline void require_nd_format(const at::Tensor& tensor)
{
    // Ascend/pytorch v2.10.0 npu_native_functions.yaml registers this schema.
    // NPUBridge has no export annotation: do not depend on its private C++ ABI.
    static const auto get_format = c10::Dispatcher::singleton()
        .findSchemaOrThrow("npu::get_npu_format", "")
        .typed<int64_t(const at::Tensor&)>();
    constexpr int64_t ND_FORMAT = 2; // ACL_FORMAT_ND
    const auto format = get_format.call(tensor);
    TORCH_CHECK(format == ND_FORMAT,
        "raw kernel input/storage must be ND; cast to format 2 before calling; got format ", format);
}
} // namespace qsfa_storage
