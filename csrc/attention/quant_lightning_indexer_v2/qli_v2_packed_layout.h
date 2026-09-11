// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vllm_ascend {

struct QLIV2PackedLayout {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    int64_t storage_offset;
    int64_t storage_size;
};

inline int64_t QLIV2DoubleChecked(int64_t value)
{
    if (value < 0 || value > std::numeric_limits<int64_t>::max() / 2) {
        throw std::invalid_argument("QLI V2 packed FP4 descriptor exceeds the logical element range");
    }
    return value * 2;
}

// Packed E2M1 tensors expose bytes to PyTorch and logical nibbles to ACL.
// This adapter intentionally supports packing along the contiguous last axis
// used by TND/PA_BBND indexers, not arbitrary transposed low-bit tensors.
inline QLIV2PackedLayout MakeQLIV2PackedLayout(
    std::vector<int64_t> shape, std::vector<int64_t> strides,
    int64_t storage_offset, int64_t storage_bytes)
{
    if (shape.empty() || shape.size() != strides.size() || strides.back() != 1) {
        throw std::invalid_argument("QLI V2 packed FP4 requires a contiguous innermost byte axis");
    }
    shape.back() = QLIV2DoubleChecked(shape.back());
    for (size_t axis = 0; axis + 1 < strides.size(); ++axis) {
        strides[axis] = QLIV2DoubleChecked(strides[axis]);
    }
    return {shape, strides, QLIV2DoubleChecked(storage_offset), QLIV2DoubleChecked(storage_bytes)};
}

}  // namespace vllm_ascend
