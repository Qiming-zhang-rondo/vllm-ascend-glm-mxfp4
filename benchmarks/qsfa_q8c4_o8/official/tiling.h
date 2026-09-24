// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "kernel_tiling/kernel_tiling.h"

#ifndef QSFA_TILING_INLINE
#define QSFA_TILING_INLINE inline
#endif

namespace qsfa_official_contract {
constexpr uint32_t PAGE_SIZE = 256;
constexpr uint32_t CONTROL_CACHE_ROW_BYTES = 656;
constexpr uint32_t CANDIDATE_CACHE_ROW_BYTES = 416;
constexpr uint32_t CANDIDATE_QUERY_ROW_BYTES = 608;
constexpr uint32_t CANDIDATE_OUTPUT_ROW_BYTES = 544;
constexpr uint32_t CONTROL_DYNAMIC_UB_BYTES = 248 * 1024;
constexpr uint32_t CONTROL_ALLOCATED_UB_BYTES = 243456;
constexpr uint32_t CANDIDATE_DYNAMIC_UB_BYTES = 216 * 1024;
constexpr uint32_t CANDIDATE_ALLOCATED_UB_BYTES = 195328;
constexpr uint32_t TOOLKIT_RESERVED_UB_BYTES = 8 * 1024;
static_assert(CANDIDATE_ALLOCATED_UB_BYTES + TOOLKIT_RESERVED_UB_BYTES <= CANDIDATE_DYNAMIC_UB_BYTES);
static_assert(CANDIDATE_DYNAMIC_UB_BYTES + TOOLKIT_RESERVED_UB_BYTES + 32 * 1024 <= 256 * 1024);
static_assert(CONTROL_ALLOCATED_UB_BYTES <= CONTROL_DYNAMIC_UB_BYTES);
static_assert(CONTROL_DYNAMIC_UB_BYTES + TOOLKIT_RESERVED_UB_BYTES <= 256 * 1024);

constexpr bool SupportsShape(uint32_t heads, uint32_t cacheRows, uint32_t selected, bool candidate)
{
    const bool validHeads = heads == 8 || heads == 16 || heads == 32 || (!candidate && heads == 64);
    return validHeads && cacheRows > 0 && cacheRows % PAGE_SIZE == 0 &&
        selected > 0 && selected % 128 == 0 && selected <= cacheRows &&
        selected <= 8192 &&
        (((static_cast<uint64_t>(cacheRows / PAGE_SIZE) * 4 + 511) / 512) * 512 +
         static_cast<uint64_t>(selected) * 12 + TOOLKIT_RESERVED_UB_BYTES <=
         (candidate ? CANDIDATE_DYNAMIC_UB_BYTES : CONTROL_DYNAMIC_UB_BYTES));
}

// Source: QSFAMlaTiling::FillTilingBaseParamsMla/GenTilingKey/GetWorkspaceSize
// and the arch35 kernel's fixed s1BaseSize=64. This intentionally restricts the
// experiment to one Q token rather than implementing another general tiler.
QSFA_TILING_INLINE KvQuantSparseFlashAttentionTilingDataMla MakeTiling(
    uint32_t heads, uint32_t cacheRows, uint32_t selected, float softmaxScale,
    bool candidate)
{
    KvQuantSparseFlashAttentionTilingDataMla data{};
    auto& base = data.baseParams;
    const uint32_t rowBytes = candidate ? CANDIDATE_CACHE_ROW_BYTES : CONTROL_CACHE_ROW_BYTES;
    base.batchSize = 1;
    base.seqSize = cacheRows;
    base.qSeqSize = 1;
    base.blockSize = PAGE_SIZE;
    base.maxBlockNumPerBatch = cacheRows / PAGE_SIZE;
    base.actualLenDimsQ = 1;
    base.actualLenDimsKV = 1;
    base.scaleValue = softmaxScale;
    base.nNumOfQInOneGroup = heads;
    base.outputLayout = 1; // QSFALayout::TND.
    base.sparseMode = 3;
    base.sparseBlockSize = 1;
    base.sparseBlockCount = selected;
    base.dSizeVInput = rowBytes;
    base.isActualLenDimsNull = 0;
    base.isActualLenDimsKVNull = 0;
    base.keyStride0 = PAGE_SIZE * rowBytes;
    base.returnSoftmaxLse = 0;
    data.singleCoreParams.usedCoreNum = 1;
    data.innerSplitParams.mBaseSize = heads;
    data.innerSplitParams.s2BaseSize = candidate ? 64 : 128;
    // arch35 does not read singleCoreTensorSize or splitKVParams for FD=false.
    return data;
}

constexpr uint64_t WorkspaceBytes(uint32_t selected)
{
    // IS_SPLIT_G=false: only the official int64 sparse physical-offset table.
    return static_cast<uint64_t>(selected) * sizeof(int64_t);
}
} // namespace qsfa_official_contract
