// Derived from CANN QSFA tiling schema, copyright Huawei Technologies Co., Ltd.
// Distributed under the CANN Open Software License Agreement Version 2.0.
// See vendor/LICENSE and UPSTREAM.json.
#pragma once
// Use a QSFA-specific filename. ASC/CANN supplies its own generic
// kernel_tiling/kernel_tiling.h (including TCubeTiling); do not shadow it.
#include <cstdint>

// These local C++ structs retain the upstream field names and types. They are
// NOT a serialized ACLNN tiling ABI. entry.h constructs them inside the kernel
// from scalar launch arguments, then calls upstream Init/Process directly.

struct KvQuantSparseFlashAttentionBaseParamsMla {
    uint32_t batchSize;
    uint32_t seqSize;
    uint32_t qSeqSize;
    int64_t blockSize;
    uint32_t maxBlockNumPerBatch;
    uint32_t actualLenDimsQ;
    uint32_t actualLenDimsKV;
    float scaleValue;
    uint32_t nNumOfQInOneGroup;
    uint32_t outputLayout;
    uint32_t sparseMode;
    int64_t sparseBlockSize;
    uint32_t sparseBlockCount;
    int64_t dSizeVInput;
    uint32_t isActualLenDimsNull;
    uint32_t isActualLenDimsKVNull;
    uint32_t keyStride0;
    uint32_t returnSoftmaxLse;
};

struct KvQuantSparseFlashAttentionSingleCoreParamsMla {
    uint32_t usedCoreNum;
};

struct KvQuantSparseFlashAttentionSingleCoreTensorSizeMla {
    uint32_t mmResUbSize;
    uint32_t bmm2ResUbSize;
};

struct KvQuantSparseFlashAttentionSplitKVParamsMla {
    uint32_t s2;
    uint32_t accumOutSize;
    uint32_t logSumExpSize;
};

struct KvQuantSparseFlashAttentionInnerSplitParams {
    uint32_t mBaseSize;
    uint32_t s2BaseSize;
};

struct KvQuantSparseFlashAttentionTilingDataMla {
    KvQuantSparseFlashAttentionBaseParamsMla baseParams;
    KvQuantSparseFlashAttentionSplitKVParamsMla splitKVParams;
    KvQuantSparseFlashAttentionSingleCoreParamsMla singleCoreParams;
    KvQuantSparseFlashAttentionSingleCoreTensorSizeMla singleCoreTensorSize;
    KvQuantSparseFlashAttentionInnerSplitParams innerSplitParams;
};
