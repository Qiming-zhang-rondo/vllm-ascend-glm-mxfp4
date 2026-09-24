// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include "acl/acl.h"

// Fixed single-query, single-sequence, one KV head, PA block size 256.
// Control heads: 8/16/32/64; candidate heads: 8/16/32. selected: positive
// multiple of 128; cacheRows: multiple of
// 256 and >= selected. Inputs, output and workspace reside on the same NPU.
// indices is int32[selected]; blockTable is int32[cacheRows / 256]; cuQ is
// int32[1] == 1; kvLen is int32[1] == cacheRows. Workspace >= selected * 8 bytes.
// Control: Q BF16[heads,576], cache bytes[cacheRows,656], O BF16[heads,512].
// Candidate: Q bytes[heads,608], cache bytes[cacheRows,416], O bytes[heads,544].
// Packing/validation is the caller's responsibility and excluded from timing.
extern "C" void qsfa_official_fp8_launch(
    aclrtStream stream, void* query, void* cache, void* indices, void* blockTable,
    void* cuQ, void* kvLen, void* output, void* workspace, uint32_t heads,
    uint32_t cacheRows, uint32_t selected, float softmaxScale);
extern "C" void qsfa_official_q8c4_launch(
    aclrtStream stream, void* query, void* cache, void* indices, void* blockTable,
    void* cuQ, void* kvLen, void* output, void* workspace, uint32_t heads,
    uint32_t cacheRows, uint32_t selected, float softmaxScale);
