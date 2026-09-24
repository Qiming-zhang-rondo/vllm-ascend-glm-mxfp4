// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "kernel_operator.h"
#include "launch.h"
#define QSFA_TILING_INLINE __aicore__ inline
#include "tiling.h"
#define KVQSFA_VERSION (-1)
#include "vendor/attention/kv_quant_sparse_flash_attention/op_kernel/arch35/kv_quant_sparse_flash_attention_kernel_mla_arch35.h"

// Adapted from upstream kv_quant_sparse_flash_attention.cpp::QSFA_OP_IMPL.
// Only the wrapper replaces generated CANN dispatch/tiling. The class below is
// the vendored official fused producer/QK/online-softmax/PV/output pipeline.
__global__ __aicore__ void QSFA_OFFICIAL_KERNEL(
    GM_ADDR query, GM_ADDR cache, GM_ADDR indices, GM_ADDR blockTable,
    GM_ADDR cuQ, GM_ADDR kvLen, GM_ADDR output, GM_ADDR workspace,
    uint32_t heads, uint32_t cacheRows, uint32_t selected, float softmaxScale)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
#if QSFA_Q8C4_CANDIDATE
    using CacheElement = uint8_t;
#else
    using CacheElement = fp8_e4m3fn_t;
#endif
    using Cube = BaseApi::QSFAMatmulService<bfloat16_t, CacheElement, float, bfloat16_t,
        false, true, QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND, QSFATemplateMode::CFA_TEMPLATE_MODE, false, true>;
    using CubeDummy = BaseApi::QSFAMatmulServiceDummy<bfloat16_t, CacheElement, float, bfloat16_t,
        false, true, QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND, QSFATemplateMode::CFA_TEMPLATE_MODE, false, true>;
    using Vec = BaseApi::QSFAVectorService<bfloat16_t, CacheElement, float, bfloat16_t,
        false, true, QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND, QSFATemplateMode::CFA_TEMPLATE_MODE, false, true>;
    using VecDummy = BaseApi::QSFAVectorServiceDummy<bfloat16_t, CacheElement, float, bfloat16_t,
        false, true, QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND, QSFATemplateMode::CFA_TEMPLATE_MODE, false, true>;
    using CubeBlock = typename std::conditional<g_coreType == AscendC::AIC, Cube, CubeDummy>::type;
    using VecBlock = typename std::conditional<g_coreType == AscendC::AIC, VecDummy, Vec>::type;
    BaseApi::KvQuantSparseFlashAttentionMla<CubeBlock, VecBlock> op;
#if defined(__DAV_C310_CUBE__)
    const KvQuantSparseFlashAttentionTilingDataMla* tiling = nullptr;
#else
    const auto localTiling = qsfa_official_contract::MakeTiling(
        heads, cacheRows, selected, softmaxScale, QSFA_Q8C4_CANDIDATE != 0);
    const KvQuantSparseFlashAttentionTilingDataMla* tiling = &localTiling;
#endif
    // workspace is already user workspace, not ACLNN's prefixed system region.
    // Combined cache embeds scales; separate key/value scale pointers unused.
    op.Init(query, cache, cache, indices, nullptr, nullptr, blockTable, cuQ, kvLen,
            output, workspace, tiling, &pipe);
    op.Process();
}

extern "C" void QSFA_OFFICIAL_LAUNCH(
    aclrtStream stream, void* query, void* cache, void* indices, void* blockTable,
    void* cuQ, void* kvLen, void* output, void* workspace, uint32_t heads,
    uint32_t cacheRows, uint32_t selected, float softmaxScale)
{
    // Q=1 and G<=64 yield exactly one official outer work item, so no split-KV
    // reduction or inter-Cube KV scratch is needed. This is not a claim that
    // the installed CANN binary chooses this same specialization.
#if QSFA_Q8C4_CANDIDATE
    constexpr uint32_t ubBytes = qsfa_official_contract::CANDIDATE_DYNAMIC_UB_BYTES;
#else
    // Native launch makes TPipe's dynamic UB allocation explicit. This TU has
    // no SIMT functions, so it does not reserve the SIMT 32-KiB Data Cache.
    constexpr uint32_t ubBytes = qsfa_official_contract::CONTROL_DYNAMIC_UB_BYTES;
#endif
    QSFA_OFFICIAL_KERNEL<<<1, ubBytes, stream>>>(
        (GM_ADDR)query, (GM_ADDR)cache, (GM_ADDR)indices, (GM_ADDR)blockTable,
        (GM_ADDR)cuQ, (GM_ADDR)kvLen, (GM_ADDR)output, (GM_ADDR)workspace,
        heads, cacheRows, selected, softmaxScale);
}
