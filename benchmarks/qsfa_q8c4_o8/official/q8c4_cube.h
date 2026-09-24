// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: CANN-Open-Software-License-Agreement-2.0
#ifndef QSFA_OFFICIAL_Q8C4_CUBE_H
#define QSFA_OFFICIAL_Q8C4_CUBE_H
#include "q8c4_layout.h"

namespace BaseApi {
// The official Buffer policies still own every L0 slot and its hard events.
// Only QK's operand loads/compute differ: E4M3/E8M0 MX NoPE and BF16 RoPE
// accumulate into the SAME FP32 L0C. PV retains the official BF16 MatmulN.
__aicore__ inline LoadData2DParamsV2 QsfaMxNtLoad(uint32_t rows, uint32_t sourceRows,
                                                uint32_t k, uint32_t c0)
{
    LoadData2DParamsV2 p{};
    p.mStep = (rows + 15) / 16;
    p.kStep = k / c0;
    p.srcStride = sourceRows / 16;
    p.dstStride = (rows + 15) / 16;
    p.ifTranspose = false;
    return p;
}
__aicore__ inline LoadData2DMxParams QsfaMxScaleLoad(uint32_t rows)
{
    LoadData2DMxParams p{};
    p.xStep = (rows + 15) / 16;
    p.yStep = 2; // K128 / group32 / two E8M0 bytes per carrier.
    p.srcStride = 2;
    p.dstStride = 2;
    return p;
}

template <typename L0AType, typename L0BType>
__aicore__ inline void QsfaMxQk(const LocalTensor<uint8_t> &qL1,
                                const LocalTensor<uint8_t> &kvL1,
                                L0AType &aBuffers, L0BType &bBuffers,
                                const LocalTensor<float> &cL0,
                                uint32_t rows, uint32_t tokens)
{
    using namespace qsfa_q8c4_layout;
    for (uint32_t chunk = 0; chunk < 4; ++chunk) {
        auto aBuf = aBuffers.Get();
        auto bBuf = bBuffers.Get();
        aBuf.template Wait<HardEvent::M_MTE1>();
        bBuf.template Wait<HardEvent::M_MTE1>();
        auto aL0 = aBuf.template GetTensor<mx_fp8_e4m3_t>();
        auto bL0 = bBuf.template GetTensor<mx_fp8_e4m3_t>();
        auto aData = qL1[Q_DATA + chunk * M * K].template ReinterpretCast<fp8_e4m3fn_t>();
        auto bData = kvL1[K_DATA + chunk * N * K].template ReinterpretCast<fp8_e4m3fn_t>();
        auto aScale = qL1[Q_SCALE + chunk * M * 4].template ReinterpretCast<fp8_e8m0_t>();
        auto bScale = kvL1[K_SCALE + chunk * N * 4].template ReinterpretCast<fp8_e8m0_t>();
        LoadData(aL0, aData, aScale, QsfaMxNtLoad(rows, M, K, 32), QsfaMxScaleLoad(rows));
        LoadData(bL0, bData, bScale, QsfaMxNtLoad(tokens, N, K, 32), QsfaMxScaleLoad(tokens));
        bBuf.template Set<HardEvent::MTE1_M>();
        bBuf.template Wait<HardEvent::MTE1_M>();
        MmadParams p{};
        p.m = rows == 1 ? 16 : rows;
        p.n = tokens;
        p.k = K;
        p.cmatrixInitVal = chunk == 0;
        p.cmatrixSource = false;
        Mmad(cL0, aL0, bL0, p);
        aBuf.template Set<HardEvent::M_MTE1>();
        bBuf.template Set<HardEvent::M_MTE1>();
    }

    auto aBuf = aBuffers.Get();
    auto bBuf = bBuffers.Get();
    aBuf.template Wait<HardEvent::M_MTE1>();
    bBuf.template Wait<HardEvent::M_MTE1>();
    auto aL0 = aBuf.template GetTensor<bfloat16_t>();
    auto bL0 = bBuf.template GetTensor<bfloat16_t>();
    auto aRope = qL1[Q_ROPE].template ReinterpretCast<bfloat16_t>();
    auto bRope = kvL1[ROPE].template ReinterpretCast<bfloat16_t>();
    LoadData(aL0, aRope, QsfaMxNtLoad(rows, M, 64, 16));
    LoadData(bL0, bRope, QsfaMxNtLoad(tokens, N, 64, 16));
    bBuf.template Set<HardEvent::MTE1_M>();
    bBuf.template Wait<HardEvent::MTE1_M>();
    MmadParams p{};
    p.m = rows == 1 ? 16 : rows;
    p.n = tokens;
    p.k = 64;
    p.cmatrixInitVal = false;
    p.cmatrixSource = false;
    Mmad(cL0, aL0, bL0, p);
    aBuf.template Set<HardEvent::M_MTE1>();
    bBuf.template Set<HardEvent::M_MTE1>();
}
} // namespace BaseApi
#endif
