// SPDX-License-Identifier: Apache-2.0
#pragma once
// Pure per-thread gather/codec math. Host tests exercise these exact producers;
// that does not validate ASC compilation, device memory transfers or Cube layout.
#include "codec.h"
#include "tiled_layout.h"
namespace qsfa_tiled_gather {
using namespace qsfa_tiled;
using namespace qsfa_codec;

__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void PrepareQk(
    __gm__ uint8_t* q, __gm__ uint8_t* qs, __gm__ uint8_t* kv, __gm__ uint8_t* ks,
    __gm__ uint16_t* rope, __gm__ int32_t* idx, __gm__ int32_t* status,
    __ubuf__ uint8_t* ub, uint32_t h, uint32_t cacheRows, uint32_t selected,
    uint32_t mOffset, uint32_t nOffset, uint32_t sub, bool checkStatus)
{
    const uint32_t tid = threadIdx.x;
    if (checkStatus && tid == 0) {
        int32_t invalid = 0;
        for (uint32_t i = 0; i < selected; ++i)
            if (idx[i] < 0 || uint32_t(idx[i]) >= cacheRows) invalid = 1;
        status[0] = invalid;
    }
    if (sub == 0) {
        for (uint32_t i = tid; i < M * 512; i += THREADS) {
            const uint32_t row = i / 512, d = i % 512, chunk = d / K;
            ub[UQ + chunk * M * K + Nz(row, d % K, M, 32)] =
                mOffset + row < h ? q[(mOffset + row) * 576 + d] : 0;
        }
        for (uint32_t i = tid; i < M * 16; i += THREADS) {
            const uint32_t row = i / 16, g = i % 16;
            ub[UQS + (g / 4) * M * 4 + MxScale(row, g % 4)] =
                mOffset + row < h ? qs[(mOffset + row) * 18 + g] : 127;
        }
        auto qr = (__ubuf__ uint16_t*)(ub + UQR);
        for (uint32_t i = tid; i < M * 64; i += THREADS) {
            const uint32_t row = i / 64, d = i % 64;
            qr[Nz(row, d, M, 16)] = mOffset + row < h ?
                Bf16(DecodeFp8(q[(mOffset + row) * 576 + 512 + d]) *
                     Scale(qs[(mOffset + row) * 18 + 16 + d / 32])) : 0;
        }
    }
    for (uint32_t i = tid; i < HALF_N * 512; i += THREADS) {
        const uint32_t row = i / 512, d = i % 512;
        const int32_t slot = idx[nOffset + sub * HALF_N + row];
        uint8_t code = 0;
        if (slot >= 0 && uint32_t(slot) < cacheRows) {
            const uint8_t byte = kv[uint64_t(slot) * 256 + d / 2];
            code = ExpandFp4((byte >> ((d & 1) * 4)) & 15);
        }
        ub[UK + (d / K) * HALF_N * K + Nz(row, d % K, HALF_N, 32)] = code;
    }
    for (uint32_t i = tid; i < HALF_N * 16; i += THREADS) {
        const uint32_t row = i / 16, g = i % 16;
        const int32_t slot = idx[nOffset + sub * HALF_N + row];
        ub[UKS + (g / 4) * HALF_N * 4 + MxScale(row, g % 4)] =
            slot >= 0 && uint32_t(slot) < cacheRows ? ks[uint64_t(slot) * 16 + g] : 127;
    }
    auto kr = (__ubuf__ uint16_t*)(ub + UKR);
    for (uint32_t i = tid; i < HALF_N * 64; i += THREADS) {
        const uint32_t row = i / 64, d = i % 64;
        const int32_t slot = idx[nOffset + sub * HALF_N + row];
        kr[Nz(row, d, HALF_N, 16)] =
            slot >= 0 && uint32_t(slot) < cacheRows ? rope[uint64_t(slot) * 64 + d] : 0;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void PreparePv(
    __gm__ uint8_t* kv, __gm__ uint8_t* ks, __gm__ int32_t* idx,
    __ubuf__ uint16_t* ub, uint32_t cacheRows, uint32_t tokenOffset, uint32_t dOffset)
{
    for (uint32_t i = threadIdx.x; i < HALF_K * N; i += THREADS) {
        const uint32_t token = i / N, d = i % N;
        const int32_t slot = idx[tokenOffset + token];
        uint16_t value = 0;
        if (slot >= 0 && uint32_t(slot) < cacheRows) {
            const uint8_t byte = kv[uint64_t(slot) * 256 + (dOffset + d) / 2];
            const uint8_t code = ExpandFp4((byte >> ((d & 1) * 4)) & 15);
            value = Bf16(DecodeFp8(code) * Scale(ks[uint64_t(slot) * 16 + (dOffset + d) / 32]));
        }
        // Small on-chip transpose only: V operand rows=D64, reduction=tokens64.
        ub[Nz(d, token, N, 16)] = value;
    }
}
}  // namespace qsfa_tiled_gather
