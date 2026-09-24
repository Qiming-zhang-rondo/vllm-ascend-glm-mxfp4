// SPDX-License-Identifier: Apache-2.0
#pragma once
// Candidate-only on-chip producers. They preserve the official NZ/bank-padding
// contract; neither expanded K/V nor scores/P are materialized in global memory.
#include "simt_api/asc_simt.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_functions.h"
#define QSFA_INLINE __simt_callee__ inline __attribute__((always_inline))
#include "../csrc/codec.h"
#undef QSFA_INLINE
#include "q8c4_layout.h"

namespace qsfa_official_vector {
constexpr uint32_t THREADS = 128;
constexpr uint32_t MX_STAGE_BYTES = 16 * 512 + 16 * 16;
constexpr uint32_t OUTPUT_ROW_BYTES = 544;

__simt_callee__ inline uint32_t Nz(uint32_t row, uint32_t col, uint32_t rows, uint32_t c0)
{
    return (col / c0) * rows * c0 + row * c0 + col % c0;
}
__simt_callee__ inline uint32_t MxScale(uint32_t row, uint32_t group)
{
    return ((row / 16) * 32 + (group / 2) * 16 + row % 16) * 2 + group % 2;
}

// E2M1 times an E8M0 power of two has at most two significant bits. Construct
// BF16 directly instead of decoding via FP8 -> FP32 multiply -> BF16 round.
// The two possible subnormal shifts lose no bits. Preserve signed zero,
// E8M0 exponent zero (2^-127), overflow to infinity, and the NaN scale code.
__simt_callee__ inline uint16_t Fp4ScaledBf16(uint8_t nibble, uint8_t scale)
{
    const uint32_t magnitude = nibble & 7U;
    const uint32_t sign = uint32_t(nibble & 8U) << 12;
    if (scale == 255) return 0x7fc0U;
    if (magnitude == 0) return sign;
    const int exponent = int(scale) + int(magnitude >> 1) - 1;
    const uint32_t fraction = magnitude > 1 ? (magnitude & 1U) << 6 : 0U;
    if (exponent <= 0) return sign | ((128U + fraction) >> (1 - exponent));
    if (exponent >= 255) return sign | 0x7f80U;
    return sign | (uint32_t(exponent) << 7) | fraction;
}

__simt_callee__ inline uint8_t Fp4ExpandedFp8(uint8_t nibble)
{
    const uint32_t magnitude = nibble & 7U;
    const uint32_t code = magnitude == 0 ? 0 : (magnitude == 1 ? 0x30 : (magnitude + 12) << 2);
    return code | ((nibble & 8U) << 4);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void DecodeCache(
    __ubuf__ uint8_t* cache, __ubuf__ uint16_t* values,
    __ubuf__ uint8_t* mx, uint32_t rows)
{
    // One thread owns a packed byte (two adjacent values). The NZ pair offsets
    // are naturally aligned: write one B32 V pair and one B16 K pair, avoiding
    // adjacent threads' subword writes and duplicate payload/scale loads.
    auto valuePairs = (__ubuf__ uint32_t*)values;
    auto mxPairs = (__ubuf__ uint16_t*)mx;
    for (uint32_t i = threadIdx.x; i < rows * 256; i += THREADS) {
        const uint32_t row = i / 256, d = (i % 256) * 2;
        const auto base = cache + row * qsfa_q8c4_layout::CACHE_ROW_BYTES;
        const uint8_t packed = base[d / 2], scale = base[384 + d / 32];
        const uint8_t lo = packed & 15U, hi = packed >> 4;
        // Existing PV producer is 17-row bank-padded BF16 NZ in UB.
        valuePairs[Nz(row, d, 17, 16) / 2] =
            uint32_t(Fp4ScaledBf16(lo, scale)) | (uint32_t(Fp4ScaledBf16(hi, scale)) << 16);
        // QK's independent operand retains E8M0 and losslessly expands E2M1.
        mxPairs[((d / 128) * 16 * 128 + Nz(row, d % 128, 16, 32)) / 2] =
            uint16_t(Fp4ExpandedFp8(lo)) | (uint16_t(Fp4ExpandedFp8(hi)) << 8);
    }
    for (uint32_t i = threadIdx.x; i < rows * 16; i += THREADS) {
        const uint32_t row = i / 16, g = i % 16;
        mx[8192 + (g / 4) * 64 + MxScale(row, g % 4)] =
            cache[row * qsfa_q8c4_layout::CACHE_ROW_BYTES + 384 + g];
    }
    for (uint32_t i = threadIdx.x; i < rows * 64; i += THREADS) {
        const uint32_t row = i / 64, d = i % 64;
        const auto rope = (__ubuf__ uint16_t*)(cache + row * qsfa_q8c4_layout::CACHE_ROW_BYTES + 256);
        values[512 * 17 + Nz(row, d, 17, 16)] = rope[d];
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void PrepareQuery(
    __gm__ uint8_t* query, __ubuf__ uint8_t* ub, uint32_t heads)
{
    using namespace qsfa_codec;
    // This initial staging uses the otherwise-idle final FP32 accumulator UB.
    for (uint32_t i = threadIdx.x; i < 64 * 512; i += THREADS) {
        const uint32_t row = i / 512, d = i % 512;
        ub[(d / 128) * 64 * 128 + Nz(row, d % 128, 64, 32)] =
            row < heads ? query[row * 608 + d] : 0;
    }
    for (uint32_t i = threadIdx.x; i < 64 * 16; i += THREADS) {
        const uint32_t row = i / 16, g = i % 16;
        ub[32768 + (g / 4) * 256 + MxScale(row, g % 4)] =
            row < heads ? query[row * 608 + 576 + g] : 127;
    }
    auto rope = (__ubuf__ uint16_t*)(ub + 33792);
    for (uint32_t i = threadIdx.x; i < 64 * 64; i += THREADS) {
        const uint32_t row = i / 64, d = i % 64;
        rope[Nz(row, d, 64, 16)] = row < heads ?
            Bf16(DecodeFp8(query[row * 608 + 512 + d]) * Scale(query[row * 608 + 592 + d / 32])) : 0;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void EncodeOutput(
    __ubuf__ float* acc, __ubuf__ uint8_t* out, uint32_t rows)
{
    using namespace qsfa_codec;
    // Each thread owns a complete D32 group, avoiding cross-warp reductions.
    for (uint32_t group = threadIdx.x; group < rows * 16; group += THREADS) {
        const uint32_t row = group / 16, d = (group % 16) * 32;
        float maximum = 0.0f;
        for (uint32_t j = 0; j < 32; ++j) {
            const float value = acc[row * 512 + d + j];
            if (!Finite(value)) { maximum = Float(0x7fc00000U); break; }
            if (Abs(value) > maximum) maximum = Abs(value);
        }
        const uint8_t exponent = SharedExponent(maximum);
        const float scale = Scale(exponent);
        out[row * OUTPUT_ROW_BYTES + 512 + group % 16] = exponent;
        out[row * OUTPUT_ROW_BYTES + 528 + group % 16] = 0;
        for (uint32_t j = 0; j < 32; ++j)
            out[row * OUTPUT_ROW_BYTES + d + j] = EncodeFp8(acc[row * 512 + d + j] / scale);
    }
}
} // namespace qsfa_official_vector
