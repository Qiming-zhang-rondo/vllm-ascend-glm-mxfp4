// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#ifndef QSFA_LAYOUT_INLINE
#define QSFA_LAYOUT_INLINE inline
#endif

namespace qsfa_tiled {
constexpr uint32_t M = 16;
constexpr uint32_t N = 64;
constexpr uint32_t K = 128;
constexpr uint32_t THREADS = 256;
constexpr uint32_t MAX_BLOCKS = 32;
constexpr uint32_t HALF_N = N / 2;
constexpr uint32_t HALF_K = K / 2;

// ND [rows, reduction] -> NZ [reduction/C0, rows, C0].
QSFA_LAYOUT_INLINE uint32_t Nz(uint32_t row, uint32_t col, uint32_t rows, uint32_t c0)
{
    return (col / c0) * rows * c0 + row * c0 + col % c0;
}

// Dn2Nz on BF16 carriers, each holding TWO E8M0 bytes. Four scales per
// K128 chunk: NZ [rows/16, 2 pairs, 16 rows, 2 bytes]. No numeric BF16 cast.
QSFA_LAYOUT_INLINE uint32_t MxScale(uint32_t row, uint32_t group)
{
    return ((row / 16) * 2 * 16 + (group / 2) * 16 + row % 16) * 2 + group % 2;
}

// Both AIVs and their AIC use these SAME byte offsets. All are 32B aligned.
constexpr uint32_t Q_DATA = 0;
constexpr uint32_t K_DATA = Q_DATA + M * 512;
constexpr uint32_t Q_SCALE = K_DATA + N * 512;
constexpr uint32_t K_SCALE = Q_SCALE + M * 16;
constexpr uint32_t Q_ROPE = K_SCALE + N * 16;
constexpr uint32_t K_ROPE = Q_ROPE + M * 64 * 2;
constexpr uint32_t QK_L1_BYTES = K_ROPE + N * 64 * 2;

// Per-AIV staging: full Q (AIV0 only), half of K/scale/RoPE.
constexpr uint32_t UQ = 0;
constexpr uint32_t UK = UQ + M * 512;
constexpr uint32_t UQS = UK + HALF_N * 512;
constexpr uint32_t UKS = UQS + M * 16;
constexpr uint32_t UQR = UKS + HALF_N * 16;
constexpr uint32_t UKR = UQR + M * 64 * 2;
constexpr uint32_t QK_UB_BYTES = UKR + HALF_N * 64 * 2;

constexpr uint32_t PV_P = 0;
constexpr uint32_t PV_V = M * K * 2;
constexpr uint32_t PV_L1_BYTES = PV_V + N * K * 2;
constexpr uint32_t PV_DECODE_UB_BYTES = N * HALF_K * 2;
constexpr uint32_t PV_OUT_UB = PV_DECODE_UB_BYTES;
constexpr uint32_t PV_UB_BYTES = PV_OUT_UB + (M / 2) * N * 4;
// LocalTensor's fixed offsets describe addresses; they do NOT reserve UB for
// a SIMD/SIMT kernel. Both native launches must request this dynamic region.
// 32 KiB covers both stages, leaving space for the 8 KiB runtime reserve and
// at least 32 KiB SIMT DCache (the hybrid limit is 216 KiB, not 248 KiB).
constexpr uint32_t DYNAMIC_UB_BYTES = 32 * 1024;
static_assert(QK_L1_BYTES <= 512 * 1024 && PV_L1_BYTES <= 512 * 1024);
static_assert(QK_UB_BYTES <= DYNAMIC_UB_BYTES && PV_UB_BYTES <= DYNAMIC_UB_BYTES);
static_assert(DYNAMIC_UB_BYTES <= (256 - 8 - 32) * 1024);
static_assert((Q_DATA | K_DATA | Q_SCALE | K_SCALE | Q_ROPE | K_ROPE |
               UQ | UK | UQS | UKS | UQR | UKR | PV_P | PV_V | PV_OUT_UB) % 32 == 0);
}  // namespace qsfa_tiled
