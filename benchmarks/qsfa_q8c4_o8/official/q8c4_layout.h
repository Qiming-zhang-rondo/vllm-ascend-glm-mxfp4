// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: CANN-Open-Software-License-Agreement-2.0
// Candidate-only producer/consumer layout; offsets and sizes are BYTES.
#ifndef QSFA_OFFICIAL_Q8C4_LAYOUT_H
#define QSFA_OFFICIAL_Q8C4_LAYOUT_H
#include <cstdint>

#ifndef QSFA_Q8C4_CANDIDATE
#define QSFA_Q8C4_CANDIDATE 0
#endif

namespace qsfa_q8c4_layout {
constexpr uint32_t M = 64;
constexpr uint32_t N = 64;
constexpr uint32_t K = 128;
constexpr uint32_t V = 0;
constexpr uint32_t ROPE = V + N * 512 * 2;
constexpr uint32_t K_DATA = ROPE + N * 64 * 2;
constexpr uint32_t K_SCALE = K_DATA + N * 512;
constexpr uint32_t KV_BYTES = K_SCALE + N * 16;
constexpr uint32_t Q_BASE = 3 * KV_BYTES;
constexpr uint32_t Q_DATA = 0;
constexpr uint32_t Q_SCALE = Q_DATA + M * 512;
constexpr uint32_t Q_ROPE = Q_SCALE + M * 16;
constexpr uint32_t Q_BYTES = Q_ROPE + M * 64 * 2;
constexpr uint32_t L1_BYTES = Q_BASE + Q_BYTES;
constexpr uint32_t Q_READY_FLAG = 14;
constexpr uint32_t Q_ROW_BYTES = 608;
constexpr uint32_t CACHE_ROW_BYTES = 416;

// Four independent K=128 chunks; DN2NZ represents each E8M0 pair using a
// bfloat16 byte carrier. This is not numeric BF16 scale storage.
constexpr uint32_t MxScale(uint32_t row, uint32_t groupInChunk)
{
    return ((row / 16) * 32 + (groupInChunk / 2) * 16 + row % 16) * 2 + groupInChunk % 2;
}
constexpr uint32_t Nz(uint32_t row, uint32_t col, uint32_t rows, uint32_t c0)
{
    return (col / c0) * rows * c0 + row * c0 + col % c0;
}
static_assert(KV_BYTES == 107520 && Q_BYTES == 41984);
static_assert(L1_BYTES == 364544 && L1_BYTES <= 512 * 1024);
static_assert(Q_BASE % 32 == 0 && K_DATA % 32 == 0 && K_SCALE % 32 == 0);
static_assert(Q_SCALE % 32 == 0 && Q_ROPE % 32 == 0);
} // namespace qsfa_q8c4_layout
#endif
