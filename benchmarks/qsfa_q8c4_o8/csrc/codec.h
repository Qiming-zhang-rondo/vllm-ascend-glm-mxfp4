// SPDX-License-Identifier: Apache-2.0
// Byte-level interchange codec. The same functions are CPU-tested and inlined
// into SIMT VFs. No assumptions about torch/CANN float4 storage aliases.
#pragma once
#include <cstdint>

#ifndef QSFA_INLINE
#define QSFA_INLINE inline
#endif

namespace qsfa_codec {
QSFA_INLINE uint32_t Bits(float x)
{
    union { float f; uint32_t u; } v;
    v.f = x;
    return v.u;
}
QSFA_INLINE float Float(uint32_t x)
{
    union { uint32_t u; float f; } v;
    v.u = x;
    return v.f;
}
QSFA_INLINE float Abs(float x) { return Float(Bits(x) & 0x7fffffffU); }
QSFA_INLINE bool Finite(float x) { return (Bits(x) & 0x7f800000U) != 0x7f800000U; }
QSFA_INLINE float Scale(uint8_t e)
{
    // E8M0 exponent 0 is 2^-127 (NOT zero); 255 is NaN.
    return e == 0 ? Float(0x00400000U) : (e == 255 ? Float(0x7fc00000U) : Float(uint32_t(e) << 23));
}
QSFA_INLINE uint8_t ExpandFp4(uint8_t nibble)
{
    const uint8_t magnitudes[8] = {0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c};
    return magnitudes[nibble & 7U] | ((nibble & 8U) << 4);
}
QSFA_INLINE float DecodeFp8(uint8_t code)
{
    const uint32_t e = (code >> 3) & 15U;
    const uint32_t f = code & 7U;
    float value;
    if (e == 0) value = float(f) * (1.0f / 512.0f);
    else if (e == 15 && f == 7) value = Float(0x7fc00000U);
    else value = Float((e + 120U) << 23) * (1.0f + float(f) * 0.125f);
    return Float(Bits(value) | (uint32_t(code & 128U) << 24));
}
QSFA_INLINE uint16_t Bf16(float value)
{
    uint32_t bits = Bits(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) return uint16_t(bits >> 16) | 0x0040U;
    return uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
QSFA_INLINE uint32_t RoundEven(float x)
{
    const uint32_t integer = uint32_t(x);
    const float fraction = x - float(integer);
    return integer + ((fraction > 0.5f || (fraction == 0.5f && (integer & 1U))) ? 1U : 0U);
}
QSFA_INLINE uint8_t EncodeFp8(float value)
{
    const uint8_t sign = uint8_t((Bits(value) >> 24) & 128U);
    float x = Abs(value);
    if (!Finite(x)) return sign | 0x7fU;
    if (x >= 448.0f) return sign | 0x7eU;
    if (x < 1.0f / 64.0f) return sign | uint8_t(RoundEven(x * 512.0f));
    int exponent = int((Bits(x) >> 23) & 255U) - 127;
    const float step = Scale(uint8_t(exponent - 3 + 127));
    uint32_t significand = RoundEven(x / step);
    if (significand == 16U) { ++exponent; significand = 8U; }
    return sign | uint8_t((uint32_t(exponent + 7) << 3) | (significand - 8U));
}
QSFA_INLINE uint8_t SharedExponent(float maximum)
{
    if (!Finite(maximum)) return 255;
    if (maximum == 0.0f) return 0;
    const uint32_t bits = Bits(maximum);
    int exponent = int(bits >> 23) - 127;
    if ((bits >> 23) == 0) {
        uint32_t mantissa = bits & 0x7fffffU;
        exponent = -149;
        while (mantissa > 1U) { mantissa >>= 1; ++exponent; }
    }
    exponent -= 8; // scale_alg=0: floor(log2(amax)) - E4M3 max exponent.
    if (exponent < -127) exponent = -127;
    return uint8_t(exponent + 127);
}
} // namespace qsfa_codec
