# SPDX-License-Identifier: Apache-2.0
"""Compile the actual kernel scalar codec with a host C++ compiler.

This verifies byte/rounding math only. It cannot verify ASC compilation,
instruction lowering, SIMD/SIMT execution spaces, Cube layout, or NPU behavior.
"""

import ctypes
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]


class CppCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if not compiler:
            raise unittest.SkipTest("Host C++ compiler unavailable")
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        source = Path(cls.directory.name) / "codec.cpp"
        source.write_text(
            '#include "codec.h"\n'
            'extern "C" void encode(const float* x, uint8_t* y, int n) {\n'
            "for(int i=0;i<n;++i) y[i]=qsfa_codec::EncodeFp8(x[i]); }\n"
            'extern "C" void decode(const uint8_t* x, float* y, int n) {\n'
            "for(int i=0;i<n;++i) y[i]=qsfa_codec::DecodeFp8(x[i]); }\n"
            'extern "C" void bf16(const float* x, uint16_t* y, int n) {\n'
            "for(int i=0;i<n;++i) y[i]=qsfa_codec::Bf16(x[i]); }\n"
            'extern "C" void scale(const uint8_t* x, float* y, int n) {\n'
            "for(int i=0;i<n;++i) y[i]=qsfa_codec::Scale(x[i]); }\n"
            'extern "C" void exponent(const float* x, uint8_t* y, int n) {\n'
            "for(int i=0;i<n;++i) y[i]=qsfa_codec::SharedExponent(x[i]); }\n"
            'extern "C" void expand(const uint8_t* x, uint8_t* y, int n) {\n'
            "for(int i=0;i<n;++i) y[i]=qsfa_codec::ExpandFp4(x[i]); }\n"
        )
        library = Path(cls.directory.name) / "codec.so"
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O2",
                "-shared",
                "-fPIC",
                "-I",
                str(ROOT / "benchmarks/qsfa_q8c4_o8/csrc"),
                str(source),
                "-o",
                str(library),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        cls.library = ctypes.CDLL(str(library))
        for name in ("encode", "decode", "bf16", "scale", "exponent", "expand"):
            method = getattr(cls.library, name)
            method.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            method.restype = None

    def apply_codec(self, name, source, dtype):
        source = source.contiguous()
        result = torch.empty(source.shape, dtype=dtype)
        getattr(self.library, name)(source.data_ptr(), result.data_ptr(), source.numel())
        return result

    def test_all_fp8_codes_decode_and_all_finite_codes_roundtrip(self):
        codes = torch.arange(256, dtype=torch.int32).to(torch.uint8)
        reference = codes.view(torch.float8_e4m3fn).float()
        decoded = self.apply_codec("decode", codes, torch.float32)
        torch.testing.assert_close(decoded, reference, rtol=0, atol=0, equal_nan=True)
        finite = torch.isfinite(reference)
        torch.testing.assert_close(self.apply_codec("encode", decoded[finite], torch.uint8), codes[finite])
        self.assertTrue(decoded[128].signbit())

    def test_fp8_midpoints_neighbours_and_saturation_match_torch(self):
        positive = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        midpoints = (positive[:-1] + positive[1:]) / 2
        values = torch.cat(
            (
                positive,
                midpoints,
                torch.nextafter(midpoints, torch.full_like(midpoints, float("inf"))),
                torch.nextafter(midpoints, torch.zeros_like(midpoints)),
                torch.tensor([0.0, 2**-149, 500, torch.finfo(torch.float32).max]),
            )
        )
        values = torch.cat((values, -values))
        reference = values.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        torch.testing.assert_close(self.apply_codec("encode", values, torch.uint8), reference)

    def test_all_e8m0_scales_and_fp4_expand_values(self):
        codes = torch.arange(255, dtype=torch.int32).to(torch.uint8)
        reference = torch.ldexp(torch.ones(255), codes.int() - 127)
        torch.testing.assert_close(self.apply_codec("scale", codes, torch.float32), reference, rtol=0, atol=0)
        codes = torch.arange(16, dtype=torch.uint8)
        expected = [0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C]
        reference = torch.tensor(expected + [x | 0x80 for x in expected], dtype=torch.uint8)
        torch.testing.assert_close(self.apply_codec("expand", codes, torch.uint8), reference)

    def test_shared_exponents_at_all_fp32_boundaries_and_subnormals(self):
        powers = torch.ldexp(torch.ones(277), torch.arange(-149, 128, dtype=torch.int32))
        values = torch.cat((powers, torch.nextafter(powers, torch.zeros_like(powers)), torch.tensor([0.0])))
        _, exponents = torch.frexp(values)
        reference = ((exponents - 1 - 8).clamp_min(-127) + 127).to(torch.uint8)
        reference[values == 0] = 0
        torch.testing.assert_close(self.apply_codec("exponent", values, torch.uint8), reference)

    def test_bf16_rounding_at_midpoints_and_random_float_bits(self):
        generator = torch.Generator().manual_seed(734)
        bits = torch.randint(0, 2**32, (40000,), generator=generator, dtype=torch.int64).to(torch.uint32)
        values = bits.view(torch.float32)
        values = values[torch.isfinite(values)]
        values = torch.cat((values, torch.tensor([0.0, -0.0, 1.00390625, 1.01171875])))
        reference = values.bfloat16().view(torch.uint16)
        torch.testing.assert_close(self.apply_codec("bf16", values, torch.uint16), reference)


if __name__ == "__main__":
    unittest.main()
