# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU checks of fake-quantization inputs; these do not execute QSFA or CANN."""

import importlib.util
import unittest
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[3] / "benchmarks/qsfa_fake_quant/quantization.py"
SPEC = importlib.util.spec_from_file_location("qsfa_quantization_under_test", MODULE_PATH)
quantization = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quantization)


class QsfaQuantizationTests(unittest.TestCase):
    def test_fp4_ties_away_saturation_and_signed_zero(self):
        x = torch.full((2, 32), 6.0)
        x[0, :9] = torch.tensor([0.0, -0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        x[1, :9] = -x[0, :9]
        x[:, 9] = torch.tensor([7.5, -7.5])
        result = quantization.mxfp4_roundtrip(x)
        expected = torch.tensor([0.0, -0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 6.0])
        torch.testing.assert_close(result[0, :10], expected, rtol=0, atol=0)
        torch.testing.assert_close(result[1, :10], -expected, rtol=0, atol=0)
        torch.testing.assert_close(result[:, :2].signbit(), x[:, :2].signbit())

    def test_fp8_rint_uses_e8m0_group_scale_and_saturates(self):
        x = torch.full((1, 32), 256.0)
        x[0, :6] = torch.tensor([1.0625, 1.1875, -1.0625, -1.1875, 500.0, -500.0])
        result = quantization.mxfp8_roundtrip(x)
        # max=500: floor(log2(500))-8 == 0, so scale=1, not 500/448.
        expected = torch.tensor([1.0, 1.25, -1.0, -1.25, 448.0, -448.0])
        torch.testing.assert_close(result[0, :6], expected, rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(result).all()))

    def test_group32_scales_tail_padding_and_noncontiguous_input(self):
        x = torch.empty((2, 65))
        x[:, :32] = 7.5
        x[:, 32:64] = 15.0
        x[:, 64:] = 30.0
        for function, expected_groups in (
            (quantization.mxfp4_roundtrip, (6.0, 12.0, 24.0)),
            (quantization.mxfp8_roundtrip, (7.0, 14.0, 28.0)),
        ):
            expected = torch.empty_like(x)
            expected[:, :32], expected[:, 32:64], expected[:, 64:] = expected_groups
            storage = torch.empty((2, 130))
            storage[:, ::2] = x
            source = storage[:, ::2]
            self.assertFalse(source.is_contiguous())
            result = function(source)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            self.assertTrue(result.is_contiguous())
            self.assertEqual(result.shape, source.shape)

    def test_zero_blocks_and_float32_subnormals(self):
        x = torch.zeros((2, 33), dtype=torch.float32)
        x[0, 0] = -0.0
        x[1, :2] = torch.tensor([2**-149, -(2**-149)])
        for function in (quantization.mxfp4_roundtrip, quantization.mxfp8_roundtrip):
            result = function(x)
            self.assertTrue(bool((result == 0).all()))
            torch.testing.assert_close(result.signbit(), x.signbit())

    def test_finite_extremes_all_supported_dtypes_and_source_unchanged(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                maximum = torch.finfo(dtype).max
                x = torch.tensor([[maximum, -maximum, 0.0, -0.0]], dtype=dtype)
                original = x.clone()
                for function in (quantization.mxfp4_roundtrip, quantization.mxfp8_roundtrip):
                    result = function(x)
                    self.assertEqual(result.dtype, torch.float32)
                    self.assertEqual(result.device.type, "cpu")
                    self.assertTrue(bool(torch.isfinite(result).all()))
                    self.assertGreater(result[0, 0].item(), 0)
                    self.assertEqual(result[0, 0].item(), -result[0, 1].item())
                    torch.testing.assert_close(result[:, 2:].signbit(), x[:, 2:].signbit())
                    torch.testing.assert_close(x, original, rtol=0, atol=0)

    def test_power_of_two_boundaries_do_not_round_scale_up(self):
        below = torch.nextafter(torch.tensor(8.0), torch.tensor(0.0))
        x = torch.zeros((2, 32))
        x[0, 0], x[1, 0] = below, 8.0
        torch.testing.assert_close(quantization.mxfp4_roundtrip(x)[:, 0], torch.tensor([6.0, 8.0]), rtol=0, atol=0)
        torch.testing.assert_close(quantization.mxfp8_roundtrip(x)[:, 0], torch.tensor([7.0, 8.0]), rtol=0, atol=0)

    def test_invalid_input_is_rejected_before_quantization(self):
        for function in (quantization.mxfp4_roundtrip, quantization.mxfp8_roundtrip):
            for source in (
                torch.tensor(1.0),
                torch.empty((2, 0)),
                torch.zeros((1, 32), dtype=torch.int8),
                torch.zeros((1, 32), dtype=torch.float64),
                torch.empty((1, 32), device="meta"),
                torch.tensor([float("nan")]),
                torch.tensor([float("inf")]),
            ):
                with (
                    self.subTest(function=function.__name__, shape=source.shape, dtype=source.dtype),
                    self.assertRaises(ValueError),
                ):
                    function(source)
            with self.assertRaises(TypeError):
                function([1.0])


if __name__ == "__main__":
    unittest.main()
