# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of native FP8 QSFA inputs, never a native/NPU execution test."""

import sys
import types
import unittest
from unittest.mock import patch

import torch

from benchmarks.qsfa_fake_quant.reference import synthetic_inputs
from benchmarks.qsfa_q8c4_o8 import native_baseline as native


class NativeBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_payload_offsets_fp32_scales_and_shared_value_pa(self):
        query = torch.ones((1, 8, 576), dtype=torch.bfloat16)
        nope = torch.tensor([448.0, 224.0, 112.0, 56.0], dtype=torch.bfloat16).repeat_interleave(128)
        kv = torch.empty((128, 576), dtype=torch.bfloat16)
        kv[:, :512] = nope
        kv[:, 512:] = torch.arange(64, dtype=torch.bfloat16)
        indices = torch.arange(128, dtype=torch.int64).reshape(1, -1)
        inputs, decoded, contract = native.prepare_inputs(query, kv, indices, 576**-0.5)
        self.assertIs(inputs["key"], inputs["value"])
        cache = inputs["key"]
        self.assertEqual(cache.dtype, torch.float8_e4m3fn)
        self.assertEqual(cache.shape, (1, 256, 1, 656))
        raw = cache.view(torch.uint8).reshape(256, 656)
        torch.testing.assert_close(raw[:128, :512], torch.full((128, 512), 0x7E, dtype=torch.uint8))
        torch.testing.assert_close(raw[:128, 512:640].contiguous().view(torch.bfloat16), kv[:, 512:])
        torch.testing.assert_close(
            raw[:128, 640:656].contiguous().view(torch.float32),
            torch.tensor([1.0, 0.5, 0.25, 0.125]).expand(128, -1),
        )
        self.assertTrue(bool((raw[128:] == 0).all()))
        torch.testing.assert_close(decoded, kv.float(), rtol=0, atol=0)
        self.assertEqual(inputs["layout_query"], "TND")
        self.assertEqual(inputs["sparse_indices"].shape, (1, 1, 128))
        self.assertEqual(inputs["actual_seq_lengths_kv"].tolist(), [128])
        self.assertEqual(inputs["block_table"].tolist(), [[0]])
        self.assertFalse(contract["is_mxfp8_d32_e8m0"])
        self.assertFalse(contract["device_quantizer_verified"])
        self.assertEqual(contract["cache_offsets_bytes"], {"nope": 0, "rope": 512, "scale": 640})
        for value in inputs.values():
            if isinstance(value, torch.Tensor):
                self.assertEqual(value.device.type, "cpu")
                self.assertTrue(value.is_contiguous())

    def test_rne_ties_and_zero_blocks_preserve_official_producer_contract(self):
        nope = torch.full((2, 512), 448.0, dtype=torch.bfloat16)
        nope[0, :4] = torch.tensor([1.0625, 1.1875, -1.0625, -1.1875])
        nope[1, :128] = 0
        nope[1, 0] = -0.0
        payload, scales = native.quantize_nope(nope)
        self.assertEqual(payload.view(torch.uint8)[0, :4].tolist(), [0x38, 0x3A, 0xB8, 0xBA])
        self.assertEqual(scales[0].tolist(), [1.0] * 4)
        self.assertEqual(scales[1].tolist(), [0.0, 1.0, 1.0, 1.0])
        self.assertTrue(bool((payload.view(torch.uint8)[1, :128] == 0).all()))

    def test_d128_scale_is_not_d32_or_power_of_two(self):
        nope = torch.ones((2, 512), dtype=torch.bfloat16)
        nope[0, 0] = 300.0
        nope[1, 0] = 150.0
        _, scales = native.quantize_nope(nope)
        self.assertEqual(scales.shape, (2, 4))
        torch.testing.assert_close(scales[:, 0], torch.tensor([300.0, 150.0]) / 448.0, rtol=0, atol=0)
        torch.testing.assert_close(scales[:, 1:], torch.full((2, 3), 1.0 / 448.0), rtol=0, atol=0)

    def test_page_padding_retains_real_k_and_indices(self):
        q, kv, indices, scale = synthetic_inputs(1, 300, 16, 256, 129)
        inputs, decoded, contract = native.prepare_inputs(q, kv, indices, scale)
        self.assertEqual(inputs["key"].shape, (2, 256, 1, 656))
        self.assertEqual(decoded.shape, (300, 576))
        self.assertEqual(inputs["block_table"].tolist(), [[0, 1]])
        self.assertEqual(inputs["actual_seq_lengths_kv"].tolist(), [300])
        torch.testing.assert_close(inputs["sparse_indices"].reshape(1, -1).long(), indices)
        self.assertEqual(contract["padded_key_tokens"], 512)
        raw = inputs["key"].view(torch.uint8).reshape(512, 656)
        self.assertTrue(bool((raw[300:] == 0).all()))

    def test_golden_uses_decoded_payload_and_bf16_probability(self):
        q, kv, indices, scale = synthetic_inputs(1, 128, 8, 128, 17)
        _, decoded, _ = native.prepare_inputs(q, kv, indices, scale)
        output = native.reference_output(q, decoded, indices, scale)
        selected = decoded[indices[0]].bfloat16().float()
        scores = q[0].float() @ selected.T * scale
        exponentials = (scores - scores.amax(-1, keepdim=True)).exp()
        p = (exponentials / exponentials.sum(-1, keepdim=True)).bfloat16().float()
        expected = (p @ selected[:, :512]).bfloat16().float().unsqueeze(0)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertEqual(output.shape, (1, 8, 512))

    def test_invalid_inputs_scales_or_payload_fail_on_cpu(self):
        q, kv, indices, scale = synthetic_inputs(1, 128, 8, 128, 17)
        bad_indices = indices.clone()
        bad_indices[0, 0] = -1
        for bad_q, bad_kv, bad_idx in (
            (q[:, :4], kv, indices),
            (q, torch.full_like(kv, float("inf")), indices),
            (q, kv, bad_indices),
        ):
            with self.assertRaises(ValueError):
                native.prepare_inputs(bad_q, bad_kv, bad_idx, scale)
        inputs, _, _ = native.prepare_inputs(q, kv, indices, scale)
        for invalid_scale in (-1.0, float("nan")):
            raw = inputs["key"].view(torch.uint8).clone().reshape(256, 656)
            raw[0, 640:656] = torch.full((4,), invalid_scale, dtype=torch.float32).view(torch.uint8)
            with self.assertRaises(ValueError):
                native.decode_cache(raw.reshape(1, 256, 1, 656).view(torch.float8_e4m3fn), 128)
        raw = inputs["key"].view(torch.uint8).clone()
        raw[0, 0, 0, 0] = 0x7F
        with self.assertRaises(ValueError):
            native.decode_cache(raw.view(torch.float8_e4m3fn), 128)

    def test_load_operation_uses_only_installed_public_api(self):
        module = types.ModuleType("torch_npu")
        sentinel = lambda **_: None
        module.npu_kv_quant_sparse_flash_attention = sentinel
        with patch.dict(sys.modules, {"torch_npu": module}):
            self.assertIs(native.load_operation(), sentinel)
            del module.npu_kv_quant_sparse_flash_attention
            with self.assertRaisesRegex(RuntimeError, "no fallback or install"):
                native.load_operation()


if __name__ == "__main__":
    unittest.main()
