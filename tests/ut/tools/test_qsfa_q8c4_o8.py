# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts and mocked orchestration only; no hardware/kernel claims."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.qsfa_fake_quant.quantization import mxfp4_roundtrip, mxfp8_roundtrip
from benchmarks.qsfa_fake_quant.reference import synthetic_inputs
from benchmarks.qsfa_q8c4_o8 import packing, run


class PackingTests(unittest.TestCase):
    def test_fp4_low_nibble_ties_and_signed_zero(self):
        values = torch.full((1, 32), 6.0)
        values[0, :16] = torch.tensor(
            [0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0]
        )
        payload, scales = packing.pack_mxfp4(values)
        torch.testing.assert_close(
            payload[0, :8], torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=torch.uint8)
        )
        self.assertEqual(scales.tolist(), [[127]])
        decoded = packing.decode_mxfp4(payload, scales)
        torch.testing.assert_close(decoded, mxfp4_roundtrip(values), rtol=0, atol=0)
        torch.testing.assert_close(decoded[:, (0, 8)].signbit(), values[:, (0, 8)].signbit())

    def test_fp4_to_fp8_is_bit_exact_and_keeps_scale_and_signed_zero(self):
        payload = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2], dtype=torch.uint8)
        expanded = packing.expand_mxfp4_to_fp8(payload)
        magnitude_bits = [0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C]
        expected = torch.tensor([magnitude_bits + [x | 0x80 for x in magnitude_bits]] * 2, dtype=torch.uint8).reshape(
            1, 32
        )
        torch.testing.assert_close(expanded, expected, rtol=0, atol=0)
        scales = torch.tensor([[125]], dtype=torch.uint8)
        decoded4 = packing.decode_mxfp4(payload, scales)
        decoded8 = packing.decode_mxfp8(expanded, scales)
        torch.testing.assert_close(decoded4, decoded8, rtol=0, atol=0)
        torch.testing.assert_close(decoded4.signbit(), decoded8.signbit())

    def test_nonuniform_d32_scales_zero_and_fp8_rne(self):
        values = torch.tensor([0.0, 0.5, 2.0, 8.0], dtype=torch.bfloat16).repeat_interleave(32).reshape(1, 128)
        for pack, decode, reference, codes in (
            (packing.pack_mxfp4, packing.decode_mxfp4, mxfp4_roundtrip, [0, 124, 126, 128]),
            (packing.pack_mxfp8, packing.decode_mxfp8, mxfp8_roundtrip, [0, 118, 120, 122]),
        ):
            payload, scales = pack(values)
            self.assertEqual(scales.tolist(), [codes])
            torch.testing.assert_close(decode(payload, scales), reference(values), rtol=0, atol=0)
        values = torch.full((1, 32), 500.0)
        values[0, :4] = torch.tensor([1.0625, 1.1875, -1.0625, -1.1875])
        payload, scales = packing.pack_mxfp8(values)
        self.assertEqual(scales.tolist(), [[127]])
        self.assertEqual(payload[0, :4].tolist(), [0x38, 0x3A, 0xB8, 0xBA])
        self.assertEqual(packing.decode_mxfp8(payload, scales)[0, 4].item(), 448.0)

    def test_decode_rejects_nan_scales_payload_and_wrong_layout(self):
        payload, scales = packing.pack_mxfp8(torch.ones(8, 512))
        self.assertEqual(packing.decode_e8m0(torch.tensor([0], dtype=torch.uint8)).item(), 2**-127)
        bad_payload = payload.clone()
        bad_payload[0, 0] = 0x7F
        for data, factors in (
            (payload, torch.full_like(scales, 255)),
            (bad_payload, scales),
            (payload, scales[..., :15]),
            (payload.to(torch.int8), scales),
        ):
            with self.assertRaises(ValueError):
                packing.decode_mxfp8(data, factors)
        with self.assertRaises(ValueError):
            packing.pack_mxfp4(torch.ones(8, 33))

    def test_logical_shapes_and_source_bf16_conversion(self):
        for heads in packing.SUPPORTED_HEADS:
            query, kv, indices, scale = synthetic_inputs(1, 128, heads, 128, 12)
            prepared, decoded_q, decoded_kv, original_q, original_kv = packing.prepare_inputs(query, kv, indices, scale)
            expected = {
                "q": ((heads, 576), torch.uint8),
                "qs": ((heads, 18), torch.uint8),
                "kv": ((128, 256), torch.uint8),
                "ks": ((128, 16), torch.uint8),
                "rope": ((128, 64), torch.bfloat16),
                "idx": ((128,), torch.int32),
            }
            for name, (shape, dtype) in expected.items():
                self.assertEqual(tuple(prepared[name].shape), shape)
                self.assertEqual(prepared[name].dtype, dtype)
                self.assertTrue(prepared[name].is_contiguous())
                self.assertEqual(prepared[name].device.type, "cpu")
            torch.testing.assert_close(decoded_q, mxfp8_roundtrip(original_q), rtol=0, atol=0)
            torch.testing.assert_close(decoded_kv[:, :512], mxfp4_roundtrip(original_kv[:, :512]), rtol=0, atol=0)
            torch.testing.assert_close(decoded_kv[:, 512:], original_kv[:, 512:].float(), rtol=0, atol=0)

    def test_invalid_shapes_indices_and_nonfinite_rejected_on_cpu(self):
        query, kv, indices, scale = synthetic_inputs(1, 128, 8, 128, 12)
        duplicate = indices.clone()
        duplicate[0, 0] = duplicate[0, 1]
        padded = indices.clone()
        padded[0, -1] = -1
        for q, k, idx, attention_scale in (
            (query[:, :4], kv, indices, scale),
            (query.expand(2, -1, -1), kv, indices, scale),
            (query, kv, indices[:, :127], scale),
            (query, kv, duplicate, scale),
            (query, kv, padded, scale),
            (query, kv, indices, float("nan")),
            (query, torch.full_like(kv, float("inf")), indices, scale),
            (query, torch.full(kv.shape, torch.finfo(torch.float32).max), indices, scale),
        ):
            with self.assertRaises(ValueError):
                packing.prepare_inputs(q, k, idx, attention_scale)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_official_golden_keeps_bf16_probability_and_pv_input_but_not_output(self):
        query, kv, indices, scale = synthetic_inputs(1, 128, 8, 128, 27)
        _, q8, kv4, original_q, original_kv = packing.prepare_inputs(query, kv, indices, scale)
        actual, original = run.reference_outputs(q8, kv4, original_q, original_kv, indices, scale)
        selected_k = kv4[indices[0]].bfloat16().float()
        scores = q8[0].float() @ selected_k.T * scale
        shifted = scores - scores.amax(-1, keepdim=True)
        probabilities = shifted.exp() / shifted.exp().sum(-1, keepdim=True)
        accumulator = probabilities.bfloat16().float() @ selected_k[:, :512]
        torch.testing.assert_close(actual, mxfp8_roundtrip(accumulator), rtol=0, atol=0)
        self.assertEqual(original.dtype, torch.float32)
        self.assertTrue(bool((original.bfloat16().float() == original).all()))
        # FP32 O8 and BF16-rounded-then-O8 are distinguishable at an FP8 tie.
        sentinel = torch.full((1, 8, 512), 256.0)
        sentinel[..., 0] = 1.0626
        with patch.object(run, "attention_reference", side_effect=[sentinel, sentinel]):
            result, _ = run.reference_outputs(q8, kv4, original_q, original_kv, indices, scale)
        self.assertEqual(result[0, 0].item(), 1.125)
        self.assertEqual(mxfp8_roundtrip(sentinel.bfloat16().float())[0, 0, 0].item(), 1.0)

    def test_output_byte_dtype_scale_and_device_status_are_checked(self):
        payload, scales = packing.pack_mxfp8(torch.ones(8, 512))
        good = payload, scales, torch.tensor([0], dtype=torch.int32)
        torch.testing.assert_close(run.decode_output(good, 8), torch.ones(8, 512))
        for result in (
            (payload, scales, torch.tensor([1], dtype=torch.int32)),
            (payload, scales, torch.tensor([0], dtype=torch.int64)),
            (payload.to(torch.float8_e4m3fn), scales, good[2]),
            (payload, scales[:, :15], good[2]),
            (payload, torch.full_like(scales, 255), good[2]),
        ):
            with self.assertRaises((AssertionError, RuntimeError, ValueError)):
                run.decode_output(result, 8)

    def test_default_cli_shape_and_capture_contract(self):
        args = run.parse_args(["--library", "candidate.so"])
        self.assertEqual(
            (args.heads, args.key_tokens, args.selected_tokens, args.warmup, args.iters), (8, 8192, 2048, 5, 20)
        )
        self.assertFalse(args.profile)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run.parse_args(["--library", "candidate.so", "--selected-tokens", "129"])

    def test_cpu_mock_orchestration_still_times_when_only_quantization_screening_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            args = run.parse_args(
                [
                    "--library",
                    "candidate.so",
                    "--output",
                    str(output),
                    "--key-tokens",
                    "128",
                    "--selected-tokens",
                    "128",
                    "--warmup",
                    "1",
                    "--iters",
                    "2",
                    "--threads",
                    "2",
                ]
            )
            query, kv, indices, scale = run.load_inputs(args)
            _, dq, dk, oq, ok = packing.prepare_inputs(query, kv, indices, scale)
            golden, _ = run.reference_outputs(dq, dk, oq, ok, indices, scale)
            c4_golden = run.attention_reference(oq, dk, indices, scale).bfloat16().float()[0]
            payload, scales = packing.pack_mxfp8(golden)
            calls = []

            def fake_operator(*inputs):
                self.assertEqual(
                    [value.dtype for value in inputs[:6]], [torch.uint8] * 4 + [torch.bfloat16, torch.int32]
                )
                self.assertEqual(inputs[0].shape, (8, 576))
                self.assertEqual(inputs[2].shape, (128, 256))
                self.assertEqual(inputs[-1], scale)
                calls.append(True)
                return payload, scales, torch.zeros(1, dtype=torch.int32)

            runtime = fake_operator, torch.device("cpu"), lambda: None, {"cpu_mock": True}
            real_compare = run.compare_output

            def quantization_failure(*arguments):
                result = real_compare(*arguments)
                self.assertTrue(result["operator_correctness_passed"])
                result["quantization_screening_passed"] = False
                return result

            with (
                patch.object(run, "load_runtime", return_value=runtime),
                patch.object(run, "compare_output", side_effect=quantization_failure),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(run.run(args), 1)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "quantization_failed")
            self.assertTrue(report["compute_verified"])
            expected_incremental = run.error_metrics(packing.decode_mxfp8(payload, scales), c4_golden)
            self.assertEqual(report["accuracy"]["vs_c4_bf16"], expected_incremental)
            self.assertEqual(
                report["accuracy"]["incremental_screening_passed_vs_c4"],
                expected_incremental["cosine"] >= 0.99 and expected_incremental["relative_rmse"] <= 0.10,
            )
            self.assertEqual(report["gates"]["vs_c4_bf16"], {"min_cosine": 0.99, "max_relative_rmse": 0.10})
            self.assertFalse(report["accuracy"]["quantization_screening_passed"])
            self.assertEqual(len(calls), 4)
            self.assertTrue(report["performance"]["measured"])
            self.assertEqual(len(report["performance"]["samples_ms"]), 2)
            self.assertTrue(report["performance"]["not_kernel_task_duration"])

    def test_failed_actual_call_retains_stage_traceback_and_never_times(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed.json"
            args = run.parse_args(
                [
                    "--library",
                    "candidate.so",
                    "--output",
                    str(output),
                    "--key-tokens",
                    "128",
                    "--selected-tokens",
                    "128",
                    "--threads",
                    "2",
                ]
            )

            def failing_operator(*_):
                raise RuntimeError("mock device call failed")

            runtime = failing_operator, torch.device("cpu"), lambda: None, {"cpu_mock": True}
            with (
                patch.object(run, "load_runtime", return_value=runtime),
                patch.object(run, "benchmark") as timing,
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(run.run(args), 1)
            timing.assert_not_called()
            report = json.loads(output.read_text())
            self.assertEqual(report["stage"], "First actual candidate invocation and synchronization")
            self.assertFalse(report["compute_verified"])
            self.assertFalse(report["performance"]["measured"])
            self.assertIn("mock device call failed", report["error"]["traceback"])

    def test_status_success_cannot_bypass_numerical_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "incorrect.json"
            args = run.parse_args(
                [
                    "--library",
                    "candidate.so",
                    "--output",
                    str(output),
                    "--key-tokens",
                    "128",
                    "--selected-tokens",
                    "128",
                    "--threads",
                    "2",
                ]
            )
            zero_result = (
                torch.zeros((8, 512), dtype=torch.uint8),
                torch.full((8, 16), 127, dtype=torch.uint8),
                torch.zeros(1, dtype=torch.int32),
            )
            runtime = lambda *_: zero_result, torch.device("cpu"), lambda: None, {"cpu_mock": True}
            with (
                patch.object(run, "load_runtime", return_value=runtime),
                patch.object(run, "benchmark") as timing,
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(run.run(args), 1)
            timing.assert_not_called()
            report = json.loads(output.read_text())
            self.assertFalse(report["accuracy"]["operator_correctness_passed"])
            self.assertFalse(report["compute_verified"])
            self.assertFalse(report["performance"]["measured"])

    def test_native_worker_uses_fp8_cache_and_only_times_validated_output(self):
        from benchmarks.qsfa_q8c4_o8 import native_baseline

        with tempfile.TemporaryDirectory() as directory:
            args = run.parse_args(
                [
                    "--library",
                    "unused.so",
                    "--variant",
                    "native",
                    "--output",
                    str(Path(directory) / "native.json"),
                    "--key-tokens",
                    "256",
                    "--selected-tokens",
                    "128",
                    "--warmup",
                    "1",
                    "--iters",
                    "2",
                    "--threads",
                    "2",
                ]
            )
            query, kv, indices, scale = run.load_inputs(args)
            _, decoded_kv, _ = native_baseline.prepare_inputs(query, kv, indices, scale)
            expected = run.attention_reference(query, decoded_kv, indices, scale).bfloat16()
            calls = []

            def fake_native(**kwargs):
                self.assertIs(kwargs["key"], kwargs["value"])
                self.assertEqual(kwargs["key"].dtype, torch.float8_e4m3fn)
                self.assertEqual(kwargs["query"].dtype, torch.bfloat16)
                self.assertEqual(kwargs["layout_query"], "TND")
                calls.append(True)
                return expected

            runtime = fake_native, torch.device("cpu"), lambda: None, {"cpu_mock": True}
            with (
                patch.object(run, "load_runtime", return_value=runtime) as loader,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(run.run(args), 0)
            loader.assert_called_once_with(args.library, args.device, native=True)
            report = json.loads(args.output.read_text())
            self.assertTrue(report["compute_verified"])
            self.assertTrue(report["performance"]["measured"])
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(report["performance"]["samples_ms"]), 2)
            self.assertGreater(report["accuracy"]["native_elementwise_check"]["cosine"], 0.999)

    def test_native_zero_or_malformed_output_cannot_produce_a_performance_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            args = run.parse_args(
                [
                    "--library",
                    "unused.so",
                    "--variant",
                    "native",
                    "--output",
                    str(Path(directory) / "native.json"),
                    "--key-tokens",
                    "256",
                    "--selected-tokens",
                    "128",
                    "--threads",
                    "2",
                ]
            )
            for output in (torch.zeros(1, 8, 512, dtype=torch.bfloat16), torch.zeros(1, 8, 512), (torch.ones(1),)):
                runtime = lambda value=output, **_: value, torch.device("cpu"), lambda: None, {"cpu_mock": True}
                with (
                    self.subTest(type=str(type(output))),
                    patch.object(run, "load_runtime", return_value=runtime),
                    patch.object(run, "benchmark") as timing,
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    self.assertEqual(run.run(args), 1)
                    timing.assert_not_called()
                    report = json.loads(args.output.read_text())
                    self.assertEqual(report["status"], "failed")
                    self.assertFalse(report["compute_verified"])
                    self.assertFalse(report["performance"]["measured"])


if __name__ == "__main__":
    unittest.main()
