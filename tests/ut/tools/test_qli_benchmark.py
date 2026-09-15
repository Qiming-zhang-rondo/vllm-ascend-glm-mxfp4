# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU regression checks; run directly with Python to avoid NPU conftest imports."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

RUNNER = Path(__file__).resolve().parents[3] / "tools/test_qli_v2_mxfp4_a5.py"
SPEC = importlib.util.spec_from_file_location("qli_benchmark_under_test", RUNNER)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class QLIBenchmarkTests(unittest.TestCase):
    def test_prefill_parameters_describe_seven_8192_token_chunks(self):
        args = benchmark.parse_args(["--prefill-tokens", "57344", "--chunk-size", "8192"])
        self.assertEqual((args.query_tokens, args.key_tokens, args.reference_rows), (8192, 57344, 16))
        self.assertEqual(args.prefill_tokens // args.chunk_size, 7)
        for options in (
            ["--prefill-tokens", "57345"],
            ["--prefill-tokens", "57344", "--chunk-size", "0"],
            ["--prefill-tokens", "57344", "--reference-rows", "4"],
            ["--prefill-tokens", "57344", "--check-cache-layout"],
            ["--prefill-tokens", "57344", "--max-mxfp4-p50-ms", "1"],
        ):
            with self.subTest(options=options), self.assertRaises(SystemExit), patch("sys.stderr"):
                benchmark.parse_args(options)

    def test_sampled_reference_preserves_original_prefill_causal_positions(self):
        generator = torch.Generator().manual_seed(91)
        # Exactly representable data isolates mask positions from BLAS rounding
        # differences when the sampled and complete batches have different M.
        q = torch.randint(-2, 3, (8, 3, 128), generator=generator).float()
        k = torch.randint(-2, 3, (8, 1, 128), generator=generator).float()
        w = torch.randint(1, 4, (8, 3), generator=generator).float() * 0.5
        rows = torch.tensor([0, 3, 7])
        for mode in (1, 5):
            full = benchmark.reference_scores(q, k, w, quant_mode=mode)
            sampled = benchmark.reference_scores(q[rows], k, w[rows], quant_mode=mode, causal_lengths=rows + 1)
            torch.testing.assert_close(sampled, full[rows], rtol=0, atol=0)
            self.assertTrue(torch.isneginf(sampled[0, 1:]).all())
            self.assertEqual(torch.isfinite(sampled).sum(dim=1).tolist(), [1, 4, 8])
        for lengths in ([0, 4, 8], [1, 4, 9], [1, 4], [1.0, 4.0, 8.0], [[1, 4, 8]]):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                benchmark.reference_scores(q[rows], k, w[rows], causal_lengths=lengths)

    def test_cpu_preparation_accuracy_and_timing_for_both_modes_without_npu_quantizers(self):
        # Exercise orchestration with CPU outputs standing in for ACLNN, not a
        # hardware correctness test. The NPU namespace has no quantization API.
        args = benchmark.parse_args(["--heads", "4", "--key-tokens", "4096", "--warmup", "1", "--iters", "2"])
        q, k, weights = benchmark.make_host_inputs(args)
        metadata_modes, completed = [], []

        def create_metadata(**kwargs):
            metadata_modes.append(kwargs["quant_mode"])
            return torch.empty(1024, dtype=torch.int32)

        with (
            patch.object(torch, "npu", SimpleNamespace(synchronize=lambda: None), create=True),
            patch("builtins.print"),
        ):
            for name, mode in (("MXFP4", 5), ("FP8", 1)):
                case = benchmark.prepare_case(
                    name,
                    mode,
                    q,
                    k,
                    torch.device("cpu"),
                    SimpleNamespace(create_metadata=create_metadata),
                    torch.tensor([0, 1], dtype=torch.int32),
                    torch.tensor([4096], dtype=torch.int32),
                    {},
                )
                self.assertEqual(case["query"].dtype, torch.uint8 if mode == 5 else torch.float8_e4m3fn)
                self.assertEqual(case["key_scale"].dtype, torch.uint8 if mode == 5 else torch.float32)
                scores = benchmark.reference_scores(
                    case["decoded_query"], case["decoded_key"], weights, quant_mode=mode
                )
                selected = scores.topk(benchmark.TOPK, dim=-1)
                indices = selected.indices.int().unsqueeze(1)
                values = selected.values.bfloat16().unsqueeze(1)
                calls = []

                def invoke(*, return_value=0, calls=calls, indices=indices, values=values):
                    # Passing strided key/scale arguments would fail this test.
                    calls.append(return_value)
                    return indices, values if return_value else torch.empty(0, dtype=torch.bfloat16)

                accuracy = {}
                # Force the original-input gate to fail while keeping valid
                # decoded-payload scores, so timing must still be reachable.
                benchmark.check_case(case, invoke, args, torch.zeros_like(scores), weights, accuracy)
                self.assertTrue(accuracy["operator_correctness_passed"])
                self.assertFalse(accuracy["quantization_thresholds_passed"])
                self.assertEqual(accuracy["cache_layout_check"], "not requested")
                latency = benchmark.benchmark_compute(invoke, args.warmup, args.iters)
                self.assertEqual(latency["iterations"], 2)
                self.assertEqual(calls, [1, 0, 0, 0, 0])
                completed.append(name)
        self.assertEqual(completed, ["MXFP4", "FP8"])
        self.assertEqual(metadata_modes, [5, 1])

    def test_cpu_mxfp4_ties_away_from_zero_and_low_nibble_first(self):
        x = torch.full((1, 1, 128), 6.0, dtype=torch.float16)
        x[0, 0, :16] = torch.tensor(
            [
                0.0,
                0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5.0,
                -0.0,
                -0.25,
                -0.75,
                -1.25,
                -1.75,
                -2.5,
                -3.5,
                -5.0,
            ]
        )
        payload, scales = benchmark.quantize_mxfp4_cpu(x)
        expected = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=torch.uint8)
        torch.testing.assert_close(payload[0, 0, :8], expected, rtol=0, atol=0)
        torch.testing.assert_close(scales, torch.full((1, 1, 2, 2), 127, dtype=torch.uint8), rtol=0, atol=0)
        for tensor in (payload, scales):
            self.assertEqual(tensor.device.type, "cpu")
            self.assertEqual(tensor.dtype, torch.uint8)
            self.assertTrue(tensor.is_contiguous())

    def test_cpu_mxfp4_zero_scale_and_fp16_min_subnormal(self):
        x = torch.zeros((1, 1, 128), dtype=torch.float16)
        x[0, 0, 1] = -0.0
        x[0, 0, 32] = 2**-24
        payload, scales = benchmark.quantize_mxfp4_cpu(x)
        self.assertEqual(payload[0, 0, 0], 0x80)
        self.assertEqual(payload[0, 0, 16], 0x06)
        torch.testing.assert_close(scales.flatten(), torch.tensor([0, 101, 0, 0], dtype=torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(benchmark.decode_mxfp4(payload, scales), x.float(), rtol=0, atol=0)

    def test_cpu_mxfp4_independent_d32_scales_and_saturation(self):
        x = (torch.tensor([0.5, 2.0, 8.0, 32.0], dtype=torch.float16).repeat_interleave(32)).reshape(1, 1, 128)
        payload, scales = benchmark.quantize_mxfp4_cpu(x)
        torch.testing.assert_close(
            scales.flatten(), torch.tensor([124, 126, 128, 130], dtype=torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(payload, torch.full_like(payload, 0x66), rtol=0, atol=0)
        torch.testing.assert_close(benchmark.decode_mxfp4(payload, scales), x.float(), rtol=0, atol=0)
        x = torch.full((1, 1, 128), 7.5, dtype=torch.float16)
        x[..., 1::2] = -7.5
        payload, scales = benchmark.quantize_mxfp4_cpu(x)
        torch.testing.assert_close(payload, torch.full_like(payload, 0xF7), rtol=0, atol=0)
        torch.testing.assert_close(scales, torch.full_like(scales, 127), rtol=0, atol=0)

    def test_cpu_fp8_per_head_scales_and_nearest_even_ties(self):
        x = torch.full((1, 2, 128), 448.0, dtype=torch.float16)
        x[:, 1].mul_(0.5)
        x[0, 0, :4] = torch.tensor([1.0625, 1.1875, -1.0625, -1.1875])
        payload, scales = benchmark.quantize_fp8_cpu(x)
        torch.testing.assert_close(scales, torch.tensor([[1.0, 0.5]]), rtol=0, atol=0)
        torch.testing.assert_close(payload[0, 0, :4].float(), torch.tensor([1.0, 1.25, -1.0, -1.25]), rtol=0, atol=0)
        self.assertEqual(payload.dtype, torch.float8_e4m3fn)
        self.assertEqual(scales.dtype, torch.float32)
        for tensor in (payload, scales):
            self.assertEqual(tensor.device.type, "cpu")
            self.assertTrue(tensor.is_contiguous())

    def test_cpu_fp8_fixture_defines_exact_zero_rows(self):
        x = torch.zeros((2, 1, 128), dtype=torch.float16)
        payload, scales = benchmark.quantize_fp8_cpu(x)
        # This fixture convention deliberately avoids DynamicQuant's 0/0 for
        # all-zero rows; scale 1 and payload 0 are exact legal QLI inputs.
        torch.testing.assert_close(scales, torch.ones((2, 1)), rtol=0, atol=0)
        torch.testing.assert_close(payload.float(), x.float(), rtol=0, atol=0)

    def test_cpu_quantizers_reject_wrong_shape_nonfinite_and_non_cpu(self):
        for quantize in (benchmark.quantize_mxfp4_cpu, benchmark.quantize_fp8_cpu):
            for x in (
                torch.zeros(1, 1, 64, dtype=torch.float16),
                torch.full((1, 1, 128), float("nan"), dtype=torch.float16),
                torch.full((1, 1, 128), float("inf"), dtype=torch.float16),
                torch.empty(1, 1, 128, dtype=torch.float16, device="meta"),
            ):
                with self.assertRaises(ValueError):
                    quantize(x)

    def test_bf16_rounding_avoids_fp32_double_rounding_at_ties(self):
        values = torch.tensor([1 + 2**-8, 1 + 3 * 2**-8, 1 + 2**-8 + 2**-30, 2**-134, 3 * 2**-134], dtype=torch.float64)
        expected = torch.tensor([1.0, 1 + 2**-6, 1 + 2**-7, 0.0, 2**-132], dtype=torch.bfloat16)
        torch.testing.assert_close(benchmark.round_fp64_to_bf16(values), expected, rtol=0, atol=0)
        self.assertNotEqual(values[2].float().bfloat16(), expected[2])

    def test_mxfp4_reference_rounds_each_head_before_causal_mask(self):
        query = torch.zeros(2, 3, 128)
        query[:, :, 0] = 1
        key = torch.zeros(3, 1, 128)
        key[:, 0, 0] = 1
        weights = torch.tensor([[1.0, 2**-8, 2**-8], [1.0, 2**-8, 2**-8]])
        actual = benchmark.reference_scores(query, key, weights, quant_mode=benchmark.QLI_MXFP4)
        # Both half-ULP additions round back to the even BF16 value 1.0.
        expected = torch.tensor([[1.0, 1.0, -torch.inf], [1.0, 1.0, 1.0]])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(benchmark.reference_scores(query, key, weights)[0, 0], 1 + 2**-7)

    def test_smoke_prepares_random_inputs_on_cpu_and_reports_compute_stage(self):
        original_randn = torch.randn

        def cpu_random(*args, **kwargs):
            self.assertEqual(kwargs.get("device"), "cpu")
            return original_randn(*args, **kwargs)

        case = {name: torch.zeros(1) for name in ("query", "key", "query_scale", "key_scale", "metadata")}
        indices = torch.arange(benchmark.TOPK, dtype=torch.int32).view(1, 1, -1)
        values = torch.ones(indices.shape, dtype=torch.bfloat16)
        backend = SimpleNamespace(invoke=lambda **kwargs: (indices, values))
        report = {}
        with (
            patch.object(torch, "randn", side_effect=cpu_random) as random,
            patch.object(torch, "npu", SimpleNamespace(synchronize=lambda: None), create=True),
            patch.object(benchmark, "prepare_case", return_value=case),
        ):
            result = benchmark.smoke_test(backend, None, torch.device("cpu"), report=report)
        self.assertEqual(random.call_count, 2)
        self.assertTrue(result["passed"])
        self.assertEqual(report["stage"], "smoke: QLI compute")

    def test_smoke_input_sync_failure_stops_before_quantization(self):
        report = {}

        def fail_sync():
            raise RuntimeError("input transfer failed")

        with (
            patch.object(torch, "npu", SimpleNamespace(synchronize=fail_sync), create=True),
            patch.object(benchmark, "prepare_case") as prepare,
            self.assertRaisesRegex(RuntimeError, "input transfer failed"),
        ):
            benchmark.smoke_test(None, None, torch.device("cpu"), report=report)
        prepare.assert_not_called()
        self.assertEqual(report["stage"], "smoke: CPU input preparation and copy to NPU")

    def test_source_inputs_are_reproducible_cpu_tensors(self):
        args = SimpleNamespace(query_tokens=2, key_tokens=2048, heads=4, seed=123)
        first = benchmark.make_host_inputs(args)
        second = benchmark.make_host_inputs(args)
        for actual, repeated, shape, dtype in zip(
            first, second, ((2, 4, 128), (2048, 1, 128), (2, 4)), (torch.float16, torch.float16, torch.float32)
        ):
            self.assertEqual(actual.device.type, "cpu")
            self.assertEqual(tuple(actual.shape), shape)
            self.assertEqual(actual.dtype, dtype)
            self.assertTrue(actual.is_contiguous())
            torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
        args.seed += 1
        self.assertFalse(torch.equal(first[0], benchmark.make_host_inputs(args)[0]))
        scores = benchmark.reference_scores(*first)
        self.assertEqual(scores.device.type, "cpu")
        self.assertTrue(torch.isfinite(scores[:, :2047]).all())

    def test_decode_all_codes_signed_zero_and_nibble_order(self):
        packed = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=torch.uint8).repeat(8)
        decoded = benchmark.decode_mxfp4(packed.reshape(1, 1, 64), torch.full((1, 1, 2, 2), 127, dtype=torch.uint8))
        expected = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
        torch.testing.assert_close(decoded.flatten()[:16], expected, rtol=0, atol=0)
        self.assertTrue(decoded.flatten()[8].signbit())

    def test_decode_nonuniform_scale_groups(self):
        packed = torch.full((2, 1, 64), 0x22, dtype=torch.uint8)
        scales = torch.tensor([[[[124, 126], [128, 130]]], [[[130, 128], [126, 124]]]], dtype=torch.uint8)
        decoded = benchmark.decode_mxfp4(packed, scales)
        expected = torch.tensor([[0.125, 0.5, 2, 8], [8, 2, 0.5, 0.125]]).repeat_interleave(32, -1).unsqueeze(1)
        torch.testing.assert_close(decoded, expected, rtol=0, atol=0)

    def test_decode_rejects_nonfinite_scale_and_wrong_layout(self):
        packed = torch.zeros((1, 1, 64), dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            benchmark.decode_mxfp4(packed, torch.full((1, 1, 2, 2), 255, dtype=torch.uint8))
        with self.assertRaisesRegex(ValueError, "Expected packed"):
            benchmark.decode_mxfp4(packed, torch.zeros((1, 1, 4), dtype=torch.uint8))

    def test_weighted_relu_and_right_aligned_causal_reference(self):
        query = torch.zeros((2, 2, 128))
        query[:, 0, 0], query[:, 1, 0] = 1, -1
        key = torch.zeros((4, 1, 128))
        key[:, 0, 0] = torch.tensor([-1, 2, -3, 4])
        weights = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
        result = benchmark.reference_scores(query, key, weights)
        torch.testing.assert_close(result, torch.tensor([[3.0, 4.0, 9.0, -torch.inf], [7.0, 10.0, 21.0, 20.0]]))

    def test_nan_cannot_pass_accuracy(self):
        indices = torch.tensor([[[0, 1]]], dtype=torch.int32)
        values = torch.tensor([[[float("nan"), 1.0]]])
        with self.assertRaisesRegex(AssertionError, "Nonfinite"):
            benchmark.accuracy_metrics(indices, values, torch.tensor([[1.0, 2.0, 0.0]]))

    def test_metrics_catch_wrong_values_and_wrong_selection(self):
        indices = torch.tensor([[[0, 1]]], dtype=torch.int32)
        scores = torch.tensor([[1.0, 2.0, 50.0, 100.0]])
        result = benchmark.accuracy_metrics(indices, torch.tensor([[[10.0, 20.0]]]), scores)
        self.assertEqual(result["topk_recall"], 0)
        self.assertFalse(result["selected_scores_close"])
        self.assertFalse(result["selection_above_tolerated_cutoff"])

    def test_validate_rejects_bad_padding_duplicate_and_causal_index(self):
        for indices in ([[-2, 0]], [[0, 0]], [[0, 4]]):
            with self.assertRaises(AssertionError):
                benchmark.validate_indices(torch.tensor([indices], dtype=torch.int32), 1, 4, topk=2)

    def test_cli_overrides_legacy_environment(self):
        with patch.dict(os.environ, {"QLI_ITERS": "9", "QLI_QUERY_TOKENS": "7"}):
            args = benchmark.parse_args(["--iters", "3", "--query-tokens", "2"])
        self.assertEqual((args.iters, args.query_tokens), (3, 2))

    def test_direct_script_help_does_not_shadow_stdlib_bisect(self):
        completed = subprocess.run([sys.executable, str(RUNNER), "--help"], text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--iterations", completed.stdout)

    def test_failure_produces_valid_json_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "result.json"
            with patch.object(benchmark, "run", side_effect=AssertionError("injected accuracy failure")):
                code = benchmark.main(["--output", str(result)])
            self.assertEqual(code, 1)
            report = json.loads(result.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["error"]["type"], "AssertionError")
            self.assertIn("Traceback (most recent call last)", report["error"]["traceback"])
            self.assertIn("injected accuracy failure", report["error"]["traceback"])

    def test_runtime_errors_do_not_trigger_operator_build(self):
        for phase, api, status in (
            ("execute", "aclnnQuantLightningIndexerV2", 161002),
            ("GetWorkspaceSize", "aclnnQuantLightningIndexerV2Metadata", 161002),
            ("GetWorkspaceSize", "aclnnQuantLightningIndexerV2", 999999),
        ):
            self.assertFalse(benchmark.can_probe_fp8_control(SimpleNamespace(phase=phase, api_name=api, status=status)))

    def test_missing_operator_and_container_exit_codes_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            for exception_type, expected in (
                (benchmark.ExistingOperatorUnavailable, 78),
                (benchmark.ContainerPrerequisiteError, 2),
            ):
                with patch.object(benchmark, "run", side_effect=exception_type("injected")):
                    code = benchmark.main(["--output", str(Path(temporary) / "result.json")])
                self.assertEqual(code, expected)

    def test_build_candidate_requires_successful_fp8_compute_control(self):
        class FakeCallError(RuntimeError):
            api_name = "aclnnQuantLightningIndexerV2"
            phase = "GetWorkspaceSize"
            status = 161002

        for control_result in ({"quant_mode": 1, "passed": True}, AssertionError("bad FP8 indices")):
            report = {}
            with patch.object(benchmark, "smoke_test", side_effect=[FakeCallError("unsupported"), control_result]):
                expected_error = (
                    FakeCallError if isinstance(control_result, Exception) else benchmark.ExistingOperatorUnavailable
                )
                with self.assertRaises(expected_error):
                    benchmark.run_smoke(None, None, None, report, FakeCallError)
            self.assertEqual("build_candidate" in report, not isinstance(control_result, Exception))


if __name__ == "__main__":
    unittest.main()
