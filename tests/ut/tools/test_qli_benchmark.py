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
