# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coordinator tests: no NPU imports, builds or actual child kernels."""

import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from benchmarks.qsfa_q8c4_o8 import compare


def worker_report(status="passed", p50=1.0, mean=1.2):
    return {
        "status": status,
        "compute_verified": True,
        "performance": {"measured": True, "p50_ms": p50, "mean_ms": mean},
        "accuracy": {"operator_correctness_passed": True, "quantization_screening_passed": status == "passed"},
    }


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "results.json"
        self.inputs = (
            torch.full((1, 8, 576), 1.003, dtype=torch.float32),
            torch.linspace(-2.0, 2.0, 128 * 576).reshape(128, 576),
            torch.arange(127, -1, -1, dtype=torch.int64).reshape(1, 128),
            576**-0.5,
        )
        self.commands = []
        self.snapshots = []

    def invoke(self, outcomes, *, mutate_snapshot=False):
        def launch(command, **kwargs):
            self.assertEqual(kwargs, {"check": False})  # Environment and log streams are inherited.
            self.commands.append(command)
            snapshot = Path(command[command.index("--input") + 1])
            self.snapshots.append((snapshot, hashlib.sha256(snapshot.read_bytes()).hexdigest()))
            self.assertEqual(snapshot.stat().st_mode & 0o222, 0)
            data = torch.load(snapshot, map_location="cpu", weights_only=True)
            self.assertEqual(data["query"].dtype, torch.bfloat16)
            self.assertEqual(data["kv"].dtype, torch.bfloat16)
            self.assertEqual(data["indices"].dtype, torch.int32)
            torch.testing.assert_close(data["query"], self.inputs[0].bfloat16(), rtol=0, atol=0)
            torch.testing.assert_close(data["kv"], self.inputs[1].bfloat16(), rtol=0, atol=0)
            torch.testing.assert_close(data["indices"], self.inputs[2].int(), rtol=0, atol=0)
            self.assertEqual(data["scale_value"], self.inputs[3])
            returncode, result = outcomes[len(self.commands) - 1]
            if result is not None:
                path = Path(command[command.index("--output") + 1])
                path.write_text(json.dumps(result))
            if mutate_snapshot:
                snapshot.chmod(0o644)
                with snapshot.open("ab") as stream:
                    stream.write(b"changed")
            return SimpleNamespace(returncode=returncode)

        stdout = io.StringIO()
        with (
            patch.object(compare, "load_inputs", return_value=self.inputs) as load,
            patch.object(compare.subprocess, "run", side_effect=launch) as children,
            patch.dict(sys.modules, {"torch_npu": None}),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = compare.main(
                [
                    "--library",
                    str(self.root / "operator.so"),
                    "--output",
                    str(self.output),
                    "--warmup",
                    "2",
                    "--iters",
                    "3",
                    "--threads",
                    "1",
                    "--device",
                    "2",
                    "--profile",
                ]
            )
        self.assertEqual(load.call_count, 1)
        self.assertEqual(children.call_count, len(self.commands))
        return code, json.loads(self.output.read_text()), stdout.getvalue()

    def test_success_uses_one_snapshot_and_fresh_isolated_workers(self):
        code, report, text = self.invoke([(0, worker_report(p50=2, mean=3)), (0, worker_report(p50=4, mean=9))])
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(self.commands), 2)
        self.assertEqual(self.snapshots[0], self.snapshots[1])
        for command, variant, name in zip(
            self.commands, ("candidate", "native"), ("candidate.json", "native_baseline.json")
        ):
            self.assertEqual(command[:3], [sys.executable, "-I", "-c"])
            self.assertIn('sys.modules["benchmarks"]=package', command[3])
            self.assertEqual(command[5], "benchmarks.qsfa_q8c4_o8.run")
            self.assertEqual(command[command.index("--variant") + 1], variant)
            self.assertEqual(Path(command[command.index("--output") + 1]).name, name)
            for flag, value in (("--warmup", "2"), ("--iters", "3"), ("--threads", "1"), ("--device", "2")):
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertIn("--profile", command)
        self.assertEqual(report["shared_input"]["sha256"], self.snapshots[0][1])
        self.assertEqual(report["candidate"]["input_sha256"], report["baseline"]["input_sha256"])
        self.assertTrue(report["compute_verified"])
        self.assertTrue(report["differing_cache_types"])
        self.assertTrue(report["not_pure_kernel_task_duration"])
        self.assertTrue(report["not_end_to_end_model_performance"])
        self.assertEqual(report["comparison"]["p50_speedup_ratio"], 2)
        self.assertEqual(report["comparison"]["p50_latency_reduction_percent"], 50)
        self.assertEqual(report["comparison"]["mean_speedup_ratio"], 3)
        self.assertIn("speedup=2.000x", text)

    def test_quantization_failure_still_compares_and_reports_slowdown(self):
        candidate = worker_report("quantization_failed", p50=4, mean=6)
        code, report, text = self.invoke([(1, candidate), (0, worker_report(p50=2, mean=3))])
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "quantization_failed")
        self.assertEqual(len(self.commands), 2)
        self.assertEqual(report["candidate"]["result"], candidate)
        self.assertTrue(report["comparison"]["available"])
        self.assertEqual(report["comparison"]["p50_speedup_ratio"], 0.5)
        self.assertEqual(report["comparison"]["mean_speedup_ratio"], 0.5)
        self.assertEqual(report["comparison"]["p50_latency_reduction_percent"], -100)
        self.assertEqual(report["comparison"]["mean_latency_reduction_percent"], -100)
        self.assertIn("speedup=0.500x", text)
        self.assertIn("latency reduction=-100.00%", text)

    def test_candidate_failures_stop_before_native_and_ignore_stale_reports(self):
        bad_compute = worker_report("quantization_failed")
        bad_compute["compute_verified"] = False
        no_timing = worker_report("quantization_failed")
        no_timing["performance"]["measured"] = False
        for code, candidate in (
            (1, worker_report("failed")),
            (1, worker_report("passed")),
            (2, worker_report("quantization_failed")),
            (1, bad_compute),
            (1, no_timing),
            (0, worker_report(p50=0)),
            (0, worker_report(mean=float("nan"))),
            (1, None),
        ):
            with self.subTest(returncode=code, candidate=candidate):
                self.commands.clear()
                self.snapshots.clear()
                stale = self.root / "candidate.json"
                stale.write_text(json.dumps(worker_report()))
                result, report, _ = self.invoke([(code, copy.deepcopy(candidate))])
                self.assertEqual(result, 1)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(len(self.commands), 1)
                self.assertNotIn("baseline", report)
                self.assertFalse(report["comparison"]["available"])
                self.assertNotIn("p50_speedup_ratio", report["comparison"])
                if candidate is None:
                    self.assertFalse(stale.exists())

    def test_native_quantization_failure_keeps_valid_comparison_and_failure_status(self):
        baseline = worker_report("quantization_failed", p50=4, mean=6)
        code, report, text = self.invoke([(0, worker_report(p50=2, mean=3)), (1, baseline)])
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "quantization_failed")
        self.assertTrue(report["compute_verified"])
        self.assertTrue(report["comparison"]["available"])
        self.assertEqual(report["comparison"]["p50_speedup_ratio"], 2)
        self.assertEqual(report["baseline"]["result"], baseline)
        self.assertIn("status=quantization_failed", text)

    def test_native_failure_keeps_candidate_report_without_speedup(self):
        candidate = worker_report(p50=2)
        code, report, text = self.invoke([(0, candidate), (1, worker_report("failed", p50=100))])
        self.assertEqual(code, 1)
        self.assertEqual(report["candidate"]["result"], candidate)
        self.assertEqual(json.loads((self.root / "candidate.json").read_text()), candidate)
        self.assertEqual(report["baseline"]["returncode"], 1)
        self.assertFalse(report["comparison"]["available"])
        self.assertNotIn("p50_speedup_ratio", report["comparison"])
        self.assertNotIn("speedup=", text)

    def test_mutated_snapshot_invalidates_comparison(self):
        code, report, _ = self.invoke([(0, worker_report())], mutate_snapshot=True)
        self.assertEqual(code, 1)
        self.assertEqual(len(self.commands), 1)
        self.assertIn("snapshot changed", report["error"]["message"])
        self.assertFalse(report["comparison"]["available"])

    def test_native_variant_is_reserved_for_worker(self):
        with patch.object(compare.subprocess, "run") as launch, contextlib.redirect_stderr(io.StringIO()):
            code = compare.main(["--library", "unused.so", "--variant", "native"])
        self.assertEqual(code, 2)
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
