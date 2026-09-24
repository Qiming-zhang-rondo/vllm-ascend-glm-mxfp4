# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU carrier/launcher regression tests; these do not execute A5 kernels."""

import contextlib
import io
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

from benchmarks.qsfa_fake_quant.reference import synthetic_inputs
from benchmarks.qsfa_q8c4_o8 import compare, native_baseline, official_inputs, packing, run


class OfficialInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_physical_pack_roundtrip_offsets_padding_and_shuffled_pages(self):
        q, kv, indices, scale = synthetic_inputs(1, 600, 16, 256, 2026)
        split, dq, dk, oq, ok = packing.prepare_inputs(q, kv, indices, scale)
        table = torch.tensor([[2, 0, 1]], dtype=torch.int64)
        inputs, actual_q, actual_kv, original_q, original_kv, contract = official_inputs.prepare_inputs(
            q, kv, indices, scale, block_table=table
        )
        self.assertEqual(tuple(inputs), official_inputs.ARGUMENT_ORDER)
        self.assertEqual(inputs["q"].shape, (16, 608))
        self.assertEqual(inputs["cache"].shape, (3, 256, 1, 416))
        self.assertEqual(inputs["idx"].shape, (1, 1, 256))
        self.assertEqual(inputs["table"].tolist(), [[2, 0, 1]])
        self.assertEqual(inputs["cuq"].tolist(), [1])
        self.assertEqual(inputs["kvlen"].tolist(), [600])
        torch.testing.assert_close(inputs["q"][:, :576], split["q"], rtol=0, atol=0)
        torch.testing.assert_close(inputs["q"][:, 576:594], split["qs"], rtol=0, atol=0)
        self.assertFalse(bool(inputs["q"][:, 594:].any()))
        logical = inputs["cache"][table[0]].reshape(-1, 416)
        torch.testing.assert_close(logical[:600, :256], split["kv"], rtol=0, atol=0)
        torch.testing.assert_close(
            logical[:600, 256:384].contiguous().view(torch.bfloat16), ok[:, 512:], rtol=0, atol=0
        )
        torch.testing.assert_close(logical[:600, 384:400], split["ks"], rtol=0, atol=0)
        self.assertFalse(bool(logical[:, 400:].any()))
        self.assertFalse(bool(logical[600:].any()))
        for actual, expected in ((actual_q, dq), (actual_kv, dk), (original_q, oq), (original_kv, ok)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Incorrectly assuming identity pages must change the decoded golden.
        wrong = {**inputs, "table": torch.arange(3, dtype=torch.int32).reshape(1, -1)}
        _, wrong_kv = official_inputs.decode_inputs(wrong)
        self.assertFalse(torch.equal(wrong_kv, dk))
        self.assertTrue(contract["experimental_carrier_not_installed_cann_contract"])
        self.assertFalse(contract["installed_native_tiling_equivalence_claimed"])
        for tensor in inputs.values():
            self.assertEqual(tensor.device.type, "cpu")
            self.assertTrue(tensor.is_contiguous())

    def test_invalid_mapping_scales_and_h64_are_rejected_before_npu(self):
        q, kv, idx, scale = synthetic_inputs(1, 300, 8, 128, 17)
        for table in (torch.tensor([[0, 0]]), torch.tensor([[0, 2]]), torch.tensor([[0.0, 1.0]])):
            with self.assertRaisesRegex(ValueError, "permutation"):
                official_inputs.prepare_inputs(q, kv, idx, scale, block_table=table)
        inputs, *_ = official_inputs.prepare_inputs(q, kv, idx, scale)
        inputs["cache"][0, 0, 0, 384] = 255
        with self.assertRaisesRegex(ValueError, "NaN"):
            official_inputs.decode_inputs(inputs)
        with self.assertRaisesRegex(ValueError, "UB budget"):
            official_inputs.prepare_inputs(q.repeat(1, 8, 1), kv, idx, scale)

    def test_strided_output_carrier_decodes_without_padding_becoming_values(self):
        values = torch.linspace(-2, 2, 8 * 512).reshape(8, 512)
        payload, scales = packing.pack_mxfp8(values)
        physical = torch.full((8, 544), 0xFF, dtype=torch.uint8)
        physical[:, :512], physical[:, 512:528] = payload, scales
        actual = run.decode_output((physical[:, :512], physical[:, 512:528], torch.zeros(1, dtype=torch.int32)), 8)
        torch.testing.assert_close(actual, packing.decode_mxfp8(payload, scales), rtol=0, atol=0)


class OfficialWorkerTests(unittest.TestCase):
    def make_args(self, directory, variant):
        return run.parse_args(
            [
                "--library",
                "unused.so",
                "--implementation",
                "official",
                "--variant",
                variant,
                "--output",
                str(Path(directory) / "result.json"),
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

    def test_source_control_uses_exact_seven_arguments_and_original_native_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(directory, "source_control")
            q, kv, idx, scale = run.load_inputs(args)
            inputs, decoded_kv, _ = native_baseline.prepare_inputs(q, kv, idx, scale)
            expected = native_baseline.reference_output(q, decoded_kv, idx, scale).bfloat16()
            calls = []

            def operation(query, cache, indices, table, cuq, kvlen, scale_value):
                calls.append(True)
                self.assertEqual(query.dtype, torch.bfloat16)
                self.assertEqual(cache.dtype, torch.uint8)
                torch.testing.assert_close(cache, inputs["key"].view(torch.uint8), rtol=0, atol=0)
                torch.testing.assert_close(indices, inputs["sparse_indices"], rtol=0, atol=0)
                torch.testing.assert_close(table, inputs["block_table"], rtol=0, atol=0)
                self.assertEqual(cuq.tolist(), [1])
                self.assertEqual(kvlen.tolist(), [256])
                self.assertEqual(scale_value, scale)
                return expected, torch.empty(0, dtype=torch.uint8), torch.zeros(1, dtype=torch.int32)

            runtime = operation, torch.device("cpu"), lambda: None, {"cpu_mock": True}
            with (
                patch.object(run, "load_runtime", return_value=runtime) as loader,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(run.run(args), 0)
            loader.assert_called_once_with(args.library, args.device, implementation="official_fp8")
            report = json.loads(args.output.read_text())
            self.assertEqual(len(calls), 4)
            self.assertTrue(report["compute_verified"])
            self.assertTrue(report["cache_bytes_verified"])
            self.assertEqual(report["contract"]["specialization"]["s2_tile"], 128)
            self.assertEqual(
                report["accuracy"]["native_elementwise_check"]["engineering_smoke_gate"]["min_cosine"], 0.999
            )

    def test_source_control_zero_output_stops_before_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(directory, "source_control")
            result = (
                torch.zeros((1, 8, 512), dtype=torch.bfloat16),
                torch.empty(0, dtype=torch.uint8),
                torch.zeros(1, dtype=torch.int32),
            )
            runtime = lambda *_: result, torch.device("cpu"), lambda: None, {"cpu_mock": True}
            with (
                patch.object(run, "load_runtime", return_value=runtime),
                patch.object(run, "benchmark") as timing,
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(run.run(args), 1)
            timing.assert_not_called()
            report = json.loads(args.output.read_text())
            self.assertFalse(report["compute_verified"])
            self.assertEqual(report["status"], "failed")

    def test_official_candidate_preserves_packed_bytes_and_times_after_correctness(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(directory, "candidate")
            q, kv, idx, scale = run.load_inputs(args)
            inputs, dq, dk, oq, ok, _ = official_inputs.prepare_inputs(q, kv, idx, scale)
            expected, _ = run.reference_outputs(dq, dk, oq, ok, idx, scale)
            payload, scales = packing.pack_mxfp8(expected)
            output = torch.zeros((8, 544), dtype=torch.uint8)
            output[:, :512], output[:, 512:528] = payload, scales
            calls = []

            def operation(*values):
                calls.append(True)
                self.assertEqual(len(values), 7)
                for name, value in zip(official_inputs.ARGUMENT_ORDER, values):
                    torch.testing.assert_close(value, inputs[name], rtol=0, atol=0)
                self.assertEqual(values[-1], scale)
                return output[:, :512], output[:, 512:528], torch.zeros(1, dtype=torch.int32)

            runtime = operation, torch.device("cpu"), lambda: None, {"cpu_mock": True}
            with patch.object(run, "load_runtime", return_value=runtime), contextlib.redirect_stdout(io.StringIO()):
                code = run.run(args)
            self.assertIn(code, (0, 1))  # Quantization loss retains its independent gate.
            report = json.loads(args.output.read_text())
            self.assertNotEqual(report["status"], "failed")
            self.assertTrue(report["compute_verified"])
            self.assertTrue(report["input_bytes_verified"])
            self.assertEqual(len(calls), 4)
            self.assertEqual(report["implementation"]["launches_per_call"], 1)
            self.assertFalse(report["implementation"]["scores_and_probabilities_in_gm"])


class OfficialComparisonTests(unittest.TestCase):
    def invoke(self, outcomes):
        with tempfile.TemporaryDirectory() as directory:
            args = run.parse_args(
                [
                    "--library",
                    "unused.so",
                    "--implementation",
                    "official",
                    "--output",
                    str(Path(directory) / "results.json"),
                    "--key-tokens",
                    "256",
                    "--selected-tokens",
                    "128",
                    "--threads",
                    "2",
                ]
            )
            calls = []

            def launch(command, **_):
                variant = command[command.index("--variant") + 1]
                calls.append(variant)
                result, code = outcomes[len(calls) - 1]
                Path(command[command.index("--output") + 1]).write_text(json.dumps(result))
                return SimpleNamespace(returncode=code)

            with (
                patch.object(compare.subprocess, "run", side_effect=launch),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                code = compare.run(args)
            return code, json.loads(args.output.read_text()), calls

    @staticmethod
    def outcome(p50=1, status="passed"):
        return {
            "status": status,
            "compute_verified": status != "failed",
            "performance": {"measured": True, "p50_ms": p50, "mean_ms": p50},
        }, 0 if status == "passed" else 1

    def test_control_first_three_processes_and_two_unclamped_comparisons(self):
        code, report, calls = self.invoke([self.outcome(1), self.outcome(2, "quantization_failed"), self.outcome(4)])
        self.assertEqual(calls, ["source_control", "candidate", "native"])
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "quantization_failed")
        self.assertEqual(report["comparison"]["p50_speedup_ratio"], 2)
        self.assertEqual(report["source_control_comparison"]["p50_speedup_ratio"], 0.5)
        self.assertEqual(report["source_control_comparison"]["p50_latency_reduction_percent"], -100)
        self.assertFalse(report["source_control_comparison"]["identical_tiling"])
        self.assertEqual(report["source_control"]["input_sha256"], report["candidate"]["input_sha256"])
        self.assertEqual(report["baseline"]["input_sha256"], report["candidate"]["input_sha256"])

    def test_failed_source_control_prevents_candidate_and_any_speedup(self):
        code, report, calls = self.invoke([self.outcome(1, "failed")])
        self.assertEqual(code, 1)
        self.assertEqual(calls, ["source_control"])
        self.assertNotIn("candidate", report)
        self.assertFalse(report["comparison"]["available"])
        self.assertFalse(report["source_control_comparison"]["available"])

    def test_h64_loaded_input_is_rejected_before_any_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query, kv, indices, scale = synthetic_inputs(1, 128, 64, 128, 17)
            path = root / "input.pt"
            torch.save({"query": query, "kv": kv, "indices": indices, "scale_value": scale}, path)
            args = run.parse_args(
                [
                    "--library",
                    "unused.so",
                    "--implementation",
                    "official",
                    "--input",
                    str(path),
                    "--output",
                    str(root / "result.json"),
                ]
            )
            with patch.object(compare.subprocess, "run") as worker, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(compare.run(args), 1)
            worker.assert_not_called()
            report = json.loads(args.output.read_text())
            self.assertIn("UB budget", report["error"]["message"])


class OfficialShellTests(unittest.TestCase):
    def test_shell_builds_official_only_when_requested_and_rejects_control_bypass(self):
        source = Path(__file__).resolve().parents[3] / "benchmarks/qsfa_q8c4_o8/run_qsfa.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "repo/benchmarks/qsfa_q8c4_o8/run_qsfa.sh"
            target.parent.mkdir(parents=True)
            # macOS sandbox denies /dev/fd process substitution. Remove only
            # logging fanout in this fixture; execute the real parser/build/run.
            script = source.read_text()
            logging_line = 'exec > >(tee "$task_run_dir/run.log") 2>&1'
            self.assertEqual(script.count(logging_line), 1)
            target.write_text(script.replace(logging_line, ": # fixture uses captured stdout"))
            fakebin = root / "bin"
            fakebin.mkdir()
            for name, text in (("uname", "Linux"), ("git", "cpu-shell-fixture")):
                command = fakebin / name
                command.write_text(f"#!/bin/sh\nprintf '%s\\n' '{text}'\n")
                command.chmod(0o755)
            log = root / "calls.jsonl"
            python = fakebin / "fixture_python"
            python.write_text(
                f"#!{sys.executable}\n"
                "import json,pathlib,sys\n"
                f"log=pathlib.Path({str(log)!r})\n"
                "with log.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
                "if '--result-file' in sys.argv:\n"
                " p=pathlib.Path(sys.argv[sys.argv.index('--result-file')+1])\n"
                " p.write_text(json.dumps({'library':'/cpu/fixture.so'}))\n"
                "elif sys.argv[-1].endswith('build.json'): print('/cpu/fixture.so')\n"
            )
            python.chmod(0o755)
            environment = {**os.environ, "PATH": f"{fakebin}:{os.environ['PATH']}"}
            for arguments, official in (
                (["--heads", "8"], False),
                (["--implementation", "official"], True),
                (["--implementation=official"], True),
            ):
                log.unlink(missing_ok=True)
                result = subprocess.run(
                    ["bash", str(target), "--python", str(python), *arguments],
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                self.assertEqual(len(calls), 3)
                self.assertEqual("--official" in calls[0], official)
                self.assertIn("benchmarks.qsfa_q8c4_o8.compare", calls[2])
            log.unlink()
            result = subprocess.run(
                ["bash", str(target), "--python", str(python), "--implementation", "official", "--candidate-only"],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("source-control case first", result.stderr)
            self.assertFalse(log.exists())


if __name__ == "__main__":
    unittest.main()
