# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU numerical/regression checks; not a substitute for an A5 run."""

import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "benchmarks"))
from qsfa_fake_quant import official_baseline, reference  # noqa: E402
from qsfa_fake_quant.run import evaluate_inputs, parse_args  # noqa: E402


class QsfaReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_official_functions_remain_verbatim(self):
        folder = REPO / "benchmarks/qsfa_fake_quant/vendor"
        manifest = json.loads((folder / "SOURCES.json").read_text())
        source = (folder / "official_qsfa_golden.py").read_text()
        lines = source.splitlines(keepends=True)
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef):
                actual = "".join(lines[node.lineno - 1 : node.end_lineno])
                self.assertEqual(
                    hashlib.sha256(actual.encode()).hexdigest(), manifest["functions"][node.name]["sha256"]
                )

    def test_official_sparse_reference_matches_independent_dense_mask(self):
        q, kv, _, scale = reference.synthetic_inputs(3, 7, 2, 4, 14)
        indices = torch.tensor([[1, 3, 6, -1], [0, 5, -1, -1], [-1, -1, -1, -1]])
        actual = reference.attention_reference(q, kv, indices, scale)
        expected = torch.zeros(3, 2, 512)
        for row in range(3):
            visible = 7 - 3 + row + 1
            selected = indices[row][(indices[row] >= 0) & (indices[row] < visible)]
            if selected.numel():
                scores = q[row].float() @ kv[selected].float().T * scale
                probabilities = torch.softmax(scores, -1).bfloat16().float()
                expected[row] = probabilities @ kv[selected, :512].float()
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.equal(actual[2], torch.zeros_like(actual[2])))

    def test_p_injection_only_changes_probability_representation(self):
        q, kv, indices, scale = reference.synthetic_inputs(2, 7, 2, 5, 18)
        expected = reference.attention_reference(q, kv, indices, scale)
        with mock.patch.object(reference, "mxfp8_roundtrip", side_effect=lambda x: x.bfloat16().float()):
            actual = reference.attention_reference(q, kv, indices, scale, quantize_probability=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_all_roundtrips_disabled_reproduce_baseline(self):
        q, kv, indices, scale = reference.synthetic_inputs(2, 7, 2, 5, 18)
        with (
            mock.patch("qsfa_fake_quant.run.mxfp4_roundtrip", side_effect=lambda x: x.float()),
            mock.patch("qsfa_fake_quant.run.mxfp8_roundtrip", side_effect=lambda x: x.bfloat16().float()),
            mock.patch.object(reference, "mxfp8_roundtrip", side_effect=lambda x: x.bfloat16().float()),
        ):
            cases = evaluate_inputs(q, kv, indices, scale)
        for case in cases.values():
            self.assertEqual(case["vs_bf16"]["relative_rmse"], 0)

    def test_native_baseline_rejects_zero_wrong_dtype_and_wrong_shape(self):
        case = official_baseline.build_case()
        expected = official_baseline.golden(case)
        torch.testing.assert_close(official_baseline.golden(case), expected, rtol=0, atol=0)
        official_baseline.validate_native_output(expected.bfloat16(), expected)
        for invalid in (torch.zeros_like(expected).bfloat16(), expected.float(), expected.bfloat16()[..., :256]):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype), self.assertRaises(AssertionError):
                official_baseline.validate_native_output(invalid, expected)

    def test_invalid_indices_and_nonfinite_data_are_rejected(self):
        q, kv, indices, scale = reference.synthetic_inputs(1, 7, 2, 4, 18)
        for invalid in (torch.tensor([[0, -1, 1, -1]]), torch.tensor([[0, 0, 1, 2]]), indices + 100):
            with self.assertRaises(ValueError):
                reference.validate_inputs(q, kv, invalid, scale)
        q[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            reference.validate_inputs(q, kv, indices, scale)

    def test_native_execution_requires_explicit_opt_in(self):
        for flags, reference_only in (([], True), (["--reference-only"], True), (["--native-baseline"], False)):
            with self.subTest(flags=flags), mock.patch.object(sys, "argv", ["qsfa", *flags]):
                self.assertEqual(parse_args().reference_only, reference_only)

    def test_default_cli_replay_persists_failed_gates_without_importing_npu(self):
        q, kv, indices, scale = reference.synthetic_inputs(1, 7, 2, 5, 18)
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "inputs.pt", Path(temp) / "result.json"
            torch.save({"query": q, "kv": kv, "indices": indices, "scale_value": scale}, source)
            command = [
                sys.executable,
                "-I",
                "-c",
                "import runpy,sys;sys.modules['torch_npu']=None;sys.path.insert(0,sys.argv.pop(1));"
                'runpy.run_module("qsfa_fake_quant.run",run_name="__main__")',
                str(REPO / "benchmarks"),
                "--input",
                str(source),
                "--min-cosine",
                "1",
                "--max-relative-rmse",
                "0",
                "--threads",
                "2",
                "--output",
                str(output),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "screening_failed")
            self.assertEqual(report["native_baseline"]["status"], "skipped")
            self.assertFalse(report["performance"]["measured"])
            self.assertEqual(len(report["experiments"]), 1)
            self.assertEqual(report["experiments"][0]["scale_value"], scale)
            self.assertFalse(report["experiments"][0]["cases"]["q8_kv4_o8"]["screening_passed_vs_bf16"])


if __name__ == "__main__":
    unittest.main()
