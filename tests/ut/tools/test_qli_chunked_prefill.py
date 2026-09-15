# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU orchestration tests. Fake ACLNN outputs do not verify an A5 kernel."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

TOOLS = Path(__file__).resolve().parents[3] / "tools"


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = load_tool("test_qli_v2_mxfp4_a5")
prefill = load_tool("qli_chunked_prefill")


class CpuQliBackend:
    """Independently decode the submitted payload and model one-head causal QLI."""

    def __init__(self, fail_at=None):
        self.calls = []
        self.metadata = []
        self.fail_at = fail_at

    def create_metadata(self, **kwargs):
        self.metadata.append(kwargs)
        return torch.tensor(
            [kwargs["cu_seqlens_q"][-1], kwargs["seqused_k"][0], kwargs["quant_mode"]], dtype=torch.int32
        )

    @staticmethod
    def decode(payload, scales, mode):
        if mode == 1:
            return payload.float() * scales.float().unsqueeze(-1)
        raw = payload.to(torch.int64)
        codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2)
        magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        elements = magnitudes[codes & 7] * torch.where(codes < 8, 1, -1)
        exponents = scales.to(torch.int32).flatten(-2) - 127
        factors = torch.pow(2.0, exponents.float()).repeat_interleave(32, dim=-1)
        return elements * factors

    def invoke(self, **kwargs):
        q = self.decode(kwargs["query"], kwargs["query_scale"], kwargs["quant_mode"])
        k = self.decode(kwargs["key"], kwargs["key_scale"], kwargs["quant_mode"]).reshape(-1, 1, 128)
        query_tokens, heads = q.shape[:2]
        if heads != 1:
            raise AssertionError("The independent test golden implements exactly one Q head")
        key_tokens = int(kwargs["seqused_k"][0])
        mode = kwargs["quant_mode"]
        self.calls.append((mode, query_tokens, key_tokens, kwargs.get("return_value", 0)))
        if self.fail_at == len(self.calls):
            raise RuntimeError("injected ACLNN compute failure")
        torch.testing.assert_close(kwargs["cu_seqlens_q"], torch.tensor([0, query_tokens], dtype=torch.int32))
        torch.testing.assert_close(
            kwargs["metadata"], torch.tensor([query_tokens, key_tokens, mode], dtype=torch.int32)
        )
        torch.testing.assert_close(kwargs["block_table"].flatten(), torch.arange(key_tokens // 128, dtype=torch.int32))
        if k.shape[0] != key_tokens:
            raise AssertionError("A chunk must submit its visible K prefix only")
        weights = kwargs["weights"][:, 0]
        # H=1: an independent dot, ReLU, and BF16 multiply are sufficient;
        # this does not reuse reference_scores or its masking implementation.
        scores = (q[:, 0] @ k[:, 0].T).clamp_min(0)
        if mode == 5:
            scores = (scores.bfloat16().float() * weights.bfloat16().float()[:, None]).bfloat16().float()
        else:
            scores *= weights[:, None]
        topk = kwargs["topk"]
        indices = torch.full((query_tokens, 1, topk), -1, dtype=torch.int32)
        values = torch.full((query_tokens, 1, topk), -torch.inf, dtype=torch.bfloat16)
        for local_row in range(query_tokens):
            visible = key_tokens - query_tokens + local_row + 1
            selected = scores[local_row, :visible].topk(min(visible, topk))
            indices[local_row, 0, : selected.indices.numel()] = selected.indices.int()
            values[local_row, 0, : selected.values.numel()] = selected.values.bfloat16()
        if not kwargs.get("return_value", 0):
            values = torch.empty(0, dtype=torch.bfloat16)
        return indices, values


class QliChunkedPrefillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_real_56k_plan_and_sample_budget_without_allocating_payloads(self):
        args = benchmark.parse_args(["--prefill-tokens", "57344", "--chunk-size", "8192"])
        self.assertEqual(args.prefill_tokens, 7 * args.chunk_size)
        self.assertEqual(args.reference_rows, 16)
        for start in range(0, args.prefill_tokens, args.chunk_size):
            rows = prefill.select_reference_rows(start, args.chunk_size, args.reference_rows)
            self.assertEqual(len(rows), 16)
            self.assertEqual(rows, sorted(set(rows)))
            self.assertEqual(rows[0], 0)
            self.assertEqual(rows[-1], 8191)
            self.assertTrue(all(0 <= row < args.chunk_size for row in rows))
            if start == 0:
                self.assertTrue({2046, 2047, 2048}.issubset(rows))
        self.assertEqual(prefill.select_reference_rows(0, 3, 16), [0, 1, 2])

    def test_padding_zero_values_and_invalid_outputs(self):
        indices = torch.tensor([[[0, -1, -1, -1]], [[1, 0, -1, -1]], [[2, 0, 1, -1]]], dtype=torch.int32)
        values = torch.tensor(
            [[[0, -torch.inf, -torch.inf, -torch.inf]], [[1, 0, -torch.inf, -torch.inf]], [[2, 0, 1, -torch.inf]]],
            dtype=torch.bfloat16,
        )
        with patch.object(prefill, "TOPK", 4):
            prefill.validate_outputs(indices, values, 0, 3)
            for kind in ("future", "duplicate", "wrong_padding_value", "negative_index", "nan_valid"):
                bad_indices, bad_values = indices.clone(), values.clone()
                if kind == "future":
                    bad_indices[0, 0, 0] = 1
                elif kind == "duplicate":
                    bad_indices[1, 0, 1] = 1
                elif kind == "wrong_padding_value":
                    bad_values[0, 0, 1] = 0
                elif kind == "negative_index":
                    bad_indices[0, 0, 1] = -2
                else:
                    bad_values[0, 0, 0] = torch.nan
                with self.subTest(kind=kind), self.assertRaises(AssertionError):
                    prefill.validate_outputs(bad_indices, bad_values, 0, 3)

    def run_small_prefill(self, *, strict_quantization=False, fail_at=None):
        args = benchmark.parse_args(
            ["--prefill-tokens", "384", "--chunk-size", "128", "--heads", "1", "--warmup", "1", "--iters", "2"]
        )
        args.min_topk_recall = 1 if strict_quantization else 0
        args.min_score_cosine = 1 if strict_quantization else 0
        args.max_score_nmae = 0 if strict_quantization else 10
        backend = CpuQliBackend(fail_at)
        report, sources, timings, references = {}, [], [], []

        def prepare(name, mode, q, k, *args, **kwargs):
            sources.append((name, q.clone(), k.clone()))
            return benchmark.prepare_case(name, mode, q, k, *args, **kwargs)

        def reference(q, k, weights, *args, **kwargs):
            lengths = kwargs["causal_lengths"]
            references.append((q.shape[0], k.numel() // 128, lengths.clone()))
            self.assertLessEqual(q.shape[0], 16)
            return benchmark.reference_scores(q, k, weights, *args, **kwargs)

        def timing(compute, warmup, iterations):
            # Run a submitted compute even though timing numbers are controlled.
            compute()
            number = len(timings) + 1
            result = {
                "mean_ms": float(number),
                "p50_ms": 100.0 + number,
                "p90_ms": 200.0 + number,
                "iterations": iterations,
            }
            timings.append(result)
            return result

        error = None
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(prefill, "TOPK", 4),
            patch.object(benchmark, "TOPK", 4),
            patch.object(torch, "npu", SimpleNamespace(synchronize=lambda: None), create=True),
            patch("builtins.print"),
        ):
            args.output = Path(directory) / "result.json"
            try:
                prefill.run_prefill(
                    args,
                    report,
                    backend,
                    torch.device("cpu"),
                    prepare_case=prepare,
                    reference_scores=reference,
                    benchmark_compute=timing,
                )
            except (AssertionError, RuntimeError) as caught:
                error = caught
        return report, backend, sources, timings, references, error

    def test_cpu_chunk_wiring_causal_samples_shared_sources_and_mean_summary(self):
        report, backend, sources, timings, references, error = self.run_small_prefill()
        self.assertIsNone(error)
        self.assertEqual(len(report["chunks"]), 3)
        self.assertEqual(len(timings), 6)
        self.assertEqual({call[2] for call in backend.calls}, {128, 256, 384})
        self.assertEqual(len(sources), 6)
        for first in range(0, len(sources), 2):
            pair = sources[first : first + 2]
            self.assertEqual({item[0] for item in pair}, {"MXFP4", "FP8"})
            torch.testing.assert_close(pair[0][1], pair[1][1], rtol=0, atol=0)
            torch.testing.assert_close(pair[0][2], pair[1][2], rtol=0, atol=0)
        for chunk in report["chunks"]:
            start = chunk["query_start"]
            self.assertEqual(chunk["key_tokens"], start + chunk["query_tokens"])
            self.assertEqual(chunk["reference_global_rows"], [start + row for row in chunk["reference_local_rows"]])
            expected = torch.tensor([row + 1 for row in chunk["reference_global_rows"]])
            matching = [lengths for _, k_tokens, lengths in references if k_tokens == chunk["key_tokens"]]
            self.assertGreaterEqual(len(matching), 3)
            for lengths in matching:
                torch.testing.assert_close(lengths, expected, check_dtype=False)
        for mode in ("MXFP4", "FP8"):
            entries = [chunk["cases"][mode] for chunk in report["chunks"]]
            total = report["cases"][mode]
            self.assertTrue(total["operator_correctness_passed"])
            self.assertEqual(
                total["sum_chunk_mean_wall_ms"], sum(entry["wall_latency"]["mean_ms"] for entry in entries)
            )
            self.assertFalse(any("p50" in key.lower() for key in total))

    def test_quantization_failure_keeps_all_chunk_timings_and_failed_gates(self):
        report, _, _, timings, _, error = self.run_small_prefill(strict_quantization=True)
        self.assertIsInstance(error, AssertionError)
        self.assertEqual(len(timings), 6)
        self.assertTrue(report["cases"]["MXFP4"]["operator_correctness_passed"])
        self.assertFalse(report["cases"]["MXFP4"]["quantization_thresholds_passed"])
        self.assertFalse(report["cases"]["MXFP4"]["passed"])

    def test_indices_only_accepts_permutation_and_alternative_boundary_ties(self):
        original_invoke = CpuQliBackend.invoke
        permuted_calls = []

        def invoke_permuted(backend, **kwargs):
            indices, values = original_invoke(backend, **kwargs)
            if not kwargs.get("return_value", 0):
                # Move valid indices and padding together, without changing
                # selection. RV0 is not required to match RV1's ordering.
                indices = indices.flip(-1)
                permuted_calls.append(kwargs["quant_mode"])
            return indices, values

        with patch.object(CpuQliBackend, "invoke", invoke_permuted):
            report, _, _, timings, _, error = self.run_small_prefill()
        self.assertIsNone(error)
        self.assertEqual(set(permuted_calls), {1, 5})
        self.assertEqual(len(timings), 6)
        for mode in ("MXFP4", "FP8"):
            self.assertTrue(report["cases"][mode]["operator_correctness_passed"])

        # Three indices tie for the last two slots. These different sets are
        # both exact top-k answers, while replacing one with index 5 is wrong.
        scores = torch.tensor([[9.0, 8.0, 7.0, 7.0, 7.0, 0.0]])
        with_values = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.int32)
        indices_only = torch.tensor([[[4, 2, 1, 0]]], dtype=torch.int32)
        values = torch.tensor([[[9.0, 8.0, 7.0, 7.0]]], dtype=torch.bfloat16)
        with patch.object(prefill, "TOPK", 4):
            prefill.validate_outputs(with_values, values, 5, 1)
            prefill.validate_outputs(indices_only, None, 5, 1)
            self.assertTrue(prefill.accuracy_metrics(with_values, values, scores)["selected_scores_close"])
            self.assertTrue(prefill.selection_above_tolerated_cutoff(indices_only, scores))
            bad = torch.tensor([[[5, 2, 1, 0]]], dtype=torch.int32)
            prefill.validate_outputs(bad, None, 5, 1)
            self.assertFalse(prefill.selection_above_tolerated_cutoff(bad, scores))

    def test_fatal_compute_error_stops_without_fake_timing_success(self):
        report, backend, sources, timings, _, error = self.run_small_prefill(fail_at=1)
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("injected ACLNN", str(error))
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(len(sources), 1)
        self.assertEqual(timings, [])
        self.assertNotEqual(report.get("status"), "passed")


if __name__ == "__main__":
    unittest.main()
