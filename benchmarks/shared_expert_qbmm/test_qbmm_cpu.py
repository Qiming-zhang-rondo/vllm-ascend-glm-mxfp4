# SPDX-License-Identifier: Apache-2.0
"""CPU contract/reference tests; these do not validate A5 kernel execution."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

import qbmm_case as worker
import reference_qbmm as reference
import run_qbmm as runner


class InputTests(unittest.TestCase):
    def test_fp4_all_codes_and_nibble_order(self):
        raw = np.array([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=np.uint8)
        decoded = worker.decode_fp4(raw)
        np.testing.assert_array_equal(decoded[0, :8], [0, .5, 1, 1.5, 2, 3, 4, 6])
        np.testing.assert_array_equal(worker.pack_fp4(decoded), raw)
        with self.assertRaises(ValueError):
            worker.pack_fp4([[.25, 1]])

    def test_e8m0_roundtrip_including_subnormal(self):
        raw = np.arange(255, dtype=np.uint8)
        np.testing.assert_array_equal(worker.encode_scale(worker.decode_scale(raw)), raw)
        with self.assertRaises(ValueError):
            worker.decode_scale([255])
        with self.assertRaises(ValueError):
            worker.encode_scale([3])

    def test_metrics_nonfinite_and_zero(self):
        self.assertIsNone(worker.error_metrics([0], [0])["relative_l2"])
        self.assertEqual(worker.error_metrics([1, 2], [1, 2])["max_abs"], 0)
        with self.assertRaises(ValueError):
            worker.error_metrics([np.nan], [0])

    def test_rank_shapes(self):
        shapes = runner.glm_shapes([1, 128], [1, 8, 16])
        self.assertEqual(len(shapes), 12)
        self.assertEqual((shapes[-1]["k"], shapes[-1]["n"]), (128, 6144))
        self.assertEqual((shapes[-2]["k"], shapes[-2]["n"]), (6144, 256))
        with self.assertRaises(ValueError):
            runner.glm_shapes([1], [3])


class BridgeTests(unittest.TestCase):
    def test_exact_vllm_layout_and_kwargs(self):
        for kind in ("mxa8w4", "mxa8w8"):
            with self.subTest(kind=kind):
                seen = {}
                def cast(weight, fmt, **kwargs):
                    seen["format"] = (fmt, kwargs)
                    return weight
                def qmm(a, w, scale, **kwargs):
                    seen["qmm"] = (a, w, scale, kwargs)
                    return "called"
                npu = types.SimpleNamespace(npu_format_cast=cast, npu_quant_matmul=qmm,
                    float8_e8m0fnu="e8m0", float4_e2m1fn_x2="fp4")
                a = torch.zeros((10, 64), dtype=torch.float8_e4m3fn)
                w = torch.zeros((32, 32 if kind == "mxa8w4" else 64),
                                dtype=torch.uint8 if kind == "mxa8w4" else torch.float8_e4m3fn)
                scale = torch.arange(64, dtype=torch.uint8).reshape(32, 2)
                a_scale = torch.zeros((10, 1, 2), dtype=torch.uint8)
                call, layout = worker.prepare_call(torch, npu, kind, a, w, a_scale, scale)
                self.assertEqual(call(), "called")
                _, actual_w, actual_s, kwargs = seen["qmm"]
                self.assertTrue(torch.equal(actual_s, scale.reshape(32, 1, 2).transpose(0, 1)))
                self.assertIs(kwargs["pertoken_scale"], a_scale)
                self.assertEqual(kwargs["output_dtype"], torch.bfloat16)
                self.assertIsNone(kwargs["bias"])
                if kind == "mxa8w4":
                    self.assertEqual(seen["format"], (29, {"customize_dtype": torch.float8_e4m3fn, "input_dtype": "fp4"}))
                    self.assertEqual(kwargs["group_sizes"], [0, 0, 32])
                    self.assertEqual(kwargs["x2_dtype"], "fp4")
                    self.assertFalse(actual_w.is_contiguous())
                    self.assertEqual(layout["weight_format"], "NZ")
                else:
                    self.assertNotIn("format", seen)
                    self.assertNotIn("x2_dtype", kwargs)
                    self.assertEqual(kwargs["group_sizes"], [1, 1, 32])
                    self.assertTrue(actual_w.is_contiguous())

    def test_child_failure_and_environment_preserved(self):
        def child(command, **kwargs):
            self.assertEqual(kwargs["env"]["ASCEND_LAUNCH_BLOCKING"], "1")
            self.assertEqual(kwargs["env"]["FLA_NPU_DISABLE_PTH"], "1")
            self.assertTrue(Path(kwargs["env"]["ASCEND_PROCESS_LOG_PATH"]).exists())
            return types.SimpleNamespace(returncode=7)
        with tempfile.TemporaryDirectory() as path, patch("run_qbmm.subprocess.run", child):
            result = runner.run_child({"case": {"name": "test"}, "kind": "mxa8w4"}, Path(path) / "case", 5)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["returncode"], 7)

    def test_speedup_direction(self):
        results = [{"case": {"name": "x", "m": 1, "k": 64, "n": 32}, "kind": kind,
                    "status": "passed", "source_input_sha256": ["a", "w"],
                    "performance": {"p50_us": latency, "scope": "test"}}
                   for kind, latency in (("mxa8w4", 2), ("mxa8w8", 3))]
        self.assertEqual(runner.comparisons(results)[0]["w8_over_w4_speedup"], 1.5)
        results[1]["source_input_sha256"] = ["different"]
        with self.assertRaises(ValueError):
            runner.comparisons(results)


class OriginalGoldenTests(unittest.TestCase):
    def test_worker_accuracy_gate_precedes_profiling(self):
        cases = json.loads((Path(__file__).parent / "official_cases.json").read_text())
        case = cases[-1]
        d = reference.make_official_inputs(case, 2026)
        golden = reference.official_reference("mxa8w4", d["a_values"], d["w_values"], d["a_scale"], d["w_scale"])
        fake_npu = types.SimpleNamespace(__version__="CPU MOCK", float8_e8m0fnu="e8m0",
            float4_e2m1fn_x2="fp4", npu_dynamic_mx_quant=None, npu_format_cast=None, npu_quant_matmul=None)
        fake_device = types.SimpleNamespace(is_available=lambda: True, set_device=lambda x: None,
            get_device_name=lambda x: "CPU MOCK", synchronize=lambda: None)
        original_to = torch.Tensor.to
        def cpu_to(tensor, *args, **kwargs):
            if args and isinstance(args[0], str) and args[0].startswith("npu:"):
                args = ("cpu", *args[1:])
            return original_to(tensor, *args, **kwargs)
        for passes in (True, False):
            result = torch.from_numpy(golden if passes else golden + 1000).to(torch.bfloat16)
            spec = {"case": case, "kind": "mxa8w4", "device": 0, "seed": 2026,
                    "accuracy_only": False, "warmup": 2, "iterations": 3}
            with tempfile.TemporaryDirectory() as tmp, \
                 patch.dict("sys.modules", {"torch_npu": fake_npu}), \
                 patch.object(torch, "npu", fake_device, create=True), \
                 patch.object(torch.Tensor, "to", cpu_to), \
                 patch("qbmm_case.prepare_call", return_value=(lambda: result, {"mock": True})), \
                 patch("profile_qbmm.profile_call", return_value={"mock": True}) as profiler:
                report = {}
                if passes:
                    worker.execute(spec, Path(tmp), report)
                    self.assertEqual(report["status"], "passed")
                    profiler.assert_called_once()
                    json.dumps(report, allow_nan=False)
                else:
                    with self.assertRaisesRegex(RuntimeError, "accuracy gate failed"):
                        worker.execute(spec, Path(tmp), report)
                    profiler.assert_not_called()
                    self.assertTrue((Path(tmp) / "accuracy_failure.npz").exists())

    def test_official_cases_and_original_comparator(self):
        cases = json.loads((Path(__file__).parent / "official_cases.json").read_text())
        for case in cases:
            with self.subTest(case=case["name"]):
                d = reference.make_official_inputs(case)
                np.testing.assert_array_equal(worker.decode_fp4(worker.pack_fp4(d["w_values"])), d["w_values"])
                for name in ("a_scale", "w_scale"):
                    np.testing.assert_array_equal(worker.decode_scale(worker.encode_scale(d[name])), d[name])
                y = reference.official_reference("mxa8w4", d["a_values"], d["w_values"], d["a_scale"], d["w_scale"])
                self.assertTrue(reference.official_compare(y, y, case)["pass"])
                self.assertFalse(reference.official_compare(y + 1000, y, case)["pass"])
                y8 = reference.official_reference("mxa8w8", d["a_values"], d["w_values"], d["a_scale"], d["w_scale"])
                np.testing.assert_array_equal(y8, y)

    def test_simple_known_answer_uses_original_goldens(self):
        a = np.ones((2, 64), np.float32)
        w = np.ones((32, 64), np.float32)
        for kind in ("mxa8w4", "mxa8w8"):
            y = reference.official_reference(kind, a, w, np.ones((2, 2), np.float32) * 2,
                                             np.ones((32, 2), np.float32) * 4)
            np.testing.assert_array_equal(y, np.full((2, 32), 512, np.float32))


if __name__ == "__main__":
    unittest.main()
