# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only contract tests for dispatch diagnostics, never operator execution."""

import contextlib
import functools
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HELPER = Path(__file__).resolve().parents[3] / "tools/qli_dispatch_debug.py"
LOG_PREFIX = "[QLI-DISPATCH] "


class TensorDescriptorOnly:
    """Any attempt to read or represent payload data is a test failure."""

    def __init__(self, shape, dtype="torch.uint8"):
        self.shape = shape
        self.dtype = dtype
        self.device = "npu:0"

    def stride(self):
        result = []
        size = 1
        for extent in reversed(self.shape):
            result.append(size)
            size *= extent
        return tuple(reversed(result))

    def _payload_forbidden(self, *args, **kwargs):
        raise AssertionError("dispatch diagnostic attempted to read tensor payload")

    cpu = item = numpy = tolist = __array__ = __iter__ = __repr__ = __str__ = _payload_forbidden


class BaseSFAFixture:
    def forward(self, *args, **kwargs):
        raise AssertionError("diagnostics must not execute forward")

    def forward_mqa(self, *args, **kwargs):
        raise AssertionError("diagnostics must not execute forward_mqa")

    def indexer_select_post_process(self, *args, **kwargs):
        raise AssertionError("diagnostics must not execute indexer selection")


class CountingContext(dict):
    def __init__(self):
        super().__init__()
        self.items_calls = 0

    def items(self):
        self.items_calls += 1
        return super().items()


class FakeAscendConfig:
    enable_sparse_li_c8 = True
    sfa_indexer_quant_mode = "mxfp4"
    _sparse_li_c8_layer_filter_enabled = True
    _sparse_li_c8_layer_names = {"model.layers.0.self_attn.indexer"}
    _sparse_li_c8_layer_ids = {0}

    def is_sparse_li_c8_layer(self, prefix):
        return prefix in self._sparse_li_c8_layer_names

    def _parse_sparse_li_c8_layers_from_quant_config(self):
        return self._sparse_li_c8_layer_ids, self._sparse_li_c8_layer_names


class TestQliDispatchDebug(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="qli-dispatch-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config = FakeAscendConfig()
        config_module = types.ModuleType("vllm_ascend.ascend_config")
        config_module.get_ascend_config = lambda: self.config
        config_module.AscendConfig = FakeAscendConfig
        config_module.__file__ = __file__
        self.context = CountingContext()
        self.vllm_config = types.SimpleNamespace(
            compilation_config=types.SimpleNamespace(static_forward_context=self.context),
            model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type="glm_moe_dsa")),
            quant_config=types.SimpleNamespace(
                quant_description={
                    "indexer_quant_type": "MXFP4",
                    "model.layers.0.self_attn.indexer.quant_type": "INT8_DYNAMIC",
                    "model.layers.1.self_attn.indexer.wq_b_weight": "W4A4_MXFP4",
                    "model.layers.0.mlp.weight": "W4A4_MXFP4",
                }
            ),
        )
        self.config.vllm_config = self.vllm_config
        vllm_config_module = types.ModuleType("vllm.config")
        vllm_config_module.get_current_vllm_config = lambda: self.vllm_config
        packages = {}
        for name in ("vllm", "vllm_ascend", "vllm_ascend.device", "vllm_ascend.attention"):
            module = types.ModuleType(name)
            module.__path__ = []
            packages[name] = module
        self.enterContext(
            patch.dict(
                sys.modules,
                {**packages, "vllm_ascend.ascend_config": config_module, "vllm.config": vllm_config_module},
            )
        )
        spec = importlib.util.spec_from_file_location("qli_dispatch_debug_fixture", HELPER)
        self.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.helper)
        self.builder = types.SimpleNamespace(
            vllm_config=self.vllm_config,
            model_config=self.vllm_config.model_config,
            layer_names=[],
            qli_v2_builder=types.SimpleNamespace(quant_mode=5),
            quant_mode=5,
        )

    @staticmethod
    def capture(function, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            function(*args)
        return [
            json.loads(line[len(LOG_PREFIX) :])
            for line in output.getvalue().splitlines()
            if line.startswith(LOG_PREFIX)
        ]

    def implementation(self, index, *, quantized, implementation_class=BaseSFAFixture):
        impl = implementation_class()
        impl.has_indexer = True
        impl.skip_topk = False
        impl.enable_sparse_li_c8 = quantized
        impl.indexer_quant_mode = "mxfp4"
        impl.use_torch_npu_lightning_indexer = False
        impl.prefix = f"model.layers.{index}.self_attn"
        impl.indexer = types.SimpleNamespace(k_cache=types.SimpleNamespace(prefix=f"{impl.prefix}.indexer"))
        return impl

    def install_av_subclass(self, *, wrapped=False):
        path = self.directory / "av" / "patch_sfa.py"
        path.parent.mkdir(exist_ok=True)
        source = (
            "def select_sfa_topk(*args, **kwargs):\n"
            "    raise AssertionError('AV compute must not execute')\n"
            "class AVDeviceOperator:\n"
            "    @staticmethod\n"
            "    def indexer_select_post_process(*args, **kwargs):\n"
            "        raise AssertionError('AV device method must not execute')\n"
            "DeviceOperator = AVDeviceOperator\n"
            "class AVSFA(BaseSFAFixture):\n"
            "    def forward(self, *args, **kwargs):\n"
            "        raise AssertionError('AV forward must not execute')\n"
            "    def forward_mqa(self, *args, **kwargs):\n"
            "        raise AssertionError('AV forward_mqa must not execute')\n"
            "    def indexer_select_post_process(self, *args, **kwargs):\n"
            "        raise AssertionError('AV indexer must not execute')\n"
        )
        path.write_text(source)
        module = types.ModuleType("av.patch_sfa_dispatch_fixture")
        module.__file__ = str(path)
        module.BaseSFAFixture = BaseSFAFixture
        self.enterContext(patch.dict(sys.modules, {module.__name__: module}))
        exec(compile(source, str(path), "exec"), module.__dict__)
        if wrapped:
            for name in ("forward", "forward_mqa", "indexer_select_post_process"):
                original = getattr(module.AVSFA, name)

                @functools.wraps(original)
                def wrapper(*args, **kwargs):
                    raise AssertionError("wrapped method must not execute")

                setattr(module.AVSFA, name, wrapper)
        return module.AVSFA, path

    def test_worker_snapshot_finds_actual_overrides_and_direct_or_nested_impl(self):
        av_class, av_path = self.install_av_subclass()
        direct = self.implementation(0, quantized=True, implementation_class=av_class)
        nested = self.implementation(1, quantized=False)
        self.context.update(
            {
                "direct": types.SimpleNamespace(impl=direct),
                "nested": types.SimpleNamespace(mla_attn=types.SimpleNamespace(impl=nested)),
                "duplicate": types.SimpleNamespace(impl=direct),
                "not_attention": object(),
            }
        )
        events = self.capture(self.helper.report_worker_dispatch, self.builder)
        snapshot = next(event for event in events if event["event"] == "worker_snapshot")
        self.assertTrue(snapshot["snapshot_only"])
        self.assertFalse(any(event["event"] == "compute_entry" for event in events))
        self.assertEqual(snapshot["builder_quant_mode"], 5)
        self.assertEqual(snapshot["config"]["sfa_indexer_quant_mode"], "mxfp4")
        self.assertTrue(snapshot["config"]["filter_enabled"])
        description = snapshot["config"]["quant_description"]
        self.assertEqual(description["indexer_quant_type"], "MXFP4")
        self.assertEqual(
            {(group["suffix"], group["value"], group["count"]) for group in description["groups"]},
            {(".indexer.quant_type", "INT8_DYNAMIC", 1), (".indexer.wq_b_weight", "W4A4_MXFP4", 1)},
        )
        self.assertNotIn("model.layers.0.mlp.weight", json.dumps(description))
        groups = snapshot["groups"]
        self.assertEqual(sum(group["count"] for group in groups), 2)
        enabled = next(group for group in groups if group["gates"]["enable_sparse_li_c8"])
        disabled = next(group for group in groups if not group["gates"]["enable_sparse_li_c8"])
        self.assertEqual(enabled["method_sources"]["forward"]["file"], str(av_path))
        self.assertEqual(enabled["method_sources"]["forward_mqa"]["file"], str(av_path))
        self.assertEqual(enabled["device_operator"]["file"], str(av_path))
        self.assertEqual(enabled["select_sfa_topk"]["file"], str(av_path))
        self.assertEqual(enabled["prefixes"], ["model.layers.0.self_attn.indexer"])
        self.assertEqual(disabled["prefixes"], ["model.layers.1.self_attn.indexer"])
        self.assertEqual(disabled["method_sources"]["forward"]["file"], __file__)
        external = [item for item in snapshot["external_sources"].values() if item["origin"]["file"] == str(av_path)]
        self.assertEqual(len(external), 2)
        self.assertTrue(all(0 < len(item["lines"]) <= 60 for item in external))
        self.assertEqual(self.capture(self.helper.report_worker_dispatch, self.builder), [])

    def test_wrapped_method_reports_real_av_source(self):
        av_class, av_path = self.install_av_subclass(wrapped=True)
        impl = self.implementation(0, quantized=True, implementation_class=av_class)
        self.context["wrapped"] = types.SimpleNamespace(impl=impl)
        events = self.capture(self.helper.report_worker_dispatch, self.builder)
        snapshot = next(event for event in events if event["event"] == "worker_snapshot")
        source = snapshot["groups"][0]["method_sources"]["forward"]
        self.assertEqual(source["file"], __file__)
        self.assertEqual(source["wrapped"]["file"], str(av_path))

    def test_empty_context_retries_without_unbounded_empty_logs(self):
        events = []
        for _ in range(12):
            events.extend(self.capture(self.helper.report_worker_dispatch, self.builder))
        empties = [event for event in events if event["event"] == "worker_snapshot_empty"]
        self.assertEqual(len(empties), 3)
        self.assertEqual([event["attempt"] for event in empties], [1, 2, 3])
        self.assertFalse(any(event["event"] == "worker_snapshot" for event in events))
        self.assertEqual(self.context.items_calls, 3)
        self.context["late_layer"] = types.SimpleNamespace(impl=self.implementation(0, quantized=True))
        late = self.capture(self.helper.report_worker_dispatch, self.builder)
        self.assertEqual(late, [])
        self.assertEqual(self.context.items_calls, 3)

    def test_empty_context_can_be_followed_by_actual_worker_snapshot(self):
        first = self.capture(self.helper.report_worker_dispatch, self.builder)
        self.assertEqual(first[0]["event"], "worker_snapshot_empty")
        self.context["layer"] = types.SimpleNamespace(impl=self.implementation(0, quantized=True))
        events = self.capture(self.helper.report_worker_dispatch, self.builder)
        self.assertTrue(any(event["event"] == "worker_snapshot" for event in events))

    def test_device_entry_reads_descriptors_only_and_reports_effective_false_gate_once(self):
        impl = self.implementation(0, quantized=True)
        query = TensorDescriptorOnly((1, 64, 64))
        events = self.capture(self.helper.report_device_entry, impl, query, None, False)
        entry = next(event for event in events if event["event"] == "device_entry")
        self.assertTrue(entry["attempt_only"])
        self.assertEqual(entry["query"]["shape"], [1, 64, 64])
        self.assertEqual(entry["query"]["dtype"], "torch.uint8")
        self.assertIsNone(entry["query_scale"])
        self.assertTrue(entry["gates"]["enable_sparse_li_c8"])
        self.assertFalse(entry["enable_sparse_li_c8_argument"])
        self.assertEqual(self.capture(self.helper.report_device_entry, impl, query, None, False), [])

    def test_compute_entry_is_attempt_not_success_and_never_reads_metadata_payload(self):
        metadata = types.SimpleNamespace(quant_mode=5, schedule=TensorDescriptorOnly((1024,), "torch.int32"))
        query = TensorDescriptorOnly((1, 64, 64))
        key = TensorDescriptorOnly((16, 128, 1, 64))
        query_scale = TensorDescriptorOnly((1, 64, 2, 2))
        key_scale = TensorDescriptorOnly((16, 128, 1, 2, 2))
        arguments = (metadata, query, key, query_scale, key_scale)
        events = self.capture(self.helper.report_compute_entry, *arguments)
        entry = next(event for event in events if event["event"] == "compute_entry")
        self.assertTrue(entry["attempt_only"])
        self.assertEqual(entry["quant_mode"], 5)
        self.assertEqual(entry["key_scale"]["shape"], [16, 128, 1, 2, 2])
        self.assertFalse(entry.get("compute_verified", False))
        self.assertFalse(entry.get("completed", False))
        self.assertEqual(self.capture(self.helper.report_compute_entry, *arguments), [])
        second_metadata = types.SimpleNamespace(quant_mode=5)
        self.assertEqual(self.capture(self.helper.report_compute_entry, second_metadata, *arguments[1:]), [])
        fp8_metadata = types.SimpleNamespace(quant_mode=1)
        fp8_events = self.capture(self.helper.report_compute_entry, fp8_metadata, *arguments[1:])
        self.assertEqual(fp8_events[0]["quant_mode"], 1)
        self.assertTrue(fp8_events[0]["attempt_only"])


if __name__ == "__main__":
    unittest.main()
