# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the dispatch-trace installer without torch, CANN, or an NPU."""

import ast
import builtins
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[3]
INSTALLER = REPO / "tools/trace_qli_dispatch.py"
SFA_PATH = Path("vllm_ascend/attention/sfa_indexer.py")
DEVICE_PATH = Path("vllm_ascend/device/device_op.py")
HELPER_PATH = Path("vllm_ascend/attention/qli_dispatch_debug.py")
CONFIG_PATH = Path("vllm_ascend/ascend_config.py")

SFA_SOURCE = '''"""Existing SFA module documentation."""
import torch

QLI_V2_MXFP4 = 5


class SFAIndexerMetadataBuilder:
    def build(self, cumulative_query_lens, seq_lens, draft_index=None):
        """Keep the builder's original docstring."""
        return ("build", cumulative_query_lens, seq_lens, draft_index)


def select_sfa_topk(query, key, weights, query_scale, key_scale, block_table, metadata):
    """Keep the compute function's original docstring."""
    return ("compute", query, key, weights, query_scale, key_scale, block_table, metadata)
'''

DEVICE_SOURCE = '''"""Existing device module documentation."""
import torch


class A5DeviceAdaptor:
    @staticmethod
    def indexer_select_post_process(
        sfa_impl, q_li, q_li_scale, q_li_shape_ori, weights, kv_cache,
        attn_metadata, actual_seq_lengths_query, actual_seq_lengths_key,
        enable_sparse_li_c8, use_torch_npu_lightning_indexer,
    ):
        """Keep the device function's original docstring."""
        return ("device", sfa_impl, q_li, q_li_scale, q_li_shape_ori, weights,
                kv_cache, attn_metadata, actual_seq_lengths_query,
                actual_seq_lengths_key, enable_sparse_li_c8,
                use_torch_npu_lightning_indexer)
'''

CONFIG_SOURCE = '''"""Existing configuration module documentation."""
DEFAULT_INDEXER_QUANT_MODE = "FP8_DYNAMIC"


class AscendConfig:
    """Preserve configuration documentation."""

    def __init__(self, vllm_config):
        """Preserve initializer documentation."""
        self.enable_sparse_li_c8 = vllm_config.enabled
        quant_config = vllm_config.quant_config
        (
            self._sparse_li_c8_layer_ids,
            self._sparse_li_c8_layer_names,
        ) = self._parse_sparse_li_c8_layers_from_quant_config(quant_config)
        self._sparse_li_c8_layer_filter_enabled = self._has_sparse_li_c8_layer_config(quant_config)
        self.initialization_finished = True

    @staticmethod
    def _parse_sparse_li_c8_layers_from_quant_config(quant_config):
        VALID_QUANT_TYPES = ("INT8_DYNAMIC", "W8A8_MXFP8", "FP8_DYNAMIC")
        return set(quant_config["ids"]), set(quant_config["names"])

    @staticmethod
    def _has_sparse_li_c8_layer_config(quant_config):
        return quant_config["filtered"]
'''


def docstrings(source):
    return {
        node.name: ast.get_docstring(node)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }


class TestQliDispatchTraceInstaller(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="qli trace installer ")
        self.addCleanup(temporary.cleanup)
        self.va_root = Path(temporary.name) / "VA checkout with spaces"
        self.originals = {
            SFA_PATH: SFA_SOURCE,
            DEVICE_PATH: DEVICE_SOURCE,
            CONFIG_PATH: CONFIG_SOURCE,
        }
        for relative, content in self.originals.items():
            path = self.va_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    def snapshot(self):
        return {
            str(path.relative_to(self.va_root)): path.read_bytes() for path in self.va_root.rglob("*") if path.is_file()
        }

    def run_installer(self, *arguments):
        return subprocess.run(
            [sys.executable, "-I", str(INSTALLER), "--va-root", str(self.va_root), *arguments],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=self.va_root,
        )

    def assert_succeeds(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_check_only_validates_without_creating_or_changing_files(self):
        before = self.snapshot()
        self.assert_succeeds(self.run_installer("--check-only"))
        self.assertEqual(self.snapshot(), before)

    def test_install_is_idempotent_and_preserves_backups_and_config(self):
        self.assert_succeeds(self.run_installer())
        installed = self.snapshot()
        for relative in (SFA_PATH, DEVICE_PATH, CONFIG_PATH):
            source = (self.va_root / relative).read_text()
            self.assertIn("QLI_DISPATCH_DEBUG_BEGIN:", source)
            self.assertEqual(docstrings(source), docstrings(self.originals[relative]))
            backup = self.va_root / (str(relative) + ".before-qli-dispatch-trace")
            self.assertEqual(backup.read_text(), self.originals[relative])
        self.assertTrue((self.va_root / HELPER_PATH).is_file())
        config = (self.va_root / CONFIG_PATH).read_text()
        self.assertIn('DEFAULT_INDEXER_QUANT_MODE = "FP8_DYNAMIC"', config)
        self.assertIn('VALID_QUANT_TYPES = ("INT8_DYNAMIC", "W8A8_MXFP8", "FP8_DYNAMIC")', config)
        self.assert_succeeds(self.run_installer())
        self.assertEqual(self.snapshot(), installed)

    def test_install_upgrades_existing_three_hooks_without_replacing_backups(self):
        self.assert_succeeds(self.run_installer())
        (self.va_root / CONFIG_PATH).write_text(CONFIG_SOURCE)
        config_backup = self.va_root / (str(CONFIG_PATH) + ".before-qli-dispatch-trace")
        config_backup.unlink()
        (self.va_root / HELPER_PATH).write_text("# QLI_DISPATCH_DEBUG_HELPER_V1\n# Previous three-hook helper.\n")
        original_hooks = {relative: (self.va_root / relative).read_bytes() for relative in (SFA_PATH, DEVICE_PATH)}
        self.assert_succeeds(self.run_installer())
        for relative, content in original_hooks.items():
            self.assertEqual((self.va_root / relative).read_bytes(), content)
            backup = self.va_root / (str(relative) + ".before-qli-dispatch-trace")
            self.assertEqual(backup.read_text(), self.originals[relative])
        self.assertEqual(config_backup.read_text(), CONFIG_SOURCE)
        self.assertEqual((self.va_root / CONFIG_PATH).read_text().count("QLI_DISPATCH_DEBUG_BEGIN:selector_init"), 1)

    def test_remove_preserves_unrelated_edits_and_keeps_helper(self):
        self.assert_succeeds(self.run_installer())
        edits = {
            SFA_PATH: '\n# User edit after installation.\nUSER_SETTING = "FP8_DYNAMIC"\n',
            DEVICE_PATH: "\n# Independent device fix.\nUSER_DEVICE_SETTING = 7\n",
        }
        for relative, edit in edits.items():
            path = self.va_root / relative
            path.write_text(path.read_text() + edit)
        config = self.va_root / CONFIG_PATH
        config.write_text(config.read_text() + "USER_CONFIG = True\n")
        self.assert_succeeds(self.run_installer("--remove"))
        for relative, edit in edits.items():
            self.assertEqual((self.va_root / relative).read_text(), self.originals[relative] + edit)
        self.assertTrue((self.va_root / HELPER_PATH).is_file())
        self.assertEqual(config.read_text(), self.originals[CONFIG_PATH] + "USER_CONFIG = True\n")

    def test_runtime_returns_and_docstrings_are_unchanged_and_compile_skips_hooks(self):
        self.assert_succeeds(self.run_installer())
        calls = []
        compiling = [False]
        fake_torch = types.ModuleType("torch")
        fake_torch.compiler = types.SimpleNamespace(is_compiling=lambda: compiling[0])
        helper = types.ModuleType("vllm_ascend.attention.qli_dispatch_debug")

        def helper_attribute(name):
            if name.startswith("__"):
                raise AttributeError(name)

            def record(*args, **kwargs):
                calls.append((name, args, kwargs))

            return record

        helper.__getattr__ = helper_attribute
        package = types.ModuleType("vllm_ascend")
        package.__path__ = []
        attention = types.ModuleType("vllm_ascend.attention")
        attention.__path__ = []
        attention.qli_dispatch_debug = helper
        package.attention = attention
        modules = {
            "torch": fake_torch,
            "vllm_ascend": package,
            "vllm_ascend.attention": attention,
            "vllm_ascend.attention.qli_dispatch_debug": helper,
        }
        with mock.patch.dict(sys.modules, modules):
            sfa, device = {}, {}
            exec(compile((self.va_root / SFA_PATH).read_text(), str(SFA_PATH), "exec"), sfa)
            exec(compile((self.va_root / DEVICE_PATH).read_text(), str(DEVICE_PATH), "exec"), device)
            builder = sfa["SFAIndexerMetadataBuilder"]()
            builder.quant_mode = 5
            builder.num_heads = 64
            builder.head_dim = 128
            builder.max_num_reqs = 16
            builder.device = "npu:0"
            build_args = ([1, 2], [2, 4], None)
            compute_args = tuple(object() for _ in range(7))
            device_args = tuple(object() for _ in range(11))
            for is_compiling in (False, True):
                compiling[0] = is_compiling
                calls.clear()
                self.assertEqual(builder.build(*build_args), ("build", *build_args))
                self.assertEqual(sfa["select_sfa_topk"](*compute_args), ("compute", *compute_args))
                self.assertEqual(
                    device["A5DeviceAdaptor"].indexer_select_post_process(*device_args),
                    ("device", *device_args),
                )
                if is_compiling:
                    self.assertEqual(calls, [])
                else:
                    self.assertEqual(len(calls), 3)

    def test_selector_initialization_runs_after_assignments_without_torch_and_swallows_diagnostic_errors(self):
        self.assert_succeeds(self.run_installer())
        source = (self.va_root / CONFIG_PATH).read_text()
        self.assertNotIn("torch", source)
        namespace = {}
        exec(compile(source, str(CONFIG_PATH), "exec"), namespace)
        quant_config = {"ids": [2, 4], "names": ["layer.2", "layer.4"], "filtered": True}
        vllm_config = types.SimpleNamespace(enabled=True, quant_config=quant_config)
        calls = []

        def report(selector, quant):
            self.assertIs(quant, quant_config)
            self.assertEqual(selector._sparse_li_c8_layer_ids, {2, 4})
            self.assertEqual(selector._sparse_li_c8_layer_names, {"layer.2", "layer.4"})
            self.assertTrue(selector._sparse_li_c8_layer_filter_enabled)
            self.assertFalse(hasattr(selector, "initialization_finished"))
            calls.append(selector)
            raise RuntimeError("Diagnostic reporting must not abort configuration")

        original_import = builtins.__import__
        helper = types.SimpleNamespace(report_selector_initialization=report)
        import_failure = [False]

        def intercept_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("Selector initialization must not import torch")
            if name == "vllm_ascend.attention.qli_dispatch_debug":
                if import_failure[0]:
                    raise ImportError("Diagnostic helper unavailable")
                return helper
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=intercept_import):
            for unavailable in (False, True):
                import_failure[0] = unavailable
                initialized = namespace["AscendConfig"](vllm_config)
                self.assertTrue(initialized.initialization_finished)
                self.assertTrue(initialized.enable_sparse_li_c8)
                self.assertEqual(initialized._sparse_li_c8_layer_ids, {2, 4})
        self.assertEqual(len(calls), 1)

    def test_missing_or_wrong_target_refuses_without_partial_writes(self):
        cases = (
            (SFA_PATH, SFA_SOURCE.replace("class SFAIndexerMetadataBuilder:", "class UnrelatedBuilder:")),
            (
                SFA_PATH,
                SFA_SOURCE.replace("cumulative_query_lens, seq_lens, draft_index=None", "cumulative_query_lens"),
            ),
            (SFA_PATH, SFA_SOURCE.replace("block_table, metadata):", "block_table):")),
            (DEVICE_PATH, DEVICE_SOURCE.replace("class A5DeviceAdaptor:", "class UnrelatedAdaptor:")),
            (
                DEVICE_PATH,
                DEVICE_SOURCE.replace("sfa_impl, q_li, q_li_scale, q_li_shape_ori", "sfa_impl, q_li, q_li_shape_ori"),
            ),
            (DEVICE_PATH, None),
            (CONFIG_PATH, CONFIG_SOURCE.replace("class AscendConfig:", "class OtherConfig:")),
            (CONFIG_PATH, None),
        )
        for relative, invalid in cases:
            with self.subTest(relative=relative, invalid=invalid):
                for original_path, source in self.originals.items():
                    (self.va_root / original_path).write_text(source)
                path = self.va_root / relative
                if invalid is None:
                    path.unlink()
                else:
                    path.write_text(invalid)
                before = self.snapshot()
                result = self.run_installer()
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self.snapshot(), before)

    def test_selector_contract_changes_refuse_before_any_file_is_written(self):
        filter_line = (
            "        self._sparse_li_c8_layer_filter_enabled = self._has_sparse_li_c8_layer_config(quant_config)\n"
        )
        parse_start = "        (\n            self._sparse_li_c8_layer_ids,\n"
        invalid_sources = (
            CONFIG_SOURCE.replace(filter_line, ""),
            CONFIG_SOURCE.replace(filter_line, filter_line * 2),
            CONFIG_SOURCE.replace(
                "_has_sparse_li_c8_layer_config(quant_config)\n", "_has_sparse_li_c8_layer_config(None)\n"
            ),
            CONFIG_SOURCE.replace(
                "_parse_sparse_li_c8_layers_from_quant_config(quant_config)\n",
                "_parse_sparse_li_c8_layers_from_quant_config(other_config)\n",
            ),
            CONFIG_SOURCE.replace("self._sparse_li_c8_layer_ids,", "self.unrelated_ids,"),
            CONFIG_SOURCE.replace("self._sparse_li_c8_layer_names,", "self._sparse_li_c8_layer_ids,"),
            CONFIG_SOURCE.replace(filter_line, "").replace(parse_start, filter_line + parse_start),
            CONFIG_SOURCE.replace(filter_line, "        if True:\n    " + filter_line),
            CONFIG_SOURCE.replace(filter_line, filter_line.rstrip() + "; self.extra = True\n"),
        )
        for invalid in invalid_sources:
            with self.subTest(source=invalid):
                (self.va_root / CONFIG_PATH).write_text(invalid)
                before = self.snapshot()
                result = self.run_installer()
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self.snapshot(), before)

    def test_existing_unrecognized_helper_refuses_without_overwriting(self):
        helper = self.va_root / HELPER_PATH
        helper.write_text("# User's existing unrelated module.\nDO_NOT_OVERWRITE = True\n")
        before = self.snapshot()
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
