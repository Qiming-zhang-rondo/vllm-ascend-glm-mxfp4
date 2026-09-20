# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the real Indexer layer selector in isolation, without torch or CANN."""

import ast
import copy
import importlib.util
import re
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO / "vllm_ascend/ascend_config.py"
FIXER_PATH = REPO / "tools/fix_qli_fp8_dynamic.py"
SELECTOR_METHODS = {
    "_has_sparse_li_c8_layer_config",
    "_parse_sparse_li_c8_layers_from_quant_config",
    "is_sparse_li_c8_layer",
}


def selector_tree(source):
    tree = ast.parse(source)
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendConfig")
    methods = [
        copy.deepcopy(node)
        for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name in SELECTOR_METHODS
    ]
    if {node.name for node in methods} != SELECTOR_METHODS:
        raise AssertionError("The real AscendConfig selector methods could not be isolated")
    selector = ast.ClassDef(name="AscendConfig", bases=[], keywords=[], body=methods, decorator_list=[])
    isolated = ast.parse("from __future__ import annotations\n")
    isolated.body.append(selector)
    return ast.fix_missing_locations(isolated)


def legacy_source():
    """Retain current production method bodies, with the historical two labels."""
    tree = selector_tree(CONFIG_PATH.read_text())
    parser = next(node for node in tree.body[-1].body if node.name == "_parse_sparse_li_c8_layers_from_quant_config")
    assignment = next(
        node
        for node in ast.walk(parser)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "VALID_QUANT_TYPES" for target in node.targets)
    )
    assignment.value = ast.Tuple(elts=[ast.Constant("INT8_DYNAMIC"), ast.Constant("W8A8_MXFP8")], ctx=ast.Load())
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def make_selector(source, description, *, enabled=True):
    utils = types.ModuleType("vllm.model_executor.models.utils")

    def extract_layer_index(name):
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
        if match is None:
            raise ValueError(f"Test layer has no index: {name}")
        return int(match[1])

    utils.extract_layer_index = extract_layer_index
    namespace = {}
    exec(compile(selector_tree(source), str(CONFIG_PATH), "exec"), namespace)
    selector = namespace["AscendConfig"]()
    quant_config = types.SimpleNamespace(quant_description=description)
    with patch.dict(sys.modules, {utils.__name__: utils}):
        selector._sparse_li_c8_layer_ids, selector._sparse_li_c8_layer_names = (
            selector._parse_sparse_li_c8_layers_from_quant_config(quant_config)
        )
    selector._sparse_li_c8_layer_filter_enabled = selector._has_sparse_li_c8_layer_config(quant_config)
    selector.enable_sparse_li_c8 = enabled
    return selector, utils


class TestQliFp8DynamicFix(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("qli_fp8_dynamic_fixer_fixture", FIXER_PATH)
        self.fixer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.fixer)
        self.original = legacy_source()

    def selected(self, selector, utils, name):
        with patch.dict(sys.modules, {utils.__name__: utils}):
            return selector.is_sparse_li_c8_layer(name)

    def test_fp8_labels_reproduce_empty_filter_then_fix_preserves_layer_exclusions(self):
        description = {f"model.layers.{index}.self_attn.indexer.quant_type": "FP8_DYNAMIC" for index in range(16)}
        description.update(
            {
                "indexer_quant_type": "INT8_DYNAMIC",
                "model.layers.16.self_attn.indexer.quant_type": "BF16",
                "model.layers.17.self_attn.indexer.quant_type": "FLOAT",
                "model.layers.18.mlp.weight": "FP8_DYNAMIC",
            }
        )
        old, utils = make_selector(self.original, description)
        self.assertTrue(old._sparse_li_c8_layer_filter_enabled)
        self.assertEqual(old._sparse_li_c8_layer_ids, set())
        self.assertFalse(self.selected(old, utils, "model.layers.0.self_attn.indexer.k_cache"))
        fixed, utils = make_selector(self.fixer.transform(self.original), description)
        self.assertEqual(fixed._sparse_li_c8_layer_ids, set(range(16)))
        for index in range(16):
            self.assertTrue(self.selected(fixed, utils, f"model.layers.{index}.self_attn.indexer.k_cache"))
        for index in (16, 17, 18, 19):
            self.assertFalse(self.selected(fixed, utils, f"model.layers.{index}.self_attn.indexer.k_cache"))
        self.assertFalse(self.selected(fixed, utils, None))

    def test_existing_labels_and_global_disable_remain_effective(self):
        description = {
            "model.layers.2.self_attn.indexer.quant_type": "INT8_DYNAMIC",
            "model.layers.4.self_attn.indexer.wq_b_weight": "W8A8_MXFP8",
            "model.layers.6.self_attn.indexer.wq_b_weight": "FP8_DYNAMIC",
        }
        fixed, utils = make_selector(self.fixer.transform(self.original), description)
        self.assertEqual(fixed._sparse_li_c8_layer_ids, {2, 4, 6})
        self.assertTrue(self.selected(fixed, utils, "model.layers.6.self_attn.indexer.k_cache"))
        fixed.enable_sparse_li_c8 = False
        for index in (2, 4, 6):
            self.assertFalse(self.selected(fixed, utils, f"model.layers.{index}.self_attn.indexer.k_cache"))

    def test_repository_parser_accepts_fp8_dynamic(self):
        description = {
            "model.layers.3.self_attn.indexer.quant_type": "FP8_DYNAMIC",
            "model.layers.7.self_attn.indexer.quant_type": "BF16",
        }
        actual, utils = make_selector(CONFIG_PATH.read_text(), description)
        self.assertEqual(actual._sparse_li_c8_layer_ids, {3})
        self.assertTrue(self.selected(actual, utils, "model.layers.3.self_attn.indexer.k_cache"))
        self.assertFalse(self.selected(actual, utils, "model.layers.7.self_attn.indexer.k_cache"))

    def test_transform_is_targeted_idempotent_and_preserves_unrelated_manual_edits(self):
        unrelated = (
            "\n# Existing deployment diagnostics and an unrelated manual FP8 edit.\n"
            "def diagnostic_hook():\n"
            "    VALID_QUANT_TYPES = ('FP8_DYNAMIC', 'FLOAT')\n"
            "    return VALID_QUANT_TYPES\n"
        )
        original = "# 保留部署中的诊断注释。\n" + self.original + unrelated
        fixed = self.fixer.transform(original)
        self.assertTrue(fixed.endswith(unrelated))
        self.assertEqual(self.fixer.transform(fixed), fixed)
        before = selector_tree(original)
        after = selector_tree(fixed)
        for tree in (before, after):
            parser = next(node for node in tree.body[-1].body if node.name.startswith("_parse_sparse"))
            for node in ast.walk(parser):
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "VALID_QUANT_TYPES" for target in node.targets
                ):
                    node.value = ast.Constant("ONLY_ALLOWED_EDIT")
        self.assertEqual(ast.dump(before, include_attributes=False), ast.dump(after, include_attributes=False))

    def test_transform_refuses_missing_dynamic_or_malformed_target(self):
        dynamic = self.original.replace("('INT8_DYNAMIC', 'W8A8_MXFP8')", "runtime_quant_types()")
        self.assertNotEqual(dynamic, self.original)
        for source in (
            "class AscendConfig:\n    pass\n",
            self.original.replace("class AscendConfig:", "class OtherConfig:"),
            self.original + self.original,
            self.original.replace("'W8A8_MXFP8'", "'FLOAT'"),
            dynamic,
            "class AscendConfig(\n",
        ):
            with self.subTest(source=source[:80]), self.assertRaises((ValueError, RuntimeError, SyntaxError)):
                self.fixer.transform(source)

    def test_cli_check_only_then_backup_and_idempotent_apply(self):
        with tempfile.TemporaryDirectory(prefix="qli fp8 deployment ") as directory:
            package = Path(directory) / "vllm_ascend"
            package.mkdir()
            target = package / "ascend_config.py"
            unrelated = "\n# Keep deployment diagnostics.\nMANUAL_FP8_LABEL = 'FP8_DYNAMIC'\n"
            original = self.original + unrelated
            target.write_text(original)
            command = [sys.executable, str(FIXER_PATH), "--va-root", str(package.parent)]
            check = subprocess.run([*command, "--check-only"], text=True, capture_output=True, timeout=20)
            self.assertEqual(check.returncode, 0, check.stdout + check.stderr)
            self.assertEqual(target.read_text(), original)
            self.assertEqual(list(package.iterdir()), [target])
            applied = subprocess.run(command, text=True, capture_output=True, timeout=20)
            self.assertEqual(applied.returncode, 0, applied.stdout + applied.stderr)
            self.assertEqual(target.read_text(), self.fixer.transform(original))
            backups = [path for path in package.iterdir() if path != target and path.is_file()]
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), original)
            repeated = subprocess.run(command, text=True, capture_output=True, timeout=20)
            self.assertEqual(repeated.returncode, 0, repeated.stdout + repeated.stderr)
            self.assertEqual(target.read_text(), self.fixer.transform(original))
            self.assertEqual(backups[0].read_text(), original)
            self.assertEqual(len(list(package.iterdir())), 2)
            # A pre-existing backup must also survive an apply that does change
            # the target, e.g. after a container checkout was restored manually.
            target.write_text(original + "# Later deployment edit.\n")
            reapplied = subprocess.run(command, text=True, capture_output=True, timeout=20)
            self.assertEqual(reapplied.returncode, 0, reapplied.stdout + reapplied.stderr)
            self.assertEqual(target.read_text(), self.fixer.transform(original + "# Later deployment edit.\n"))
            self.assertEqual(backups[0].read_text(), original)


if __name__ == "__main__":
    unittest.main()
