# SPDX-License-Identifier: Apache-2.0
"""Exercise AV import guards without importing vLLM or Mooncake."""

import builtins
import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("av_mooncake_patcher", REPO / "tools/patch_av_optional_mooncake.py")
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)

LEGACY = """# _AV_OPTIONAL_MOONCAKE_PATCH = True
import os as _av_os
_av_false_values = {"0", "false", "off", "no"}
_av_mooncake_disabled = (
    _av_os.environ.get("VLLM_ASCEND_ENABLE_MOONCAKE", "").lower() in _av_false_values
    or _av_os.environ.get("VLLM_ENABLE_MOONCAKE", "").lower() in _av_false_values
)
if not _av_mooncake_disabled:
    from ascend_vllm.patch.platform import patch_mooncake_connector as patch_mooncake_connector
else:
    patch_mooncake_connector = None
from ascend_vllm.patch.platform import patch_health
from ascend_vllm.patch.platform import patch_mooncake_backend # noqa: F401
from ascend_vllm.patch.platform import patch_spec_anom
"""


class OptionalMooncakeTests(unittest.TestCase):
    def test_direct_script_entrypoint_outside_repository(self):
        # Test the user's exact launch style. Loading the module in this test
        # process alone hides tools/bisect shadowing the standard library.
        script = REPO / "tools/patch_av_optional_mooncake.py"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "__init__.py"
            target.write_text(LEGACY)
            for args, expected in (
                (["--help"], "usage:"),
                (["--target", str(target)], "Patched:"),
                (["--target", str(target)], "Already patched:"),
            ):
                with self.subTest(args=args):
                    result = subprocess.run(
                        [sys.executable, "-S", str(script), *args],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(expected, result.stdout)
            self.assertEqual(
                self.execute(target.read_text(), {"VLLM_ASCEND_ENABLE_MOONCAKE": "0"}, fail_mooncake=True),
                ["patch_health", "patch_spec_anom"],
            )

    def test_direct_script_discovers_av_without_importing_it(self):
        script = REPO / "tools/patch_av_optional_mooncake.py"
        with tempfile.TemporaryDirectory() as directory:
            av_package = Path(directory) / "ascend_vllm"
            platform = av_package / "patch/platform"
            platform.mkdir(parents=True)
            (av_package / "__init__.py").write_text("raise AssertionError('AV must not be imported by the patcher')\n")
            target = platform / "__init__.py"
            target.write_text(LEGACY)
            env = dict(os.environ, PYTHONPATH=directory)
            result = subprocess.run(
                [sys.executable, "-S", str(script)],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(str(target.resolve()), result.stdout)
            self.assertEqual(
                self.execute(target.read_text(), {"VLLM_ASCEND_ENABLE_MOONCAKE": "0"}, fail_mooncake=True),
                ["patch_health", "patch_spec_anom"],
            )

    def execute(self, source, env, fail_mooncake=False):
        imports = []
        real_import = builtins.__import__

        def record_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "os":
                return real_import(name, globals, locals, fromlist, level)
            for item in fromlist or (name,):
                if fail_mooncake and "mooncake" in item:
                    raise ImportError("synthetic libldap EVP_md2 error")
                imports.append(item)
            return types.SimpleNamespace(**{item: object() for item in fromlist})

        with patch.dict(os.environ, env, clear=True), patch("builtins.__import__", side_effect=record_import):
            exec(compile(source, "av_platform", "exec"), {})
        return imports

    def test_upgrade_connector_only_patch_skips_backend_too(self):
        updated = PATCHER.transform(LEGACY)
        for flag in ("VLLM_ASCEND_ENABLE_MOONCAKE", "VLLM_ENABLE_MOONCAKE"):
            with self.subTest(flag=flag):
                self.assertEqual(
                    self.execute(updated, {flag: "0"}, fail_mooncake=True),
                    ["patch_health", "patch_spec_anom"],
                )
        self.assertEqual(PATCHER.transform(updated), updated)

    def test_enabled_and_default_keep_imports_and_errors(self):
        updated = PATCHER.transform(LEGACY)
        expected = ["patch_mooncake_connector", "patch_health", "patch_mooncake_backend", "patch_spec_anom"]
        for env in ({}, {"VLLM_ASCEND_ENABLE_MOONCAKE": "1"}):
            self.assertEqual(self.execute(updated, env), expected)
            with self.assertRaisesRegex(ImportError, "EVP_md2"):
                self.execute(updated, env, fail_mooncake=True)

    def test_legacy_backend_before_connector_gets_initialized_guard(self):
        updated = PATCHER.transform("from . import patch_mooncake_backend\n" + LEGACY)
        self.assertEqual(
            self.execute(updated, {"VLLM_ASCEND_ENABLE_MOONCAKE": "0"}, fail_mooncake=True),
            ["patch_health", "patch_spec_anom"],
        )
        self.assertEqual(PATCHER.transform(updated), updated)

    def test_multiline_mixed_import_keeps_non_mooncake_order(self):
        source = """from ascend_vllm.patch.platform import (
    patch_health,
    patch_mooncake_connector as connector,
    patch_mooncake_backend,
    patch_spec_anom,
)
"""
        updated = PATCHER.transform(source)
        self.assertEqual(
            self.execute(updated, {"VLLM_ASCEND_ENABLE_MOONCAKE": " false "}, fail_mooncake=True),
            ["patch_health", "patch_spec_anom"],
        )
        self.assertEqual(PATCHER.transform(updated), updated)
        self.assertEqual(
            self.execute(updated, {}),
            ["patch_health", "patch_mooncake_connector", "patch_mooncake_backend", "patch_spec_anom"],
        )

    def test_alias_relative_and_nested_imports(self):
        source = """from . import patch_mooncake_backend as backend
if True:
    from ascend_vllm.patch.platform.patch_mooncake_connector import helper
import ascend_vllm.patch.platform.patch_mooncake_other as other
from ascend_vllm.patch.platform import patch_health
"""
        updated = PATCHER.transform(source)
        self.assertEqual(self.execute(updated, {"VLLM_ENABLE_MOONCAKE": "OFF"}), ["patch_health"])
        self.assertEqual(PATCHER.transform(updated), updated)

    def test_preserves_docstring_future_import_and_backup(self):
        original = '"""AV platform."""\nfrom __future__ import annotations\n' + LEGACY
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "__init__.py"
            target.write_text(original)
            self.assertTrue(PATCHER.patch_target(target))
            self.assertFalse(PATCHER.patch_target(target))
            self.assertEqual(target.with_name(target.name + ".before-optional-mooncake").read_text(), original)
            compile(target.read_text(), str(target), "exec")
        fresh = '"""AV platform."""\nfrom __future__ import annotations\nfrom . import patch_mooncake_backend\n'
        compile(PATCHER.transform(fresh), "fresh", "exec")

    def test_unrecognized_or_unsafe_source_is_not_written(self):
        for source in ("x = 1\n", "from . import patch_mooncake_backend; x = 1\n"):
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "__init__.py"
                target.write_text(source)
                with self.assertRaises(RuntimeError):
                    PATCHER.patch_target(target)
                self.assertEqual(target.read_text(), source)


if __name__ == "__main__":
    unittest.main()
