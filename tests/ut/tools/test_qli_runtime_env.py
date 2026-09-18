# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise real shell activation with symbol-only local C libraries.

These tests check environment propagation and ACLNN symbol discovery, never
execute an operator, and do not require torch, CANN, or NPU hardware.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SYMBOLS = (
    "aclnnQuantLightningIndexerV2GetWorkspaceSize",
    "aclnnQuantLightningIndexerV2",
    "aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize",
    "aclnnQuantLightningIndexerV2Metadata",
)
ENV_KEYS = (
    "ASCEND_CUSTOM_OPP_PATH",
    "LD_LIBRARY_PATH",
    "ASCEND_OPP_PATH",
    "FLA_NPU_DISABLE_PTH",
    "TORCH_DEVICE_BACKEND_AUTOLOAD",
)


class TestQliRuntimeEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("cc") or shutil.which("clang")
        if compiler is None:
            raise unittest.SkipTest("A local C compiler is required for symbol-only library tests")
        cls.compiled_directory = tempfile.TemporaryDirectory(prefix="qli symbols ")
        cls.addClassCleanup(cls.compiled_directory.cleanup)
        cls.libraries = {}
        for missing in (None, *SYMBOLS):
            name = missing or "all_symbols"
            source = Path(cls.compiled_directory.name) / f"{name}.c"
            source.write_text("\n".join(f"void {symbol}(void) {{}}" for symbol in SYMBOLS if symbol != missing))
            library = source.with_suffix(".so")
            flags = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
            subprocess.run([compiler, *flags, str(source), "-o", str(library)], check=True, capture_output=True)
            cls.libraries[missing] = library
        source = Path(cls.compiled_directory.name) / "optional_unresolved.c"
        source.write_text(
            "\n".join(f"void {symbol}(void) {{}}" for symbol in SYMBOLS)
            + "\nextern void qli_missing_optional_dependency(void);\n"
            + "void qli_unused_optional_entry(void) { qli_missing_optional_dependency(); }\n"
        )
        library = source.with_suffix(".so")
        flags = ["-dynamiclib", "-undefined", "dynamic_lookup"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
        subprocess.run([compiler, *flags, str(source), "-o", str(library)], check=True, capture_output=True)
        cls.optional_unresolved_library = library

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="qli activation ")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.repo = self.directory / "checkout with spaces"
        self.tools = self.repo / "tools"
        self.tools.mkdir(parents=True)
        for name in ("qli_runtime_env.py", "activate_qli_mxfp4_a5.sh"):
            shutil.copy2(REPO / "tools" / name, self.tools / name)
        self.activation = self.tools / "activate_qli_mxfp4_a5.sh"
        self.helper = self.tools / "qli_runtime_env.py"
        self.env_dump = self.directory / "dump child environment.py"
        self.env_dump.write_text(
            "import json, os\n"
            f"keys = {ENV_KEYS!r}\n"
            "print('CHILD_ENV=' + json.dumps({key: os.environ.get(key) for key in keys}))\n"
            "print('SOURCE_STATUS=' + os.environ['QLI_TEST_SOURCE_STATUS'])\n"
        )
        self.env = os.environ.copy()
        for key in (
            *ENV_KEYS,
            "ASCEND_HOME_PATH",
            "ASCEND_AICPU_PATH",
            "ASCEND_TOOLKIT_HOME",
            "PYTHONPATH",
        ):
            self.env.pop(key, None)
        python_bin = self.directory / "bin"
        python_bin.mkdir()
        (python_bin / "python3").symlink_to(sys.executable)
        self.env.update(
            PATH=str(python_bin) + os.pathsep + os.environ.get("PATH", ""),
            ASCEND_CUSTOM_OPP_PATH=str(self.directory / "existing vendor"),
            LD_LIBRARY_PATH=str(self.directory / "existing library"),
            ASCEND_OPP_PATH=str(self.directory / "system opp"),
            FLA_NPU_DISABLE_PTH="previous-fla-value",
            TORCH_DEVICE_BACKEND_AUTOLOAD="previous-autoload-value",
            QLI_TEST_PYTHON=sys.executable,
            QLI_TEST_ENV_DUMP=str(self.env_dump),
        )

    def install(self, name="private vendor's $literal", missing=None):
        vendor = self.directory / name
        library = vendor / "op_api/lib/libcust_opapi.so"
        library.parent.mkdir(parents=True)
        shutil.copy2(self.libraries[missing], library)
        return vendor, library

    def manifest(self, vendor, library, path=None):
        path = path or self.repo / ".qli-op-build/install.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"opp_root": str(vendor), "opapi_lib": str(library)}))
        return path

    def source(self, *arguments, repeat=False):
        script = """
qli_test_activation=$1
shift
if source "$qli_test_activation" "$@"; then
    qli_test_status=0
else
    qli_test_status=$?
fi
"""
        if repeat:
            script += """
if [[ $qli_test_status == 0 ]]; then
    if source "$qli_test_activation" "$@"; then
        qli_test_status=0
    else
        qli_test_status=$?
    fi
fi
"""
        script += """
export QLI_TEST_SOURCE_STATUS=$qli_test_status
"$QLI_TEST_PYTHON" "$QLI_TEST_ENV_DUMP"
"""
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-c", script, "qli-test", str(self.activation), *map(str, arguments)],
            env=self.env,
            cwd=self.repo,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = result.stdout.splitlines()
        child = json.loads(next(line.removeprefix("CHILD_ENV=") for line in lines if line.startswith("CHILD_ENV=")))
        status = int(next(line.removeprefix("SOURCE_STATUS=") for line in lines if line.startswith("SOURCE_STATUS=")))
        return status, child, result

    def test_source_manifest_preserves_paths_and_exports_to_child(self):
        vendor, library = self.install()
        manifest = self.manifest(vendor, library, self.directory / "explicit manifest.json")
        status, child, result = self.source("--manifest", manifest)
        self.assertEqual(status, 0, result.stdout + result.stderr)
        self.assertEqual(
            child["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep),
            [str(vendor), self.env["ASCEND_CUSTOM_OPP_PATH"]],
        )
        self.assertEqual(
            child["LD_LIBRARY_PATH"].split(os.pathsep),
            [str(library.parent), self.env["LD_LIBRARY_PATH"]],
        )
        self.assertEqual(child["ASCEND_OPP_PATH"], self.env["ASCEND_OPP_PATH"])
        self.assertEqual(child["FLA_NPU_DISABLE_PTH"], "1")
        self.assertEqual(child["TORCH_DEVICE_BACKEND_AUTOLOAD"], "0")

    def test_repeated_source_uses_local_manifest_without_duplicate_paths(self):
        vendor, library = self.install()
        self.manifest(vendor, library)
        status, child, result = self.source(repeat=True)
        self.assertEqual(status, 0, result.stdout + result.stderr)
        self.assertEqual(
            child["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep),
            [str(vendor), self.env["ASCEND_CUSTOM_OPP_PATH"]],
        )
        self.assertEqual(
            child["LD_LIBRARY_PATH"].split(os.pathsep),
            [str(library.parent), self.env["LD_LIBRARY_PATH"]],
        )

    def test_missing_selected_api_fails_without_mutating_parent_environment(self):
        fallback_vendor, _ = self.install("valid fallback vendor")
        self.env["ASCEND_CUSTOM_OPP_PATH"] = str(fallback_vendor)
        for missing in SYMBOLS:
            with self.subTest(missing=missing):
                _, library = self.install(missing, missing=missing)
                status, child, result = self.source("--opapi-lib", library)
                self.assertNotEqual(status, 0, result.stdout + result.stderr)
                self.assertEqual(child, {key: self.env.get(key) for key in ENV_KEYS})
                self.assertIn(missing, result.stdout + result.stderr)

    def test_explicit_missing_manifest_does_not_fall_back_to_valid_local_manifest(self):
        vendor, library = self.install()
        self.manifest(vendor, library)
        missing = self.directory / "missing install.json"
        status, child, result = self.source("--manifest", missing)
        self.assertNotEqual(status, 0, result.stdout + result.stderr)
        self.assertEqual(child, {key: self.env.get(key) for key in ENV_KEYS})
        self.assertIn(str(missing), result.stdout + result.stderr)

    def test_unused_unresolved_optional_function_does_not_block_lazy_symbol_check(self):
        vendor, library = self.install()
        shutil.copy2(self.optional_unresolved_library, library)
        status, child, result = self.source("--opapi-lib", library)
        self.assertEqual(status, 0, result.stdout + result.stderr)
        self.assertEqual(child["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep)[0], str(vendor))
        for symbol in SYMBOLS:
            self.assertIn(symbol, result.stdout + result.stderr)

    def test_invalid_library_fails_without_mutating_parent_environment(self):
        _, library = self.install()
        library.write_text("This is not a shared library.\n")
        status, child, result = self.source("--opapi-lib", library)
        self.assertNotEqual(status, 0, result.stdout + result.stderr)
        self.assertEqual(child, {key: self.env.get(key) for key in ENV_KEYS})
        self.assertIn(str(library), result.stdout + result.stderr)

    def test_resolve_is_shell_exports_only_and_check_reports_actual_symbol_owners(self):
        _, library = self.install()
        command = [sys.executable, str(self.helper)]
        resolved = subprocess.run(
            [*command, "resolve", "--opapi-lib", str(library)],
            env=self.env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(resolved.returncode, 0, resolved.stdout + resolved.stderr)
        self.assertTrue(resolved.stdout.strip())
        self.assertTrue(all(line.startswith("export ") for line in resolved.stdout.splitlines()), resolved.stdout)
        checked = subprocess.run(
            [*command, "check", "--opapi-lib", str(library)],
            env=self.env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
        for symbol in SYMBOLS:
            self.assertIn(symbol, checked.stdout + checked.stderr)
        self.assertIn(str(library), checked.stdout + checked.stderr)

    def test_existing_cann_vendor_config_can_supply_apis_without_private_install(self):
        opp = Path(self.env["ASCEND_OPP_PATH"])
        vendor = opp / "vendors" / "container_vendor"
        library = vendor / "op_api/lib/libcust_opapi.so"
        library.parent.mkdir(parents=True)
        shutil.copy2(self.libraries[None], library)
        (opp / "vendors/config.ini").write_text("load_priority=container_vendor\n")
        result = subprocess.run(
            [sys.executable, str(self.helper), "check"],
            env=self.env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for symbol in SYMBOLS:
            self.assertIn(symbol, result.stdout + result.stderr)
        self.assertIn(str(library), result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
