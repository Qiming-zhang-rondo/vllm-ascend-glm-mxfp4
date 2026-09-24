# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Toolchain discovery fixtures; no torch import, compiler or device execution."""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

BUILD_SOURCE = Path(__file__).resolve().parents[3] / "benchmarks/qsfa_q8c4_o8/build.py"
SPEC = importlib.util.spec_from_file_location("qsfa_build_under_test", BUILD_SOURCE)
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


class QsfaBuildDiscoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def make_file(self, relative, *, executable=False):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 77\n" if executable else "# fixture\n", encoding="utf-8")
        path.chmod(0o755 if executable else 0o644)
        return path

    def test_direct_bisheng_does_not_require_asc_package(self):
        sdk = self.root / "CANN 9.1.1"
        compiler = self.make_file("CANN 9.1.1/compiler/bisheng_compiler/bin/bisheng", executable=True)
        self.make_file("CANN 9.1.1/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
        found_root, config, found_compiler = build.find_cann([sdk])
        self.assertEqual(found_root, sdk)
        self.assertIsNone(config)
        self.assertEqual(found_compiler, compiler)
        self.assertEqual(build._toolchain_options(config, found_compiler), [f"-DQSFA_BISHENG={compiler}"])

    def test_compiler_locations_and_native_precedence(self):
        locations = (
            "compiler/bisheng_compiler/bin/bisheng",
            "tools/bisheng_compiler/bin/bisheng",
            "toolkit/tools/bisheng_compiler/bin/bisheng",
            "aarch64-linux/bisheng_compiler/bin/bisheng",
            "arm64-linux/bisheng_compiler/bin/bisheng",
            "bin/bisheng",
            "compiler/ccec_compiler/bin/bisheng",
            "tools/ccec_compiler/bin/bisheng",
        )
        for index, location in enumerate(locations):
            with self.subTest(location=location):
                compiler = self.make_file(f"sdk{index}/{location}", executable=True)
                root, config, found = build.find_cann([self.root / f"sdk{index}"])
                self.assertEqual(root, self.root / f"sdk{index}")
                self.assertIsNone(config)
                self.assertEqual(found, compiler)
        preferred = self.make_file("precedence/compiler/bisheng_compiler/bin/bisheng", executable=True)
        self.make_file("precedence/bin/bisheng", executable=True)
        self.make_file("precedence/compiler/ccec_compiler/bin/bisheng", executable=True)
        self.assertEqual(build.find_cann([self.root / "precedence"])[2], preferred)

    def test_findasc_uses_module_path_while_config_uses_asc_dir(self):
        module = self.make_file("module-sdk/aarch64-linux/asc/cmake/FindASC.cmake")
        root, config, compiler = build.find_cann([self.root / "module-sdk"])
        self.assertEqual(root, self.root / "module-sdk")
        self.assertEqual(config, module)
        self.assertIsNone(compiler)
        self.assertEqual(build._toolchain_options(config, compiler), [f"-DCMAKE_MODULE_PATH={module.parent}"])
        preferred = self.make_file("module-sdk/aarch64-linux/asc/lib/cmake/ASC/ASCConfig.cmake")
        _, config, compiler = build.find_cann([self.root / "module-sdk"])
        self.assertEqual(config, preferred)
        self.assertEqual(build._toolchain_options(config, compiler), [f"-DASC_DIR={preferred.parent}"])
        compiler = self.make_file("module-sdk/bin/bisheng", executable=True)
        self.assertIn(f"-DCMAKE_ASC_COMPILER={compiler}", build._toolchain_options(config, compiler))

    def test_component_symlinks_are_followed_without_revisiting_cycle(self):
        sdk = self.root / "linked-sdk"
        sdk.mkdir()
        config = self.make_file("component/asc/cmake/ASCConfig.cmake")
        (sdk / "compiler").symlink_to(self.root / "component", target_is_directory=True)
        (sdk / "duplicate-component").symlink_to(self.root / "component", target_is_directory=True)
        (self.root / "component" / "loop-to-sdk").symlink_to(sdk, target_is_directory=True)
        matches = list(build._walk_files(sdk, {"ASCConfig.cmake", "FindASC.cmake"}))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].resolve(), config)
        root, found, compiler = build.find_cann([sdk])
        self.assertEqual(root, sdk)
        self.assertEqual(found.resolve(), config)
        self.assertIsNone(compiler)

    def test_active_root_direct_compiler_wins_over_later_asc_package(self):
        compiler = self.make_file("active/compiler/bisheng_compiler/bin/bisheng", executable=True)
        self.make_file("inactive/asc/cmake/ASCConfig.cmake")
        self.make_file("inactive/compiler/bisheng_compiler/bin/bisheng", executable=True)
        active, inactive = self.root / "active", self.root / "inactive"
        found = build.find_cann([active, inactive])
        self.assertEqual(found, (active, None, compiler))
        with patch.dict(os.environ, {"ASCEND_HOME_PATH": str(active), "ASCEND_CANN_PACKAGE_PATH": str(inactive)}):
            self.assertEqual(build._cann_candidates()[:2], [active, inactive])
            self.assertEqual(build.find_cann(), (active, None, compiler))

    def test_nonexecutable_or_missing_compiler_cannot_be_replaced_from_path(self):
        sdk = self.root / "incomplete"
        self.make_file("incomplete/compiler/bisheng_compiler/bin/bisheng", executable=False)
        unrelated = self.make_file("another-toolkit/bin/bisheng", executable=True)
        with (
            patch.dict(os.environ, {"PATH": str(unrelated.parent)}),
            self.assertRaisesRegex(RuntimeError, "No installed native ASC CMake package or bisheng compiler found"),
        ):
            build.find_cann([sdk])

    def test_nonexecutable_compiler_does_not_block_a_real_asc_package(self):
        self.make_file("sdk/compiler/bisheng_compiler/bin/bisheng", executable=False)
        config = self.make_file("sdk/asc/cmake/ASCConfig.cmake")
        self.assertEqual(build.find_cann([self.root / "sdk"]), (self.root / "sdk", config, None))

    def test_missing_toolkit_reports_checked_paths_without_download(self):
        missing = self.root / "not installed"
        with self.assertRaises(RuntimeError) as error:
            build.find_cann([missing, missing])
        self.assertIn("No toolkit will be downloaded", str(error.exception))
        self.assertEqual(str(error.exception).count(str(missing)), 1)

    @contextlib.contextmanager
    def mocked_build_environment(self):
        project, sdk = self.root / "project", self.root / "sdk"
        compilers = [
            self.make_file("sdk/compiler/bisheng_compiler/bin/bisheng", executable=True),
            self.make_file("sdk/compiler/ccec_compiler/bin/bisheng", executable=True),
        ]
        for name in (
            "CMakeLists.txt",
            "build.py",
            "csrc/vector.asc",
            "csrc/matmul.asc",
            "csrc/torch_binding.cpp",
            "csrc/launch.h",
        ):
            self.make_file(f"project/{name}")
        fake_torch = types.ModuleType("torch")
        fake_torch.__version__ = "fixture-2.10"
        fake_torch.__file__ = str(self.make_file("torch/__init__.py"))
        fake_torch._C = types.SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True)
        fake_npu = types.ModuleType("torch_npu")
        fake_npu.__version__ = "fixture-2.10"
        fake_npu.__file__ = str(self.make_file("torch_npu/__init__.py"))
        for header in (
            "NPUBridge.h",
            "npu/NPUStream.h",
            "npu/NPUGuard.h",
            "npu/NPUCachingAllocator.h",
        ):
            self.make_file(f"torch_npu/include/torch_npu/csrc/core/{header}")
        self.make_file("torch_npu/lib/libtorch_npu.so")
        cmake = self.make_file("bin/cmake", executable=True)
        with (
            patch.object(build, "ROOT", project),
            patch.object(build.platform, "system", return_value="Linux"),
            patch.object(build, "_cann_candidates", return_value=[sdk]),
            patch.object(
                build.shutil, "which", side_effect=lambda name: str(cmake) if name in ("cmake", "make") else None
            ),
            patch.dict(sys.modules, {"torch": fake_torch, "torch_npu": fake_npu}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            yield project, sdk, compilers, cmake

    def test_probe_failure_retries_only_next_installed_compiler_and_stamps_success(self):
        with self.mocked_build_environment() as (project, sdk, compilers, cmake):
            commands, configured = [], []

            def fake_cmake(command, *, check):
                self.assertTrue(check)
                self.assertEqual(command[0], str(cmake))
                commands.append(command)
                if "--build" in command:
                    directory = Path(command[command.index("--build") + 1])
                    (directory / "libqsfa_q8c4_o8.so").write_bytes(b"locally-built-fixture")
                    return subprocess.CompletedProcess(command, 0)
                directory = Path(command[command.index("-B") + 1])
                configured.append(directory)
                self.assertIn(f"-DASCEND_HOME_PATH={sdk}", command)
                self.assertIn(f"-DQSFA_BISHENG={compilers[len(configured) - 1]}", command)
                if len(configured) == 1:
                    (directory / "toolchain_probe.failed").touch()
                    raise subprocess.CalledProcessError(1, command)
                return subprocess.CompletedProcess(command, 0)

            with patch.object(build.subprocess, "run", side_effect=fake_cmake):
                result = build.build_library(jobs=2)
            self.assertEqual(len(commands), 3)
            self.assertEqual(len(set(configured)), 2)
            self.assertFalse((configured[0] / "manifest.json").exists())
            self.assertEqual(Path(result["library"]), configured[1] / "libqsfa_q8c4_o8.so")
            self.assertEqual(Path(result["library"]).read_bytes(), b"locally-built-fixture")
            manifest = json.loads((configured[1] / "manifest.json").read_text())
            self.assertEqual(manifest["fingerprint"], result["fingerprint"])
            self.assertFalse(result["reused"])
            self.assertEqual(len(list(project.rglob("libqsfa_q8c4_o8.so"))), 1)

    def test_kernel_build_failure_is_not_retried_or_marked_installed(self):
        with self.mocked_build_environment() as (project, _, compilers, _):
            commands = []

            def fake_cmake(command, *, check):
                self.assertTrue(check)
                commands.append(command)
                if "--build" in command:
                    raise subprocess.CalledProcessError(2, command)
                self.assertIn(f"-DQSFA_BISHENG={compilers[0]}", command)
                # Even a leftover probe marker must not authorize retrying a
                # genuine kernel compilation failure from the build phase.
                directory = Path(command[command.index("-B") + 1])
                (directory / "toolchain_probe.failed").touch()
                return subprocess.CompletedProcess(command, 0)

            with (
                patch.object(build.subprocess, "run", side_effect=fake_cmake),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                build.build_library()
            self.assertEqual(len(commands), 2)
            self.assertFalse(list(project.rglob("manifest.json")))
            self.assertFalse(list(project.rglob("libqsfa_q8c4_o8.so")))

    def test_unrelated_configure_failure_has_no_compiler_retry(self):
        with self.mocked_build_environment() as (project, _, _, _):
            with (
                patch.object(
                    build.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["cmake"])
                ) as execute,
                self.assertRaises(subprocess.CalledProcessError),
            ):
                build.build_library()
            self.assertEqual(execute.call_count, 1)
            self.assertFalse(list(project.rglob("manifest.json")))


if __name__ == "__main__":
    unittest.main()
