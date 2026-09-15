# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression checks for the no-download standalone build launcher."""

import os
import sys

if sys.path and os.path.realpath(sys.path[0]) == os.path.dirname(os.path.realpath(__file__)):
    sys.path.pop(0)

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("qli_offline_build", REPO / "tools/build_qli_v2_mxfp4_a5.py")
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def make_install(prefix):
    vendor = prefix / "packages/vendors/qli_mxfp4_transformer"
    files = (
        "op_api/lib/libcust_opapi.so",
        "op_impl/ai_core/tbe/kernel/ascend950/quant_lightning_indexer_v2/QuantLightningIndexerV2_fp4.o",
        "op_impl/cpu/aicpu_kernel/impl/libtransformer_aicpu_kernels.so",
        "op_impl/cpu/config/cust_aicpu_kernel.json",
        "op_impl/ai_core/tbe/kernel/config/ascend950/binary_info_config.json",
        "op_impl/ai_core/tbe/kernel/config/ascend950/quant_lightning_indexer_v2.json",
    )
    for relative in files:
        path = vendor / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mock artifact")
    return vendor


def git_commit(repo):
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=QLI Test", "-c", "user.email=qli@example.invalid", "commit", "-qm", "test"],
        cwd=repo,
        check=True,
    )


def make_clean_source(repo):
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "csrc").mkdir()
    (repo / "csrc/kernel.cpp").write_text("original kernel\n")
    (repo / "benchmark.py").write_text("original benchmark\n")
    git_commit(repo)
    return BUILD.source_identity(repo)


class TestQliOfflineBuild(unittest.TestCase):
    def test_a5_soc_variant_normalizes_to_family(self):
        args = BUILD.parse_args(["--soc", "ascend950dt_9582"])
        self.assertEqual(args.requested_soc, "ascend950dt_9582")
        self.assertEqual(args.soc, "ascend950")

    def test_build_sequence_has_two_configures_and_no_installers(self):
        args = BUILD.parse_args(["--jobs", "3"])
        prereqs = {
            "commands": {"cmake": "/container/cmake", "gcc": "/container/gcc", "g++": "/container/g++"},
            "generator": "Ninja",
            "json_include": Path("/container/torch/include"),
        }
        commands = BUILD.build_commands(
            Path("/repo with space"),
            Path("/private/build"),
            Path("/private/install"),
            Path("/container/CANN"),
            prereqs,
            args,
        )
        self.assertEqual(len(commands), 5)
        self.assertEqual(commands[0], commands[2])
        self.assertIn("prepare_build", commands[1])
        self.assertIn("-DQLI_STANDALONE_OFFLINE=ON", commands[0])
        self.assertIn("-DASCEND_OP_NAME=" + BUILD.OPS, commands[0])
        self.assertIn(f"-DPython3_EXECUTABLE={sys.executable}", commands[0])
        self.assertEqual(
            commands[-1], ["/container/cmake", "--install", "/private/build", "--prefix", "/private/install"]
        )
        for command in commands:
            self.assertEqual(command[0], "/container/cmake")
            self.assertFalse(any(token in {"pip", "curl", "wget", "docker", "package"} for token in command))

    def test_runtime_only_container_fails_without_invoking_subprocess(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(BUILD.subprocess, "run") as run:
            with self.assertRaisesRegex(BUILD.PrerequisiteError, "No toolkit will be downloaded"):
                BUILD.find_cann_root(Path(directory), {})
            run.assert_not_called()

    def test_explicit_json_does_not_silently_fall_back(self):
        paths = BUILD.json_candidates(Path("/repo"), Path("/cann"), Path("/requested/include"))
        self.assertEqual(paths, [Path("/requested/include")])

    def test_offline_prerequisites_report_headers_tools_and_no_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(BUILD.shutil, "which", return_value=None),
                patch.object(BUILD.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "no tbe")),
                self.assertRaises(BUILD.PrerequisiteError) as raised,
            ):
                BUILD.check_prerequisites(root, root, {"PATH": ""}, root / "json")
            message = str(raised.exception)
            self.assertIn("nothing will be installed or downloaded", message)
            for detail in ("nlohmann/json.hpp", "bisheng", "libbase_ascend_protobuf.a", "acl_base.h", "no tbe"):
                self.assertIn(detail, message)

    def test_artifact_check_rejects_api_only_and_missing_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            vendor = make_install(prefix)
            kernel = next(vendor.rglob("*.o"))
            kernel.unlink()
            with self.assertRaisesRegex(RuntimeError, "API-only"):
                BUILD.installed_artifacts(prefix)
            kernel.write_bytes(b"mock kernel")
            (vendor / "op_impl/cpu/aicpu_kernel/impl/libtransformer_aicpu_kernels.so").unlink()
            with self.assertRaisesRegex(RuntimeError, "metadata AICPU"):
                BUILD.installed_artifacts(prefix)

    def test_artifact_check_rejects_missing_kernel_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            vendor = make_install(prefix)
            (vendor / "op_impl/cpu/config/cust_aicpu_kernel.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "registration/configuration"):
                BUILD.installed_artifacts(prefix)

    def test_reuse_requires_source_cann_soc_and_complete_install(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            make_install(prefix)
            opapi, vendor = BUILD.installed_artifacts(prefix)
            manifest = prefix / "install.json"
            manifest.write_text(
                json.dumps(
                    {
                        "source_digest": "source",
                        "cann_digest": "cann",
                        "soc": "ascend950",
                        "opapi_lib": str(opapi),
                        "opp_root": str(vendor),
                    }
                )
            )
            self.assertTrue(BUILD.reusable_manifest(manifest, "source", "cann", "ascend950"))
            self.assertFalse(BUILD.reusable_manifest(manifest, "changed", "cann", "ascend950"))
            self.assertFalse(BUILD.reusable_manifest(manifest, "source", "changed", "ascend950"))
            self.assertFalse(BUILD.reusable_manifest(manifest, "source", "cann", "ascend910b"))
            opapi.unlink()
            self.assertFalse(BUILD.reusable_manifest(manifest, "source", "cann", "ascend950"))

    def test_full_mock_build_writes_manifest_only_after_device_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = []
            prereqs = {
                "commands": {"cmake": "/mock/cmake", "gcc": "/mock/gcc", "g++": "/mock/g++"},
                "generator": "Ninja",
                "json_include": root / "json",
            }

            def run(command, **kwargs):
                commands.append(command)
                self.assertEqual(command[0], "/mock/cmake")
                if "--install" in command:
                    make_install(Path(command[command.index("--prefix") + 1]))
                return subprocess.CompletedProcess(command, 0)

            with (
                patch.object(BUILD, "__file__", str(root / "tools/build.py")),
                patch.object(BUILD.platform, "system", return_value="Linux"),
                patch.object(BUILD, "find_cann_root", return_value=root / "CANN"),
                patch.object(BUILD, "container_environment", return_value={}),
                patch.object(BUILD, "check_prerequisites", return_value=prereqs),
                patch.object(BUILD, "source_identity", return_value=("git-ref", "source-digest")),
                patch.object(BUILD, "cann_identity", return_value="cann-digest"),
                patch.object(BUILD.subprocess, "run", side_effect=run),
            ):
                self.assertEqual(BUILD.main([]), 0)
            self.assertEqual(len(commands), 5)
            data = json.loads((root / ".qli-op-build/install.json").read_text())
            self.assertEqual(data["ref"], "git-ref")
            self.assertFalse(data["device_tested"])
            self.assertTrue(Path(data["opapi_lib"]).is_file())
            self.assertTrue(Path(data["opp_root"]).is_dir())

    def test_python_only_commit_reuses_clean_schema_one_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            ref, old_digest = make_clean_source(repo)
            self.assertEqual(old_digest, hashlib.sha256(ref.encode()).hexdigest())
            make_install(root / "install")
            opapi, vendor = BUILD.installed_artifacts(root / "install")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "ref": ref,
                        "source_digest": old_digest,
                        "cann_digest": "cann",
                        "soc": "ascend950",
                        "opapi_lib": str(opapi),
                        "opp_root": str(vendor),
                    }
                )
            )
            (repo / "benchmark.py").write_text("CPU input preparation\n")
            git_commit(repo)
            _, current_digest = BUILD.source_identity(repo)
            self.assertNotEqual(old_digest, current_digest)
            self.assertTrue(BUILD.reusable_manifest(manifest, current_digest, "cann", "ascend950", repo))
            self.assertFalse(BUILD.reusable_manifest(manifest, current_digest, "new-cann", "ascend950", repo))
            self.assertFalse(BUILD.reusable_manifest(manifest, current_digest, "cann", "ascend910b", repo))
            opapi.unlink()
            self.assertFalse(BUILD.reusable_manifest(manifest, current_digest, "cann", "ascend950", repo))

    def test_legacy_clean_build_rejects_all_csrc_changes(self):
        for change in ("unstaged", "staged", "committed", "untracked"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                repo = Path(directory) / "repo"
                ref, digest = make_clean_source(repo)
                data = {"schema_version": 1, "ref": ref, "source_digest": digest}
                if change == "untracked":
                    (repo / "csrc/new_kernel.h").write_text("new header\n")
                else:
                    (repo / "csrc/kernel.cpp").write_text("changed kernel\n")
                if change == "staged":
                    subprocess.run(["git", "add", "."], cwd=repo, check=True)
                elif change == "committed":
                    git_commit(repo)
                self.assertFalse(BUILD.unchanged_clean_build_source(data, repo))

    def test_legacy_dirty_build_missing_ref_and_unknown_schema_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            ref, digest = make_clean_source(repo)
            data = {"schema_version": 1, "ref": ref, "source_digest": digest}
            self.assertFalse(BUILD.unchanged_clean_build_source({**data, "source_digest": "dirty"}, repo))
            self.assertFalse(BUILD.unchanged_clean_build_source({**data, "schema_version": 2}, repo))
            missing_ref = "a" * 40
            self.assertFalse(
                BUILD.unchanged_clean_build_source(
                    {**data, "ref": missing_ref, "source_digest": hashlib.sha256(missing_ref.encode()).hexdigest()},
                    repo,
                )
            )

    def test_non_linux_does_not_reuse_a_stale_success_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / ".qli-op-build/install.json"
            manifest.parent.mkdir()
            manifest.write_text('{"stale":true}')
            with (
                patch.object(BUILD, "__file__", str(root / "tools/build.py")),
                patch.object(BUILD.platform, "system", return_value="Darwin"),
                patch.object(BUILD.subprocess, "run") as run,
            ):
                self.assertEqual(BUILD.main([]), 78)
            self.assertFalse(manifest.exists())
            run.assert_not_called()

    def test_offline_cmake_omits_hidden_recursive_prepare_and_fetch_includes(self):
        # These are the paths that previously launched downloads even when the
        # outer build passed FETCHCONTENT_FULLY_DISCONNECTED.
        top = (REPO / "csrc/CMakeLists.txt").read_text()
        for dependency in ("abseil-cpp", "ascend_protobuf", "makeself-fetch"):
            gated = top.split("if(NOT QLI_STANDALONE_OFFLINE)", 1)[1].split("endif()", 1)[0]
            self.assertIn(f"include(cmake/third_party/{dependency}.cmake)", gated)
        common = (REPO / "csrc/common/CMakeLists.txt").read_text()
        self.assertIn("if (NOT BUILD_OPS_RTY_KERNEL AND NOT QLI_STANDALONE_OFFLINE)", common)
        config = (REPO / "csrc/cmake/config.cmake").read_text()
        self.assertIn("if (NOT PREPARE_BUILD AND ENABLE_OPS_KERNEL AND NOT QLI_STANDALONE_OFFLINE)", config)
        offline = (REPO / "csrc/cmake/qli_standalone_offline.cmake").read_text()
        self.assertIn("set(json_FOUND TRUE)", offline)
        self.assertNotIn("ExternalProject_Add(", offline)
        self.assertNotIn("FetchContent_Declare(", offline)
        self.assertNotIn("https://", offline)
        self.assertIn("set(quant_lightning_indexer_v2_depends attention/lightning_indexer_v2)", offline)
        function_source = (REPO / "csrc/cmake/func.cmake").read_text()
        self.assertIn('list(APPEND _BUILD_COMMAND export HI_PYTHON="${HI_PYTHON}" &&)', function_source)
        for kind in ("op_api", "op_tiling"):
            stub = (REPO / f"csrc/common/stub/{kind}/CMakeLists.txt").read_text()
            self.assertIn(
                "if(QLI_STANDALONE_OFFLINE)\n    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/stubs)", stub
            )


if __name__ == "__main__":
    unittest.main()
