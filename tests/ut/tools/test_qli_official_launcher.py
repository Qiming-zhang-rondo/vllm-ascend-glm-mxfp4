# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

LAUNCHER_PATH = Path(__file__).resolve().parents[3] / "tools/run_qli_official_a5.py"
SPEC = importlib.util.spec_from_file_location("qli_official_launcher", LAUNCHER_PATH)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class TestOfficialLauncher(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name).resolve()
        self.source = self.repo / "tools/vendor/cann_qli_v2"
        self.test_dir = self.source / "pytest"
        self.vendor = self.repo / "private-opp"
        self.opapi = self.vendor / "op_api/lib/libcust_opapi.so"
        self.cann = self.repo / "existing-cann"
        for directory in (self.test_dir, self.opapi.parent, self.cann / "include", self.cann / "lib64"):
            directory.mkdir(parents=True)
        self.opapi.touch()
        manifest_dir = self.repo / ".qli-op-build"
        manifest_dir.mkdir()
        (manifest_dir / "install.json").write_text(
            json.dumps({"opp_root": str(self.vendor), "opapi_lib": str(self.opapi), "cann_root": str(self.cann)})
        )
        self.process = SimpleNamespace(pid=43210, stdout=iter(["official test output\n"]), wait=Mock(return_value=0))
        self.run = self.enterContext(patch.object(launcher.subprocess, "run"))
        self.run.return_value = subprocess.CompletedProcess([], 0, "preflight passed\n", "")
        self.popen = self.enterContext(patch.object(launcher.subprocess, "Popen", return_value=self.process))
        self.enterContext(patch.object(launcher, "__file__", str(self.repo / "tools/run_qli_official_a5.py")))
        self.enterContext(patch.object(launcher.shutil, "which", return_value="/existing/tool"))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(
            patch.dict(
                os.environ,
                {
                    "QLIV2_PARAMSET": "default",
                    "QLIV2_CASE_NAMES": "old_fp8_case",
                    "QLIV2_RUN_MODE": "graph",
                    "QLIV2_SINGLE_RESULT_PATH": "old.xlsx",
                    "QLIV2_SINGLE_SAVE_PT_DIR": "/old-pt",
                    "PYTEST_ADDOPTS": "--collect-only",
                    "PYTHONPATH": "/container/python",
                    "LD_LIBRARY_PATH": "/container/lib",
                    "ASCEND_OPP_PATH": "/old-cann/opp",
                },
                clear=True,
            )
        )

    def test_success_runs_official_mxfp4_cases_with_isolated_settings(self):
        self.assertEqual(launcher.main([]), 0)
        self.run.assert_called_once()
        self.popen.assert_called_once()
        command = self.popen.call_args.args[0]
        self.assertEqual(command[:3], [launcher.sys.executable, "-m", "pytest"])
        self.assertEqual(command[-1], str(self.test_dir / "test_quant_lightning_indexer_v2_single.py"))
        self.assertEqual(command[command.index("-c") + 1], str(self.test_dir / "pytest.ini"))
        self.assertIn("-x", command)
        self.assertEqual(self.popen.call_args.kwargs["cwd"], self.test_dir)
        environment = self.popen.call_args.kwargs["env"]
        expected = {
            "QLIV2_PARAMSET": "stc",
            "QLIV2_CASE_NAMES": "MXFP4_PA_20,MXFP4_META_70_002",
            "QLIV2_RUN_MODE": "eager",
            "QLIV2_SINGLE_RESULT_PATH": "",
            "QLIV2_SINGLE_SAVE_PT_DIR": "",
            "PYTEST_ADDOPTS": "",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "ASCEND_HOME_PATH": str(self.cann),
            "ASCEND_OPP_PATH": str(self.cann / "opp"),
            "ASCEND_CUSTOM_OPP_PATH": str(self.vendor),
            "ASCEND_LAUNCH_BLOCKING": "1",
        }
        for name, value in expected.items():
            self.assertEqual(environment[name], value, name)
        self.assertEqual(environment["PYTHONPATH"].split(os.pathsep), [str(self.source), "/container/python"])
        self.assertEqual(
            environment["LD_LIBRARY_PATH"].split(os.pathsep),
            [str(self.opapi.parent), str(self.cann / "lib64"), "/container/lib"],
        )
        self.assertEqual(self.run.call_args.kwargs["env"], environment)
        self.assertEqual(self.run.call_args.args[0][:2], [launcher.sys.executable, "-c"])
        run_dir = Path(environment["ASCEND_PROCESS_LOG_PATH"]).parent
        self.assertEqual((run_dir / "official.log").read_text(), "official test output\n")
        self.assertFalse((run_dir / "collection.log").exists())

    def test_preflight_failure_does_not_start_pytest(self):
        self.run.return_value = subprocess.CompletedProcess([], 2, "", "missing pandas\n")
        self.assertEqual(launcher.main([]), 2)
        self.run.assert_called_once()
        self.popen.assert_not_called()
        run_dir = next((self.repo / ".qli-official").iterdir())
        self.assertEqual((run_dir / "preflight.log").read_text(), "missing pandas\n")
        self.assertFalse((run_dir / "official.log").exists())
        self.assertFalse((run_dir / "run.json").exists())

    def test_pytest_failure_preserves_status_and_collects_exact_run(self):
        self.process.wait.return_value = 7
        self.run.side_effect = [
            subprocess.CompletedProcess([], 0, "preflight passed\n", ""),
            subprocess.CompletedProcess([], 1, "collector output\n", "collector warning\n"),
        ]
        self.assertEqual(launcher.main(["--cases", "MXFP4_META_70_002"]), 7)
        self.assertEqual(self.run.call_count, 2)
        environment = self.popen.call_args.kwargs["env"]
        self.assertEqual(environment["QLIV2_CASE_NAMES"], "MXFP4_META_70_002")
        plog = Path(environment["ASCEND_PROCESS_LOG_PATH"])
        run_dir = plog.parent
        self.assertEqual(
            self.run.call_args.args[0],
            [
                launcher.sys.executable,
                str(self.repo / "tools/collect_qli_timeout.py"),
                "--log",
                str(run_dir / "official.log"),
                "--plog-dir",
                str(plog),
                "--pid",
                "43210",
                "--output",
                str(run_dir / "diagnostic.json"),
            ],
        )
        metadata = json.loads((run_dir / "run.json").read_text())
        self.assertEqual(metadata["pid"], self.process.pid)
        self.assertEqual(metadata["plog_dir"], str(plog))
        self.assertEqual(metadata["command"], self.popen.call_args.args[0])
        self.assertEqual((run_dir / "collection.log").read_text(), "collector output\ncollector warning\n")


if __name__ == "__main__":
    unittest.main()
