# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
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
        self.assertEqual(command[:4], [launcher.sys.executable, "-c", launcher.PYTEST_BOOTSTRAP, str(self.vendor)])
        self.assertEqual(command[-1], str(self.test_dir / "test_quant_lightning_indexer_v2_single.py"))
        pytest_args = command[4:]
        self.assertEqual(pytest_args[pytest_args.index("-c") + 1], str(self.test_dir / "pytest.ini"))
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
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "FLA_NPU_DISABLE_PTH": "1",
        }
        for name, value in expected.items():
            self.assertEqual(environment[name], value, name)
        self.assertEqual(environment["PYTHONPATH"].split(os.pathsep), [str(self.source), "/container/python"])
        self.assertEqual(
            environment["LD_LIBRARY_PATH"].split(os.pathsep),
            [str(self.opapi.parent), str(self.cann / "lib64"), "/container/lib"],
        )
        self.assertEqual(self.run.call_args.kwargs["env"], environment)
        self.assertEqual(
            self.run.call_args.args[0], [launcher.sys.executable, "-c", launcher.PREFLIGHT, str(self.vendor)]
        )
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

    def mock_perf_process(self, command, *, env, **unused):
        name = env["QLIV2_CASE_NAMES"]
        output = Path(command[command.index("--qli-perf-output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "cases": {
                        name: {
                            "status": "passed",
                            "official_accuracy_passed": True,
                            "shape": {"q_seq": 1, "k_seq": 128},
                            "performance": {"p50_us": 2.0 if name.startswith("MXFP4") else 4.0},
                        }
                    },
                }
            )
        )
        return SimpleNamespace(
            pid=43210 + self.popen.call_count, stdout=iter([f"passed {name}\n"]), wait=Mock(return_value=0)
        )

    def test_performance_isolates_cases_and_preserves_verified_launch_mode(self):
        self.popen.side_effect = self.mock_perf_process
        self.assertEqual(launcher.main(["--perf", "--warmup", "3", "--iters", "7"]), 0)
        self.assertEqual(self.popen.call_count, 4)
        names, plogs, outputs = [], [], []
        for call in self.popen.call_args_list:
            environment = call.kwargs["env"]
            self.assertEqual(environment["ASCEND_LAUNCH_BLOCKING"], "1")
            self.assertEqual(environment["TORCH_DEVICE_BACKEND_AUTOLOAD"], "0")
            self.assertEqual(environment["ASCEND_GLOBAL_LOG_LEVEL"], "3")
            names.append(environment["QLIV2_CASE_NAMES"])
            plogs.append(environment["ASCEND_PROCESS_LOG_PATH"])
            command = call.args[0]
            self.assertEqual(
                command[:5],
                [
                    launcher.sys.executable,
                    "-c",
                    launcher.PERF_BOOTSTRAP,
                    str(self.vendor),
                    str(self.repo / "tools/qli_official_perf_plugin.py"),
                ],
            )
            pytest_args = command[5:]
            self.assertEqual(pytest_args[pytest_args.index("-c") + 1], str(self.test_dir / "pytest.ini"))
            self.assertIn(str(self.test_dir / "test_quant_lightning_indexer_v2_single.py"), command)
            self.assertEqual(command[command.index("--qli-perf-warmup") + 1], "3")
            self.assertEqual(command[command.index("--qli-perf-iters") + 1], "7")
            outputs.append(Path(command[command.index("--qli-perf-output") + 1]))
        self.assertEqual(names, launcher.PERF_CASES.split(","))
        self.assertEqual(len(set(plogs)), 4)
        self.assertEqual(len(set(outputs)), 4)
        aggregate = json.loads((outputs[0].parents[1] / "performance.json").read_text())
        self.assertEqual(aggregate["status"], "passed")
        self.assertEqual(len(aggregate["cases"]), 4)
        self.assertEqual(len(aggregate["comparisons"]), 2)
        self.assertTrue(all(item["fp8_over_mxfp4_p50"] == 2.0 for item in aggregate["comparisons"]))
        self.assertIn("--perf", self.run.call_args.args[0])

    def test_performance_stops_after_first_failed_case(self):
        def fail_first(command, **kwargs):
            process = self.mock_perf_process(command, **kwargs)
            output = Path(command[command.index("--qli-perf-output") + 1])
            name = kwargs["env"]["QLIV2_CASE_NAMES"]
            output.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "cases": {
                            name: {
                                "status": "accuracy_failed",
                                "official_accuracy_passed": False,
                            }
                        },
                    }
                )
            )
            process.wait.return_value = 1
            return process

        self.popen.side_effect = fail_first
        self.assertEqual(launcher.main(["--perf"]), 1)
        self.popen.assert_called_once()
        command = self.popen.call_args.args[0]
        output = Path(command[command.index("--qli-perf-output") + 1])
        aggregate = json.loads((output.parents[1] / "performance.json").read_text())
        self.assertEqual(aggregate["status"], "failed")
        self.assertEqual(aggregate["failed_case"], "MXFP4_PA_20")
        self.assertEqual(aggregate["cases"]["MXFP4_PA_20"]["status"], "accuracy_failed")
        self.assertEqual(aggregate["comparisons"], [])

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


class TestRuntimeBootstrap(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / "observed.json"
        self.events = self.root / "events.jsonl"
        self.private_opp = str(self.root / "private operator package")
        self.environment = dict(
            os.environ,
            PYTHONPATH=str(self.root),
            TORCH_DEVICE_BACKEND_AUTOLOAD="0",
            ASCEND_CUSTOM_OPP_PATH="/old/container/vendor",
            BOOTSTRAP_EXPECTED_OPP=self.private_opp,
            BOOTSTRAP_OUTPUT=str(self.output),
            BOOTSTRAP_EVENTS=str(self.events),
        )
        modules = {
            "bootstrap_checks.py": """
                import json, os
                def check(stage):
                    actual = os.environ.get('ASCEND_CUSTOM_OPP_PATH')
                    assert actual == os.environ['BOOTSTRAP_EXPECTED_OPP'], (stage, actual)
                    with open(os.environ['BOOTSTRAP_EVENTS'], 'a') as output:
                        output.write(json.dumps(stage) + '\\n')
                def forbid_npu(*args, **kwargs):
                    raise AssertionError('Bootstrap must not execute NPU operations')
            """,
            "torch.py": """
                import os
                from types import SimpleNamespace
                from bootstrap_checks import check, forbid_npu
                assert os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] == '0'
                check('torch_import')
                float4_e2m1fn_x2 = object()
                float8_e8m0fnu = object()
                ops = SimpleNamespace(cann_ops_transformer=SimpleNamespace(
                    quant_lightning_indexer=forbid_npu,
                    quant_lightning_indexer_metadata=forbid_npu,
                ))
                class NPU:
                    def __getattr__(self, name):
                        forbid_npu()
                npu = NPU()
            """,
            "torch_npu.py": """
                import os
                from types import SimpleNamespace
                from bootstrap_checks import check, forbid_npu
                check('torch_npu_import')
                os.environ['ASCEND_CUSTOM_OPP_PATH'] = (
                    '/bundled/torch_npu/vendor:' + os.environ['ASCEND_CUSTOM_OPP_PATH']
                )
                profiler = SimpleNamespace(**{name: forbid_npu for name in (
                    'profile', 'tensorboard_trace_handler', '_ExperimentalConfig', 'ProfilerLevel', 'ExportType'
                )})
            """,
            "numpy.py": "__version__ = 'fake-cpu-only'\n",
            "pandas.py": "__version__ = 'fake-cpu-only'\n",
            "cann_ops_transformer.py": """
                from bootstrap_checks import check
                check('official_registration')
            """,
            "pytest.py": """
                import json, os
                from bootstrap_checks import check
                if os.environ['BOOTSTRAP_KIND'] != 'preflight':
                    check('pytest_import')
                def main(args, plugins=None):
                    check('pytest_main')
                    import cann_ops_transformer
                    with open(os.environ['BOOTSTRAP_OUTPUT'], 'w') as output:
                        json.dump({'args': args, 'plugins': [item.__name__ for item in plugins or []]}, output)
                    return 19
            """,
            "fake_plugin.py": """
                import pytest
                from bootstrap_checks import check
                check('plugin_import')
            """,
        }
        for filename, content in modules.items():
            (self.root / filename).write_text(textwrap.dedent(content))

    def run_bootstrap(self, bootstrap, arguments, kind):
        return subprocess.run(
            [sys.executable, "-S", "-c", bootstrap, self.private_opp, *arguments],
            env=dict(self.environment, BOOTSTRAP_KIND=kind),
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_preflight_restores_private_package_before_official_registration(self):
        result = self.run_bootstrap(launcher.PREFLIGHT, ["--perf"], "preflight")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"errors": []', result.stdout)
        self.assertIn("QLIV2_IGNORED_IMPORT_OPP_PATH: /bundled/torch_npu/vendor:", result.stdout)
        self.assertEqual(
            [json.loads(line) for line in self.events.read_text().splitlines()],
            ["torch_import", "torch_npu_import", "official_registration"],
        )
        self.assertFalse(self.output.exists(), "Preflight must not invoke pytest or NPU operators")

    def test_accuracy_and_performance_restore_private_package_and_forward_pytest_arguments(self):
        pytest_args = ["-c", "fixture with spaces/pytest.ini", "-x", "-m", "ci", "official_case.py"]
        for perf in (False, True):
            with self.subTest(perf=perf):
                self.events.unlink(missing_ok=True)
                arguments = list(pytest_args)
                if perf:
                    arguments += ["--qli-perf-output", "result with spaces.json", "--qli-perf-iters", "7"]
                expected_args = list(arguments)
                if perf:
                    arguments.insert(0, str(self.root / "fake_plugin.py"))
                result = self.run_bootstrap(
                    launcher.PERF_BOOTSTRAP if perf else launcher.PYTEST_BOOTSTRAP,
                    arguments,
                    "performance" if perf else "accuracy",
                )
                self.assertEqual(result.returncode, 19, result.stdout + result.stderr)
                self.assertEqual(
                    json.loads(self.output.read_text()),
                    {"args": expected_args, "plugins": ["qli_official_perf_plugin"] if perf else []},
                )
                self.assertEqual(
                    [json.loads(line) for line in self.events.read_text().splitlines()],
                    ["torch_import", "torch_npu_import", "pytest_import"]
                    + (["plugin_import"] if perf else [])
                    + ["pytest_main", "official_registration"],
                )
                self.assertIn("QLIV2_IGNORED_IMPORT_OPP_PATH: /bundled/torch_npu/vendor:", result.stdout)


if __name__ == "__main__":
    unittest.main()
