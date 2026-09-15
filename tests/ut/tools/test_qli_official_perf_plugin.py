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
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PLUGIN_PATH = Path(__file__).resolve().parents[3] / "tools/qli_official_perf_plugin.py"
SPEC = importlib.util.spec_from_file_location("qli_official_perf_plugin_test", PLUGIN_PATH)
plugin = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plugin)


class OfficialPerfTests(unittest.TestCase):
    def test_argument_snapshot_reads_descriptors_without_device_operations(self):
        tensor = SimpleNamespace(
            shape=(1, 64, 64),
            dtype="torch.float4_e2m1fn_x2",
            device="npu:0",
            stride=lambda: (4096, 64, 1),
            storage_offset=lambda: 0,
        )
        self.assertEqual(
            plugin.describe_argument(tensor),
            {
                "shape": [1, 64, 64],
                "dtype": "torch.float4_e2m1fn_x2",
                "device": "npu:0",
                "stride": [4096, 64, 1],
                "storage_offset": 0,
            },
        )
        self.assertEqual(plugin.describe_argument(5), 5)
        self.assertIsNone(plugin.describe_argument(None))

    def test_profiler_excludes_warmup_and_reuses_exact_call_arguments(self):
        events = []
        args, kwargs = (object(), object()), {"metadata": object()}
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory)

            def op(*actual_args, **actual_kwargs):
                self.assertEqual(actual_args, args)
                self.assertEqual(actual_kwargs, kwargs)
                events.append("op")
                return object()

            @contextlib.contextmanager
            def profile(**options):
                self.assertFalse(options["experimental_config"]["data_simplification"])
                events.append("profile-start")
                yield
                events.append("profile-stop")
                (trace / "op_summary.csv").write_text(
                    "Op Name,Task Duration(us)\n" + "QuantLightningIndexerV2,4.5\n" * 3
                )

            def trace_handler(path, async_mode):
                self.assertEqual(path, directory)
                self.assertFalse(async_mode)

            profiler = SimpleNamespace(
                profile=profile,
                tensorboard_trace_handler=trace_handler,
                _ExperimentalConfig=lambda **kwargs: kwargs,
                ProfilerLevel=SimpleNamespace(Level1=1),
                ExportType=SimpleNamespace(Text="text"),
                ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
            )
            with patch.dict(
                sys.modules,
                {
                    "torch": SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: events.append("sync"))),
                    "torch_npu": SimpleNamespace(profiler=profiler),
                },
            ):
                result = plugin.profile_call(op, args, kwargs, trace, warmup=2, iterations=3)
            self.assertEqual(events, ["op", "op", "sync", "profile-start", "op", "op", "op", "sync", "profile-stop"])
            self.assertEqual(result["sample_count"], 3)
            self.assertEqual(result["p50_us"], 4.5)

    def test_real_pytest_hook_profiles_only_after_accuracy_passes(self):
        # Exercise actual pytest plugin registration/dispatch without importing NPU
        # libraries. Only the device call and profiler are substituted.
        bootstrap = r"""
import importlib.util, json, sys, types
spec = importlib.util.spec_from_file_location('qli_test_plugin', sys.argv[1])
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)
state = {'op': 0, 'perf': 0, 'accuracy': False}
def op(*args, **kwargs):
    state['op'] += 1
    return 42
namespace = types.SimpleNamespace(quant_lightning_indexer=op)
sys.modules['torch'] = types.SimpleNamespace(ops=types.SimpleNamespace(cann_ops_transformer=namespace), state=state)
def profile(original_op, args, kwargs, trace_dir, warmup, iterations):
    assert state['accuracy'] and original_op is op
    assert args == ('original-query',) and kwargs == {'metadata': 'original-metadata'}
    state['perf'] += 1
    return {'p50_us': 2., 'p90_us': 3., 'mean_us': 2.5}
plugin.profile_call = profile
import pytest
status = pytest.main(sys.argv[2:], plugins=[plugin])
print('CAPTURED_STATE=' + json.dumps(state))
sys.exit(status)
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pytest.ini").write_text("[pytest]\n")
            case = root / "test_quant_lightning_indexer_v2_single.py"
            output = root / "performance.json"
            for passed in (True, False):
                case.write_text(
                    "import pytest, torch\n"
                    "@pytest.mark.parametrize('param_combinations', [{'case_name': 'MXFP4_PA_20', 'quant_mode': 5}])\n"
                    "def test_qliv2(param_combinations):\n"
                    "    result = torch.ops.cann_ops_transformer.quant_lightning_indexer(\n"
                    "        'original-query', metadata='original-metadata')\n"
                    f"    assert result == {42 if passed else 99}\n"
                    "    torch.state['accuracy'] = True\n"
                )
                environment = os.environ.copy()
                environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
                environment["PYTEST_ADDOPTS"] = ""
                environment["PYTHONDONTWRITEBYTECODE"] = "1"
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        bootstrap,
                        str(PLUGIN_PATH),
                        "-c",
                        str(root / "pytest.ini"),
                        "-s",
                        str(case),
                        "--qli-perf-output",
                        str(output),
                    ],
                    cwd=root,
                    env=environment,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0 if passed else 1, result.stdout + result.stderr)
                state = json.loads(result.stdout.split("CAPTURED_STATE=")[-1])
                self.assertEqual(state["op"], 1)
                self.assertEqual(state["perf"], 1 if passed else 0)
                report = json.loads(output.read_text())
                self.assertEqual(report["status"], "passed" if passed else "failed")
                self.assertEqual(len(report["cases"]), 1)
                entry = report["cases"]["MXFP4_PA_20"]
                self.assertEqual(entry["status"], "passed" if passed else "accuracy_failed")
                self.assertEqual(entry["official_accuracy_passed"], passed)
                self.assertEqual(entry["compute_arguments"]["positional"], ["original-query"])
                self.assertEqual(entry["compute_arguments"]["keyword"], {"metadata": "original-metadata"})
                if not passed:
                    self.assertNotIn("performance", entry)

    def test_profile_failure_preserves_accuracy_success(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            options = {
                "--qli-perf-output": str(Path(directory) / "performance.json"),
                "--qli-perf-warmup": 5,
                "--qli-perf-iters": 20,
            }
            config = SimpleNamespace(getoption=options.__getitem__)
            plugin.pytest_configure(config)
            namespace = SimpleNamespace(quant_lightning_indexer=lambda *args, **kwargs: None)
            item = SimpleNamespace(
                module=SimpleNamespace(__name__="test_quant_lightning_indexer_v2_single"),
                config=config,
                funcargs={"param_combinations": {"case_name": "MXFP4_PA_20", "quant_mode": 5}},
                obj=lambda **unused: namespace.quant_lightning_indexer("q", metadata="m"),
            )
            with (
                patch.dict(
                    sys.modules, {"torch": SimpleNamespace(ops=SimpleNamespace(cann_ops_transformer=namespace))}
                ),
                patch.object(plugin, "profile_call", side_effect=RuntimeError("profiling failed")),
                self.assertRaisesRegex(RuntimeError, "profiling failed"),
            ):
                plugin.pytest_pyfunc_call(item)
            report = json.loads(Path(options["--qli-perf-output"]).read_text())
            entry = report["cases"]["MXFP4_PA_20"]
            self.assertTrue(entry["official_accuracy_passed"])
            self.assertEqual(entry["status"], "performance_failed")
            self.assertNotIn("performance", entry)
