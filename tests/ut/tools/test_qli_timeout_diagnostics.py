# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[3] / "tools/collect_qli_timeout.py"
SPEC = importlib.util.spec_from_file_location("qli_diagnostics", PATH)
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


class TimeoutDiagnosticsTests(unittest.TestCase):
    def test_explicit_run_ignores_previous_probe_and_reads_its_own_plog(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / ".qli-op-build").mkdir()
            (repo / ".qli-op-build/probe-installed.json").write_text(
                json.dumps({"error": {"message": "PID: 99 aicore error exception pc start: 0x100, current: 0x200"}})
            )
            log = repo / "official.log"
            log.write_text("QLI failure without a PID in pytest output")
            plog = repo / "current-plog"
            plog.mkdir()
            (plog / "plog-123.log.1").write_text("[ERROR] kernel execution timeout\nfault kernel_name=qli_fp4\n")
            output = repo / "diagnostic.json"
            with (
                patch.object(diagnostics, "__file__", str(repo / "tools/collect_qli_timeout.py")),
                patch.object(diagnostics.shutil, "which", return_value=None),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    diagnostics.main(
                        ["--log", str(log), "--plog-dir", str(plog), "--pid", "123", "--output", str(output)]
                    ),
                    0,
                )
            report = json.loads(output.read_text())
            self.assertIsNone(report["probe_path"])
            self.assertIsNone(report["package_matches_probe_api"])
            self.assertEqual(report["pc_offsets"], {})
            self.assertEqual(report["fault_kernel_names"], ["qli_fp4"])
            self.assertEqual(report["plog"][0]["errors"], ["[ERROR] kernel execution timeout"])

    def test_offsets_are_relative_to_each_kernel_entry(self):
        text = (
            "aicore error exception, core id is 7, pc start: 0x12004f0489d0, current: 0x12004f04f9e4\n"
            "aivec error exception, core id is 68, pc start: 0x12004f1b4b00, current: 0x12004f1d00e4\n"
        )
        self.assertEqual(diagnostics.failure_offsets(text), {"mix_aic": [0x7014], "mix_aiv": [0x1B5E4]})

    def test_disassembly_adds_offset_to_matching_symbol_base(self):
        text = (
            "00001000 <qli_mix_aic>:\n  8014: wait_flag\n"
            "00020000 <qli_mix_aiv>:\n  20004: other\n  3b5e4: sync_all\n"
            "00050000 <unrelated>:\n  8014: unrelated\n"
        )
        windows = diagnostics.instruction_windows(text, {"mix_aic": [0x7014], "mix_aiv": [0x1B5E4]})
        self.assertEqual(
            [(window["symbol"], window["address"]) for window in windows],
            [("qli_mix_aic", "8014"), ("qli_mix_aiv", "3b5e4")],
        )
