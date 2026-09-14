# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the launcher against fake container tools, without CANN or network."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[3] / "tools/test_qli_v2_mxfp4_a5.sh"


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="qli launcher ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        tools = self.repo / "tools"
        tools.mkdir(parents=True)
        self.launcher = tools / LAUNCHER.name
        shutil.copy2(LAUNCHER, self.launcher)
        (tools / "test_qli_v2_mxfp4_a5.py").touch()
        (tools / "qli_container_backend.py").touch()
        self.write_executable(self.bin / "uname", "#!/bin/sh\necho Linux\n")
        for name in ("pip", "pip3", "curl", "wget", "docker", "podman"):
            self.write_executable(self.bin / name, "#!/bin/sh\necho UNEXPECTED_DEPENDENCY_DOWNLOAD >&2\nexit 99\n")
        self.write_executable(
            self.bin / "git",
            '#!/bin/sh\ncase "$*" in *rev-parse*) echo fixture-commit;; '
            "*) echo UNEXPECTED_GIT_NETWORK >&2; exit 99;; esac\n",
        )
        self.python = self.bin / "container-python"
        self.write_executable(
            self.python,
            f"#!{sys.executable}\n"
            "import json, os, sys\nfrom pathlib import Path\n"
            "if sys.argv[1] == '-c':\n    exec(sys.argv[2]); raise SystemExit(0)\n"
            "with open('calls.jsonl', 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
            "is_probe = '--check-only' in sys.argv\n"
            "if is_probe and not Path('.qli-op-build/install.json').exists():\n"
            "    raise SystemExit(int(os.environ.get('FAKE_PROBE_STATUS', '0')))\n"
            "raise SystemExit(0 if is_probe else int(os.environ.get('FAKE_TEST_STATUS', '0')))\n",
        )
        self.write_executable(
            tools / "build_qli_v2_mxfp4_a5.sh",
            "#!/usr/bin/env bash\nset -e\n"
            'if [[ " $* " == *" --reuse-only "* ]]; then printf reused > reuse-called; exit 0; fi\n'
            "printf 'built' > build-called\n"
            f'"{self.python}" -c '
            "'import json; from pathlib import Path; "
            'p=Path(".qli-op-build/opp/op_api/lib/libcust_opapi.so").resolve(); '
            "p.parent.mkdir(parents=True,exist_ok=True); p.touch(); "
            'Path(".qli-op-build/install.json").write_text(json.dumps('
            '{"opapi_lib":str(p),"opp_root":str(p.parents[2])}))' + "'\n",
        )
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", ASCEND_HOME_PATH=str(self.root / "cann"))

    @staticmethod
    def write_executable(path, text):
        path.write_text(text)
        path.chmod(0o755)

    def run_launcher(self, *args, stdin=False, **env):
        command = ["bash"] if stdin else ["bash", str(self.launcher)]
        command += (
            ["--python", str(self.python), *args] if not stdin else ["-s", "--", "--python", str(self.python), *args]
        )
        result = subprocess.run(
            command,
            input=LAUNCHER.read_text() if stdin else None,
            cwd=self.root,
            env=dict(self.env, **env),
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertNotIn("UNEXPECTED", result.stdout + result.stderr)
        return result

    def calls(self):
        return [json.loads(line) for line in (self.repo / "calls.jsonl").read_text().splitlines()]

    def test_installed_compute_reuses_container_without_build_or_network(self):
        result = self.run_launcher("--iterations", "7")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.calls()), 2)
        self.assertIn("7", self.calls()[1])
        self.assertFalse((self.repo / "build-called").exists())

    def test_only_missing_capability_triggers_offline_build_and_reprobe(self):
        result = self.run_launcher(FAKE_PROBE_STATUS="78")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.repo / "build-called").exists())
        self.assertEqual(len(self.calls()), 3)
        self.assertIn("--opapi-lib", self.calls()[-1])
        self.assertIn(".qli-op-build/probe-custom.json", self.calls()[1])

    def test_accuracy_or_runtime_failure_is_not_rebuilt(self):
        result = self.run_launcher(FAKE_PROBE_STATUS="1")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertFalse((self.repo / "build-called").exists())
        self.assertEqual(len(self.calls()), 1)

    def test_installed_only_never_builds(self):
        result = self.run_launcher("--installed-only", FAKE_PROBE_STATUS="78")
        self.assertEqual(result.returncode, 78, result.stdout + result.stderr)
        self.assertFalse((self.repo / "build-called").exists())

    def test_force_build_precedes_probe_and_reuses_result(self):
        result = self.run_launcher("--build-op", "--check-only")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.repo / "build-called").exists())
        self.assertEqual(len(self.calls()), 1)
        self.assertIn("--opapi-lib", self.calls()[0])

    def test_matching_cached_operator_is_reused_without_rebuild(self):
        first = self.run_launcher("--build-op", "--check-only")
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        (self.repo / "build-called").unlink()
        result = self.run_launcher("--check-only")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.repo / "reuse-called").exists())
        self.assertFalse((self.repo / "build-called").exists())
        self.assertIn("--opapi-lib", self.calls()[-1])

    def test_final_accuracy_failure_preserves_exit_status(self):
        result = self.run_launcher(FAKE_TEST_STATUS="1")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def test_downloaded_stdin_launcher_reuses_existing_checkout(self):
        (self.repo / ".git").mkdir()
        result = self.run_launcher("--workdir", str(self.repo), "--check-only", stdin=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.calls()), 1)

    def test_reject_conflicting_build_and_installed_options(self):
        result = self.run_launcher("--build-op", "--installed-only")
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.repo / "calls.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
