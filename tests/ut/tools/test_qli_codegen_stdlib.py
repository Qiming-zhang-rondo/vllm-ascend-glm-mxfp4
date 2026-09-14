# SPDX-License-Identifier: Apache-2.0
"""Run the QLI build generators without site-packages or the regex package."""

import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
UTIL = REPO / "csrc/cmake/scripts/util"


class CodegenStdlibTests(unittest.TestCase):
    def run_python(self, *args):
        result = subprocess.run([sys.executable, "-S", *map(str, args)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_build_entry_points_start_without_site_packages(self):
        for script in (
            UTIL / "ascendc_impl_build.py",
            UTIL / "ascendc_bin_param_build.py",
            REPO / "csrc/scripts/util/merge_proto.py",
        ):
            with self.subTest(script=script.name):
                self.run_python(script, "--help")

    def test_tiling_keys_and_generated_imports_use_stdlib(self):
        result = self.run_python(
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import ascendc_impl_build as impl; import ascendc_bin_param_build as binary; "
            "assert binary.get_tiling_keys('1; 3-5 ;9-7;x3-4;8') == {1,3,4,5,8}; "
            "assert impl.optype_snake('QuantLightningIndexerV2') == 'quant_lightning_indexer_v2'; "
            "print(impl.IMPL_HEAD.format(2025, 2026, ['query'], ['indices']))",
            UTIL,
        )
        generated = ast.parse(result.stdout)
        imports = [node for node in ast.walk(generated) if isinstance(node, ast.Import)]
        names = [alias.name for node in imports for alias in node.names]
        self.assertIn("re", names)
        self.assertNotIn("regex", names)
        compile(result.stdout, "generated_qli.py", "exec")

    def test_proto_merge_runs_without_site_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "qli_proto.h", root / "merged.h"
            definition = "REG_OP(QuantLightningIndexerV2)\n.INPUT(query, TensorType({DT_FLOAT}))\n"
            definition += "OP_END_FACTORY_REG(QuantLightningIndexerV2)"
            source.write_text(definition)
            self.run_python(REPO / "csrc/scripts/util/merge_proto.py", source, "--output-file", output)
            self.assertIn(definition, output.read_text())


if __name__ == "__main__":
    unittest.main()
