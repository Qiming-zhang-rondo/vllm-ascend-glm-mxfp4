# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of QLI shape logging; these do not validate NPU execution."""

import ctypes
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TILING_SOURCE = REPO / "csrc/attention/quant_lightning_indexer_v2/op_host/quant_lightning_indexer_v2_tiling.cpp"


class QLITilingLoggingCPUTests(unittest.TestCase):
    def test_shape_diagnostics_do_not_call_sdk_formatter(self):
        # A declaration in the SDK does not guarantee a corresponding exported
        # symbol in the container's libraries. Check every call, not just one.
        self.assertNotRegex(TILING_SOURCE.read_text(), r"Ops\s*::\s*Base\s*::\s*ToString\s*\(")

    def test_real_local_helpers_link_load_and_format_without_sdk(self):
        compiler = shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
        if compiler is None:
            self.skipTest("A C++ compiler is required for the CPU-only shared-library check")
        if sys.platform not in ("darwin", "linux"):
            self.skipTest("Shared-library linker flags are supported on macOS and Linux")

        source = TILING_SOURCE.read_text()
        helpers = []
        for name in ("ToVector", "ToStringRaw", "FormatShape"):
            match = re.search(
                rf"^static [^\n]+ {name}\(const gert::Shape &shape\)\s*\n\{{.*?^\}}",
                source,
                re.MULTILINE | re.DOTALL,
            )
            self.assertIsNotNone(match, f"Cannot extract actual {name} helper from QLI tiling source")
            helpers.append(match.group())

        stub = """
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>
namespace gert {
class Shape {
public:
    explicit Shape(const std::vector<int64_t>& dims) : dims_(dims) {}
    size_t GetDimNum() const { return dims_.size(); }
    int64_t GetDim(size_t i) const { return dims_.at(i); }
private:
    std::vector<int64_t> dims_;
};
}
"""
        wrapper = """
extern "C" const char* format_shape(const int64_t* dims, size_t rank)
{
    std::vector<int64_t> values;
    if (rank != 0) {
        values.assign(dims, dims + rank);
    }
    static thread_local std::string output;
    output = FormatShape(gert::Shape(values));
    return output.c_str();
}
"""
        with tempfile.TemporaryDirectory(prefix="qli-logging-cpu-") as directory:
            root = Path(directory)
            translation_unit = root / "logging.cpp"
            library = root / ("logging.dylib" if sys.platform == "darwin" else "logging.so")
            translation_unit.write_text(stub + "\n".join(helpers) + wrapper)
            linker_flags = (
                ["-dynamiclib", "-Wl,-undefined,error"] if sys.platform == "darwin" else ["-shared", "-Wl,-z,defs"]
            )
            result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-fPIC",
                    *linker_flags,
                    str(translation_unit),
                    "-o",
                    str(library),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            loaded = ctypes.CDLL(str(library))
            formatter = loaded.format_shape
            formatter.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t]
            formatter.restype = ctypes.c_char_p
            for dims, expected in (
                ((), "[]"),
                ((0,), "[0]"),
                ((1, 64, 128), "[1, 64, 128]"),
                ((-1, 64, -1), "[-1, 64, -1]"),
                ((2**40, 2**63 - 1, -(2**63)), "[1099511627776, 9223372036854775807, -9223372036854775808]"),
            ):
                with self.subTest(dims=dims):
                    values = (ctypes.c_int64 * len(dims))(*dims) if dims else None
                    self.assertEqual(formatter(values, len(dims)).decode(), expected)


if __name__ == "__main__":
    unittest.main()
