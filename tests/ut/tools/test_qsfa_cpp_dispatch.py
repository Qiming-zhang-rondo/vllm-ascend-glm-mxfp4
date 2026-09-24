# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile and execute the real storage-format helper against CPU libtorch.

The dummy npu::get_npu_format CPU registration tests C++ Dispatcher schema,
int64 return type and format validation. It does not test torch_npu or an NPU.
No extension builder, ninja, package installation or device code is involved.
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.cpp_extension import include_paths, library_paths

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "benchmarks/qsfa_q8c4_o8/csrc"
CPP_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include "storage_format.h"

static at::Tensor format_tensor(int64_t format) {
    return at::scalar_tensor(format, at::TensorOptions().dtype(at::kLong));
}

static int64_t fake_format(const at::Tensor& tensor) {
    return tensor.item<int64_t>();
}

static bool rejects_format(int64_t format) {
    try {
        qsfa_storage::require_linear_base_format(format_tensor(format));
    } catch (const c10::Error& error) {
        const std::string message(error.what_without_backtrace());
        if (message.find("ND") == std::string::npos) {
            throw std::runtime_error("Unexpected format rejection: " + message);
        }
        return true;
    }
    return false;
}

static void check_rejected_layout(const at::Tensor& tensor) {
    try {
        qsfa_storage::require_linear_base_format(tensor);
    } catch (const c10::Error& error) {
        const std::string message(error.what_without_backtrace());
        if (message.find("contiguous with zero storage offset") == std::string::npos) throw;
        return;
    }
    throw std::runtime_error("Unsupported raw storage layout was accepted");
}

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "lifecycle";
        if (mode == "missing" || mode == "lifecycle") {
            bool rejected = false;
            try {
                qsfa_storage::require_linear_base_format(format_tensor(2));
            } catch (const c10::Error& error) {
                const std::string message(error.what_without_backtrace());
                if (message.find("npu::get_npu_format") == std::string::npos) throw;
                rejected = true;
            }
            if (!rejected) throw std::runtime_error("Missing schema was accepted");
            if (mode == "missing") {
                std::cout << "missing-op rejected\n";
                return 0;
            }
        }
        // Scope registration explicitly so the missing-op path above runs
        // before the static helper handle can be initialized successfully.
        torch::Library schema(torch::Library::DEF, "npu", std::nullopt, __FILE__, __LINE__);
        schema.def("get_npu_format(Tensor self) -> int");
        torch::Library implementation(torch::Library::IMPL, "npu", c10::DispatchKey::CPU, __FILE__, __LINE__);
        implementation.impl("get_npu_format", TORCH_FN(fake_format));

        auto dispatcher = c10::Dispatcher::singleton()
            .findSchemaOrThrow("npu::get_npu_format", "")
            .typed<int64_t(const at::Tensor&)>();
        constexpr int64_t wide_value = (int64_t{1} << 32) + 2;
        if (dispatcher.call(format_tensor(wide_value)) != wide_value) {
            throw std::runtime_error("Dispatcher int return was truncated");
        }
        qsfa_storage::require_linear_base_format(format_tensor(2));
        qsfa_storage::require_linear_base_format(format_tensor(0));
        if (mode == "reject" || mode == "lifecycle") {
            for (auto format : {int64_t{1}, int64_t{3}, int64_t{4}, int64_t{29}, int64_t{-1}, wide_value}) {
                if (!rejects_format(format)) {
                    throw std::runtime_error("Unsupported storage format was accepted");
                }
            }
            // A valid cached operator handle must dispatch each new tensor;
            // neither the previous error nor previous format may be cached.
            qsfa_storage::require_linear_base_format(format_tensor(2));
            qsfa_storage::require_linear_base_format(format_tensor(0));
        }
        if (mode == "layout" || mode == "lifecycle") {
            auto storage = at::zeros({12}, at::TensorOptions().dtype(at::kLong));
            check_rejected_layout(storage.view({3,4}).transpose(0,1));
            check_rejected_layout(storage.narrow(0,1,1));
        }
        std::cout << mode << " passed: int64 schema; ND/NCHW accepted; required rejects checked\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
"""


class CppStorageFormatDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("Host C++ compiler unavailable")
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        source = Path(temporary.name) / "dispatch.cpp"
        source.write_text(CPP_SOURCE, encoding="utf-8")
        cls.executable = Path(temporary.name) / "dispatch"
        command = [
            compiler,
            "-std=c++17",
            "-O0",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            "-I",
            str(SOURCE_ROOT),
        ]
        for path in include_paths():
            command.extend(("-I", path))
        command.extend((str(source), "-o", str(cls.executable)))
        for path in library_paths():
            command.extend(("-L", path, f"-Wl,-rpath,{path}"))
        command.extend(("-ltorch", "-ltorch_cpu", "-lc10"))
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise AssertionError(f"Host libtorch compile/link failed:\n{result.stdout}\n{result.stderr}")

    def run_case(self, mode, expected):
        result = subprocess.run([str(self.executable), mode], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
        self.assertIn(expected, result.stdout)

    def test_missing_operator_is_not_silently_accepted(self):
        self.run_case("missing", "missing-op rejected")

    def test_registered_int64_schema_accepts_nd_and_nchw(self):
        self.run_case("nd", "nd passed")

    def test_nz_negative_and_wide_integer_formats_are_rejected(self):
        self.run_case("reject", "reject passed")

    def test_noncontiguous_and_nonzero_offset_storage_is_rejected(self):
        self.run_case("layout", "layout passed")

    def test_registration_after_missing_lookup_recovers_and_checks_each_tensor(self):
        self.run_case("lifecycle", "lifecycle passed")


if __name__ == "__main__":
    unittest.main()
