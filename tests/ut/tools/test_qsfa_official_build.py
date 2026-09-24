# SPDX-License-Identifier: Apache-2.0
"""Offline source/tiling checks; these do not compile or execute A5 kernels."""

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = ROOT / "benchmarks/qsfa_q8c4_o8/official"
ARCH = "attention/kv_quant_sparse_flash_attention/op_kernel/arch35/"


class OfficialSourceBuildTests(unittest.TestCase):
    def test_schema_fields_match_retained_upstream_host_definition(self):
        upstream = (
            OFFICIAL / "vendor/attention/kv_quant_sparse_flash_attention/op_host/"
            "kv_quant_sparse_flash_attention_tiling.h"
        ).read_text()
        local = (OFFICIAL / "include/kernel_tiling/kernel_tiling.h").read_text()
        structs = re.findall(r"BEGIN_TILING_DATA_DEF\((\w+)\)(.*?)END_TILING_DATA_DEF", upstream, re.S)
        self.assertEqual(len(structs), 6)
        for name, body in structs:
            with self.subTest(struct=name):
                fields = re.findall(r"TILING_DATA_FIELD_DEF(?:_STRUCT)?\((\w+),\s*(\w+)\)", body)
                local_body = re.search(r"struct " + name + r" \{(.*?)\};", local, re.S)
                self.assertIsNotNone(local_body)
                self.assertEqual(fields, re.findall(r"(\w+)\s+(\w+);", local_body[1]))

    def test_unmodified_upstream_dependencies_and_license_keep_their_hashes(self):
        manifest = json.loads((OFFICIAL / "UPSTREAM.json").read_text())
        self.assertEqual(manifest["ref"], "55498d91634277d4eec912499c027818a8c167fb")
        modified = {
            ARCH + "kv_quant_sparse_flash_attention_common_arch35.h",
            ARCH + "kv_quant_sparse_flash_attention_kernel_mla_arch35.h",
            ARCH + "kv_quant_sparse_flash_attention_service_cube_mla_arch35.h",
            ARCH + "kv_quant_sparse_flash_attention_service_vector_mla_arch35.h",
        }
        self.assertIn("LICENSE", manifest["upstream_files"])
        for name, original_hash in manifest["upstream_files"].items():
            path = OFFICIAL / "vendor" / name
            with self.subTest(path=name):
                self.assertTrue(path.is_file())
                if name not in modified:
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), original_hash)

    def test_vendor_relative_include_routes_are_complete(self):
        # Upstream supports original and flattened build staging. We preserve
        # the original tree: either the primary or its fallback must resolve.
        roots = [
            OFFICIAL,
            OFFICIAL / "include",
            OFFICIAL / "vendor/attention/kv_quant_sparse_flash_attention/op_kernel",
            OFFICIAL / "vendor/common/include/op_kernel",
        ]
        checked = 0
        for header in (OFFICIAL / "vendor").rglob("*.h"):
            routes = re.findall(
                r'#if __has_include\("([^"]+)"\)\s+#include "[^"]+"\s+#else\s+#include "([^"]+)"',
                header.read_text(),
            )
            for primary, fallback in routes:
                checked += 1
                with self.subTest(header=header.name, include=primary):
                    self.assertTrue(
                        any(
                            (base / include).is_file()
                            for base in [header.parent, *roots]
                            for include in (primary, fallback)
                        )
                    )
        self.assertGreaterEqual(checked, 12)

    def test_device_header_closure_has_no_unresolved_local_include(self):
        roots = [
            OFFICIAL,
            OFFICIAL / "include",
            OFFICIAL / "vendor/attention/kv_quant_sparse_flash_attention/op_kernel",
            OFFICIAL / "vendor/common/include/op_kernel",
        ]
        sdk_headers = {
            "acl/acl.h",
            "adv_api/utils/init_global_memory.h",
            "kernel_basic_intf.h",
            "kernel_cube_intf.h",
            "kernel_operator.h",
            "kernel_operator_list_tensor_intf.h",
            "kernel_tensor.h",
            "kernel_vec_intf.h",
            "lib/matmul_intf.h",
            "lib/matrix/matmul/tiling.h",
            "simt_api/asc_simt.h",
            "simt_api/common_functions.h",
            "simt_api/device_functions.h",
        }
        pending = [OFFICIAL / "entry.h"]
        seen = set()
        while pending:
            path = pending.pop()
            if path in seen:
                continue
            seen.add(path)
            source = path.read_text()
            conditional_alternatives = set()
            for primary, fallback in re.findall(
                r'#if __has_include\("([^"]+)"\)\s+#include "[^"]+"\s+#else\s+#include "([^"]+)"',
                source,
            ):
                if any((base / primary).is_file() for base in [path.parent, *roots]):
                    conditional_alternatives.add(fallback)
                else:
                    conditional_alternatives.add(primary)
            for include in re.findall(r'^#include\s+"([^"]+)"', source, re.M):
                if include in conditional_alternatives or include in sdk_headers:
                    continue
                candidates = [(base / include).resolve() for base in [path.parent, *roots]]
                resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
                self.assertIsNotNone(resolved, f"Missing include {include!r} from {path}")
                pending.append(resolved)
        self.assertGreaterEqual(len(seen), 35)

    def test_actual_cpp_tiling_and_shape_contract(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        source = r"""
#include "tiling.h"
#include <cassert>
#include <initializer_list>
using namespace qsfa_official_contract;
static_assert(WorkspaceBytes(2048) == 16384);
static_assert(CONTROL_DYNAMIC_UB_BYTES >= CONTROL_ALLOCATED_UB_BYTES);
static_assert(CANDIDATE_DYNAMIC_UB_BYTES >= CANDIDATE_ALLOCATED_UB_BYTES + 8192);
static_assert(CANDIDATE_DYNAMIC_UB_BYTES + 8192 + 32768 <= 262144);
int main() {
    for (bool candidate : {false, true}) {
        for (uint32_t h : {8U, 16U, 32U}) {
            assert(SupportsShape(h, 8192, 2048, candidate));
            const auto t = MakeTiling(h, 8192, 2048, 0.04f, candidate);
            assert(t.baseParams.batchSize == 1 && t.baseParams.qSeqSize == 1);
            assert(t.baseParams.seqSize == 8192 && t.baseParams.nNumOfQInOneGroup == h);
            assert(t.baseParams.keyStride0 == (candidate ? 106496U : 167936U));
            assert(t.baseParams.dSizeVInput == (candidate ? 416 : 656));
            assert(t.baseParams.blockSize == 256 && t.baseParams.maxBlockNumPerBatch == 32);
            assert(t.singleCoreParams.usedCoreNum == 1);
            assert(t.baseParams.sparseBlockSize == 1 && t.baseParams.sparseBlockCount == 2048);
            assert(t.baseParams.sparseMode == 3 && t.baseParams.outputLayout == 1);
            assert(t.innerSplitParams.s2BaseSize == (candidate ? 64U : 128U));
            assert(t.baseParams.isActualLenDimsNull == 0 && t.baseParams.isActualLenDimsKVNull == 0);
            assert(t.baseParams.returnSoftmaxLse == 0 && t.splitKVParams.s2 == 0);
        }
        assert(!SupportsShape(7, 8192, 2048, candidate));
        assert(!SupportsShape(8, 8191, 2048, candidate));
        assert(!SupportsShape(8, 8192, 0, candidate));
        assert(!SupportsShape(8, 8192, 2047, candidate));
        assert(!SupportsShape(8, 1024, 2048, candidate));
        assert(!SupportsShape(8, 16384, 16384, candidate));
    }
    assert(SupportsShape(64, 8192, 2048, false));
    assert(!SupportsShape(64, 8192, 2048, true));
}
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "contract.cpp"
            binary = root / "contract"
            src.write_text(source)
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    f"-I{OFFICIAL}",
                    f"-I{OFFICIAL / 'include'}",
                    str(src),
                    "-o",
                    str(binary),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run([str(binary)], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
