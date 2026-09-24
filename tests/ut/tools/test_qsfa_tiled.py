# SPDX-License-Identifier: Apache-2.0
"""Execute actual tiled gather producers on CPU, with independent layout inverses.

No ASC compiler, NPU instructions, cross-core synchronization or performance
are verified here. Device correctness remains gated by the existing A5 runner.
"""

import ctypes
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from benchmarks.qsfa_fake_quant.reference import synthetic_inputs
from benchmarks.qsfa_q8c4_o8 import compare, packing, run

ROOT = Path(__file__).resolve().parents[3]
CPP = r"""
#include <cstdint>
#define __simt_vf__
#define __aicore__
#define LAUNCH_BOUND(x)
#define __gm__
#define __ubuf__
struct ThreadId { uint32_t x; };
static ThreadId threadIdx;
#include "tiled_gather.h"
extern "C" void qk(uint8_t* q, uint8_t* qs, uint8_t* kv, uint8_t* ks,
    uint16_t* rope, int32_t* idx, int32_t* status, uint8_t* ub,
    uint32_t h, uint32_t cache, uint32_t s, uint32_t mo, uint32_t no, uint32_t sub, uint32_t check) {
    for (threadIdx.x=0; threadIdx.x<qsfa_tiled::THREADS; ++threadIdx.x)
        qsfa_tiled_gather::PrepareQk(q,qs,kv,ks,rope,idx,status,ub,h,cache,s,mo,no,sub,check);
}
extern "C" void pv(uint8_t* kv, uint8_t* ks, int32_t* idx, uint16_t* ub,
    uint32_t cache, uint32_t token, uint32_t dim) {
    for (threadIdx.x=0; threadIdx.x<qsfa_tiled::THREADS; ++threadIdx.x)
        qsfa_tiled_gather::PreparePv(kv,ks,idx,ub,cache,token,dim);
}
extern "C" uint32_t constants(uint32_t i) {
    using namespace qsfa_tiled;
    const uint32_t values[]={UQ,UK,UQS,UKS,UQR,UKR,QK_UB_BYTES,PV_DECODE_UB_BYTES};
    return values[i];
}
"""


class TiledProducerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if not compiler:
            raise unittest.SkipTest("Host C++ compiler unavailable")
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        source, library = Path(temporary.name) / "gather.cpp", Path(temporary.name) / "gather.so"
        source.write_text(CPP)
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O2",
                "-shared",
                "-fPIC",
                "-I",
                str(ROOT / "benchmarks/qsfa_q8c4_o8/csrc"),
                str(source),
                "-o",
                str(library),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.qk.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_uint32] * 7
        cls.lib.qk.restype = None
        cls.lib.pv.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint32] * 3
        cls.lib.pv.restype = None
        cls.lib.constants.argtypes = [ctypes.c_uint32]
        cls.lib.constants.restype = ctypes.c_uint32
        cls.offsets = [cls.lib.constants(i) for i in range(8)]

    @staticmethod
    def inverse_nz(data, rows, reduction, c0):
        return data.reshape(reduction // c0, rows, c0).permute(1, 0, 2).reshape(rows, reduction)

    @staticmethod
    def inverse_mx_scales(data, rows):
        # Four K128 chunks, each DN2NZ on B16 byte-pair carriers.
        return data.reshape(4, rows // 16, 2, 16, 2).permute(1, 3, 0, 2, 4).reshape(rows, 16)

    def inputs(self, h):
        q, kv, idx, scale = synthetic_inputs(1, 512, h, 256, 319)
        # Nonuniform D32 scale and unsorted sparse slots expose layout/axis swaps.
        q = q * torch.logspace(-2, 2, 18).repeat_interleave(32)
        kv = kv * torch.logspace(-2, 2, 18).repeat_interleave(32)
        return packing.prepare_inputs(q, kv, idx, scale)

    def test_qk_producer_all_chunks_scales_rope_padding_and_sparse_indices(self):
        uq, uk, uqs, uks, uqr, ukr, size, _ = self.offsets
        for h, mo in ((8, 0), (16, 0), (32, 16), (64, 48)):
            prepared, dq, _, _, _ = self.inputs(h)
            for sub in (0, 1):
                with self.subTest(heads=h, m_offset=mo, sub=sub):
                    guard = torch.full((size + 64,), 0xA5, dtype=torch.uint8)
                    ub = guard[32:-32]
                    status = torch.tensor([123], dtype=torch.int32)
                    self.lib.qk(
                        *[prepared[n].data_ptr() for n in run.ARGUMENT_ORDER],
                        status.data_ptr(),
                        ub.data_ptr(),
                        h,
                        512,
                        256,
                        mo,
                        64,
                        sub,
                        int(sub == 0),
                    )
                    self.assertTrue(bool((guard[:32] == 0xA5).all() and (guard[-32:] == 0xA5).all()))
                    self.assertEqual(status.item(), 0 if sub == 0 else 123)
                    slots = prepared["idx"][64 + sub * 32 : 96 + sub * 32].long()
                    expanded = packing.expand_mxfp4_to_fp8(prepared["kv"][slots])
                    k = ub[uk : uk + 32 * 512].reshape(4, 4, 32, 32).permute(2, 0, 1, 3).reshape(32, 512)
                    torch.testing.assert_close(k, expanded, rtol=0, atol=0)
                    torch.testing.assert_close(
                        self.inverse_mx_scales(ub[uks : uks + 32 * 16], 32), prepared["ks"][slots], rtol=0, atol=0
                    )
                    kr = self.inverse_nz(ub[ukr : ukr + 32 * 64 * 2].view(torch.bfloat16), 32, 64, 16)
                    torch.testing.assert_close(kr, prepared["rope"][slots], rtol=0, atol=0)
                    if sub:
                        self.assertTrue(bool((ub[uq : uq + 16 * 512] == 0xA5).all()))
                        continue
                    expected_q = torch.zeros((16, 512), dtype=torch.uint8)
                    expected_qs = torch.full((16, 16), 127, dtype=torch.uint8)
                    valid = min(16, h - mo)
                    expected_q[:valid] = prepared["q"][mo : mo + valid, :512]
                    expected_qs[:valid] = prepared["qs"][mo : mo + valid, :16]
                    actual_q = ub[uq : uq + 16 * 512].reshape(4, 4, 16, 32).permute(2, 0, 1, 3).reshape(16, 512)
                    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
                    torch.testing.assert_close(self.inverse_mx_scales(ub[uqs : uqs + 256], 16), expected_qs)
                    qr = self.inverse_nz(ub[uqr : uqr + 16 * 64 * 2].view(torch.bfloat16), 16, 64, 16)
                    expected_rope = torch.zeros((16, 64), dtype=torch.bfloat16)
                    expected_rope[:valid] = dq[0, mo : mo + valid, 512:].bfloat16()
                    torch.testing.assert_close(qr, expected_rope, rtol=0, atol=0)

    def test_pv_tile_transpose_and_nonuniform_scales_match_decoded_cache(self):
        prepared, _, decoded, _, _ = self.inputs(8)
        _, _, _, _, _, _, _, size = self.offsets
        for token in (0, 64, 128, 192):
            for dim in (0, 64, 192, 448):
                guard = torch.full((size + 64,), 0xA5, dtype=torch.uint8)
                ub = guard[32:-32]
                self.lib.pv(*[prepared[n].data_ptr() for n in ("kv", "ks", "idx")], ub.data_ptr(), 512, token, dim)
                actual = self.inverse_nz(ub.view(torch.bfloat16), 64, 64, 16)
                slots = prepared["idx"][token : token + 64].long()
                expected = decoded[slots, dim : dim + 64].bfloat16().T
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(bool((guard[:32] == 0xA5).all() and (guard[-32:] == 0xA5).all()))

    def test_invalid_slots_are_flagged_and_never_dereferenced(self):
        prepared, _, _, _, _ = self.inputs(8)
        prepared["idx"][:2] = torch.tensor([-1, 512], dtype=torch.int32)
        uq, uk, uqs, uks, uqr, ukr, size, pv_size = self.offsets
        ub = torch.full((size,), 0xA5, dtype=torch.uint8)
        status = torch.tensor([123], dtype=torch.int32)
        self.lib.qk(
            *[prepared[n].data_ptr() for n in run.ARGUMENT_ORDER],
            status.data_ptr(),
            ub.data_ptr(),
            8,
            512,
            256,
            0,
            0,
            0,
            1,
        )
        self.assertEqual(status.item(), 1)
        k = ub[uk : uk + 32 * 512].reshape(4, 4, 32, 32).permute(2, 0, 1, 3).reshape(32, 512)
        self.assertTrue(bool((k[:2] == 0).all()))
        self.assertTrue(bool((self.inverse_mx_scales(ub[uks : uks + 512], 32)[:2] == 127).all()))
        pv = torch.full((pv_size,), 0xA5, dtype=torch.uint8)
        self.lib.pv(*[prepared[n].data_ptr() for n in ("kv", "ks", "idx")], pv.data_ptr(), 512, 0, 0)
        values = self.inverse_nz(pv.view(torch.bfloat16), 64, 64, 16)
        self.assertTrue(bool((values[:, :2] == 0).all()))


class TiledDispatchTests(unittest.TestCase):
    def test_explicit_dispatch_has_no_silent_fallback(self):
        tiled, prototype = object(), object()
        with (
            patch.object(torch.ops, "qsfa_q8c4_o8", SimpleNamespace(forward=tiled)),
            self.assertRaises(AttributeError),
        ):
            run.candidate_operation("tiled")
        with patch.object(torch.ops, "qsfa_q8c4_o8", SimpleNamespace(forward=prototype, forward_tiled=tiled)):
            self.assertIs(run.candidate_operation("tiled"), tiled)
            self.assertIs(run.candidate_operation("prototype"), prototype)
            with self.assertRaises(ValueError):
                run.candidate_operation("unknown")

    def test_comparison_forwards_candidate_selection_to_fresh_worker(self):
        for implementation in ("tiled", "prototype"):
            args = run.parse_args(["--library", "candidate.so", "--implementation", implementation])
            command = compare.worker_command(args, "candidate", Path("input.pt"), Path("out.json"))
            self.assertEqual(command[command.index("--implementation") + 1], implementation)
        self.assertEqual(run.parse_args(["--library", "candidate.so"]).implementation, "tiled")


if __name__ == "__main__":
    unittest.main()
