"""Execute the actual fused candidate's VFs on host with independent inverses.

This verifies producer/consumer bytes, padding, rounding and guard regions.
It does not simulate Ascend pipelines, cache coherency, or device execution.
"""

import ctypes
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

OFFICIAL = Path(__file__).resolve().parents[3] / "benchmarks/qsfa_q8c4_o8/official"
CPP = r"""
#include <cstdint>
#define __simt_vf__
#define __simt_callee__
#define __aicore__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(x)
struct Thread { uint32_t x; };
static Thread threadIdx;
#include "lowbit_vector.h"
extern "C" void cache(uint8_t* src,uint16_t* values,uint8_t* mx,uint32_t rows) {
    for(threadIdx.x=0;threadIdx.x<qsfa_official_vector::THREADS;++threadIdx.x)
        qsfa_official_vector::DecodeCache(src,values,mx,rows);
}
extern "C" void query(uint8_t* src,uint8_t* dst,uint32_t heads) {
    for(threadIdx.x=0;threadIdx.x<qsfa_official_vector::THREADS;++threadIdx.x)
        qsfa_official_vector::PrepareQuery(src,dst,heads);
}
extern "C" void output(float* src,uint8_t* dst,uint32_t rows) {
    for(threadIdx.x=0;threadIdx.x<qsfa_official_vector::THREADS;++threadIdx.x)
        qsfa_official_vector::EncodeOutput(src,dst,rows);
}
"""


@pytest.fixture(scope="module")
def actual_vfs(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("official_vector")
    includes = directory / "simt_api"
    includes.mkdir()
    for header in ("asc_simt.h", "common_functions.h", "device_functions.h"):
        (includes / header).write_text("// Host qualification shim; no device behavior simulated.\n")
    source, binary = directory / "vf.cpp", directory / "vf.so"
    source.write_text(CPP)
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-O2",
            "-I",
            str(directory),
            "-I",
            str(OFFICIAL),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    library = ctypes.CDLL(str(binary))
    library.cache.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint32]
    library.query.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_uint32]
    library.output.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_uint32]
    for name in ("cache", "query", "output"):
        getattr(library, name).restype = None
    return library


def guarded(size):
    storage = torch.full((size + 128,), 0xA5, dtype=torch.uint8)
    return storage, storage[64:-64]


def check_guard(storage):
    assert torch.all(storage[:64] == 0xA5)
    assert torch.all(storage[-64:] == 0xA5)


def nz_inverse(storage, rows, cols, c0):
    return storage.reshape(cols // c0, rows, c0).permute(1, 0, 2).reshape(rows, cols)


def scale_inverse(storage, rows):
    # chunks, row-blocks, scale-pairs, rows-in-block, byte-in-pair -> rows,D32.
    return storage.reshape(4, rows // 16, 2, 16, 2).permute(1, 3, 0, 2, 4).reshape(rows, 16)


def random_cache(rows, seed=32):
    generator = torch.Generator().manual_seed(seed)
    cache = torch.randint(0, 256, (rows, 416), dtype=torch.uint8, generator=generator)
    cache[:, 384:400] = torch.randint(117, 134, (rows, 16), dtype=torch.uint8, generator=generator)
    rope = torch.randn(rows, 64, generator=generator).to(torch.bfloat16)
    cache[:, 256:384] = rope.view(torch.uint8)
    return cache, rope


@pytest.mark.parametrize("rows", [1, 8, 16])
def test_cache_decodes_bf16_and_independent_mx_views(actual_vfs, rows):
    cache, rope = random_cache(rows)
    values_guard, values = guarded(576 * 17 * 2)
    mx_guard, mx = guarded(8448)
    actual_vfs.cache(cache.data_ptr(), values.data_ptr(), mx.data_ptr(), rows)
    unpacked = torch.stack((cache[:, :256] & 15, cache[:, :256] >> 4), dim=-1).reshape(rows, 512)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    expected = (lut[unpacked.long()] * torch.exp2(cache[:, 384:400].float() - 127).repeat_interleave(32, -1)).to(
        torch.bfloat16
    )
    decoded = nz_inverse(values[: 512 * 17 * 2].view(torch.bfloat16), 17, 512, 16)
    torch.testing.assert_close(decoded[:rows], expected, rtol=0, atol=0)
    decoded_rope = nz_inverse(values[512 * 17 * 2 :].view(torch.bfloat16), 17, 64, 16)
    torch.testing.assert_close(decoded_rope[:rows], rope, rtol=0, atol=0)
    # The seventeenth bank-padding row must not be touched by any VF thread.
    assert torch.all(decoded[16].view(torch.uint8) == 0xA5)
    assert torch.all(decoded_rope[16].view(torch.uint8) == 0xA5)
    fp8 = nz_inverse(mx[:8192], 16, 512, 32)
    torch.testing.assert_close(fp8[:rows].view(torch.float8_e4m3fn).float(), lut[unpacked.long()], rtol=0, atol=0)
    assert torch.equal(scale_inverse(mx[8192:], 16)[:rows], cache[:, 384:400])
    check_guard(values_guard)
    check_guard(mx_guard)


def test_16row_producer_copies_reconstruct_full_l1_slot(actual_vfs):
    cache, rope = random_cache(64, seed=77)
    l1_guard, l1 = guarded(107520)
    for row in range(0, 64, 16):
        _, values = guarded(576 * 17 * 2)
        _, mx = guarded(8448)
        actual_vfs.cache(cache[row:].data_ptr(), values.data_ptr(), mx.data_ptr(), 16)
        # Exact DataCopy block lengths/strides from CopyOutKvUb2L1.
        for block in range(36):
            dest = (block * 64 + row) * 32
            l1[dest : dest + 16 * 32] = values[block * 17 * 32 : block * 17 * 32 + 16 * 32]
        for chunk in range(4):
            for block in range(4):
                dest = 73728 + chunk * 64 * 128 + (block * 64 + row) * 32
                src = chunk * 16 * 128 + block * 16 * 32
                l1[dest : dest + 512] = mx[src : src + 512]
            dest = 106496 + chunk * 64 * 4 + row * 4
            l1[dest : dest + 64] = mx[8192 + chunk * 64 : 8192 + (chunk + 1) * 64]
    k = nz_inverse(l1[73728:106496], 64, 512, 32).view(torch.float8_e4m3fn).float()
    scales = scale_inverse(l1[106496:], 64)
    assert torch.equal(scales, cache[:, 384:400])
    expected_v = (k * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)).to(torch.bfloat16)
    torch.testing.assert_close(nz_inverse(l1[:65536].view(torch.bfloat16), 64, 512, 16), expected_v, rtol=0, atol=0)
    torch.testing.assert_close(nz_inverse(l1[65536:73728].view(torch.bfloat16), 64, 64, 16), rope, rtol=0, atol=0)
    check_guard(l1_guard)


@pytest.mark.parametrize("heads", [8, 16, 32])
def test_query_nz_mx_pairs_padding_and_bf16_rope(actual_vfs, heads):
    generator = torch.Generator().manual_seed(99 + heads)
    query = torch.randint(0, 255, (heads, 608), dtype=torch.uint8, generator=generator)
    query[:, :576] &= 0xFE  # Exclude E4M3 NaN; all signs/exponents remain covered.
    query[:, 576:594] = torch.randint(116, 134, (heads, 18), dtype=torch.uint8, generator=generator)
    guard, result = guarded(41984)
    actual_vfs.query(query.data_ptr(), result.data_ptr(), heads)
    q = nz_inverse(result[:32768], 64, 512, 32)
    assert torch.equal(q[:heads], query[:, :512])
    assert torch.all(q[heads:] == 0)
    scales = scale_inverse(result[32768:33792], 64)
    assert torch.equal(scales[:heads], query[:, 576:592])
    assert torch.all(scales[heads:] == 127)
    qrope = nz_inverse(result[33792:].view(torch.bfloat16), 64, 64, 16)
    expected = (
        query[:, 512:576].contiguous().view(torch.float8_e4m3fn).float()
        * torch.exp2(query[:, 592:594].float() - 127).repeat_interleave(32, -1)
    ).to(torch.bfloat16)
    torch.testing.assert_close(qrope[:heads], expected, rtol=0, atol=0)
    assert torch.all(qrope[heads:] == 0)
    check_guard(guard)


@pytest.mark.parametrize("rows", [4, 8, 16])
def test_final_fp32_output_quantization_groups_and_zero_padding(actual_vfs, rows):
    generator = torch.Generator().manual_seed(131 + rows)
    acc = torch.randn(rows, 512, generator=generator) * torch.exp2(torch.arange(16).repeat_interleave(32) - 8)
    acc[:, :32] = 0
    guard, result = guarded(rows * 544)
    actual_vfs.output(acc.data_ptr(), result.data_ptr(), rows)
    packed = result.reshape(rows, 544)
    maxima = acc.reshape(rows, 16, 32).abs().amax(-1)
    exponent = torch.where(maxima == 0, -127, torch.floor(torch.log2(maxima)) - 8).clamp_min(-127)
    expected_scale = (exponent + 127).to(torch.uint8)
    expected_data = (acc / torch.exp2(exponent).repeat_interleave(32, -1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(packed[:, :512], expected_data.view(torch.uint8))
    assert torch.equal(packed[:, 512:528], expected_scale)
    assert torch.all(packed[:, 528:] == 0)
    check_guard(guard)
