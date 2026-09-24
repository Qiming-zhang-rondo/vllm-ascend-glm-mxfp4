# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare the installed native BF16-Q/O, FP8-KV QSFA baseline on CPU.

This is E4M3FN with four FP32 scales per 512-D cache token (D128 blocks),
not MXFP8 D32/E8M0. The combined PA row is 512 payload bytes, 128 BF16 RoPE
bytes and 16 FP32 scale bytes. Only the NoPE payload is numerically FP8;
the remaining bytes are transported through the same FP8-typed tensor.

Sources: cann/ops-transformer@55498d91634277d4eec912499c027818a8c167fb,
attention/kv_quant_sparse_flash_attention/tests/pytest/
kv_quant_sparse_flash_attention_golden.py (combined PA packing and golden),
and cann/ops-nn@2a77283db46e6648ff47bc8277442cf9c721e3c2,
quant/dynamic_block_quant/{tests/assets/golden.py,docs/aclnnDynamicBlockQuant.md}.
VA custom_kv_rmsnorm_rope uses DynamicBlockQuant row=1,col=128; its public
A5DeviceAdaptor call uses key=value=combined cache and TND/PA_BSND.
"""

import torch

from benchmarks.qsfa_fake_quant.reference import attention_reference
from benchmarks.qsfa_q8c4_o8.packing import NOPE_DIM, QUERY_DIM, ROPE_DIM, validate_logical_inputs

BLOCK_SIZE = 256
TILE_SIZE = 128
MAX_FP8 = 448.0
ROPE_OFFSET = NOPE_DIM
SCALE_OFFSET = NOPE_DIM + ROPE_DIM * 2
CACHE_ROW_BYTES = SCALE_OFFSET + (NOPE_DIM // TILE_SIZE) * 4


def quantize_nope(nope):
    """CPU DynamicBlockQuant contract: FP32 absmax/448, E4M3 RNE.

    Zero blocks retain scale=0 and positive-zero payload, matching the official
    golden's NaN-to-zero handling of 0/0. This prepares legal inputs; it does
    not verify the installed device quantizer's instruction-level rounding.
    """
    if (
        not isinstance(nope, torch.Tensor)
        or nope.device.type != "cpu"
        or nope.ndim != 2
        or nope.shape[1] != NOPE_DIM
        or nope.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        raise ValueError("NoPE quantization requires CPU floating-point [K,512]")
    values = nope.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("NoPE quantization input must be finite")
    blocks = values.reshape(values.shape[0], NOPE_DIM // TILE_SIZE, TILE_SIZE)
    maxima = blocks.abs().amax(dim=-1)
    scales = maxima / MAX_FP8
    if bool(((maxima != 0) & (scales == 0)).any()):
        raise ValueError("Nonzero block scale underflowed FP32; unsupported native baseline input")
    safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    normalized = blocks / safe_scales.unsqueeze(-1)
    normalized = torch.where((maxima == 0).unsqueeze(-1), torch.zeros_like(normalized), normalized)
    payload = normalized.clamp(-MAX_FP8, MAX_FP8).to(torch.float8_e4m3fn).reshape(nope.shape).contiguous()
    return payload, scales.contiguous()


def decode_cache(cache, key_tokens):
    """Decode real PA bytes to logical [K,576] FP32 for the official golden."""
    if (
        not isinstance(cache, torch.Tensor)
        or cache.device.type != "cpu"
        or cache.dtype != torch.float8_e4m3fn
        or cache.ndim != 4
        or tuple(cache.shape[1:]) != (BLOCK_SIZE, 1, CACHE_ROW_BYTES)
        or not cache.is_contiguous()
        or not 1 <= key_tokens <= cache.shape[0] * BLOCK_SIZE
    ):
        raise ValueError("Expected CPU contiguous FP8 PA[blocks,256,1,656] and a valid K length")
    raw = cache.view(torch.uint8).reshape(-1, CACHE_ROW_BYTES)[:key_tokens]
    payload = raw[:, :ROPE_OFFSET].contiguous().view(torch.float8_e4m3fn).float()
    rope = raw[:, ROPE_OFFSET:SCALE_OFFSET].contiguous().view(torch.bfloat16).float()
    scales = raw[:, SCALE_OFFSET:].contiguous().view(torch.float32)
    if not bool(torch.isfinite(scales).all()) or bool((scales < 0).any()):
        raise ValueError("Native cache scales must be finite, nonnegative FP32")
    nope = payload * scales.repeat_interleave(TILE_SIZE, dim=-1)
    decoded = torch.cat((nope, rope), dim=-1)
    if not bool(torch.isfinite(decoded).all()) or not bool(torch.isfinite(decoded.bfloat16()).all()):
        raise ValueError("Native cache decodes to nonfinite or BF16-overflowing values")
    return decoded


def prepare_inputs(query, kv, indices, scale):
    """Return (native_kwargs, decoded_kv, contract), all tensors on CPU.

    Query is TND [1,H,576], sparse indices are [1,1,S], output is [1,H,512].
    Both key and value point to the same FP8-typed combined PA tensor. Any last
    page is zero-padded; actual_seq_lengths_kv retains the original K length.
    """
    validate_logical_inputs(query, kv, indices, scale)
    query, kv = query.bfloat16().contiguous(), kv.bfloat16().contiguous()
    validate_logical_inputs(query, kv, indices, scale)
    key_tokens = kv.shape[0]
    pages = (key_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    payload, scales = quantize_nope(kv[:, :NOPE_DIM])
    raw = torch.zeros((pages * BLOCK_SIZE, CACHE_ROW_BYTES), dtype=torch.uint8, device="cpu")
    raw[:key_tokens, :ROPE_OFFSET] = payload.view(torch.uint8)
    raw[:key_tokens, ROPE_OFFSET:SCALE_OFFSET] = kv[:, NOPE_DIM:].contiguous().view(torch.uint8)
    raw[:key_tokens, SCALE_OFFSET:] = scales.view(torch.uint8)
    cache = raw.reshape(pages, BLOCK_SIZE, 1, CACHE_ROW_BYTES).view(torch.float8_e4m3fn)
    decoded_kv = decode_cache(cache, key_tokens)
    native_kwargs = {
        "query": query,
        "key": cache,
        "value": cache,
        "sparse_indices": indices.to(torch.int32).reshape(1, 1, -1).contiguous(),
        "scale_value": float(scale),
        "sparse_block_size": 1,
        "actual_seq_lengths_query": torch.tensor([1], dtype=torch.int32, device="cpu"),
        "actual_seq_lengths_kv": torch.tensor([key_tokens], dtype=torch.int32, device="cpu"),
        "layout_query": "TND",
        "layout_kv": "PA_BSND",
        "sparse_mode": 3,
        "block_table": torch.arange(pages, dtype=torch.int32, device="cpu").reshape(1, pages),
        "attention_mode": 2,
        "quant_scale_repo_mode": 1,
        "tile_size": TILE_SIZE,
        "key_quant_mode": 2,
        "value_quant_mode": 2,
        "rope_head_dim": ROPE_DIM,
    }
    contract = {
        "api": "torch_npu.npu_kv_quant_sparse_flash_attention",
        "label": "Native BF16 Q/O, E4M3FN FP8 KV with D128 FP32 scales",
        "is_mxfp8_d32_e8m0": False,
        "query_dtype": "bfloat16",
        "output_dtype": "bfloat16",
        "output_shape": [1, query.shape[1], NOPE_DIM],
        "kv_payload_dtype": "float8_e4m3fn",
        "kv_scale_dtype": "float32",
        "kv_scale_group_size": TILE_SIZE,
        "kv_scales_per_token": NOPE_DIM // TILE_SIZE,
        "producer": "CPU DynamicBlockQuant formula: min_scale=0, row_block_size=1, col_block_size=128, RNE",
        "zero_block": "FP32 scale=0 and positive-zero E4M3 payload",
        "device_quantizer_verified": False,
        "cache_layout": "PA_BSND",
        "cache_shape": list(cache.shape),
        "cache_row_bytes": CACHE_ROW_BYTES,
        "cache_offsets_bytes": {"nope": 0, "rope": ROPE_OFFSET, "scale": SCALE_OFFSET},
        "key_value_share_storage": True,
        "block_size": BLOCK_SIZE,
        "actual_key_tokens": key_tokens,
        "padded_key_tokens": pages * BLOCK_SIZE,
        "query_dim": QUERY_DIM,
        "no_int8_fallback": True,
    }
    return native_kwargs, decoded_kv, contract


def reference_output(query, decoded_kv, indices, scale):
    """Unchanged official QSFA golden via the existing logical-input adapter.

    Preserve its BF16 K/V and P casts, FP32 accumulation and final BF16 output.
    Return float32 values of that BF16 output in the public TND output shape.
    """
    return attention_reference(query.bfloat16(), decoded_kv, indices, scale).bfloat16().float()


def load_operation():
    """Resolve only the installed public native operator, with no fallback."""
    import torch_npu

    operation = getattr(torch_npu, "npu_kv_quant_sparse_flash_attention", None)
    if not callable(operation):
        raise RuntimeError("Installed torch_npu lacks npu_kv_quant_sparse_flash_attention; no fallback or install")
    return operation
