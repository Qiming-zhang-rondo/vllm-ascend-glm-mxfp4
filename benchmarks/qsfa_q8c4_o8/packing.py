# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU payload codecs; no vLLM, torch_npu or hardware quantizer imports.

Uses the pinned CANN DynamicMxQuant scale_alg=0 arithmetic already validated
by qsfa_fake_quant.quantization. Scales here are flat D32 E8M0 byte arrays,
not the paired [..., D/64, 2] descriptor used by DynamicMxQuant itself.
"""

import math

import torch

from benchmarks.qsfa_fake_quant.quantization import (
    FP4_MAX_EXPONENT,
    FP8_MAX_EXPONENT,
    FP8_MAX_VALUE,
    GROUP_SIZE,
    _prepare_groups,
)

NOPE_DIM = 512
ROPE_DIM = 64
QUERY_DIM = NOPE_DIM + ROPE_DIM
SUPPORTED_HEADS = (8, 16, 32, 64)
MIN_SELECTED = 128
MAX_SELECTED = 8192


def _quantization_groups(x, max_exponent):
    if not isinstance(x, torch.Tensor) or x.ndim == 0 or x.shape[-1] % GROUP_SIZE:
        raise ValueError("Packed inputs require a last dimension divisible by 32")
    normalized, scales = _prepare_groups(x, max_exponent)
    _, exponent = torch.frexp(scales.squeeze(-1))
    return normalized, (exponent + 126).to(torch.uint8).contiguous()


def pack_mxfp4(x):
    """E2M1, ties away, with consecutive even/odd D in low/high nibbles."""
    normalized, scales = _quantization_groups(x, FP4_MAX_EXPONENT)
    thresholds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32, device="cpu")
    codes = torch.bucketize(normalized.abs().contiguous(), thresholds, right=True).to(torch.uint8)
    codes |= normalized.signbit().to(torch.uint8) << 3
    codes = codes.reshape(x.shape)
    return (codes[..., ::2] | (codes[..., 1::2] << 4)).contiguous(), scales


def pack_mxfp8(x):
    """E4M3FN RNE with explicit ±448 saturation and D32 E8M0 scales."""
    normalized, scales = _quantization_groups(x, FP8_MAX_EXPONENT)
    payload = normalized.clamp(-FP8_MAX_VALUE, FP8_MAX_VALUE).to(torch.float8_e4m3fn)
    return payload.reshape(x.shape).view(torch.uint8).contiguous(), scales


def _check_bytes(tensor, name):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.uint8:
        raise ValueError(f"{name} must be CPU uint8 bytes")
    if tensor.ndim < 1 or tensor.shape[-1] == 0:
        raise ValueError(f"{name} must have a nonempty last dimension")


def decode_e8m0(scales):
    _check_bytes(scales, "scales")
    if bool((scales == 255).any()):
        raise ValueError("E8M0 byte 255 is NaN, not a valid scale")
    return torch.ldexp(torch.ones_like(scales, dtype=torch.float32), scales.int() - 127)


def _apply_scales(values, scales):
    expected = (*values.shape[:-1], values.shape[-1] // GROUP_SIZE)
    if values.shape[-1] % GROUP_SIZE or scales.shape != expected:
        raise ValueError(f"Expected D32 scales of shape {expected}, got {tuple(scales.shape)}")
    result = values * decode_e8m0(scales).repeat_interleave(GROUP_SIZE, dim=-1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Decoded payload contains NaN or infinity")
    return result


def decode_mxfp4(payload, scales):
    _check_bytes(payload, "FP4 payload")
    _check_bytes(scales, "FP4 scales")
    codes = torch.stack((payload & 15, payload >> 4), dim=-1).flatten(-2).long()
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device="cpu")
    values = torch.copysign(magnitudes[codes & 7], torch.where(codes < 8, 1.0, -1.0))
    return _apply_scales(values, scales)


def expand_mxfp4_to_fp8(payload):
    """Losslessly expand E2M1 element bits to E4M3FN, without applying scales.

    This is the CPU contract for the candidate's C4-to-FP8 gather stage. The
    original E8M0 scales are unchanged; the resulting elements are not MXFP8
    requantization of the decoded values.
    """
    _check_bytes(payload, "FP4 payload")
    codes = torch.stack((payload & 15, payload >> 4), dim=-1).flatten(-2).long()
    table = torch.tensor([0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C], dtype=torch.uint8, device="cpu")
    return (table[codes & 7] | ((codes & 8).to(torch.uint8) << 4)).contiguous()


def decode_mxfp8(payload, scales):
    _check_bytes(payload, "FP8 payload")
    _check_bytes(scales, "FP8 scales")
    return _apply_scales(payload.contiguous().view(torch.float8_e4m3fn).float(), scales)


def validate_logical_inputs(query, kv, indices, scale):
    if not isinstance(query, torch.Tensor) or query.ndim != 3:
        raise ValueError("query must have logical shape [1, H, 576]")
    if query.shape[0] != 1 or query.shape[1] not in SUPPORTED_HEADS or query.shape[2] != QUERY_DIM:
        raise ValueError("Only Q=1, H in {8,16,32,64}, and D=576 are supported")
    if not isinstance(kv, torch.Tensor) or kv.ndim != 2 or kv.shape[-1] != QUERY_DIM:
        raise ValueError("kv must have logical shape [K, 576]")
    for name, tensor in (("query", query), ("kv", kv)):
        if tensor.device.type != "cpu" or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"{name} must be CPU float16, bfloat16 or float32")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite")
    if not isinstance(indices, torch.Tensor) or indices.ndim != 2 or indices.shape[0] != 1:
        raise ValueError("indices must have shape [1, S]")
    if indices.device.type != "cpu" or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must be CPU int32/int64")
    count = indices.shape[1]
    if count < MIN_SELECTED or count > MAX_SELECTED or count % MIN_SELECTED or kv.shape[0] < count:
        raise ValueError("Require S in [128,8192], S divisible by 128, and K >= S")
    if kv.shape[0] > torch.iinfo(torch.int32).max:
        raise ValueError("K must fit the prototype's int32 sparse index contract")
    if bool(((indices < 0) | (indices >= kv.shape[0])).any()) or indices.unique().numel() != count:
        raise ValueError("Sparse indices must be unique valid keys; this decode case has no -1 padding")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")


def prepare_inputs(query, kv, indices, scale):
    """BF16 source activations -> validated CPU arguments for the custom op."""
    validate_logical_inputs(query, kv, indices, scale)
    query, kv = query.bfloat16(), kv.bfloat16()
    validate_logical_inputs(query, kv, indices, scale)
    q, qs = pack_mxfp8(query[0])
    c, cs = pack_mxfp4(kv[:, :NOPE_DIM])
    prepared = {
        "q": q,
        "qs": qs,
        "kv": c,
        "ks": cs,
        "rope": kv[:, NOPE_DIM:].contiguous(),
        "idx": indices[0].to(torch.int32).contiguous(),
    }
    # Decode now, before any H2D or NPU dispatch, so invalid scales/payloads
    # cannot masquerade as an operator accuracy failure.
    decoded_q = decode_mxfp8(q, qs).unsqueeze(0)
    decoded_kv = torch.cat((decode_mxfp4(c, cs), prepared["rope"].float()), dim=-1)
    if not bool(torch.isfinite(decoded_q.bfloat16()).all()) or not bool(torch.isfinite(decoded_kv.bfloat16()).all()):
        raise ValueError("Decoded inputs overflow the BF16 RoPE/PV contract")
    return prepared, decoded_q, decoded_kv, query, kv
