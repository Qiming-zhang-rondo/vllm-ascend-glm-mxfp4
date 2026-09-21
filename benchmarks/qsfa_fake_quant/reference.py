# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logical MLA sparse-attention inputs and quantization-error experiments.

Only the experimental P-quantization branch changes the official CPU formula.
All calculations here are reference calculations, never NPU kernel timings.
"""

import math

import torch

from .quantization import mxfp8_roundtrip
from .vendor.official_qsfa_golden import _t_increattention_bnsd, gatherKV, softmax

NOPE_DIM = 512
ROPE_DIM = 64
QUERY_DIM = NOPE_DIM + ROPE_DIM


def validate_inputs(query, kv, indices, scale):
    if query.ndim != 3 or query.shape[-1] != QUERY_DIM:
        raise ValueError("query must have logical shape [Q, heads, 576]")
    if kv.ndim != 2 or kv.shape[-1] != QUERY_DIM:
        raise ValueError("kv must be decoded logical [K, 576], not a packed/paged cache")
    if indices.ndim != 2 or indices.shape[0] != query.shape[0] or indices.shape[1] < 1:
        raise ValueError("indices must be [Q, selected_tokens], with optional trailing -1 padding")
    if query.shape[0] < 1 or query.shape[1] < 1 or kv.shape[0] < query.shape[0]:
        raise ValueError("Require heads > 0 and K >= Q > 0")
    for name, tensor in (("query", query), ("kv", kv)):
        if tensor.device.type != "cpu" or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be a finite floating-point CPU tensor")
    if indices.device.type != "cpu" or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must be a CPU int32/int64 tensor")
    if ((indices < -1) | (indices >= kv.shape[0])).any():
        raise ValueError("Sparse index outside [-1, K)")
    if ((indices == -1).cumsum(-1).bool() & (indices != -1)).any():
        raise ValueError("Invalid -1 indices must be trailing padding, as required by the official reference")
    for row in indices:
        valid = row[row >= 0]
        if valid.unique().numel() != valid.numel():
            raise ValueError("Duplicate sparse indices would change attention weights")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale_value must be finite and positive")


def synthetic_inputs(query_tokens, key_tokens, heads, selected_tokens, seed):
    if not 1 <= query_tokens <= key_tokens or not 1 <= selected_tokens <= key_tokens or heads < 1:
        raise ValueError("Require K >= Q > 0, K >= selected_tokens > 0, and heads > 0")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    query = torch.randn(query_tokens, heads, QUERY_DIM, generator=generator).bfloat16()
    kv = torch.randn(key_tokens, QUERY_DIM, generator=generator).bfloat16()
    indices = torch.full((query_tokens, selected_tokens), -1, dtype=torch.int64)
    for row in range(query_tokens):
        # Right-aligned causal attention: Q is the final Q tokens of this K prefix.
        visible = key_tokens - query_tokens + row + 1
        count = min(visible, selected_tokens)
        indices[row, :count] = torch.randperm(visible, generator=generator)[:count]
    return query, kv, indices, QUERY_DIM**-0.5


def official_parameters(query, kv, indices, scale):
    """Adapt decoded MLA tensors to the pinned official CPU reference.

    A unit dequant scale lets the upstream code consume already-decoded values.
    Its BF16 K/P casts intentionally remain part of the reference contract.
    """
    q, heads, _ = query.shape
    k = kv.shape[0]
    keys = kv.float().reshape(1, 1, k, QUERY_DIM)
    return {
        "b": 1,
        "numHeads": heads,
        "numKeyValueHeads": 1,
        "actualSeqLengths_q": [q],
        "actualSeqLengths_kv": [k],
        "scaleValue": scale,
        "sparse_blocksize": 1,
        "sparse_blockcount": indices.shape[-1],
        "sparse_indices_bnsd_tensor": indices.reshape(1, 1, q, -1),
        "q_bnsd_shape": [1, heads, q, QUERY_DIM],
        "v_bnsd_shape": [1, 1, k, NOPE_DIM],
        "sparse_mode": 3,
        "q_bnsd_tensor": query.float().transpose(0, 1).unsqueeze(0),
        "k_bnsd_tensor": keys.clone(),
        "v_bnsd_tensor": keys[..., :NOPE_DIM].clone(),
        "k_dequant_scale_bnsd_tensor": torch.ones(1, 1, k, NOPE_DIM // 128),
        "v_dequant_scale_bnsd_tensor": torch.ones(1, 1, k, NOPE_DIM // 128),
        "q_dtype": "bfloat16",
    }


def attention_reference(query, kv, indices, scale, *, quantize_probability=False):
    """Return [Q, heads, 512] FP32 accumulated values, one Q row at a time."""
    params = official_parameters(query, kv, indices, scale)
    if not quantize_probability:
        return _t_increattention_bnsd(params)[0].transpose(0, 1).contiguous()

    # Experimental branch: same official gather/softmax, inserting MXFP8 after
    # normalization instead of the baseline BF16 P cast. No probability
    # renormalization after quantization (that would hide its numerical error).
    keys = params["k_bnsd_tensor"].clone()
    keys[..., :NOPE_DIM] = keys[..., :NOPE_DIM].bfloat16().float()
    values = keys[..., :NOPE_DIM]
    output = torch.zeros(query.shape[0], query.shape[1], NOPE_DIM)
    for row in range(query.shape[0]):
        empty, selected_k, selected_v = gatherKV(
            keys, values, indices[row], 1, indices.shape[-1], 0, 0, row, kv.shape[0], query.shape[0], 3
        )
        if empty:
            continue
        scores = torch.matmul(query[row].float(), selected_k.T) * scale
        probabilities = mxfp8_roundtrip(softmax(scores))
        output[row] = torch.matmul(probabilities, selected_v)
    return output


def error_metrics(actual, expected):
    actual, expected = actual.double(), expected.double()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("Nonfinite attention output")
    delta = actual - expected
    norm_actual, norm_expected = actual.norm(), expected.norm()
    if norm_expected == 0:
        if norm_actual != 0:
            raise ValueError("Nonzero output against an all-zero reference")
        cosine, relative_rmse = 1.0, 0.0
    else:
        cosine = float((actual * expected).sum() / (norm_actual * norm_expected)) if norm_actual else 0.0
        relative_rmse = float(delta.norm() / norm_expected)
    # Per-query/per-head metrics expose damage hidden by a global average.
    row_norm = expected.norm(dim=-1)
    row_error = delta.norm(dim=-1)
    zero_rows = row_norm == 0
    if (zero_rows & (row_error != 0)).any():
        raise ValueError("Nonzero output against a zero reference row")
    row_relative = torch.where(zero_rows, 0.0, row_error / row_norm.clamp_min(1e-300))
    return {
        "cosine": min(1.0, max(-1.0, cosine)),
        "relative_rmse": relative_rmse,
        "rmse": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
        "p99_row_relative_rmse": float(torch.quantile(row_relative.flatten(), 0.99)),
        "max_row_relative_rmse": float(row_relative.max()),
    }
