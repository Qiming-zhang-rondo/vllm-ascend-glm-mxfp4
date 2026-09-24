# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU carriers for the experimental official-pipeline Q8/C4/O8 adaptation.

PA416 and query608 are this experiment's contracts, not an installed CANN
QSFA API. Cache rows contain C4 NoPE, BF16 RoPE, D32 E8M0 scales, then pad;
query rows contain 576 E4M3 bytes, 18 E8M0 bytes, then pad. Padding is zero.
"""

import torch

from benchmarks.qsfa_q8c4_o8.packing import (
    NOPE_DIM,
    QUERY_DIM,
    decode_mxfp4,
    decode_mxfp8,
)
from benchmarks.qsfa_q8c4_o8.packing import (
    prepare_inputs as prepare_separate_inputs,
)

BLOCK_SIZE = 256
QUERY_ROW_BYTES = 608
QUERY_SCALE_OFFSET = QUERY_DIM
QUERY_SCALE_COUNT = QUERY_DIM // 32
CACHE_ROW_BYTES = 416
CACHE_ROPE_OFFSET = NOPE_DIM // 2
CACHE_SCALE_OFFSET = CACHE_ROPE_OFFSET + 64 * 2
CACHE_SCALE_COUNT = NOPE_DIM // 32
ARGUMENT_ORDER = ("q", "cache", "idx", "table", "cuq", "kvlen")
OFFICIAL_SOURCE_REF = "55498d91634277d4eec912499c027818a8c167fb"
SUPPORTED_HEADS = (8, 16, 32)


def validate_head_count(query):
    """The initial fused adaptation reserves UB only for H <= 32."""
    if not isinstance(query, torch.Tensor) or query.ndim != 3 or query.shape[1] not in SUPPORTED_HEADS:
        raise ValueError("The official-pipeline adaptation requires H in {8,16,32}; H64 exceeds its UB budget")


def _block_table(table, pages):
    if table is None:
        return torch.arange(pages, dtype=torch.int32).reshape(1, pages)
    if (
        not isinstance(table, torch.Tensor)
        or table.device.type != "cpu"
        or table.dtype not in (torch.int32, torch.int64)
        or tuple(table.shape) != (1, pages)
        or not torch.equal(table[0].long().sort().values, torch.arange(pages))
    ):
        raise ValueError("block_table must be a CPU integer [1,pages] permutation of physical page IDs")
    return table.to(torch.int32).contiguous()


def decode_inputs(prepared):
    """Decode actual physical carriers back to logical Q/K for the golden."""
    q, cache, table, kvlen = (prepared[name] for name in ("q", "cache", "table", "kvlen"))
    if (
        q.device.type != "cpu"
        or q.dtype != torch.uint8
        or q.ndim != 2
        or q.shape[1] != QUERY_ROW_BYTES
        or not q.is_contiguous()
        or cache.device.type != "cpu"
        or cache.dtype != torch.uint8
        or cache.ndim != 4
        or tuple(cache.shape[1:]) != (BLOCK_SIZE, 1, CACHE_ROW_BYTES)
        or not cache.is_contiguous()
    ):
        raise ValueError("Expected CPU contiguous query uint8[H,608] and cache uint8[pages,256,1,416]")
    table = _block_table(table, cache.shape[0])
    if kvlen.device.type != "cpu" or kvlen.dtype != torch.int32 or tuple(kvlen.shape) != (1,):
        raise ValueError("kvlen must be CPU int32[1]")
    key_tokens = int(kvlen[0])
    if not 1 <= key_tokens <= cache.shape[0] * BLOCK_SIZE:
        raise ValueError("kvlen is outside the physical cache capacity")
    logical = cache[table[0].long()].reshape(-1, CACHE_ROW_BYTES)[:key_tokens]
    decoded_q = decode_mxfp8(q[:, :QUERY_DIM], q[:, QUERY_SCALE_OFFSET : QUERY_SCALE_OFFSET + QUERY_SCALE_COUNT])
    decoded_nope = decode_mxfp4(
        logical[:, :CACHE_ROPE_OFFSET],
        logical[:, CACHE_SCALE_OFFSET : CACHE_SCALE_OFFSET + CACHE_SCALE_COUNT],
    )
    rope = logical[:, CACHE_ROPE_OFFSET:CACHE_SCALE_OFFSET].contiguous().view(torch.bfloat16).float()
    decoded_kv = torch.cat((decoded_nope, rope), dim=-1)
    if not bool(torch.isfinite(decoded_kv.bfloat16()).all()):
        raise ValueError("Physical C4 cache decodes to nonfinite BF16 KV")
    return decoded_q.unsqueeze(0), decoded_kv


def prepare_inputs(query, kv, indices, scale, *, block_table=None):
    """Return prepared tensors, decoded Q/K, BF16 originals, and JSON contract.

    The six tensors follow ARGUMENT_ORDER; scale is the seventh op argument.
    Optional page permutations test that logical sparse IDs use block_table.
    """
    validate_head_count(query)
    if isinstance(kv, torch.Tensor) and kv.ndim == 2 and isinstance(indices, torch.Tensor) and indices.ndim == 2:
        pages = (kv.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE
        address_ub = ((pages * 4 + 511) // 512) * 512 + indices.shape[1] * 12
        if address_ub + 8192 > 216 * 1024:
            raise ValueError("PA address preparation exceeds the fixed UB budget")
    split, _, _, original_q, original_kv = prepare_separate_inputs(query, kv, indices, scale)
    heads, key_tokens = original_q.shape[1], original_kv.shape[0]
    pages = (key_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    table = _block_table(block_table, pages)
    query_raw = torch.zeros((heads, QUERY_ROW_BYTES), dtype=torch.uint8)
    query_raw[:, :QUERY_DIM] = split["q"]
    query_raw[:, QUERY_SCALE_OFFSET : QUERY_SCALE_OFFSET + QUERY_SCALE_COUNT] = split["qs"]
    logical = torch.zeros((pages * BLOCK_SIZE, CACHE_ROW_BYTES), dtype=torch.uint8)
    logical[:key_tokens, :CACHE_ROPE_OFFSET] = split["kv"]
    logical[:key_tokens, CACHE_ROPE_OFFSET:CACHE_SCALE_OFFSET] = split["rope"].view(torch.uint8)
    logical[:key_tokens, CACHE_SCALE_OFFSET : CACHE_SCALE_OFFSET + CACHE_SCALE_COUNT] = split["ks"]
    cache = torch.zeros((pages, BLOCK_SIZE, 1, CACHE_ROW_BYTES), dtype=torch.uint8)
    cache[table[0].long()] = logical.reshape(pages, BLOCK_SIZE, 1, CACHE_ROW_BYTES)
    prepared = {
        "q": query_raw,
        "cache": cache,
        "idx": split["idx"].reshape(1, 1, -1),
        "table": table,
        "cuq": torch.tensor([1], dtype=torch.int32),
        "kvlen": torch.tensor([key_tokens], dtype=torch.int32),
    }
    decoded_q, decoded_kv = decode_inputs(prepared)
    contract = {
        "api": "torch.ops.qsfa_q8c4_o8.forward_official",
        "source_repo": "https://gitcode.com/cann/ops-transformer",
        "source_ref": OFFICIAL_SOURCE_REF,
        "experimental_carrier_not_installed_cann_contract": True,
        "query_row_bytes": QUERY_ROW_BYTES,
        "query_offsets_bytes": {"payload": 0, "scale": QUERY_SCALE_OFFSET, "padding": 594},
        "cache_row_bytes": CACHE_ROW_BYTES,
        "cache_offsets_bytes": {"nope": 0, "rope": CACHE_ROPE_OFFSET, "scale": CACHE_SCALE_OFFSET, "padding": 400},
        "cache_shape": list(cache.shape),
        "block_size": BLOCK_SIZE,
        "logical_sparse_indices": True,
        "actual_key_tokens": key_tokens,
        "padded_key_tokens": pages * BLOCK_SIZE,
        "output_row_stride_bytes": 544,
        "output_offsets_bytes": {"payload": 0, "scale": 512},
        "specialization": {"aic_cores": 1, "flash_decode": False, "s2_tile": 64},
        "installed_native_tiling_equivalence_claimed": False,
    }
    return prepared, decoded_q, decoded_kv, original_q, original_kv, contract
