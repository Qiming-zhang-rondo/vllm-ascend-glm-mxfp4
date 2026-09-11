# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scatter already-quantized MXFP4 payload/scales without interpreting bytes."""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _scatter_indexer_bytes(
    key,
    scales,
    key_cache,
    scale_cache,
    slots,
    KEY_ROW_STRIDE: tl.constexpr,
    SCALE_ROW_STRIDE: tl.constexpr,
    KEY_BLOCK_STRIDE: tl.constexpr,
    SCALE_BLOCK_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(slots + token).to(tl.int64)
    # Padded graph tokens have slot=-1.  Use masks instead of a runtime
    # Python branch: Triton only permits compile-time branches in kernels.
    valid = (slot >= 0) & (slot < NUM_BLOCKS * BLOCK_SIZE)
    block = slot // BLOCK_SIZE
    row = slot % BLOCK_SIZE
    d = tl.arange(0, 64)
    payload = tl.load(key + token * KEY_ROW_STRIDE + d, mask=valid, other=0)
    tl.store(key_cache + block * KEY_BLOCK_STRIDE + row * 64 + d, payload, mask=valid)
    s = tl.arange(0, 4)
    exponents = tl.load(scales + token * SCALE_ROW_STRIDE + s, mask=valid, other=0)
    tl.store(scale_cache + block * SCALE_BLOCK_STRIDE + row * 4 + s, exponents, mask=valid)


def sfa_indexer_mxfp4_cache(
    key: torch.Tensor,
    scales: torch.Tensor,
    key_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    if key.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("MXFP4 scatter requires packed uint8 K and E8M0 byte scales.")
    if key_cache.dtype != torch.uint8 or scale_cache.dtype != torch.uint8:
        raise ValueError("MXFP4 caches must use byte storage.")
    if key.dim() != 2 or key.shape[1] != 64 or scales.shape != (key.shape[0], 4):
        raise ValueError("MXFP4 scatter expects K [tokens,64] and scales [tokens,4].")
    if key_cache.dim() != 4 or key_cache.shape[2:] != (1, 64):
        raise ValueError("MXFP4 K cache must be [blocks,block_size,1,64].")
    if scale_cache.shape != (*key_cache.shape[:2], 1, 4):
        raise ValueError("MXFP4 scale cache must be [blocks,block_size,1,4].")
    if key.stride(-1) != 1 or scales.stride(-1) != 1:
        raise ValueError("MXFP4 scatter input rows must be contiguous.")
    if key_cache.stride(1) != 64 or scale_cache.stride(1) != 4:
        raise ValueError("MXFP4 cache inner rows must be contiguous; only block padding is supported.")
    if key_cache.stride(-1) != 1 or scale_cache.stride(-1) != 1:
        raise ValueError("MXFP4 cache byte axis must be contiguous.")
    if slot_mapping.numel() != key.shape[0] or slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError("MXFP4 scatter needs one integer slot per input token.")
    if key.shape[0] == 0:
        return
    _scatter_indexer_bytes[(key.shape[0],)](
        key,
        scales,
        key_cache,
        scale_cache,
        slot_mapping.contiguous(),
        key.stride(0),
        scales.stride(0),
        key_cache.stride(0),
        scale_cache.stride(0),
        key_cache.shape[1],
        key_cache.shape[0],
    )


def _fake_sfa_indexer_mxfp4_cache(
    key: torch.Tensor,
    scales: torch.Tensor,
    key_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    return


direct_register_custom_op(
    op_name="sfa_indexer_mxfp4_cache",
    op_func=sfa_indexer_mxfp4_cache,
    fake_impl=_fake_sfa_indexer_mxfp4_cache,
    mutates_args=["key_cache", "scale_cache"],
    dispatch_key="PrivateUse1",
)
