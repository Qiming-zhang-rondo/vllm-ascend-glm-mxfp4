# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""QLI V2 contract for the uncompressed GLM SFA indexer on A5."""

from dataclasses import dataclass

import torch

SFA_INDEXER_TOPK = 2048
QLI_V2_FP8_PER_TOKEN = 1
QLI_V2_MXFP4 = 5
MXFP4_HEAD_DIM = 128
MXFP4_PACKED_DIM = 64
MXFP4_SCALE_DIM = 4
QLI_V2_METADATA_SIZE = 1024


@dataclass
class SFAIndexerMetadata:
    cu_seqlens_q: torch.Tensor
    seqused_k: torch.Tensor
    schedule: torch.Tensor
    quant_mode: int = QLI_V2_FP8_PER_TOKEN


class SFAIndexerMetadataBuilder:
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        max_num_reqs: int,
        device: torch.device,
        quant_mode: int = QLI_V2_FP8_PER_TOKEN,
    ):
        if quant_mode not in (QLI_V2_FP8_PER_TOKEN, QLI_V2_MXFP4):
            raise ValueError("SFA indexer supports QLI V2 FP8 (1) or MXFP4 (5).")
        self.quant_mode = quant_mode
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_num_reqs = max_num_reqs
        self.device = device
        # Graph replay retains tensor addresses. Draft steps must not overwrite
        # the main model's metadata or another draft step's metadata.
        self._buffers: dict[int | None, SFAIndexerMetadata] = {}

    def build(
        self,
        cumulative_query_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        draft_index: int | None = None,
    ) -> SFAIndexerMetadata:
        num_reqs = cumulative_query_lens.numel()
        if num_reqs != seq_lens.numel() or num_reqs > self.max_num_reqs:
            raise ValueError("SFA QLI V2 requires matching Q/K request counts within the metadata buffer capacity.")
        if draft_index not in self._buffers:
            self._buffers[draft_index] = SFAIndexerMetadata(
                cu_seqlens_q=torch.zeros(self.max_num_reqs + 1, dtype=torch.int32, device=self.device),
                seqused_k=torch.empty(self.max_num_reqs, dtype=torch.int32, device=self.device),
                schedule=torch.empty(QLI_V2_METADATA_SIZE, dtype=torch.int32, device=self.device),
            )
        buffers = self._buffers[draft_index]
        cu_seqlens_q = buffers.cu_seqlens_q[: num_reqs + 1]
        seqused_k = buffers.seqused_k[:num_reqs]
        # Legacy SFA stores cumulative ends without the leading zero. V2
        # requires B+1 entries. PA key lengths remain per-request lengths.
        cu_seqlens_q[1:].copy_(cumulative_query_lens)
        seqused_k.copy_(seq_lens)
        schedule = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
            num_heads_q=self.num_heads,
            num_heads_k=1,
            head_dim=self.head_dim,
            topk=SFA_INDEXER_TOPK,
            quant_mode=self.quant_mode,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            batch_size=num_reqs,
            max_seqlen_q=-1,
            max_seqlen_k=-1,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=1,
            device=str(self.device),
        )
        buffers.schedule.copy_(schedule)
        return SFAIndexerMetadata(cu_seqlens_q, seqused_k, buffers.schedule, self.quant_mode)


def select_sfa_topk(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    block_table: torch.Tensor,
    metadata: SFAIndexerMetadata,
) -> torch.Tensor:
    if metadata.quant_mode == QLI_V2_MXFP4:
        if query.shape[-1] != MXFP4_PACKED_DIM or key.shape[-1] != MXFP4_PACKED_DIM:
            raise ValueError("MXFP4 Q/K must store logical D128 in 64 packed bytes.")
        if key_scale.shape[-1] != MXFP4_SCALE_DIM or key_scale.dim() != 4:
            raise ValueError("MXFP4 K scale cache must have shape [blocks, block_size, 1, 4].")
        # Storage stays uint8 throughout gather/scatter. The C++ mode-5
        # wrapper supplies ACL E8M0 / FP4 descriptors without numerical casts.
        key_scale = key_scale.unflatten(-1, (2, 2))
    # V2 accepts axis-0 cache strides. Do not copy/repack the full cache.
    indices, _ = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
        query=query,
        key=key,
        weights=weights.float().contiguous(),
        query_dequant_scale=query_scale,
        key_dequant_scale=key_scale,
        topk=SFA_INDEXER_TOPK,
        quant_mode=metadata.quant_mode,
        cu_seqlens_q=metadata.cu_seqlens_q,
        seqused_k=metadata.seqused_k,
        block_table=block_table,
        metadata=metadata.schedule,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=1,
        return_value=0,
    )
    return indices


def quantize_sfa_indexer_mxfp4(x: torch.Tensor, npu_ops) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-32 E2M1/E8M0 quantization; keep physical storage as uint8.

    Q and K use exactly the same quantizer and default rounding/scale algorithm.
    This function runs after the existing rotary/Hadamard transform.
    """
    if x.shape[-1] != MXFP4_HEAD_DIM:
        raise ValueError("MXFP4 SFA indexer requires logical head dimension 128.")
    packed, scale = npu_ops.npu_dynamic_mx_quant(
        x.contiguous(), dst_type=npu_ops.float4_e2m1fn_x2, round_mode="round"
    )
    packed = packed.view(torch.uint8)
    scale = scale.view(torch.uint8)
    if packed.shape != (*x.shape[:-1], MXFP4_PACKED_DIM):
        raise ValueError("npu_dynamic_mx_quant must return packed FP4 with half the logical head width.")
    if scale.shape != (*x.shape[:-1], 2, 2):
        raise ValueError("npu_dynamic_mx_quant must return per-32 E8M0 scale pairs [..., 2, 2].")
    return packed, scale
