# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.attention.sfa_indexer import (
    QLI_V2_MXFP4,
    SFAIndexerMetadataBuilder,
    quantize_sfa_indexer_mxfp4,
    select_sfa_topk,
)


@pytest.fixture
def metadata_calls(monkeypatch):
    calls = []

    def schedule(**kwargs):
        calls.append(kwargs)
        return torch.full((1024,), len(calls), dtype=torch.int32)

    monkeypatch.setattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2_metadata", schedule, raising=False)
    return calls


@pytest.mark.parametrize(
    ("q_ends", "k_lengths"),
    [([3, 8], [3, 5]), ([1, 2], [4096, 8192]), ([4, 7], [3000, 6000]), ([0, 3, 3], [0, 301, 0])],
)
def test_uncompressed_request_lengths(metadata_calls, q_ends, k_lengths):
    builder = SFAIndexerMetadataBuilder(64, 128, 16, torch.device("cpu"))
    metadata = builder.build(torch.tensor(q_ends), torch.tensor(k_lengths))
    assert metadata.cu_seqlens_q.tolist() == [0, *q_ends]
    assert metadata.seqused_k.tolist() == k_lengths
    assert metadata.cu_seqlens_q.dtype == metadata.seqused_k.dtype == torch.int32
    call = metadata_calls[-1]
    assert call["quant_mode"] == 1
    assert call["cmp_ratio"] == 1
    assert call["layout_k"] == "PA_BBND"
    assert "cmp_residual_k" not in call
    assert "cu_seqlens_k" not in call


def test_graph_addresses_stay_stable_while_lengths_change(metadata_calls):
    builder = SFAIndexerMetadataBuilder(64, 128, 16, torch.device("cpu"))
    first = builder.build(torch.tensor([1, 2]), torch.tensor([127, 511]))
    pointers = (first.cu_seqlens_q.data_ptr(), first.seqused_k.data_ptr(), first.schedule.data_ptr())
    second = builder.build(torch.tensor([4, 8]), torch.tensor([131, 515]))
    assert pointers == (second.cu_seqlens_q.data_ptr(), second.seqused_k.data_ptr(), second.schedule.data_ptr())
    assert first.cu_seqlens_q.tolist() == [0, 4, 8]
    assert first.seqused_k.tolist() == [131, 515]
    assert first.schedule.tolist() == [2] * 1024
    # Changing request count must not shift the underlying graph buffers.
    smaller = builder.build(torch.tensor([1]), torch.tensor([1]))
    assert smaller.cu_seqlens_q.data_ptr() == pointers[0]
    assert smaller.seqused_k.tolist() == [1]


def test_draft_metadata_does_not_overwrite_main_or_other_steps(metadata_calls):
    builder = SFAIndexerMetadataBuilder(64, 128, 16, torch.device("cpu"))
    main = builder.build(torch.tensor([1, 2]), torch.tensor([127, 255]))
    draft = builder.build(torch.tensor([1, 2]), torch.tensor([128, 256]), draft_index=1)
    later = builder.build(torch.tensor([1, 2]), torch.tensor([129, 257]), draft_index=2)
    assert len({m.schedule.data_ptr() for m in (main, draft, later)}) == 3
    assert main.seqused_k.tolist() == [127, 255]
    builder.build(torch.tensor([1, 2]), torch.tensor([130, 258]), draft_index=1)
    assert draft.seqused_k.tolist() == [130, 258]
    assert later.seqused_k.tolist() == [129, 257]


@pytest.mark.parametrize(("q", "k"), [([1, 2], [1]), ([1, 2, 3], [1, 2, 3])])
def test_reject_incompatible_request_counts(metadata_calls, q, k):
    builder = SFAIndexerMetadataBuilder(64, 128, 2, torch.device("cpu"))
    with pytest.raises(ValueError, match="request counts"):
        builder.build(torch.tensor(q), torch.tensor(k))
    assert not metadata_calls


def test_fp8_consumer_preserves_cache_views_and_quantization(metadata_calls, monkeypatch):
    metadata = SFAIndexerMetadataBuilder(64, 128, 2, torch.device("cpu")).build(
        torch.tensor([1, 2]), torch.tensor([128, 256])
    )
    q = torch.randn(2, 64, 128).to(torch.float8_e4m3fn)
    key = torch.randn(8, 128, 1, 128).to(torch.float8_e4m3fn)[::2]
    key_scale = torch.rand(8, 128, 1)[::2]
    q_scale = torch.rand(2, 64)
    # SFA weights come from a slice of the fused WK/weights projection.
    weights = torch.randn(2, 128 + 64, dtype=torch.bfloat16)[:, 128:]
    block_table = torch.tensor([[3, 0], [1, 2]], dtype=torch.int32)
    output = torch.empty(2, 1, 2048, dtype=torch.int32)
    before = key.view(torch.uint8).clone()

    def compute(**kwargs):
        assert kwargs["query"] is q
        assert kwargs["key"] is key
        assert kwargs["query_dequant_scale"] is q_scale
        assert kwargs["key_dequant_scale"] is key_scale
        assert kwargs["key"].stride(0) == 2 * 128 * 128
        assert kwargs["key_dequant_scale"].stride(0) == 2 * 128
        assert kwargs["weights"].dtype == torch.float32
        assert kwargs["weights"].is_contiguous()
        torch.testing.assert_close(kwargs["weights"], weights.float())
        assert kwargs["metadata"] is metadata.schedule
        assert kwargs["block_table"] is block_table
        assert kwargs["quant_mode"] == kwargs["cmp_ratio"] == 1
        assert kwargs["topk"] == 2048
        assert kwargs["mask_mode"] == 3
        assert kwargs["layout_q"] == "TND"
        assert kwargs["layout_k"] == "PA_BBND"
        assert "cmp_residual_k" not in kwargs
        return output, torch.empty(0)

    monkeypatch.setattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2", compute, raising=False)
    assert select_sfa_topk(q, key, weights, q_scale, key_scale, block_table, metadata) is output
    torch.testing.assert_close(key.view(torch.uint8), before)


def test_mxfp4_quantizer_keeps_packed_bytes_and_pair_scales():
    class FakeNpu:
        float4_e2m1fn_x2 = object()

        @staticmethod
        def npu_dynamic_mx_quant(x, *, dst_type, round_mode):
            assert dst_type is FakeNpu.float4_e2m1fn_x2
            assert round_mode == "round"
            return torch.zeros((*x.shape[:-1], 64), dtype=torch.uint8), torch.zeros(
                (*x.shape[:-1], 2, 2), dtype=torch.uint8
            )

    packed, scales = quantize_sfa_indexer_mxfp4(torch.randn(3, 1, 128), FakeNpu)
    assert packed.shape == (3, 1, 64)
    assert scales.shape == (3, 1, 2, 2)
    assert packed.dtype == scales.dtype == torch.uint8


def test_mxfp4_consumer_uses_mode5_and_cache_contract(metadata_calls, monkeypatch):
    builder = SFAIndexerMetadataBuilder(1, 128, 2, torch.device("cpu"), quant_mode=QLI_V2_MXFP4)
    metadata = builder.build(torch.tensor([2, 5]), torch.tensor([128, 256]))
    assert metadata.quant_mode == QLI_V2_MXFP4
    assert metadata_calls[-1]["quant_mode"] == QLI_V2_MXFP4

    query = torch.zeros(2, 1, 64, dtype=torch.uint8)
    key = torch.zeros(4, 128, 1, 64, dtype=torch.uint8)[::2]
    query_scale = torch.zeros(2, 1, 2, 2, dtype=torch.uint8)
    key_scale = torch.zeros(4, 128, 1, 4, dtype=torch.uint8)[::2]
    weights = torch.ones(2, 1, dtype=torch.float32)
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    output = torch.empty(2, 1, 2048, dtype=torch.int32)

    def compute(**kwargs):
        assert kwargs["query"] is query
        assert kwargs["key"] is key
        assert kwargs["query_dequant_scale"] is query_scale
        assert kwargs["key_dequant_scale"].shape == (2, 128, 1, 2, 2)
        assert kwargs["key_dequant_scale"].stride(0) == key_scale.stride(0)
        assert kwargs["weights"].dtype == torch.float32
        assert kwargs["quant_mode"] == QLI_V2_MXFP4
        assert kwargs["layout_q"] == "TND" and kwargs["layout_k"] == "PA_BBND"
        return output, torch.empty(0)

    monkeypatch.setattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2", compute, raising=False)
    assert select_sfa_topk(query, key, weights, query_scale, key_scale, block_table, metadata) is output


def test_mxfp4_cache_scatter_contract_rejects_wrong_physical_width():
    # Keep this test independent of Triton; the same shape is enforced by the
    # registered op before launching the A5 kernel.
    pytest.importorskip("vllm.triton_utils")
    from vllm_ascend.ops.triton.sfa_indexer_cache import sfa_indexer_mxfp4_cache

    with pytest.raises(ValueError, match="packed uint8 K"):
        sfa_indexer_mxfp4_cache(
            torch.zeros(1, 128, dtype=torch.uint8),
            torch.zeros(1, 4, dtype=torch.uint8),
            torch.zeros(1, 16, 1, 64, dtype=torch.uint8),
            torch.zeros(1, 16, 1, 4, dtype=torch.uint8),
            torch.zeros(1, dtype=torch.int32),
        )
