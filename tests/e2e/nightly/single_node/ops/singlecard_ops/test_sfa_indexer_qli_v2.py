# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A5 gates for GLM SFA QLI V2 (FP8 parity and MXFP4 execution)."""

import pytest
import torch
import torch_npu

from vllm_ascend.attention.sfa_indexer import (
    QLI_V2_MXFP4,
    SFAIndexerMetadataBuilder,
    quantize_sfa_indexer_mxfp4,
    select_sfa_topk,
)
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(autouse=True)
def require_a5():
    if not get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION):
        pytest.skip("requires Ascend A5 FP8 attention")
    enable_custom_op()
    # Missing symbols on A5 are deployment failures, not successful skips.
    assert hasattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2")
    assert hasattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2_metadata")
    assert hasattr(torch_npu, "npu_quant_lightning_indexer")
    assert hasattr(torch_npu, "npu_dynamic_mx_quant")


def make_inputs(q_lengths, k_lengths, strided, ranked):
    torch.manual_seed(73)
    block_size, heads, dim = 128, 64, 128
    blocks_per_req = [(length + block_size - 1) // block_size for length in k_lengths]
    num_blocks = sum(blocks_per_req)
    block_ids = torch.randperm(num_blocks)
    table = torch.zeros((len(k_lengths), max(blocks_per_req)), dtype=torch.int32)
    scales = torch.rand(num_blocks, block_size, 1) + 0.5
    offset = 0
    for row, (length, count) in enumerate(zip(k_lengths, blocks_per_req)):
        table[row, :count] = block_ids[offset : offset + count]
        if ranked:
            # Unique scores despite FP8 payload ties: larger logical token ids
            # have larger FP32 K scales. Wrong causal alignment is observable.
            for page in range(count):
                scales[table[row, page], :, 0] = (torch.arange(block_size) + page * block_size + 1) / max(k_lengths)
        offset += count
    tokens = sum(q_lengths)
    q = torch.ones(tokens, heads, dim) if ranked else torch.randn(tokens, heads, dim)
    k = torch.ones(num_blocks, block_size, 1, dim) if ranked else torch.randn(num_blocks, block_size, 1, dim)
    q = q.to(torch.float8_e4m3fn).npu()
    k = k.to(torch.float8_e4m3fn).npu()
    scales = scales.npu()
    if strided:
        storage = torch.empty(num_blocks * 2, block_size, 1, dim, dtype=k.dtype, device="npu")
        scale_storage = torch.empty(num_blocks * 2, block_size, 1, dtype=torch.float32, device="npu")
        storage[::2].copy_(k)
        scale_storage[::2].copy_(scales)
        k, scales = storage[::2], scale_storage[::2]
    weights = torch.ones(tokens, heads, dtype=torch.bfloat16, device="npu")
    q_scale = torch.ones(tokens, heads, dtype=torch.float32, device="npu")
    q_ends = torch.tensor(q_lengths, dtype=torch.int32, device="npu").cumsum(0, dtype=torch.int32)
    k_lengths_npu = torch.tensor(k_lengths, dtype=torch.int32, device="npu")
    return q, k, weights, q_scale, scales, table.npu(), q_ends, k_lengths_npu


def legacy(inputs):
    q, k, weights, q_scale, k_scale, table, q_ends, k_lengths = inputs
    return torch_npu.npu_quant_lightning_indexer(
        query=q,
        key=k.contiguous(),
        weights=weights,
        query_dequant_scale=q_scale,
        key_dequant_scale=k_scale.contiguous(),
        actual_seq_lengths_query=q_ends,
        actual_seq_lengths_key=k_lengths,
        block_table=table,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=2048,
        sparse_mode=3,
    )


def assert_topk(actual, expected, q_lengths, k_lengths):
    actual, expected = actual.cpu(), expected.cpu()
    assert actual.shape == expected.shape == (sum(q_lengths), 1, 2048)
    assert actual.dtype == expected.dtype == torch.int32
    cursor = 0
    for q_len, k_len in zip(q_lengths, k_lengths):
        for index in range(q_len):
            causal_length = k_len - q_len + index + 1
            new = actual[cursor + index, 0]
            old = expected[cursor + index, 0]
            valid = new[new >= 0]
            assert valid.numel() == min(2048, causal_length)
            assert valid.unique().numel() == valid.numel()
            assert (valid < causal_length).all()
            # Index order is immaterial to SFA. Keep the set gate strict;
            # investigate cutoff ties or reduction changes before relaxing it.
            torch.testing.assert_close(new.sort().values, old.sort().values, rtol=0, atol=0)
        cursor += q_len


@pytest.mark.parametrize(
    ("q_lengths", "k_lengths"),
    [([3, 5], [3, 5]), ([3, 5], [4096, 6000]), ([1, 1], [128, 8192]), ([4, 4], [4097, 8193])],
    ids=["ragged-prefill", "chunked-prefill", "decode", "speculative-decode"],
)
@pytest.mark.parametrize("strided", [False, True], ids=["dense-blocks", "strided-blocks"])
@pytest.mark.parametrize("ranked", [True, False], ids=["unique-ranking", "random"])
@torch.inference_mode()
def test_sfa_fp8_legacy_v2(q_lengths, k_lengths, strided, ranked):
    inputs = make_inputs(q_lengths, k_lengths, strided, ranked)
    q, k, weights, q_scale, k_scale, table, q_ends, key_lengths = inputs
    builder = SFAIndexerMetadataBuilder(64, 128, len(q_lengths), torch.device("npu"))
    metadata = builder.build(q_ends, key_lengths)
    actual = select_sfa_topk(q, k, weights, q_scale, k_scale, table, metadata)
    assert_topk(actual, legacy(inputs), q_lengths, k_lengths)


@torch.inference_mode()
def test_sfa_v2_graph_replay_refreshes_lengths():
    inputs = make_inputs([1, 1], [4096, 8192], True, True)
    q, k, weights, q_scale, k_scale, table, q_ends, key_lengths = inputs
    builder = SFAIndexerMetadataBuilder(64, 128, 2, torch.device("npu"))
    metadata = builder.build(q_ends, key_lengths)
    select_sfa_topk(q, k, weights, q_scale, k_scale, table, metadata)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        output = select_sfa_topk(q, k, weights, q_scale, k_scale, table, metadata)
    for lengths in ([3000, 5000], [3500, 7000]):
        key_lengths.copy_(torch.tensor(lengths, dtype=torch.int32, device="npu"))
        refreshed = builder.build(q_ends, key_lengths)
        assert refreshed.schedule.data_ptr() == metadata.schedule.data_ptr()
        graph.replay()
        torch.npu.synchronize()
        assert_topk(output, legacy(inputs), [1, 1], lengths)


@torch.inference_mode()
def test_sfa_mxfp4_mode5_executes_with_packed_cache_contract():
    block_size, num_blocks, tokens = 128, 2, 2
    q_float = torch.randn(tokens, 1, 128, device="npu", dtype=torch.float16)
    k_float = torch.randn(num_blocks * block_size, 1, 128, device="npu", dtype=torch.float16)
    q, q_scale = quantize_sfa_indexer_mxfp4(q_float, torch_npu)
    k, k_scale = quantize_sfa_indexer_mxfp4(k_float, torch_npu)
    k = k.view(num_blocks, block_size, 1, 64)
    k_scale = k_scale.view(num_blocks, block_size, 1, 4)
    weights = torch.ones(tokens, 1, dtype=torch.float32, device="npu")
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="npu").view(1, -1)
    q_ends = torch.tensor([tokens], dtype=torch.int32, device="npu")
    k_lengths = torch.tensor([block_size], dtype=torch.int32, device="npu")

    builder = SFAIndexerMetadataBuilder(1, 128, 1, torch.device("npu"), quant_mode=QLI_V2_MXFP4)
    metadata = builder.build(q_ends, k_lengths)
    indices = select_sfa_topk(q, k, weights, q_scale, k_scale, block_table, metadata)
    assert indices.shape == (tokens, 1, 2048)
    assert indices.dtype == torch.int32
    assert (indices[:, 0, 0] >= 0).all()
