#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run the A5 QuantLightningIndexerV2 MXFP4 producer/consumer contract."""

import os
import sys
import time

import torch
import torch_npu

from vllm_ascend.attention.sfa_indexer import (
    QLI_V2_MXFP4,
    SFAIndexerMetadataBuilder,
    quantize_sfa_indexer_mxfp4,
    select_sfa_topk,
)
from vllm_ascend.device.hardware_profile import (
    HardwareCapability,
    get_current_hardware_profile,
)
from vllm_ascend.utils import enable_custom_op

TOPK = 2048


def require_a5_contract() -> None:
    profile = get_current_hardware_profile()
    if not profile.supports(HardwareCapability.FP8_ATTENTION):
        raise RuntimeError(f"This test requires Ascend A5; detected hardware profile: {profile}")

    required = {
        "_C_ascend.npu_quant_lightning_indexer_v2": hasattr(
            torch.ops._C_ascend, "npu_quant_lightning_indexer_v2"
        ),
        "_C_ascend.npu_quant_lightning_indexer_v2_metadata": hasattr(
            torch.ops._C_ascend, "npu_quant_lightning_indexer_v2_metadata"
        ),
        "torch_npu.npu_dynamic_mx_quant": hasattr(
            torch_npu, "npu_dynamic_mx_quant"
        ),
        "torch_npu.float4_e2m1fn_x2": hasattr(
            torch_npu, "float4_e2m1fn_x2"
        ),
    }
    missing = [name for name, available in required.items() if not available]
    if missing:
        raise RuntimeError("Missing A5 MXFP4 symbols: " + ", ".join(missing))


def invoke_qli_v2(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    block_table: torch.Tensor,
    metadata,
    *,
    return_value: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if metadata.quant_mode == QLI_V2_MXFP4:
        key_scale = key_scale.unflatten(-1, (2, 2))
    return torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
        query=query,
        key=key,
        weights=weights.float().contiguous(),
        query_dequant_scale=query_scale,
        key_dequant_scale=key_scale,
        topk=TOPK,
        quant_mode=metadata.quant_mode,
        cu_seqlens_q=metadata.cu_seqlens_q,
        seqused_k=metadata.seqused_k,
        block_table=block_table,
        metadata=metadata.schedule,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=1,
        return_value=return_value,
    )


def validate_indices(indices: torch.Tensor, query_tokens: int, key_tokens: int) -> None:
    expected_shape = (query_tokens, 1, TOPK)
    if indices.shape != expected_shape or indices.dtype != torch.int32:
        raise AssertionError(
            f"unexpected result: shape={tuple(indices.shape)}, dtype={indices.dtype}; "
            f"expected {expected_shape}, torch.int32"
        )

    host_indices = indices.cpu()
    for query_index in range(query_tokens):
        # mask_mode=3 uses right-aligned causal lengths for a chunked query.
        causal_length = key_tokens - query_tokens + query_index + 1
        valid = host_indices[query_index, 0]
        valid = valid[valid >= 0]
        expected_count = min(TOPK, causal_length)
        if valid.numel() != expected_count:
            raise AssertionError(
                f"query {query_index}: got {valid.numel()} valid indices; "
                f"expected {expected_count}"
            )
        if valid.unique().numel() != valid.numel():
            raise AssertionError(f"query {query_index}: duplicate sparse indices")
        if not bool((valid < causal_length).all()):
            raise AssertionError(
                f"query {query_index}: index exceeds causal length {causal_length}"
            )


def run_compute(
    label: str,
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    block_table: torch.Tensor,
    metadata,
    key_tokens: int,
) -> None:
    indices = select_sfa_topk(
        query=query,
        key=key,
        weights=weights,
        query_scale=query_scale,
        key_scale=key_scale,
        block_table=block_table,
        metadata=metadata,
    )
    torch.npu.synchronize()
    validate_indices(indices, query.shape[0], key_tokens)
    print(
        f"PASS {label}: output={tuple(indices.shape)} {indices.dtype}; "
        f"K stride0={key.stride(0)}, scale stride0={key_scale.stride(0)}"
    )


def reference_scores(
    query_fp16: torch.Tensor,
    key_fp16: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    # Keep the reference off the timed NPU path. QLI computes a per-head QK
    # product, applies ReLU, weights the heads, then reduces across heads.
    query_cpu = query_fp16.cpu().float()
    key_cpu = key_fp16[:, 0].cpu().float()
    weights_cpu = weights.cpu().float()
    correlations = torch.matmul(query_cpu, key_cpu.transpose(0, 1))
    scores = (correlations.relu() * weights_cpu.unsqueeze(-1)).sum(dim=1)
    for query_index in range(query_fp16.shape[0]):
        causal_length = key_fp16.shape[0] - query_fp16.shape[0] + query_index + 1
        scores[query_index, causal_length:] = -torch.inf
    return scores


def check_accuracy(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    block_table: torch.Tensor,
    metadata,
    query_fp16: torch.Tensor,
    key_fp16: torch.Tensor,
) -> None:
    indices, values = invoke_qli_v2(
        query,
        key,
        weights,
        query_scale,
        key_scale,
        block_table,
        metadata,
        return_value=1,
    )
    torch.npu.synchronize()
    validate_indices(indices, query.shape[0], key_fp16.shape[0])

    scores = reference_scores(query_fp16, key_fp16, weights)
    reference_indices = scores.topk(TOPK, dim=-1).indices
    actual_indices = indices[:, 0].cpu().long()
    actual_values = values[:, 0].cpu().float()
    selected_reference_values = scores.gather(1, actual_indices)

    recalls = []
    for actual, expected in zip(actual_indices, reference_indices):
        recalls.append(torch.isin(actual, expected).float().mean().item())
    recall = sum(recalls) / len(recalls)
    cosine = torch.nn.functional.cosine_similarity(
        actual_values.flatten(), selected_reference_values.flatten(), dim=0
    ).item()
    score_nmae = (
        (actual_values - selected_reference_values).abs().mean()
        / selected_reference_values.abs().mean().clamp_min(1e-12)
    ).item()

    min_recall = float(os.getenv("QLI_MIN_TOPK_RECALL", "0.90"))
    min_cosine = float(os.getenv("QLI_MIN_SCORE_COSINE", "0.98"))
    max_nmae = float(os.getenv("QLI_MAX_SCORE_NMAE", "0.10"))
    print(
        "ACCURACY MXFP4 vs FP32 reference: "
        f"top{TOPK}_recall={recall:.6f} (min {min_recall:.3f}), "
        f"score_cosine={cosine:.6f} (min {min_cosine:.3f}), "
        f"score_nmae={score_nmae:.6f} (max {max_nmae:.3f})"
    )
    if recall < min_recall or cosine < min_cosine or score_nmae > max_nmae:
        raise AssertionError("MXFP4 accuracy is outside the configured thresholds")


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def benchmark_compute(name: str, compute, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        compute()
    torch.npu.synchronize()

    latencies_ms = []
    for _ in range(iterations):
        started = time.perf_counter()
        compute()
        torch.npu.synchronize()
        latencies_ms.append((time.perf_counter() - started) * 1000)
    result = {
        "p50": percentile(latencies_ms, 0.50),
        "p90": percentile(latencies_ms, 0.90),
        "mean": sum(latencies_ms) / len(latencies_ms),
    }
    print(
        f"PERF {name} compute-only wall latency: "
        f"P50={result['p50']:.3f} ms, P90={result['p90']:.3f} ms, "
        f"mean={result['mean']:.3f} ms ({iterations} iterations)"
    )
    return result


@torch.inference_mode()
def main() -> None:
    enable_custom_op()
    require_a5_contract()

    device = torch.device("npu")
    block_size = 128
    query_tokens = int(os.getenv("QLI_QUERY_TOKENS", "1"))
    key_tokens = int(os.getenv("QLI_KEY_TOKENS", "8192"))
    if key_tokens % block_size or key_tokens < TOPK + query_tokens - 1:
        raise ValueError(
            f"QLI_KEY_TOKENS must be a multiple of {block_size} and cover top-{TOPK}"
        )
    if query_tokens < 1 or query_tokens > key_tokens:
        raise ValueError("QLI_QUERY_TOKENS must be within the key sequence")
    num_blocks = key_tokens // block_size
    heads = 64
    key_heads = 1
    logical_dim = 128

    torch.manual_seed(20260911)
    torch.npu.manual_seed_all(20260911)
    query_fp16 = torch.randn(
        query_tokens, heads, logical_dim, dtype=torch.float16, device=device
    )
    key_fp16 = torch.randn(
        key_tokens, key_heads, logical_dim, dtype=torch.float16, device=device
    )
    query, query_scale = quantize_sfa_indexer_mxfp4(query_fp16, torch_npu)
    key, key_scale = quantize_sfa_indexer_mxfp4(key_fp16, torch_npu)
    key = key.view(num_blocks, block_size, key_heads, 64)
    key_scale = key_scale.view(num_blocks, block_size, key_heads, 4)

    if query.dtype != torch.uint8 or key.dtype != torch.uint8:
        raise AssertionError("packed E2M1 payload must retain uint8 physical storage")
    if query_scale.dtype != torch.uint8 or key_scale.dtype != torch.uint8:
        raise AssertionError("E8M0 scales must retain uint8 physical storage")

    weights = torch.ones(query_tokens, heads, dtype=torch.float32, device=device)
    block_table = torch.arange(
        num_blocks, dtype=torch.int32, device=device
    ).view(1, num_blocks)
    cumulative_query_lens = torch.tensor(
        [query_tokens], dtype=torch.int32, device=device
    )
    sequence_lens = torch.tensor([key_tokens], dtype=torch.int32, device=device)
    metadata = SFAIndexerMetadataBuilder(
        num_heads=heads,
        head_dim=logical_dim,
        max_num_reqs=1,
        device=device,
        quant_mode=QLI_V2_MXFP4,
    ).build(cumulative_query_lens, sequence_lens)

    fp8_query, fp8_query_scale = torch_npu.npu_dynamic_quant(
        query_fp16.view(-1, logical_dim), dst_type=torch.float8_e4m3fn
    )
    fp8_key, fp8_key_scale = torch_npu.npu_dynamic_quant(
        key_fp16.view(-1, logical_dim), dst_type=torch.float8_e4m3fn
    )
    fp8_query = fp8_query.view(query_tokens, heads, logical_dim)
    fp8_query_scale = fp8_query_scale.float().view(query_tokens, heads)
    fp8_key = fp8_key.view(num_blocks, block_size, 1, logical_dim)
    fp8_key_scale = fp8_key_scale.float().view(num_blocks, block_size, 1)
    fp8_metadata = SFAIndexerMetadataBuilder(
        num_heads=heads,
        head_dim=logical_dim,
        max_num_reqs=1,
        device=device,
    ).build(cumulative_query_lens, sequence_lens)

    print("device:", torch.npu.get_device_name(torch.npu.current_device()))
    print("torch:", torch.__version__)
    print("torch_npu:", torch_npu.__version__)
    print(
        "contract: quant_mode=5, Q=TND uint8[E2M1x2]/E8M0 [T,N,64]/[T,N,2,2], "
        "K=PA_BBND uint8[E2M1x2]/E8M0 [B,S,1,64]/[B,S,1,4]"
    )

    run_compute(
        "dense PA cache",
        query,
        key,
        weights,
        query_scale,
        key_scale,
        block_table,
        metadata,
        key_tokens,
    )

    check_accuracy(
        query,
        key,
        weights,
        query_scale,
        key_scale,
        block_table,
        metadata,
        query_fp16,
        key_fp16,
    )

    # QLI V2 explicitly permits padding between cache blocks. Exercise that
    # descriptor path because the real allocator can group several caches in
    # one raw allocation and produce a non-contiguous axis-0 view.
    key_storage = torch.empty(
        num_blocks * 2, block_size, key_heads, 64, dtype=torch.uint8, device=device
    )
    scale_storage = torch.empty(
        num_blocks * 2, block_size, key_heads, 4, dtype=torch.uint8, device=device
    )
    key_storage[::2].copy_(key)
    scale_storage[::2].copy_(key_scale)
    run_compute(
        "axis-0-strided PA cache",
        query,
        key_storage[::2],
        weights,
        query_scale,
        scale_storage[::2],
        block_table,
        metadata,
        key_tokens,
    )

    warmup = int(os.getenv("QLI_WARMUP", "5"))
    iterations = int(os.getenv("QLI_ITERS", "20"))
    if warmup < 1 or iterations < 1:
        raise ValueError("QLI_WARMUP and QLI_ITERS must be positive")
    mxfp4_perf = benchmark_compute(
        "MXFP4 mode=5",
        lambda: select_sfa_topk(
            query,
            key,
            weights,
            query_scale,
            key_scale,
            block_table,
            metadata,
        ),
        warmup,
        iterations,
    )
    fp8_perf = benchmark_compute(
        "FP8 mode=1",
        lambda: select_sfa_topk(
            fp8_query,
            fp8_key,
            weights,
            fp8_query_scale,
            fp8_key_scale,
            block_table,
            fp8_metadata,
        ),
        warmup,
        iterations,
    )
    print(f"PERF P50 ratio FP8/MXFP4: {fp8_perf['p50'] / mxfp4_perf['p50']:.3f}x")
    max_p50_ms = os.getenv("QLI_MAX_MXFP4_P50_MS")
    if max_p50_ms is not None and mxfp4_perf["p50"] > float(max_p50_ms):
        raise AssertionError(
            f"MXFP4 P50 {mxfp4_perf['p50']:.3f} ms exceeds "
            f"QLI_MAX_MXFP4_P50_MS={max_p50_ms}"
        )
    print("QLI V2 MXFP4 A5 single-operator test passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"QLI V2 MXFP4 A5 single-operator test FAILED: {error}", file=sys.stderr)
        raise
