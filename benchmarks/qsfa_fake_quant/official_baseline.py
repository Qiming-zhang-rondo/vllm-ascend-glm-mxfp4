# SPDX-License-Identifier: Apache-2.0
"""Small native INT8 QSFA correctness gate using the official CANN golden.

This baseline validates the installed public torch_npu calling path. It does not
execute a C4 kernel, inject quantization into a fused kernel, or measure latency.
Only the container's existing torch and torch_npu are used.
"""

import copy
import math

import torch

from .vendor.official_qsfa_golden import _t_increattention_bnsd

BF16_ATOL = 6e-2
BF16_RTOL = 7.8125e-3
MAX_NORMALIZED_RMSE = 0.02
MIN_COSINE = 0.999
TOLERANCE_SOURCE = (
    "Qiming-zhang-rondo/vllm-ascend-glm-mxfp4@93720e60b:"
    "tests/e2e/nightly/single_node/ops/singlecard_ops/test_kv_quant_sparse_flash_attention.py:"
    "BF16_ATOL/BF16_RTOL; existing VA operator tolerances, not a quantization-loss gate"
)
GOLDEN_SOURCE = (
    "cann/ops-transformer@55498d91634277d4eec912499c027818a8c167fb:"
    "attention/kv_quant_sparse_flash_attention/tests/pytest/"
    "kv_quant_sparse_flash_attention_golden.py:_t_increattention_bnsd"
)


def build_case(seed=20260921):
    """Build one CPU-only official-layout INT8 PA case, ready for the native gate.

    The returned ``native_inputs`` contains the exact public API keyword inputs.
    ``fa_param`` is the official golden's BNSD dictionary. ``q_nope``, ``q_rope``,
    ``k_nope`` (decoded), and ``k_rope`` are logical BF16 BSND tensors, with K=V
    for the 512-dimensional MLA latent. These are separate from the byte cache.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch, query_tokens, key_tokens = 1, 1, 8192
    heads, kv_heads, dim, rope_dim = 8, 1, 512, 64
    block_size, tile_size, sparse_count = 256, 128, 2048
    blocks = key_tokens // block_size

    def uniform(shape):
        return torch.empty(shape, dtype=torch.float32).uniform_(-1, 1, generator=generator).to(torch.bfloat16)

    q_nope = uniform((batch, query_tokens, heads, dim))
    q_rope = uniform((batch, query_tokens, heads, rope_dim))
    k_rope = uniform((batch, key_tokens, kv_heads, rope_dim))
    payload = torch.randint(-100, 101, (blocks, block_size, kv_heads, dim), generator=generator, dtype=torch.int8)
    dequant_scale = torch.full((blocks, block_size, kv_heads, dim // tile_size), 0.01, dtype=torch.float32)
    packed_key = torch.cat(
        (
            payload,
            k_rope.reshape(blocks, block_size, kv_heads, rope_dim).view(torch.int8),
            dequant_scale.view(torch.int8),
        ),
        dim=-1,
    )
    query = torch.cat((q_nope, q_rope), dim=-1)
    sparse_indices = torch.randperm(key_tokens, generator=generator, dtype=torch.int32)[:sparse_count].reshape(
        batch, query_tokens, kv_heads, sparse_count
    )
    scale_value = 1 / math.sqrt(dim + rope_dim)
    raw_key = payload.reshape(batch, key_tokens, kv_heads, dim)
    logical_scale = dequant_scale.reshape(batch, key_tokens, kv_heads, dim // tile_size)
    k_nope = (raw_key.float() * logical_scale.repeat_interleave(tile_size, dim=-1)).to(torch.bfloat16)
    k_for_golden = torch.cat((raw_key.float(), k_rope.float()), dim=-1).permute(0, 2, 1, 3).contiguous()
    scales_bnsd = logical_scale.permute(0, 2, 1, 3).contiguous()
    fa_param = {
        "b": batch,
        "numHeads": heads,
        "numKeyValueHeads": kv_heads,
        "actualSeqLengths_q": [query_tokens],
        "actualSeqLengths_kv": [key_tokens],
        "scaleValue": scale_value,
        "sparse_blocksize": 1,
        "sparse_blockcount": sparse_count,
        "sparse_indices_bnsd_tensor": sparse_indices.permute(0, 2, 1, 3).contiguous(),
        "q_bnsd_shape": [batch, heads, query_tokens, dim + rope_dim],
        "v_bnsd_shape": [batch, kv_heads, key_tokens, dim],
        "sparse_mode": 3,
        "q_bnsd_tensor": query.permute(0, 2, 1, 3).contiguous(),
        "k_bnsd_tensor": k_for_golden,
        "k_dequant_scale_bnsd_tensor": scales_bnsd,
        "v_bnsd_tensor": raw_key.permute(0, 2, 1, 3).contiguous(),
        "v_dequant_scale_bnsd_tensor": scales_bnsd,
        "q_dtype": "bfloat16",
    }
    native_inputs = {
        "query": query,
        "key": packed_key,
        "value": payload.clone(),
        "sparse_indices": sparse_indices,
        "scale_value": scale_value,
        "sparse_block_size": 1,
        "actual_seq_lengths_query": torch.tensor([query_tokens], dtype=torch.int32),
        "actual_seq_lengths_kv": torch.tensor([key_tokens], dtype=torch.int32),
        "layout_query": "BSND",
        "layout_kv": "PA_BSND",
        "sparse_mode": 3,
        "block_table": torch.arange(blocks, dtype=torch.int32).reshape(batch, blocks),
        "attention_mode": 2,
        "quant_scale_repo_mode": 1,
        "tile_size": tile_size,
        "key_quant_mode": 2,
        "value_quant_mode": 2,
        "rope_head_dim": rope_dim,
    }
    return {
        "native_inputs": native_inputs,
        "fa_param": fa_param,
        "q_nope": q_nope,
        "q_rope": q_rope,
        "k_nope": k_nope,
        "k_rope": k_rope,
        "sparse_indices": sparse_indices,
        "scale_value": scale_value,
    }


def golden(case):
    """Evaluate unchanged official CPU golden, returning BSND FP32 output.

    The official routine mutates a shape list and its key tensor. Clone its
    dictionary so repeated calls do not dequantize the same key twice.
    """
    params = copy.deepcopy(case["fa_param"])
    return _t_increattention_bnsd(params).permute(0, 2, 1, 3).contiguous()


def validate_native_output(actual, expected):
    """Validate the native BF16 contract and return JSON-safe error metrics.

    In addition to the existing VA elementwise tolerances, independent relative
    gates reject degenerate outputs for small-magnitude synthetic inputs. These
    extra gates are engineering smoke criteria, not official CANN guarantees.
    Both inputs must already be CPU tensors; no device transfers happen here.
    """
    if not isinstance(actual, torch.Tensor):
        raise TypeError("Expected public torch_npu QSFA single-Tensor output; this is not the VA tuple-output wrapper")
    if actual.dtype != torch.bfloat16 or tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(f"Unexpected native output contract: dtype={actual.dtype}, shape={tuple(actual.shape)}")
    if actual.device.type != "cpu" or expected.device.type != "cpu":
        raise ValueError("Native output validation requires CPU tensors")
    actual = actual.float()
    expected = expected.float()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("Native QSFA output or reference contains NaN or infinity")
    reference_rms = expected.square().mean().sqrt().item()
    if reference_rms == 0:
        raise ValueError("Degenerate all-zero reference cannot establish this native smoke baseline")
    delta = actual - expected
    rmse = delta.square().mean().sqrt().item()
    normalized_rmse = rmse / reference_rms
    a_flat, e_flat = actual.flatten(), expected.flatten()
    denominator = a_flat.norm().item() * e_flat.norm().item()
    cosine = torch.dot(a_flat, e_flat).item() / denominator if denominator > 0 else 0.0
    if normalized_rmse > MAX_NORMALIZED_RMSE or cosine < MIN_COSINE:
        raise AssertionError(
            "Native QSFA engineering smoke gate failed: "
            f"normalized_rmse={normalized_rmse:.8g} (max {MAX_NORMALIZED_RMSE}), "
            f"cosine={cosine:.8g} (min {MIN_COSINE})"
        )
    torch.testing.assert_close(actual, expected, atol=BF16_ATOL, rtol=BF16_RTOL)
    return {
        "max_abs": delta.abs().max().item(),
        "rmse": rmse,
        "normalized_rmse": normalized_rmse,
        "cosine": cosine,
        "atol": BF16_ATOL,
        "rtol": BF16_RTOL,
        "tolerance_source": TOLERANCE_SOURCE,
        "engineering_smoke_gate": {
            "max_normalized_rmse": MAX_NORMALIZED_RMSE,
            "min_cosine": MIN_COSINE,
            "note": "Independent engineering smoke criteria; not official CANN accuracy guarantees",
        },
    }


@torch.inference_mode()
def run_native_baseline(case, device=0):
    """Call the installed public native QSFA API once and fail on mismatch.

    Returns JSON-serializable evidence after synchronization and correctness
    checks. Imports torch_npu lazily, so CPU reference tests need no NPU package.
    """
    import torch_npu

    if not hasattr(torch_npu, "npu_kv_quant_sparse_flash_attention"):
        raise RuntimeError("Installed torch_npu has no npu_kv_quant_sparse_flash_attention; no dependencies installed")
    print("QSFA_BASELINE: preparing official CPU golden for native INT8 PA correctness", flush=True)
    expected = golden(case)
    torch.npu.set_device(device)
    inputs = {
        name: value.to(f"npu:{device}") if isinstance(value, torch.Tensor) else value
        for name, value in case["native_inputs"].items()
    }
    torch.npu.synchronize()
    print("QSFA_BASELINE: calling torch_npu.npu_kv_quant_sparse_flash_attention (INT8, not C4)", flush=True)
    output = torch_npu.npu_kv_quant_sparse_flash_attention(**inputs)
    torch.npu.synchronize()
    if not isinstance(output, torch.Tensor):
        raise TypeError("Expected public torch_npu QSFA single-Tensor output; this is not the VA tuple-output wrapper")
    actual = output.cpu()
    accuracy = validate_native_output(actual, expected)
    result = {
        "status": "passed",
        "api": "torch_npu.npu_kv_quant_sparse_flash_attention",
        "scope": "Native INT8 combined-PA QSFA operator correctness only; not C4 execution or performance",
        "golden": GOLDEN_SOURCE,
        "query_shape": list(inputs["query"].shape),
        "key_cache_shape": list(inputs["key"].shape),
        "sparse_indices_shape": list(inputs["sparse_indices"].shape),
        "output_shape": list(actual.shape),
        "output_dtype": str(output.dtype),
        **accuracy,
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
    }
    print("QSFA_BASELINE: native INT8 accuracy passed", flush=True)
    return result
