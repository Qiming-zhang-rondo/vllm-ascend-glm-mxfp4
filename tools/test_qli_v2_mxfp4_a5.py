#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test the container's QLI V2 directly; do not import or install vLLM/VA.

MXFP4 and FP8 are prepared on CPU from the same source tensors, then consumed
as real low-bit inputs by the CANN compute operator on A5. No NPU quantizer is
part of this single-operator test.
Decoded-payload correctness and error against the original input are reported
separately. Timings include ACLNN preparation, dispatch and synchronization;
they are not isolated kernel times or model-throughput measurements.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

# Directly running a file in tools/ otherwise puts tools/bisect before Python's
# stdlib bisect; torch -> tempfile -> random then fails before the test starts.
if __name__ == "__main__":
    sys.path[:] = [entry for entry in sys.path if Path(entry).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402

TOPK = 2048
HEAD_DIM = 128
BLOCK_SIZE = 128
QLI_FP8 = 1
QLI_MXFP4 = 5
BF16_SIGNIFICAND_BITS = 8


class ExistingOperatorUnavailable(RuntimeError):
    """Exit 78 tells the launcher that a local operator build may be needed."""


class ContainerPrerequisiteError(RuntimeError):
    """A build cannot repair the selected container's missing runtime/toolkit."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    # These existing launcher variables remain accepted only for migration.
    legacy = {
        "query_tokens": ("QLI_QUERY_TOKENS", int, 1),
        "key_tokens": ("QLI_KEY_TOKENS", int, 8192),
        "warmup": ("QLI_WARMUP", int, 5),
        "iters": ("QLI_ITERS", int, 20),
        "min_topk_recall": ("QLI_MIN_TOPK_RECALL", float, 0.90),
        "min_score_cosine": ("QLI_MIN_SCORE_COSINE", float, 0.98),
        "max_score_nmae": ("QLI_MAX_SCORE_NMAE", float, 0.10),
        "max_mxfp4_p50_ms": ("QLI_MAX_MXFP4_P50_MS", float, None),
    }
    for destination, (variable, cast, default) in legacy.items():
        value = os.getenv(variable)
        flags = ["--" + destination.replace("_", "-")]
        if destination == "iters":
            flags.append("--iterations")
        parser.add_argument(
            *flags,
            type=cast,
            default=cast(value) if value is not None else default,
        )
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--check-only", action="store_true", help="Run a small real MXFP4 compute smoke test only.")
    parser.add_argument("--check-cache-layout", action="store_true", help="Also exercise padded, offset K cache views.")
    parser.add_argument("--opapi-lib", help="Existing library exporting both QLI V2 compute and metadata APIs")
    parser.add_argument("--acl-header", help="Container acl_base.h declaring FP4/E8M0 dtype enum values")
    parser.add_argument("--cann-root", help="Existing container CANN toolkit root")
    parser.add_argument("--output", type=Path, default=Path("qli_a5_results.json"))
    args = parser.parse_args(argv)
    if not 1 <= args.query_tokens <= args.key_tokens:
        parser.error("--query-tokens must be within the key sequence")
    if args.key_tokens % BLOCK_SIZE or args.key_tokens < TOPK + args.query_tokens - 1:
        parser.error(f"--key-tokens must be a multiple of {BLOCK_SIZE} and cover top-{TOPK} for every query")
    if not 1 <= args.heads <= 64:
        parser.error("--heads must be in [1, 64]")
    if args.warmup < 1 or args.iters < 1 or args.device < 0:
        parser.error("warmup/iters must be positive and device must be nonnegative")
    for name in ("min_topk_recall", "min_score_cosine"):
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) <= 1:
            parser.error(f"{name} must be finite and in [0, 1]")
    if not math.isfinite(args.max_score_nmae) or args.max_score_nmae < 0:
        parser.error("--max-score-nmae must be finite and nonnegative")
    if args.max_mxfp4_p50_ms is not None and (not math.isfinite(args.max_mxfp4_p50_ms) or args.max_mxfp4_p50_ms <= 0):
        parser.error("--max-mxfp4-p50-ms must be finite and positive")
    return args


def decode_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode low-nibble-first E2M1 and one E8M0 scale per 32 values on CPU."""
    payload = packed.detach().cpu().view(torch.uint8)
    scales = scale.detach().cpu().view(torch.uint8)
    if payload.shape[-1] != HEAD_DIM // 2 or scales.shape != (*payload.shape[:-1], 2, 2):
        raise ValueError("Expected packed D64 bytes and E8M0 [...,2,2] for logical D128")
    if bool((scales == 255).any()):
        raise ValueError("Nonfinite E8M0 scale in the generated test input")
    codes = torch.stack((payload & 15, payload >> 4), dim=-1).flatten(-2).long()
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    values = table[codes & 7] * torch.where(codes < 8, 1.0, -1.0)
    factors = torch.pow(2.0, scales.float() - 127).flatten(-2).repeat_interleave(32, dim=-1)
    return values * factors


def decode_fp8(payload: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    values = payload.detach().cpu().float()
    factors = scale.detach().cpu().float()
    if factors.shape != values.shape[:-1]:
        raise ValueError("FP8 per-token/head scales must match the payload prefix")
    return values * factors.unsqueeze(-1)


def round_fp64_to_bf16(values: torch.Tensor) -> torch.Tensor:
    """Round directly to BF16 RNE, avoiding an intermediate FP32 double rounding."""
    values = values.double()
    mantissa, exponent = torch.frexp(values)
    normal = torch.ldexp(torch.round(mantissa * (2**BF16_SIGNIFICAND_BITS)), exponent - BF16_SIGNIFICAND_BITS)
    info = torch.finfo(torch.bfloat16)
    subnormal_step = info.tiny * info.eps
    subnormal = torch.round(values / subnormal_step) * subnormal_step
    return torch.where(values.abs() >= info.tiny, normal, subnormal).float().bfloat16()


def reference_scores(query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, quant_mode=None) -> torch.Tensor:
    query_cpu = query.detach().cpu().float()
    key_cpu = key.detach().cpu().float().reshape(-1, 1, HEAD_DIM)[:, 0]
    weights_cpu = weights.detach().cpu().float()
    rows = []
    # Bound the temporary [T,N,K] allocation for chunked-query tests.
    for first in range(0, query_cpu.shape[0], 8):
        correlations = torch.matmul(query_cpu[first : first + 8], key_cpu.transpose(0, 1))
        if quant_mode == QLI_MXFP4:
            # CANN ops-transformer@632dddba, quant_lightning_indexer_v2_golden.py
            # reduce_mxfp4_weighted_qk: BF16 QK/ReLU, BF16 weights, and a BF16
            # destination FMA rounded after EACH head (not an FP32 sum).
            correlations = correlations.relu().bfloat16().double()
            head_weights = weights_cpu[first : first + 8].bfloat16().double()
            accum = torch.zeros((correlations.shape[0], correlations.shape[-1]), dtype=torch.bfloat16)
            for head in range(correlations.shape[1]):
                accum = round_fp64_to_bf16(accum.double() + head_weights[:, head : head + 1] * correlations[:, head])
            rows.append(accum.float())
        else:
            rows.append((correlations.relu() * weights_cpu[first : first + 8].unsqueeze(-1)).sum(dim=1))
    scores = torch.cat(rows)
    for query_index in range(query_cpu.shape[0]):
        causal_length = key_cpu.shape[0] - query_cpu.shape[0] + query_index + 1
        scores[query_index, causal_length:] = -torch.inf
    return scores


def make_host_inputs(args):
    """Prepare synthetic data on CPU so NPU RNG/broadcast ops are not under test."""
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    query = torch.randn(args.query_tokens, args.heads, HEAD_DIM, dtype=torch.float16, device="cpu", generator=generator)
    key = torch.randn(args.key_tokens, 1, HEAD_DIM, dtype=torch.float16, device="cpu", generator=generator)
    # Different D32 magnitudes expose incorrect FP4 scale pairing.
    amplitude = torch.tensor([0.125, 0.5, 2.0, 8.0], dtype=torch.float16, device="cpu").repeat_interleave(32)
    query.mul_(amplitude)
    key.mul_(amplitude.flip(0))
    weights = torch.rand(args.query_tokens, args.heads, dtype=torch.float32, device="cpu", generator=generator)
    weights.mul_(1.5).add_(0.25)
    return query, key, weights


def validate_indices(indices: torch.Tensor, query_tokens: int, key_tokens: int, topk: int = TOPK) -> None:
    if indices.shape != (query_tokens, 1, topk) or indices.dtype != torch.int32:
        raise AssertionError(f"Unexpected indices: {tuple(indices.shape)} {indices.dtype}")
    for query_index, row in enumerate(indices.cpu()[:, 0]):
        causal_length = key_tokens - query_tokens + query_index + 1
        if bool((row < -1).any()):
            raise AssertionError("Invalid negative index (only -1 is padding)")
        valid = row[row >= 0]
        if valid.numel() != min(topk, causal_length) or valid.unique().numel() != valid.numel():
            raise AssertionError(f"Query {query_index}: incorrect count or duplicate sparse indices")
        if not bool((valid < causal_length).all()):
            raise AssertionError(f"Query {query_index}: sparse index exceeds causal length")


def accuracy_metrics(indices: torch.Tensor, values: torch.Tensor, scores: torch.Tensor) -> dict:
    """Require finite values explicitly: NaN must never silently pass a gate."""
    actual_indices = indices.cpu()[:, 0].long()
    actual_values = values.cpu()[:, 0].float()
    if actual_values.shape != actual_indices.shape:
        raise AssertionError("Indices and values must have the same shape")
    if bool((actual_indices < 0).any()):
        raise AssertionError("Accuracy cases must cover top-k without padding")
    selected = scores.gather(1, actual_indices)
    if not bool(torch.isfinite(actual_values).all()) or not bool(torch.isfinite(selected).all()):
        raise AssertionError("Nonfinite QLI output or selected reference scores")
    topk = actual_indices.shape[-1]
    reference = scores.topk(topk, dim=-1)
    recall = (
        sum(
            torch.isin(actual, expected).float().mean().item()
            for actual, expected in zip(actual_indices, reference.indices)
        )
        / actual_indices.shape[0]
    )
    cosine = torch.nn.functional.cosine_similarity(actual_values.flatten(), selected.flatten(), dim=0).item()
    # Define zero/zero as exact rather than letting cosine's epsilon imply a failure.
    if not bool(actual_values.any()) and not bool(selected.any()):
        cosine = 1.0
    nmae = ((actual_values - selected).abs().mean() / selected.abs().mean().clamp_min(1e-12)).item()
    if not all(math.isfinite(value) for value in (recall, cosine, nmae)):
        raise AssertionError("Nonfinite accuracy metrics")
    # Close scores at the selection boundary may legitimately exchange indices.
    cutoff = reference.values[:, -1:]
    boundary_ok = selected >= cutoff - (cutoff.abs() * 0.02 + 0.02)
    return {
        "topk_recall": recall,
        "score_cosine": cosine,
        "score_nmae": nmae,
        "selected_scores_close": bool(torch.isclose(actual_values, selected, rtol=0.02, atol=0.02).all()),
        "selection_above_tolerated_cutoff": bool(boundary_ok.all()),
    }


def benchmark_compute(compute, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        compute()
    torch.npu.synchronize()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        compute()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    ordered = sorted(samples)
    return {
        "p50_ms": ordered[round((iterations - 1) * 0.50)],
        "p90_ms": ordered[round((iterations - 1) * 0.90)],
        "mean_ms": sum(samples) / iterations,
        "iterations": iterations,
    }


def report_stage(report, stage):
    if report is not None:
        report["stage"] = stage
        print("STAGE:", stage, flush=True)


def quantization_input_cpu(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cpu" or x.shape[-1] != HEAD_DIM:
        raise ValueError("Single-operator inputs must be CPU tensors with D128")
    if x.dtype != torch.float16 or not bool(torch.isfinite(x).all()):
        raise ValueError("Single-operator inputs must contain finite FP16 values")
    return x.float()


def quantize_mxfp4_cpu(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """D32 MX scale and E2M1 round-away encoding, using CPU tensor operations.

    Follows ops-nn@2a77283 quant/dynamic_mx_quant/tests/assets/golden.py:
    scale exponent = floor(log2(absmax)) - 2; round_mode='round'.
    This prepares QLI inputs; it does not validate the NPU quantizer itself.
    """
    groups = quantization_input_cpu(x).reshape(*x.shape[:-1], 4, 32)
    maxima = groups.abs().amax(dim=-1)
    _, exponents = torch.frexp(maxima)
    exponents = torch.where(maxima == 0, -127, exponents - 3).clamp(-127, 127)
    normalized = groups / torch.ldexp(torch.ones_like(maxima), exponents).unsqueeze(-1)
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device="cpu")
    # right=True chooses the larger magnitude at ties, and saturates at code 7.
    codes = torch.bucketize(normalized.abs(), midpoints, right=True).to(torch.uint8)
    codes |= normalized.signbit().to(torch.uint8) << 3
    codes = codes.reshape(x.shape)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    scales = (exponents + 127).to(torch.uint8).reshape(*x.shape[:-1], 2, 2)
    return packed.contiguous(), scales.contiguous()


def quantize_fp8_cpu(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row FP32 absmax/448 scale and E4M3FN RNE, as DynamicQuant mode 1."""
    values = quantization_input_cpu(x)
    scales = values.abs().amax(dim=-1) * (1.0 / 448.0)
    # Define an exact zero row for this QLI fixture, avoiding 0/0 in the producer.
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    payload = (values / scales.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return payload.contiguous(), scales.contiguous()


def prepare_case(name, quant_mode, q_source, k_source, device, backend, cu_q, seq_k, report=None):
    report_stage(report, f"{name}: CPU payload, scales and decoded reference")
    if quant_mode == QLI_MXFP4:
        query, query_scale = quantize_mxfp4_cpu(q_source)
        key, key_scale = quantize_mxfp4_cpu(k_source)
        key = key.reshape(-1, BLOCK_SIZE, 1, HEAD_DIM // 2)
        key_scale = key_scale.reshape(-1, BLOCK_SIZE, 1, 2, 2)
        decoded_q, decoded_k = decode_mxfp4(query, query_scale), decode_mxfp4(key, key_scale)
    else:
        query, query_scale = quantize_fp8_cpu(q_source)
        key, key_scale = quantize_fp8_cpu(k_source)
        key = key.reshape(-1, BLOCK_SIZE, 1, HEAD_DIM)
        key_scale = key_scale.reshape(-1, BLOCK_SIZE, 1)
        decoded_q, decoded_k = decode_fp8(query, query_scale), decode_fp8(key, key_scale)
    host_key, host_scale = key, key_scale
    report_stage(report, f"{name}: packed input H2D and synchronize")
    query, key, query_scale, key_scale = (tensor.to(device) for tensor in (query, key, query_scale, key_scale))
    torch.npu.synchronize()
    if report is not None:
        report["stage"] = f"{name}: QLI metadata"
        print("STAGE:", report["stage"], flush=True)
    metadata = backend.create_metadata(
        num_heads_q=q_source.shape[1],
        head_dim=HEAD_DIM,
        topk=TOPK,
        quant_mode=quant_mode,
        cu_seqlens_q=cu_q,
        seqused_k=seq_k,
    )
    return {
        "name": name,
        "query": query,
        "key": key,
        "query_scale": query_scale,
        "key_scale": key_scale,
        "metadata": metadata,
        "quant_mode": quant_mode,
        "decoded_query": decoded_q,
        "decoded_key": decoded_k,
        "host_key": host_key,
        "host_key_scale": host_scale,
    }


def check_case(case, invoke, args, original_scores, weights, result) -> dict:
    indices, values = invoke(return_value=1)
    torch.npu.synchronize()
    indices, values = indices.cpu(), values.cpu()
    validate_indices(indices, args.query_tokens, args.key_tokens)
    if values.dtype != torch.bfloat16:
        raise AssertionError(f"Expected BF16 sparse values, got {values.dtype}")
    scores = reference_scores(case["decoded_query"], case["decoded_key"], weights, quant_mode=case["quant_mode"])
    decoded = accuracy_metrics(indices, values, scores)
    original = accuracy_metrics(indices, values, original_scores)
    decoded_passed = decoded["selected_scores_close"] and decoded["selection_above_tolerated_cutoff"]
    quantization_passed = (
        original["topk_recall"] >= args.min_topk_recall
        and original["score_cosine"] >= args.min_score_cosine
        and original["score_nmae"] <= args.max_score_nmae
    )
    result.update(
        {
            "compute_reference": "BF16 QK/weights and per-head BF16 FMA" if case["quant_mode"] == QLI_MXFP4 else "FP32",
            "decoded_payload_reference": decoded,
            "original_fp16_input_reference": original,
            "operator_correctness_passed": decoded_passed,
            "quantization_thresholds_passed": quantization_passed,
            "passed": False,
        }
    )

    if args.check_cache_layout:
        # Optional cache coverage, with backing storage prepared on CPU.
        key, scale = case["host_key"], case["host_key_scale"]
        key_storage = torch.zeros((key.shape[0] * 2 + 1, *key.shape[1:]), dtype=key.dtype, device="cpu")
        scale_storage = torch.zeros((scale.shape[0] * 2 + 1, *scale.shape[1:]), dtype=scale.dtype, device="cpu")
        key_storage[1::2].copy_(key)
        scale_storage[1::2].copy_(scale)
        strided_key = key_storage.to(case["key"].device)[1::2]
        strided_scale = scale_storage.to(case["key_scale"].device)[1::2]
        strided_indices, strided_values = invoke(key=strided_key, key_scale=strided_scale, return_value=1)
        torch.npu.synchronize()
        torch.testing.assert_close(strided_indices.cpu(), indices, rtol=0, atol=0)
        torch.testing.assert_close(strided_values.cpu(), values, rtol=0, atol=0)
        result["dense_vs_strided_exact"] = True
    else:
        result["cache_layout_check"] = "not requested"
    index_only, empty_values = invoke(return_value=0)
    torch.npu.synchronize()
    torch.testing.assert_close(index_only.cpu(), indices.cpu(), rtol=0, atol=0)
    if empty_values.numel() != 0:
        raise AssertionError("return_value=0 must return empty sparse values")
    result.update(
        {
            "indices_only_vs_return_values_exact": True,
            "passed": decoded_passed and quantization_passed,
        }
    )
    print(f"ACCURACY {case['name']}: {json.dumps(result, allow_nan=False)}", flush=True)
    return result


def smoke_test(backend, npu_ops, device, quant_mode=QLI_MXFP4, report=None):
    """Test the actual mode-5 compute op, rather than just exported metadata."""
    if report is not None:
        report["stage"] = "smoke: CPU input preparation and copy to NPU"
        print("STAGE:", report["stage"], flush=True)
    generator = torch.Generator(device="cpu").manual_seed(20260911)
    query_source = torch.randn(1, 64, HEAD_DIM, dtype=torch.float16, device="cpu", generator=generator)
    key_source = torch.randn(TOPK, 1, HEAD_DIM, dtype=torch.float16, device="cpu", generator=generator)
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device="cpu").to(device)
    seq_k = torch.tensor([TOPK], dtype=torch.int32, device="cpu").to(device)
    weights = torch.ones((1, 64), dtype=torch.float32, device="cpu").to(device)
    blocks = torch.arange(TOPK // BLOCK_SIZE, dtype=torch.int32, device="cpu").view(1, -1).to(device)
    torch.npu.synchronize()
    case = prepare_case("QLI smoke", quant_mode, query_source, key_source, device, backend, cu_q, seq_k, report)
    if report is not None:
        report["stage"] = "smoke: QLI compute"
        print("STAGE:", report["stage"], flush=True)
    indices, values = backend.invoke(
        query=case["query"],
        key=case["key"],
        weights=weights,
        query_scale=case["query_scale"],
        key_scale=case["key_scale"],
        block_table=blocks,
        metadata=case["metadata"],
        cu_seqlens_q=cu_q,
        seqused_k=seq_k,
        quant_mode=quant_mode,
        topk=TOPK,
        return_value=1,
    )
    torch.npu.synchronize()
    validate_indices(indices, 1, TOPK)
    if values.shape != indices.shape or values.dtype != torch.bfloat16 or not bool(torch.isfinite(values.cpu()).all()):
        raise AssertionError(f"Mode-{quant_mode} smoke returned invalid or nonfinite sparse values")
    return {"quant_mode": quant_mode, "query_tokens": 1, "key_tokens": TOPK, "passed": True}


def can_probe_fp8_control(error):
    # PARAM_INVALID also includes bad shapes; it alone never proves missing FP4.
    return (
        error.phase == "GetWorkspaceSize"
        and error.api_name == "aclnnQuantLightningIndexerV2"
        and error.status == 161002
    )


def run_smoke(backend, npu_ops, device, report, call_error_type):
    try:
        report["smoke"] = smoke_test(backend, npu_ops, device, report=report)
    except call_error_type as error:
        report["smoke"] = {
            "quant_mode": QLI_MXFP4,
            "passed": False,
            "api": error.api_name,
            "phase": error.phase,
            "status": error.status,
        }
        if can_probe_fp8_control(error):
            try:
                report["fp8_control"] = smoke_test(backend, npu_ops, device, quant_mode=QLI_FP8, report=report)
            except Exception as control_error:
                report["fp8_control"] = {"passed": False, "error": str(control_error)}
                raise error
            report["build_candidate"] = (
                "Validated-shape MXFP4 was rejected, while same-shape FP8 compute passed. "
                "One local operator build may help; this does not prove the installed operator lacks MXFP4."
            )
            raise ExistingOperatorUnavailable(f"{error} {report['build_candidate']}") from error
        raise
    report["status"] = "compute_smoke_passed"
    print("Real MXFP4 QLI V2 smoke passed; no dependencies installed.", flush=True)


@torch.inference_mode()
def run(args, report):
    # Deliberately lazy: --help and CPU regression tests do not require torch_npu.
    try:
        import torch_npu
    except ImportError as error:
        raise ContainerPrerequisiteError(f"Cannot import container torch_npu: {error}") from error

    # Load this exact checkout's adapter without putting tools/ ahead of stdlib.
    backend_path = Path(__file__).resolve().with_name("qli_container_backend.py")
    backend_spec = importlib.util.spec_from_file_location("qli_container_backend", backend_path)
    backend_module = importlib.util.module_from_spec(backend_spec)
    backend_spec.loader.exec_module(backend_module)
    BackendUnavailable = backend_module.BackendUnavailable
    QLIBackend, QLICallError = backend_module.QLIBackend, backend_module.QLICallError

    try:
        if not torch.npu.is_available():
            raise RuntimeError("The current container has no accessible NPU")
        torch.npu.set_device(args.device)
        soc = torch_npu.npu.get_soc_version()
        device_name = torch.npu.get_device_name(args.device)
    except (RuntimeError, AttributeError) as error:
        raise ContainerPrerequisiteError(f"Cannot initialize the container NPU: {error}") from error
    if soc != 260:
        raise ContainerPrerequisiteError(f"Real MXFP4 QLI requires A5 (runtime SOC 260); got {soc}, {device_name}")
    required = ["get_npu_format"]
    missing = [name for name in required if not hasattr(torch_npu, name)]
    if not args.check_only and not hasattr(torch, "float8_e4m3fn"):
        missing.append("torch.float8_e4m3fn")
    if missing:
        raise ContainerPrerequisiteError("Container is missing required APIs/dtypes: " + ", ".join(missing))
    report["environment"] = {
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "torch_npu_path": torch_npu.__file__,
        "device": device_name,
        "runtime_soc": soc,
    }
    try:
        backend = QLIBackend(opapi_lib=args.opapi_lib, acl_header=args.acl_header, cann_root=args.cann_root)
    except BackendUnavailable as error:
        raise ExistingOperatorUnavailable(str(error)) from error
    except (RuntimeError, ValueError) as error:
        raise ContainerPrerequisiteError(str(error)) from error
    report["backend"] = backend.describe()
    report["input_preparation"] = {
        "device": "CPU",
        "mxfp4": "D32 E8M0 floor(log2(absmax))-2 scale, E2M1 round-away, low-nibble first",
        "fp8": "per-row FP32 absmax/448 scale, E4M3FN RNE",
        "scope": "QLI compute test; does not validate or time NPU quantization operators",
        "reference": "CANN ops-nn@2a77283 tests/assets/golden.py for dynamic_mx_quant and dynamic_quant",
    }
    print("CONTAINER:", json.dumps(report["environment"], default=str), flush=True)
    print("BACKEND:", json.dumps(report["backend"], default=str), flush=True)
    print("INPUT_PREPARATION: CPU packed payload/scales; A5 executes real QLI Metadata and compute", flush=True)
    if args.check_only:
        run_smoke(backend, torch_npu, torch.device("npu", args.device), report, QLICallError)
        report["backend"]["mxfp4_compute_verified"] = True
        report["backend"]["capability_note"] = (
            "Actual mode-5 smoke completed; full numerical/performance tests were not run"
        )
        return

    report["stage"] = "CPU input generation and reference"
    query_host, key_host, weights_host = make_host_inputs(args)
    original_scores = reference_scores(query_host, key_host, weights_host)
    print("CPU_INPUT_READY: synthetic inputs and FP32 reference prepared on CPU", flush=True)
    report["stage"] = "input copy to NPU"
    device = torch.device("npu", args.device)
    weights = weights_host.to(device)
    cu_q = torch.tensor([0, args.query_tokens], dtype=torch.int32, device="cpu").to(device)
    seq_k = torch.tensor([args.key_tokens], dtype=torch.int32, device="cpu").to(device)
    blocks = torch.arange(args.key_tokens // BLOCK_SIZE, dtype=torch.int32, device="cpu").view(1, -1).to(device)
    torch.npu.synchronize()
    print("INPUT_H2D_READY: input copies synchronized", flush=True)
    report["cases"] = {}
    report["timing_scope"] = "Synchronous wall latency including Python/ACLNN preparation, workspace, dispatch and sync"
    print("TIMING:", report["timing_scope"], "; quantization and metadata excluded", flush=True)
    for name, quant_mode in (("MXFP4", QLI_MXFP4), ("FP8", QLI_FP8)):
        report["stage"] = f"{name} CPU input preparation and metadata"
        case = prepare_case(name, quant_mode, query_host, key_host, device, backend, cu_q, seq_k, report)
        print(f"{name}_QUANT_READY: payload, scales and metadata prepared", flush=True)

        def invoke(*, key=None, key_scale=None, return_value=0, case=case, quant_mode=quant_mode):
            return backend.invoke(
                query=case["query"],
                key=case["key"] if key is None else key,
                weights=weights,
                query_scale=case["query_scale"],
                key_scale=case["key_scale"] if key_scale is None else key_scale,
                block_table=blocks,
                metadata=case["metadata"],
                cu_seqlens_q=cu_q,
                seqused_k=seq_k,
                quant_mode=quant_mode,
                topk=TOPK,
                return_value=return_value,
            )

        result = report["cases"][name] = {}
        result["accuracy"] = {}
        report["stage"] = f"{name} QLI accuracy checks"
        print(f"QLI_CALL_BEGIN: {name}", flush=True)
        check_case(case, invoke, args, original_scores, weights_host, result["accuracy"])
        if quant_mode == QLI_MXFP4:
            report["backend"]["mxfp4_compute_verified"] = True
            report["backend"]["mxfp4_numerical_checks_passed"] = result["accuracy"]["operator_correctness_passed"]
            report["backend"]["capability_note"] = "Actual mode-5 compute completed; see the separate accuracy gates"
        report["stage"] = f"{name} QLI timing"
        result["wall_latency"] = benchmark_compute(invoke, args.warmup, args.iters)
        print(f"PERF {name}: {json.dumps(result['wall_latency'])}", flush=True)
    mx_p50 = report["cases"]["MXFP4"]["wall_latency"]["p50_ms"]
    fp8_p50 = report["cases"]["FP8"]["wall_latency"]["p50_ms"]
    report["fp8_over_mxfp4_wall_p50_ratio"] = fp8_p50 / mx_p50
    print(f"PERF FP8/MXFP4 wall P50 ratio: {fp8_p50 / mx_p50:.3f}x", flush=True)
    if not all(case["accuracy"]["passed"] for case in report["cases"].values()):
        raise AssertionError("Accuracy gate failed; see decoded-payload and quantization metrics separately in JSON")
    if args.max_mxfp4_p50_ms is not None and mx_p50 > args.max_mxfp4_p50_ms:
        raise AssertionError("MXFP4 wall P50 exceeds the configured limit")
    report["status"] = "passed"


def main(argv=None) -> int:
    args = parse_args(argv)
    report = {
        "status": "running",
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "scope": "Synthetic QLI V2 single-operator correctness and timing; not GLM model accuracy",
        "operator_score_tolerance": {"rtol": 0.02, "atol": 0.02, "provisional": True},
    }
    try:
        run(args, report)
        return_code = 0
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(f"QLI test FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        if isinstance(error, ExistingOperatorUnavailable):
            return_code = 78
        elif isinstance(error, ContainerPrerequisiteError):
            return_code = 2
        else:
            return_code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Result: {args.output.resolve()} ({report['status']})", flush=True)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
