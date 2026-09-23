#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the actual A5 Q8/C4/O8 custom operator, with CPU input preparation.

No native INT8 baseline, dependency installation or fallback is performed.
Wall latency covers the full custom op and synchronization, not isolated Cube
instructions. The optional profiler provides device tasks for separate review.
"""

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path

# Allow both python -m benchmarks.qsfa_q8c4_o8.run and an absolute script path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from benchmarks.qsfa_fake_quant.quantization import mxfp8_roundtrip  # noqa: E402
from benchmarks.qsfa_fake_quant.reference import attention_reference, error_metrics, synthetic_inputs  # noqa: E402
from benchmarks.qsfa_q8c4_o8.packing import (  # noqa: E402
    MAX_SELECTED,
    MIN_SELECTED,
    NOPE_DIM,
    SUPPORTED_HEADS,
    decode_mxfp8,
    prepare_inputs,
)

COMPUTE_MIN_COSINE = 0.999
COMPUTE_MAX_RELATIVE_RMSE = 0.02
QUANT_MIN_COSINE = 0.99
QUANT_MAX_RELATIVE_RMSE = 0.10
ARGUMENT_ORDER = ("q", "qs", "kv", "ks", "rope", "idx")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True, help="Built custom torch operator shared library")
    parser.add_argument("--output", type=Path, default=Path("qsfa_q8c4_o8_results.json"))
    parser.add_argument("--heads", type=int, choices=SUPPORTED_HEADS, default=8)
    parser.add_argument("--key-tokens", type=int, default=8192)
    parser.add_argument("--selected-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", "--iterations", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--input", type=Path, help="Saved CPU query/kv/indices/scale_value dictionary; shapes from file"
    )
    parser.add_argument("--profile", action="store_true", help="Capture one additional full-op profiler invocation")
    args = parser.parse_args(argv)
    if not args.input and (
        not MIN_SELECTED <= args.selected_tokens <= MAX_SELECTED
        or args.selected_tokens % MIN_SELECTED
        or args.key_tokens < args.selected_tokens
    ):
        parser.error("Require S in [128,8192], S divisible by 128, and K >= S")
    if args.device < 0 or args.warmup < 1 or args.iters < 1 or args.threads < 1:
        parser.error("device must be nonnegative; warmup, iters and threads must be positive")
    return args


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stage(args, report, name):
    report["stage"] = name
    write_report(args.output, report)
    print(f"QSFA_Q8C4_O8: {name}", flush=True)


def load_inputs(args):
    if args.input:
        data = torch.load(args.input, map_location="cpu", weights_only=True)
        if not isinstance(data, dict) or not {"query", "kv", "indices", "scale_value"}.issubset(data):
            raise ValueError("--input requires query, kv, indices and the actual model scale_value")
        return data["query"], data["kv"], data["indices"], float(data["scale_value"])
    return synthetic_inputs(1, args.key_tokens, args.heads, args.selected_tokens, args.seed)


def reference_outputs(decoded_query, decoded_kv, original_query, original_kv, indices, scale):
    accumulator = attention_reference(decoded_query, decoded_kv, indices, scale, quantize_probability=False)
    decoded_golden = mxfp8_roundtrip(accumulator)[0]
    original_golden = attention_reference(original_query, original_kv, indices, scale).bfloat16().float()[0]
    return decoded_golden, original_golden


def load_runtime(library, device_index):
    if not library.is_file():
        raise FileNotFoundError(f"Custom operator library does not exist: {library}")
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("No NPU is available; the runner does not install dependencies or use a CPU fallback")
    torch.npu.set_device(device_index)
    device = torch.device(f"npu:{device_index}")
    torch.ops.load_library(str(library.resolve()))
    operation = torch.ops.qsfa_q8c4_o8.forward
    return (
        operation,
        device,
        lambda: torch.npu.synchronize(device),
        {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": str(device),
            "device_name": torch.npu.get_device_name(device_index),
            "library": str(library.resolve()),
        },
    )


def check_status(result):
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        raise AssertionError("Expected (uint8 output, uint8 output_scales, int32 status)")
    status = result[2]
    if not isinstance(status, torch.Tensor) or status.dtype != torch.int32 or tuple(status.shape) != (1,):
        raise AssertionError("Expected status tensor int32[1]")
    code = int(status.detach().cpu()[0])
    if code:
        raise RuntimeError(f"Custom operator reported status {code}; output is not valid")


def decode_output(result, heads):
    check_status(result)
    payload, scales, _ = result
    if payload.dtype != torch.uint8 or tuple(payload.shape) != (heads, NOPE_DIM):
        raise AssertionError("Expected output uint8[H,512] containing E4M3FN bytes")
    if scales.dtype != torch.uint8 or tuple(scales.shape) != (heads, NOPE_DIM // 32):
        raise AssertionError("Expected output scales uint8[H,16] containing E8M0 bytes")
    return decode_mxfp8(payload.detach().cpu(), scales.detach().cpu())


def compare_output(actual, decoded_golden, original_golden, c4_bf16_golden=None):
    compute = error_metrics(actual, decoded_golden)
    quantization = error_metrics(actual, original_golden)
    compute_passed = compute["cosine"] >= COMPUTE_MIN_COSINE and compute["relative_rmse"] <= COMPUTE_MAX_RELATIVE_RMSE
    quantization_passed = (
        quantization["cosine"] >= QUANT_MIN_COSINE and quantization["relative_rmse"] <= QUANT_MAX_RELATIVE_RMSE
    )
    result = {
        "operator_correctness": compute,
        "operator_correctness_passed": compute_passed,
        "vs_original_bf16": quantization,
        "quantization_screening_passed": quantization_passed,
    }
    if c4_bf16_golden is not None:
        incremental = error_metrics(actual, c4_bf16_golden)
        result["vs_c4_bf16"] = incremental
        result["incremental_screening_passed_vs_c4"] = (
            incremental["cosine"] >= QUANT_MIN_COSINE and incremental["relative_rmse"] <= QUANT_MAX_RELATIVE_RMSE
        )
    return result


def benchmark(invoke, synchronize, warmup, iterations):
    for _ in range(warmup):
        result = invoke()
        synchronize()
        check_status(result)
    samples = []
    for _ in range(iterations):
        synchronize()
        start = time.perf_counter()
        result = invoke()
        synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
        check_status(result)  # Status readback is outside the measured interval.
    ordered = sorted(samples)
    return {
        "scope": "Synchronous custom-op wall time: host dispatch/allocations, all device kernels and completion sync",
        "included": "Cache gather/unpack, QK NoPE and RoPE, softmax, BF16 PV, and O8 quantization/writeout",
        "excluded": "CPU source generation/packing/reference, H2D, library loading/JIT, warmup and result readback",
        "not_kernel_task_duration": True,
        "warmup": warmup,
        "iterations": iterations,
        "samples_ms": samples,
        "mean_ms": statistics.mean(samples),
        "p50_ms": statistics.median(samples),
        "p90_ms": ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)],
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def capture_profile(invoke, synchronize, output_path):
    import torch_npu

    profiler = torch_npu.profiler
    trace_dir = output_path.parent / (output_path.stem + "_profiler")
    trace_dir.mkdir(parents=True, exist_ok=True)
    synchronize()
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        on_trace_ready=profiler.tensorboard_trace_handler(str(trace_dir), async_mode=False),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ):
        result = invoke()
        synchronize()
    check_status(result)
    return {
        "directory": str(trace_dir.resolve()),
        "scope": "One extra full custom-op invocation; inspect all its device tasks, not only QK",
        "device_task_duration_automatically_summarized": False,
    }


@torch.inference_mode()
def run(args):
    report = {
        "status": "running",
        "stage": "initialization",
        "scope": "Actual custom A5 Q8/C4/O8 prototype; one Q token; no model or native INT8 baseline",
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "compute_verified": False,
        "performance": {"measured": False},
        "contract": {
            "input": "Sources rounded to BF16; Q=1; sparse indices supplied, not computed by this test",
            "query": "MXFP8 E4M3FN/E8M0 D32 including Q RoPE; scale_alg=0, RNE",
            "cache": "NoPE MXFP4 E2M1/E8M0 D32, even D in low nibble, ties away; K RoPE remains BF16",
            "qk": "C4 payload expanded exactly to FP8 with original E8M0 scales for MXFP8 NoPE QK; RoPE BF16 QK",
            "pv": "FP32 softmax -> BF16 P; decoded C4 V -> BF16; FP32 PV accumulation",
            "output": "MXFP8/E8M0 D32 from FP32 PV result without an intervening BF16 output cast",
            "reference": "Pinned official QSFA _t_increattention_bnsd via qsfa_fake_quant.reference; decoded inputs",
            "c4_bf16_reference": "Original BF16 Q with decoded C4 KV and BF16 output; isolates Q8/O8 increment",
        },
        "gates": {
            "provisional": True,
            "scope": "Engineering smoke and quantization screening; not official kernel or model certification",
            "operator": {"min_cosine": COMPUTE_MIN_COSINE, "max_relative_rmse": COMPUTE_MAX_RELATIVE_RMSE},
            "original_bf16": {"min_cosine": QUANT_MIN_COSINE, "max_relative_rmse": QUANT_MAX_RELATIVE_RMSE},
            "vs_c4_bf16": {"min_cosine": QUANT_MIN_COSINE, "max_relative_rmse": QUANT_MAX_RELATIVE_RMSE},
        },
    }
    write_report(args.output, report)
    try:
        torch.set_num_threads(args.threads)
        stage(args, report, "CPU inputs, low-bit packing and payload validation")
        query, kv, indices, scale = load_inputs(args)
        prepared, decoded_query, decoded_kv, original_query, original_kv = prepare_inputs(query, kv, indices, scale)
        heads = original_query.shape[1]
        report["case"] = {
            "query_shape": list(original_query.shape),
            "kv_shape": list(original_kv.shape),
            "indices_shape": list(indices.shape),
            "scale": scale,
            "prepared_inputs": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in prepared.items()
            },
        }
        stage(args, report, "CPU official golden, original BF16 and C4/BF16 references")
        decoded_golden, original_golden = reference_outputs(
            decoded_query, decoded_kv, original_query, original_kv, indices, scale
        )
        c4_bf16_golden = attention_reference(original_query, decoded_kv, indices, scale).bfloat16().float()[0]
        stage(args, report, "Load actual custom operator and NPU runtime")
        operation, device, synchronize, runtime_info = load_runtime(args.library, args.device)
        report["runtime"] = runtime_info
        stage(args, report, "Copy validated packed inputs to NPU")
        device_inputs = [prepared[name].to(device) for name in ARGUMENT_ORDER]
        synchronize()

        def invoke():
            return operation(*device_inputs, scale)

        stage(args, report, "First actual candidate invocation and synchronization")
        result = invoke()
        synchronize()
        stage(args, report, "Read back output and verify decoded-payload correctness")
        actual = decode_output(result, heads)
        accuracy = compare_output(actual, decoded_golden, original_golden, c4_bf16_golden)
        report["accuracy"] = accuracy
        print("ACCURACY:", json.dumps(accuracy, allow_nan=False), flush=True)
        if not accuracy["operator_correctness_passed"]:
            raise AssertionError("Candidate failed decoded-input operator correctness; timing was not run")
        report["compute_verified"] = True
        stage(args, report, "Warmup and full custom-op synchronous wall measurement")
        report["performance"] = {"measured": True, **benchmark(invoke, synchronize, args.warmup, args.iters)}
        print("PERFORMANCE:", json.dumps(report["performance"], allow_nan=False), flush=True)
        if args.profile:
            stage(args, report, "Explicit profiler capture of one complete candidate invocation")
            report["profile"] = capture_profile(invoke, synchronize, args.output)
        report["status"] = "passed" if accuracy["quantization_screening_passed"] else "quantization_failed"
        report["stage"] = "complete"
        write_report(args.output, report)
        return 0 if report["status"] == "passed" else 1
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_report(args.output, report)
        traceback.print_exc()
        return 1


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
