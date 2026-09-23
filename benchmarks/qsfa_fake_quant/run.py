# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-operator quantization sensitivity, without model/framework changes."""

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import torch

from .official_baseline import build_case, run_native_baseline
from .quantization import mxfp4_roundtrip, mxfp8_roundtrip
from .reference import (
    NOPE_DIM,
    attention_reference,
    error_metrics,
    synthetic_inputs,
    validate_inputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reference-only", action="store_true", help="CPU simulation only (default)")
    mode.add_argument(
        "--native-baseline",
        action="store_false",
        dest="reference_only",
        help="Opt in to the separate native INT8 QSFA gate before CPU simulation; requires an NPU",
    )
    parser.set_defaults(reference_only=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--query-tokens", type=int, default=1)
    parser.add_argument("--key-tokens", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8, help="Local heads on one TP rank, not global model heads")
    parser.add_argument("--selected-tokens", type=int, default=2048)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260921, 20260922, 20260923])
    parser.add_argument("--input", type=Path, help="torch.save dictionary: query/kv/indices/scale_value")
    parser.add_argument("--keep-q-rope-bf16", action="store_true", help="Quantize only Q NoPE; default quantizes all Q")
    parser.add_argument("--threads", type=int, default=8, help="CPU reference threads")
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument("--max-relative-rmse", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=Path("qsfa_fake_quant.json"))
    args = parser.parse_args()
    if args.device < 0 or args.threads < 1:
        parser.error("device must be nonnegative and threads positive")
    if not math.isfinite(args.min_cosine) or not 0 <= args.min_cosine <= 1:
        parser.error("--min-cosine must be finite and in [0,1]")
    if not math.isfinite(args.max_relative_rmse) or args.max_relative_rmse < 0:
        parser.error("--max-relative-rmse must be finite and nonnegative")
    return args


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate_inputs(query, kv, indices, scale, *, keep_q_rope_bf16=False):
    validate_inputs(query, kv, indices, scale)
    # This experiment starts from BF16 activations. Record this in the report;
    # replayed FP32 sources are deliberately rounded before any comparison.
    query, kv = query.bfloat16(), kv.bfloat16()
    validate_inputs(query, kv, indices, scale)  # BF16 conversion can overflow.
    kv4 = torch.cat((mxfp4_roundtrip(kv[..., :NOPE_DIM]), kv[..., NOPE_DIM:].float()), dim=-1)
    if keep_q_rope_bf16:
        q8 = torch.cat((mxfp8_roundtrip(query[..., :NOPE_DIM]), query[..., NOPE_DIM:].float()), dim=-1)
    else:
        q8 = mxfp8_roundtrip(query)

    full = attention_reference(query, kv, indices, scale).bfloat16().float()
    c4 = attention_reference(query, kv4, indices, scale).bfloat16().float()
    q8_accumulator = attention_reference(q8, kv4, indices, scale)
    p8_accumulator = attention_reference(q8, kv4, indices, scale, quantize_probability=True)
    outputs = {
        "kv4": c4,
        "q8_kv4": q8_accumulator.bfloat16().float(),
        "q8_kv4_o8": mxfp8_roundtrip(q8_accumulator),
        "q8_kv4_p8_o8": mxfp8_roundtrip(p8_accumulator),
    }
    records = {}
    previous = full
    for name, output in outputs.items():
        records[name] = {
            "vs_bf16": error_metrics(output, full),
            "vs_c4": error_metrics(output, c4),
            "vs_previous_stage": error_metrics(output, previous),
        }
        previous = output
    return records


def load_inputs(args):
    if args.input:
        captured = torch.load(args.input, map_location="cpu", weights_only=True)
        if not isinstance(captured, dict) or not {"query", "kv", "indices", "scale_value"}.issubset(captured):
            raise ValueError("--input requires query, kv, indices, scale_value (the actual model attention scale)")
        values = captured["query"], captured["kv"], captured["indices"], float(captured["scale_value"])
        validate_inputs(*values)
        yield {"input_file": str(args.input.resolve()), "seed": None}, values
    else:
        for seed in args.seeds:
            yield (
                {"input_file": None, "seed": seed},
                synthetic_inputs(args.query_tokens, args.key_tokens, args.heads, args.selected_tokens, seed),
            )


@torch.inference_mode()
def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    report = {
        "status": "running",
        "scope": "Single-operator numerical simulation; not a C4/FP8 QSFA kernel or model-accuracy validation",
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch": torch.__version__,
        "performance": {"measured": False, "reason": "CPU fake-quantization time cannot predict a new NPU kernel"},
        "contract": {
            "input": "Logical single-sequence MLA, source activations rounded to BF16; K NoPE is also V",
            "query_quantization": "MXFP8 E4M3FN/E8M0 group32 along D, scale_alg=0, RNE",
            "kv_quantization": "MXFP4 E2M1/E8M0 group32 along NoPE D, scale_alg=0, ties away; RoPE stays BF16",
            "probability": "FP32 softmax; baseline P casts to BF16; P8 uses MXFP8 along selected-token axis",
            "accumulation": "FP32 QK and PV reference matmul",
            "output_quantization": "MXFP8 group32 along output D, from FP32 accumulated result",
            "caveat": "This does not establish direct FP8xFP4 Cube support or a compatible PV scale layout",
        },
        "screening_gates": {
            "min_cosine": args.min_cosine,
            "max_relative_rmse": args.max_relative_rmse,
            "provisional": True,
            "scope": "Engineering screening gates only; not official QSFA tolerances or a no-model-loss guarantee",
        },
        "native_baseline": {"status": "not_run"},
        "experiments": [],
    }
    write_report(args.output, report)
    try:
        if args.reference_only:
            report["native_baseline"] = {
                "status": "skipped",
                "reason": "CPU simulation; native QSFA requires explicit --native-baseline",
            }
            print("REFERENCE ONLY: no NPU operator is executed", flush=True)
        else:
            report["stage"] = "native INT8 QSFA baseline"
            write_report(args.output, report)
            report["native_baseline"] = run_native_baseline(build_case(), device=args.device)

        for identity, (query, kv, indices, scale) in load_inputs(args):
            report["stage"] = "CPU quantization-error screening"
            print(f"EXPERIMENT: {identity}, Q={list(query.shape)}, KV={list(kv.shape)}", flush=True)
            cases = evaluate_inputs(query, kv, indices, scale, keep_q_rope_bf16=args.keep_q_rope_bf16)
            entry = {
                **identity,
                "query_shape": list(query.shape),
                "kv_shape": list(kv.shape),
                "indices_shape": list(indices.shape),
                "scale_value": scale,
                "source_dtype": {"query": str(query.dtype), "kv": str(kv.dtype)},
                "cases": cases,
            }
            for name, case in cases.items():
                for anchor in ("vs_bf16", "vs_c4"):
                    metrics = case[anchor]
                    case[f"screening_passed_{anchor}"] = (
                        metrics["cosine"] >= args.min_cosine and metrics["relative_rmse"] <= args.max_relative_rmse
                    )
                for anchor in ("vs_bf16", "vs_c4"):
                    print(
                        f"{name:18s} {anchor:8s} cosine={case[anchor]['cosine']:.6f} "
                        f"relative_rmse={case[anchor]['relative_rmse']:.6f} "
                        f"screening={'PASS' if case[f'screening_passed_{anchor}'] else 'FAIL'}",
                        flush=True,
                    )
            report["experiments"].append(entry)
            write_report(args.output, report)
        report["incremental_screening_passed_vs_c4"] = all(
            case["screening_passed_vs_c4"]
            for experiment in report["experiments"]
            for name, case in experiment["cases"].items()
            if name != "kv4"
        )
        report["screening_passed"] = all(
            case["screening_passed_vs_bf16"] and case["screening_passed_vs_c4"]
            for experiment in report["experiments"]
            for case in experiment["cases"].values()
        )
        report["status"] = "screening_passed" if report["screening_passed"] else "screening_failed"
        report["stage"] = "completed"
        code = 0 if report["screening_passed"] else 1
    except Exception as error:
        report["status"] = "error"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
        code = 2
    write_report(args.output, report)
    print(f"RESULT: {args.output.resolve()} ({report['status']})", flush=True)
    print("No candidate-kernel performance or end-to-end model accuracy was measured.", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
