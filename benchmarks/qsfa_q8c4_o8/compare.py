#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare candidate and installed native QSFA in separate fresh processes.

This coordinator prepares one CPU BF16 input snapshot. It neither loads the
NPU runtime nor builds kernels. Both timings are synchronized operator wall
latencies on the same logical inputs, with different cache/compute contracts.
"""

import hashlib
import json
import math
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from benchmarks.qsfa_q8c4_o8.packing import validate_logical_inputs  # noqa: E402
from benchmarks.qsfa_q8c4_o8.run import load_inputs, parse_args, write_report  # noqa: E402

BOOTSTRAP = """import pathlib,runpy,sys,types
root,module,*args=sys.argv[1:]
sys.path.insert(0,root)
package=types.ModuleType("benchmarks")
package.__path__=[str(pathlib.Path(root)/"benchmarks")]
sys.modules["benchmarks"]=package
sys.argv=[module,*args]
runpy.run_module(module,run_name="__main__")
"""


def input_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_nonfinite_json(value):
    raise ValueError(f"Worker report contains nonfinite JSON value {value}")


def freeze_inputs(args):
    query, kv, indices, scale = load_inputs(args)
    validate_logical_inputs(query, kv, indices, scale)
    query = query.detach().to(dtype=torch.bfloat16).contiguous().clone()
    kv = kv.detach().to(dtype=torch.bfloat16).contiguous().clone()
    indices = indices.to(dtype=torch.int32).contiguous().clone()
    validate_logical_inputs(query, kv, indices, scale)
    with tempfile.NamedTemporaryFile(
        prefix="comparison_inputs_", suffix=".pt", dir=args.output.parent, delete=False
    ) as f:
        path = Path(f.name).resolve()
        torch.save({"query": query, "kv": kv, "indices": indices, "scale_value": float(scale)}, f)
    path.chmod(0o444)
    return {
        "path": str(path),
        "sha256": input_digest(path),
        "query_shape": list(query.shape),
        "kv_shape": list(kv.shape),
        "indices_shape": list(indices.shape),
        "source_dtype": "bfloat16",
        "scale_value": float(scale),
        "contract": "Both workers read this same frozen logical input; each prepares its own cache representation",
    }


def worker_command(args, variant, snapshot, output):
    repo = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "-I",
        "-c",
        BOOTSTRAP,
        str(repo),
        "benchmarks.qsfa_q8c4_o8.run",
        "--variant",
        variant,
        "--library",
        str(args.library.resolve()),
        "--input",
        str(snapshot),
        "--output",
        str(output),
    ]
    for flag, value in (
        ("--warmup", args.warmup),
        ("--iters", args.iters),
        ("--device", args.device),
        ("--threads", args.threads),
        ("--heads", args.heads),
        ("--key-tokens", args.key_tokens),
        ("--selected-tokens", args.selected_tokens),
        ("--seed", args.seed),
    ):
        command.extend((flag, str(value)))
    if args.profile:
        command.append("--profile")
    return command


def validate_worker(result, returncode, variant):
    if not isinstance(result, dict):
        raise RuntimeError(f"{variant} did not produce a JSON object report")
    passed = returncode == 0 and result.get("status") == "passed"
    quantization_failed = returncode == 1 and result.get("status") == "quantization_failed"
    if not (passed or quantization_failed):
        raise RuntimeError(f"{variant} failed: exit={returncode}, status={result.get('status')!r}; no speedup claim")
    performance = result.get("performance", {})
    if (
        result.get("compute_verified") is not True
        or not isinstance(performance, dict)
        or performance.get("measured") is not True
    ):
        raise RuntimeError(f"{variant} lacks verified compute or a completed performance measurement")
    for key in ("p50_ms", "mean_ms"):
        value = performance.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"{variant} has invalid positive finite latency {key}={value!r}")


def run(args):
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output.parent / "candidate.json"
    baseline_path = args.output.parent / "native_baseline.json"
    if args.output in (candidate_path, baseline_path):
        raise ValueError("Comparison --output must differ from candidate.json and native_baseline.json")
    report = {
        "schema_version": 1,
        "status": "running",
        "compute_verified": False,
        "scope": "Same logical BF16 inputs; separate fresh processes; synchronized operator wall latency",
        "not_pure_kernel_task_duration": True,
        "not_end_to_end_model_performance": True,
        "differing_cache_types": True,
        "comparison": {"available": False},
    }
    write_report(args.output, report)
    try:
        torch.set_num_threads(args.threads)
        report["shared_input"] = freeze_inputs(args)
        snapshot = Path(report["shared_input"]["path"])
        expected_hash = report["shared_input"]["sha256"]
        write_report(args.output, report)
        for variant, role, output in (
            ("candidate", "candidate", candidate_path),
            ("native", "baseline", baseline_path),
        ):
            if input_digest(snapshot) != expected_hash:
                raise RuntimeError("Shared input snapshot changed before worker launch")
            # A crashed child must never be mistaken for a successful stale run.
            output.unlink(missing_ok=True)
            entry = {"report_path": str(output), "input_sha256": expected_hash}
            report[role] = entry
            report["stage"] = f"{variant} fresh-process worker"
            write_report(args.output, report)
            command = worker_command(args, variant, snapshot, output)
            print(f"COMPARISON: starting {variant}; shared input SHA256={expected_hash}", flush=True)
            # Inherit the launcher's FLA/CANN/NPU environment and log streams.
            completed = subprocess.run(command, check=False)
            entry["returncode"] = completed.returncode
            if output.is_file():
                entry["result"] = json.loads(output.read_text(encoding="utf-8"), parse_constant=reject_nonfinite_json)
            else:
                raise RuntimeError(f"{variant} exited {completed.returncode} without report: {output}")
            write_report(args.output, report)
            if input_digest(snapshot) != expected_hash:
                raise RuntimeError(f"Shared input snapshot changed during {variant}; comparison is invalid")
            validate_worker(entry["result"], completed.returncode, variant)

        candidate = report["candidate"]["result"]
        baseline = report["baseline"]["result"]
        cp, bp = candidate["performance"], baseline["performance"]
        report["comparison"] = {
            "available": True,
            "candidate_p50_ms": cp["p50_ms"],
            "baseline_p50_ms": bp["p50_ms"],
            "p50_speedup_ratio": bp["p50_ms"] / cp["p50_ms"],
            "p50_latency_reduction_percent": (1.0 - cp["p50_ms"] / bp["p50_ms"]) * 100.0,
            "candidate_mean_ms": cp["mean_ms"],
            "baseline_mean_ms": bp["mean_ms"],
            "mean_speedup_ratio": bp["mean_ms"] / cp["mean_ms"],
            "mean_latency_reduction_percent": (1.0 - cp["mean_ms"] / bp["mean_ms"]) * 100.0,
            "interpretation": "Ratio <1 and negative latency reduction mean the candidate is slower",
            "scope": "Full synchronized operator wall calls; see each report for included stages and cache dtype",
            "candidate_contract": "Q8/MXFP4 cache/O8 custom prototype",
            "baseline_contract": "Installed native QSFA; its cache/output contract is recorded in the baseline report",
        }
        report["compute_verified"] = True
        report["status"] = (
            "quantization_failed" if "quantization_failed" in (candidate["status"], baseline["status"]) else "passed"
        )
        report["stage"] = "complete"
        write_report(args.output, report)
        comparison = report["comparison"]
        print(
            "COMPARISON: "
            f"candidate p50={cp['p50_ms']:.6f} ms; native p50={bp['p50_ms']:.6f} ms; "
            f"speedup={comparison['p50_speedup_ratio']:.3f}x; "
            f"latency reduction={comparison['p50_latency_reduction_percent']:+.2f}% "
            f"(operator wall time, differing cache types; status={report['status']})",
            flush=True,
        )
        return 0 if report["status"] == "passed" else 1
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_report(args.output, report)
        print(f"COMPARISON: unavailable: {exc}", file=sys.stderr, flush=True)
        return 1


def main(argv=None):
    args = parse_args(argv)
    if getattr(args, "variant", "candidate") != "candidate":
        print("--variant native is a worker option; comparison always runs candidate then native", file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
