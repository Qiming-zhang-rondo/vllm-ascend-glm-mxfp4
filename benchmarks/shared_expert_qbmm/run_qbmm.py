# SPDX-License-Identifier: Apache-2.0
"""Single-command launcher. Parent needs only Python's standard library."""

import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def int_list(value):
    try:
        values = [int(x) for x in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated positive integers") from error
    if not values or min(values) <= 0 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must be positive and unique")
    return values


def glm_shapes(tokens, tp_sizes):
    """Shapes of one rank, Y[M,N] = X[M,K] @ W[K,N]. M is LOCAL tokens."""
    cases = []
    for tp in tp_sizes:
        if 2048 % tp or (2048 // tp) % 64:
            raise ValueError("TP must divide 2048; local intermediate size must be divisible by 64")
        for m in tokens:
            for projection, k, n in (("gate_up", 6144, 4096 // tp), ("down", 2048 // tp, 6144)):
                cases.append({"name": f"{projection}_tp{tp}_m{m}", "m": m, "k": k, "n": n,
                              "projection": projection, "tp": tp, "stage": "glm"})
    return cases


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def comparisons(results):
    grouped = {}
    for row in results:
        if row.get("status") != "passed" or not row.get("performance"):
            continue
        grouped.setdefault(row["case"]["name"], {})[row["kind"]] = row
    output = []
    for name, pair in grouped.items():
        if set(pair) != {"mxa8w4", "mxa8w8"}:
            continue
        a, b = pair["mxa8w4"], pair["mxa8w8"]
        if a.get("source_input_sha256") != b.get("source_input_sha256"):
            raise ValueError(f"Cannot compare {name}: original inputs differ")
        p4, p8 = a["performance"]["p50_us"], b["performance"]["p50_us"]
        output.append({"case": name, **{k: a["case"][k] for k in ("m", "k", "n")},
                       "mxa8w4_p50_us": p4, "mxa8w8_p50_us": p8,
                       "w8_over_w4_speedup": p8 / p4,
                       "latency_reduction_percent": 100 * (1 - p4 / p8),
                       "scope": a["performance"]["scope"]})
    return output


def save_summary(path, report):
    report["comparisons"] = comparisons(report["results"])
    write_json(path / "summary.json", report)
    rows = report["comparisons"]
    with (path / "comparison.csv").open("w", newline="") as stream:
        names = list(rows[0]) if rows else ["case", "m", "k", "n", "mxa8w4_p50_us", "mxa8w8_p50_us",
                                          "w8_over_w4_speedup", "latency_reduction_percent", "scope"]
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def run_child(spec, directory, timeout):
    directory.mkdir(parents=True)
    write_json(directory / "spec.json", spec)
    command = [sys.executable, str(ROOT / "qbmm_case.py"), "--spec", str(directory / "spec.json")]
    child_env = os.environ.copy()
    child_env.update(FLA_NPU_DISABLE_PTH="1", TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    child_env.setdefault("ASCEND_LAUNCH_BLOCKING", "1")
    (directory / "plog").mkdir()
    child_env["ASCEND_PROCESS_LOG_PATH"] = str(directory / "plog")
    started = time.monotonic()
    with (directory / "run.log").open("w") as log:
        print(f"RUN {spec['case']['name']} {spec['kind']} -> {directory}", flush=True)
        try:
            # Dedicated process avoids NPU state leaking across quantization cases.
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=timeout, check=False, env=child_env)
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = 124
            log.write(f"\nTIMEOUT after {timeout} seconds; child killed.\n")
    result_path = directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        result = {"case": spec["case"], "kind": spec["kind"], "status": "failed",
                  "error": "Child did not produce result.json; inspect run.log"}
    result["returncode"] = code
    result["wall_seconds_including_setup"] = time.monotonic() - started
    result["directory"] = str(directory)
    if code != 0 or result.get("status") != "passed":
        result["status"] = "failed"
        print((directory / "run.log").read_text(errors="replace")[-10000:], flush=True)
    else:
        perf = result.get("performance")
        suffix = f"; QBMM p50={perf['p50_us']:.3f} us" if perf else ""
        print(f"PASS {spec['case']['name']} {spec['kind']}{suffix}", flush=True)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="logical NPU index after ASCEND_RT_VISIBLE_DEVICES")
    parser.add_argument("--tokens", type=int_list, default=[1, 16, 128, 512], help="local M, default 1,16,128,512")
    parser.add_argument("--tp-sizes", type=int_list, default=[1, 8, 16], help="shape-only TP simulation, default 1,8,16")
    parser.add_argument("--shape", nargs=3, type=positive_int, metavar=("M", "K", "N"), help="one custom shape after official smoke")
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--iterations", type=positive_int, default=50, help="profiled QBMM calls per case")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--timeout", type=positive_int, default=1200, help="seconds per isolated case")
    parser.add_argument("--output", type=Path, help="new result directory; never overwrite an existing run")
    parser.add_argument("--smoke-only", action="store_true", help="official precision smoke cases only")
    parser.add_argument("--accuracy-only", action="store_true", help="skip profiler, retain accuracy gates")
    parser.add_argument("--dry-run", action="store_true", help="print planned cases; no torch/NPU needed")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error("--device must be nonnegative")
    # Official case metadata is kept separately from the vendor golden.
    smoke = json.loads((ROOT / "official_cases.json").read_text())
    control = dict(smoke[-1], name="derived_mxa8w8_control", kind="mxa8w8",
                   stage="derived_control", weight_format="ND")
    smoke.append(control)
    try:
        shapes = glm_shapes(args.tokens, args.tp_sizes)
    except ValueError as error:
        parser.error(str(error))
    if args.shape:
        m, k, n = args.shape
        if k % 64 or n % 32:
            parser.error("custom K must be a multiple of 64 and N a multiple of 32 for this vLLM layout test")
        shapes = [{"name": f"custom_m{m}_k{k}_n{n}", "m": m, "k": k, "n": n,
                   "projection": "custom", "tp": None, "stage": "glm"}]
    plan = [(case, case["kind"]) for case in smoke]
    if not args.smoke_only:
        for i, case in enumerate(shapes):
            # Alternate order between shapes to reduce a fixed-order thermal bias.
            kinds = ("mxa8w8", "mxa8w4") if i % 2 == 0 else ("mxa8w4", "mxa8w8")
            plan.extend((case, kind) for kind in kinds)
    if args.dry_run:
        print(json.dumps([{"case": c, "kind": k} for c, k in plan], indent=2))
        return 0
    if platform.system() != "Linux":
        parser.error("A5 execution requires Linux + installed CANN/torch_npu; use --dry-run or CPU tests here")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or Path("qbmm-results") / f"{stamp}-{os.getpid()}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "description": "Isolated shared-expert QBMM; no communication or SwiGLU timed",
              "baseline": "vLLM 0.26.0 / vLLM-Ascend v0.26.0rc1", "started_utc": stamp,
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "environment": {key: os.environ.get(key) for key in ("ASCEND_LAUNCH_BLOCKING", "FLA_NPU_DISABLE_PTH",
                               "TORCH_DEVICE_BACKEND_AUTOLOAD", "ASCEND_RT_VISIBLE_DEVICES")}, "results": []}
    save_summary(output, report)
    for index, (case, kind) in enumerate(plan):
        spec = {"case": case, "kind": kind, "device": args.device, "seed": args.seed,
                "warmup": args.warmup, "iterations": args.iterations,
                "accuracy_only": args.accuracy_only or case["stage"] != "glm"}
        result = run_child(spec, output / f"{index:03d}_{case['name']}_{kind}", args.timeout)
        report["results"].append(result)
        if result["status"] != "passed":
            report["status"] = "failed"
            save_summary(output, report)
            print(f"STOP: first failure. Logs and JSON preserved in {output}", flush=True)
            return 1
        save_summary(output, report)
    report["status"] = "passed"
    save_summary(output, report)
    for row in report["comparisons"]:
        print(f"{row['case']}: W8/W4 = {row['w8_over_w4_speedup']:.3f}x (>1 means W4 faster)")
    print(f"RESULTS: {output}/summary.json and comparison.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
