#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read existing QLI failure logs and kernel disassembly; never launch NPU work."""

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


def read_json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def failure_offsets(text):
    entries = re.findall(r"(aicore|aivec) error exception.*?pc start:\s*(0x[0-9a-f]+), current:\s*(0x[0-9a-f]+)", text)
    result = {}
    for kind, start, current in entries:
        result.setdefault("mix_aic" if kind == "aicore" else "mix_aiv", set()).add(int(current, 16) - int(start, 16))
    return {kind: sorted(values) for kind, values in result.items()}


def instruction_windows(disassembly, offsets):
    lines = disassembly.splitlines()
    symbol = None
    targets = []
    windows = []
    for index, line in enumerate(lines):
        label = re.match(r"\s*([0-9a-fA-F]+)\s+<(.+)>:\s*$", line)
        if label:
            base, symbol = int(label[1], 16), label[2]
            targets = [base + offset for kind, values in offsets.items() if kind in symbol for offset in values]
        address = re.match(r"\s*([0-9a-fA-F]+):", line)
        if address and int(address[1], 16) in targets:
            windows.append(
                {"symbol": symbol, "address": address[1], "instructions": lines[max(0, index - 8) : index + 9]}
            )
    return windows


def find_objdump(cann_root):
    if cann_root and Path(cann_root).is_dir():
        for candidate in Path(cann_root).rglob("llvm-objdump"):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return shutil.which("llvm-objdump")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log", type=Path, help="Read this exact run log instead of the benchmark probe")
    parser.add_argument("--plog-dir", type=Path, help="Plog directory belonging to that run")
    parser.add_argument("--pid", type=int, help="Known test process PID")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    probe_paths = [repo / ".qli-op-build" / name for name in ("probe-installed.json", "probe-custom.json")]
    probe_paths = [path for path in probe_paths if path.is_file()]
    probe_path = max(probe_paths, key=lambda path: path.stat().st_mtime) if probe_paths else None
    probe = read_json(probe_path) if probe_path and not args.log else {}
    manifest = read_json(repo / ".qli-op-build/install.json")
    error_text = probe.get("error", {}).get("message", "")
    # Never combine an older run's PID/kernel name with the current probe.
    logs = [args.log] if args.log else []
    if not logs and not failure_offsets(error_text):
        probe_pids = set(re.findall(r"PID[:=]\s*(\d+)", error_text))
        for candidate in sorted(repo.glob("qli_a5_*.log"), key=lambda path: path.stat().st_mtime, reverse=True):
            if probe_pids and any(pid in candidate.read_text(errors="replace") for pid in probe_pids):
                logs = [candidate]
                break
    if logs:
        latest = logs[0]
        with latest.open("rb") as stream:
            stream.seek(max(0, latest.stat().st_size - 256 * 1024))
            error_text += "\n" + stream.read().decode(errors="replace")
    offsets = failure_offsets(error_text)
    pids = set(re.findall(r"PID[:=]\s*(\d+)", error_text))
    if args.pid is not None:
        pids.add(str(args.pid))
    report = {
        "scope": "Read-only diagnostics: no torch/torch_npu import, NPU launch, build, install or download",
        "probe_path": str(probe_path) if probe else None,
        "source_log": str(logs[0]) if logs else None,
        "probe": {key: probe.get(key) for key in ("status", "configuration", "environment", "backend", "stage")},
        "build_manifest": manifest,
        "pc_offsets": {kind: [hex(value) for value in values] for kind, values in offsets.items()},
        "plog": [],
        "kernels": [],
    }
    report["error"] = {key: probe.get("error", {}).get(key) for key in ("type", "message")}
    if report["error"]["message"]:
        report["error"]["message"] = report["error"]["message"].splitlines()[0]
    smi = shutil.which("npu-smi")
    if smi:
        try:
            result = subprocess.run([smi, "info"], capture_output=True, text=True, timeout=10)
            report["npu_smi"] = {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            report["npu_smi"] = {"error": str(error)}
    roots = {args.plog_dir} if args.plog_dir else {Path.home() / "ascend/log"}
    if not args.plog_dir and os.environ.get("ASCEND_PROCESS_LOG_PATH"):
        roots.add(Path(os.environ["ASCEND_PROCESS_LOG_PATH"]))
    log_paths = set()
    for root in roots:
        if root.is_dir():
            for pid in pids:
                log_paths.update(path for path in root.rglob(f"*{pid}*.log*") if path.is_file())
    fault_names = set()
    for path in sorted(log_paths):
        try:
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 2 * 1024 * 1024))
                lines = stream.read().decode(errors="replace").splitlines()
            selected = [
                line for line in lines if re.search(r"kernel_name|kernelName|tiling.?key|block.?dim", line, re.I)
            ]
            fault_names.update(re.findall(r"fault kernel_name\s*[=:]\s*([^,\s]+)", "\n".join(selected)))
            errors = [line for line in lines if re.search(r"\[ERROR\]|errorCode|errcode|timeout|exception", line, re.I)]
            report["plog"].append({"path": str(path), "lines": selected[-60:], "errors": errors[-60:]})
        except OSError as error:
            report["plog"].append({"path": str(path), "error": str(error)})
    report["fault_kernel_names"] = sorted(fault_names)

    # Only inspect the private package referenced by the existing build manifest.
    vendor = Path(manifest.get("opp_root", "/nonexistent"))
    candidates = []
    fp4_candidates = []
    if vendor.is_dir():
        for path in vendor.rglob("*.json"):
            config = read_json(path)
            name = config.get("binFileName", "")
            if "quantlightningindexerv2" not in name.lower().replace("_", "") or "metadata" in name.lower():
                continue
            binary = path.parent / (Path(name).name + config.get("binFileSuffix", ".o"))
            if not binary.is_file():
                continue
            support = json.dumps(config.get("supportInfo", {})).lower()
            if "float4" in support or "e2m1" in support:
                fp4_candidates.append((binary, config))
            if fault_names:
                selected = ("float4" in support or "e2m1" in support) and any(
                    name in fault or fault in name for fault in fault_names
                )
            else:
                selected = "float4" in support or "e2m1" in support
            if selected:
                candidates.append((binary, config))
    if not candidates:
        candidates = fp4_candidates
    report["package_matches_probe_api"] = (
        probe.get("backend", {}).get("api_libraries", {}).get("aclnnQuantLightningIndexerV2")
        == manifest.get("opapi_lib")
        if probe
        else None
    )
    objdump = find_objdump(manifest.get("cann_root") or os.environ.get("ASCEND_HOME_PATH")) if offsets else None
    report["objdump"] = objdump
    report["candidate_count"] = len(candidates)
    deadline = time.monotonic() + 45
    for binary, config in candidates[:16]:
        entry = {
            "binary": str(binary),
            "binFileName": config.get("binFileName"),
            "coreType": config.get("coreType"),
            "taskRation": config.get("taskRation"),
            "supportInfo": config.get("supportInfo"),
        }
        report["kernels"].append(entry)
        if not offsets:
            entry["disassembly_note"] = "No fault PC in this run's log"
            continue
        if not objdump or time.monotonic() >= deadline:
            entry["disassembly_note"] = "objdump unavailable or collection time budget reached"
            continue
        try:
            completed = subprocess.run([objdump, "-d", str(binary)], capture_output=True, text=True, timeout=10)
            entry["objdump_exit_code"] = completed.returncode
            entry["instruction_windows"] = instruction_windows(completed.stdout, offsets)
            if not entry["instruction_windows"]:
                entry["symbols"] = re.findall(r"^\s*[0-9a-fA-F]+\s+<(.+)>:$", completed.stdout, re.M)[:24]
            if completed.stderr:
                entry["objdump_stderr"] = completed.stderr[-2000:]
        except (OSError, subprocess.TimeoutExpired) as error:
            entry["disassembly_error"] = str(error)
    output = args.output or repo / "qli_timeout_diagnostic.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
