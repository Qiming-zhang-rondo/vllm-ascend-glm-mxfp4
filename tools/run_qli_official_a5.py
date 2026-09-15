#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the pinned CANN QLI MXFP4 pytest cases and retain this run's plog."""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_CASES = "MXFP4_PA_20,MXFP4_META_70_002"
PERF_CASES = DEFAULT_CASES + ",FP8_PA_04,FP8_META_70_002"

PERF_BOOTSTRAP = r"""
import importlib.util, sys
spec = importlib.util.spec_from_file_location('qli_official_perf_plugin', sys.argv[1])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
import pytest
sys.exit(pytest.main(sys.argv[2:], plugins=[plugin]))
"""

PREFLIGHT = r"""
import importlib, json, sys
errors = []
modules = {}
for name in ('torch', 'torch_npu', 'numpy', 'pandas', 'pytest'):
    try:
        module = importlib.import_module(name)
        modules[name] = {'version': getattr(module, '__version__', None), 'path': module.__file__}
    except Exception as error:
        errors.append(f'{name}: {error}')
if not errors:
    import torch
    for dtype in ('float4_e2m1fn_x2', 'float8_e8m0fnu'):
        if not hasattr(torch, dtype):
            errors.append(f'Official test requires torch.{dtype}')
if not errors:
    try:
        import cann_ops_transformer
        modules['cann_ops_transformer'] = {'path': cann_ops_transformer.__file__}
        for name in ('quant_lightning_indexer', 'quant_lightning_indexer_metadata'):
            if not hasattr(torch.ops.cann_ops_transformer, name):
                errors.append(f'Official bridge did not register {name}')
    except Exception as error:
        errors.append(f'Official bridge import: {error}')
if not errors and '--perf' in sys.argv:
    import torch_npu
    for name in ('profile', 'tensorboard_trace_handler', '_ExperimentalConfig', 'ProfilerLevel', 'ExportType'):
        if not hasattr(torch_npu.profiler, name):
            errors.append(f'Performance collection requires existing torch_npu.profiler.{name}')
print(json.dumps({'modules': modules, 'errors': errors}, indent=2), flush=True)
sys.exit(2 if errors else 0)
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", help="Official STC case names; --perf defaults to the C4 cases plus matching FP8 cases"
    )
    parser.add_argument(
        "--perf", action="store_true", help="After each official accuracy check, collect QLI device-task timings"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.iters < 1:
        parser.error("--warmup and --iters must be positive")
    args.cases = args.cases or (PERF_CASES if args.perf else DEFAULT_CASES)
    repo = Path(__file__).resolve().parents[1]
    source = repo / "tools/vendor/cann_qli_v2"
    manifest_path = repo / ".qli-op-build/install.json"
    if not manifest_path.is_file():
        parser.error("Existing private QLI install.json is required; this launcher does not build an NPU operator")
    manifest = json.loads(manifest_path.read_text())
    vendor, opapi, cann = (Path(manifest[key]) for key in ("opp_root", "opapi_lib", "cann_root"))
    if not vendor.is_dir() or not opapi.is_file() or not (cann / "include").is_dir() or not (cann / "lib64").is_dir():
        parser.error("The existing manifest's operator/CANN paths are unavailable")
    missing = [name for name in (os.getenv("CXX", "c++"), "ninja") if not shutil.which(name)]
    if missing:
        parser.error(f"Official host bridge needs existing tools: {', '.join(missing)}. Nothing was installed.")

    run_dir = repo / ".qli-official" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    plog_dir = run_dir / "plog"
    plog_dir.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(
        {
            "ASCEND_HOME_PATH": str(cann),
            "ASCEND_OPP_PATH": str(cann / "opp"),
            "ASCEND_CUSTOM_OPP_PATH": str(vendor),
            "ASCEND_PROCESS_LOG_PATH": str(plog_dir),
            "ASCEND_GLOBAL_LOG_LEVEL": "3" if args.perf else "0",
            "ASCEND_SLOG_PRINT_TO_STDOUT": "0",
            "ASCEND_LAUNCH_BLOCKING": "0" if args.perf else "1",
            "TORCH_EXTENSIONS_DIR": str(repo / ".qli-official/torch_extensions"),
            "MAX_JOBS": environment.get("MAX_JOBS", "2"),
            "PYTHONPATH": os.pathsep.join([str(source), environment.get("PYTHONPATH", "")]).rstrip(os.pathsep),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTEST_ADDOPTS": "",
            "QLIV2_PARAMSET": "stc",
            "QLIV2_CASE_NAMES": args.cases,
            "QLIV2_RUN_MODE": "eager",
            "QLIV2_SINGLE_RESULT_PATH": "",
            "QLIV2_SINGLE_SAVE_PT_DIR": "",
        }
    )
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(opapi.parent), str(cann / "lib64"), environment.get("LD_LIBRARY_PATH", "")]
    ).rstrip(os.pathsep)
    if args.perf:
        profiler_paths = (cann / "tools/profiler/bin/msprof", cann / "toolkit/tools/profiler/bin/msprof")
        msprof = next(
            (str(path) for path in profiler_paths if path.is_file() and os.access(path, os.X_OK)),
            shutil.which("msprof"),
        )
        if not msprof:
            parser.error("--perf requires the container's CANN msprof export tool; no dependencies were installed")
        environment["PATH"] = os.pathsep.join([str(Path(msprof).parent), environment.get("PATH", "")]).rstrip(
            os.pathsep
        )
        print("Reuse CANN profiler:", msprof, flush=True)
    print("Official CANN cases:", args.cases, flush=True)
    print("Reuse NPU operator:", opapi, flush=True)
    print(
        "No pip, dependency downloads or NPU kernel rebuild. First run JIT-builds the official C++ host bridge.",
        flush=True,
    )
    checked = subprocess.run(
        [sys.executable, "-c", PREFLIGHT] + (["--perf"] if args.perf else []),
        cwd=source,
        env=environment,
        capture_output=True,
        text=True,
    )
    (run_dir / "preflight.log").write_text(checked.stdout + checked.stderr)
    print(checked.stdout + checked.stderr, end="", flush=True)
    if checked.returncode:
        print(f"Official test prerequisites are missing; see {run_dir / 'preflight.log'}. Nothing was installed.")
        return checked.returncode

    test_dir = source / "pytest"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        str(test_dir / "pytest.ini"),
        "-x",
        "-vv",
        "-s",
        "-m",
        "ci",
        str(test_dir / "test_quant_lightning_indexer_v2_single.py"),
    ]
    if args.perf:
        command = (
            [sys.executable, "-c", PERF_BOOTSTRAP, str(repo / "tools/qli_official_perf_plugin.py")]
            + command[3:]
            + [
                "--qli-perf-output",
                str(run_dir / "performance.json"),
                "--qli-perf-warmup",
                str(args.warmup),
                "--qli-perf-iters",
                str(args.iters),
            ]
        )
        print(
            "Performance: official accuracy first, then warmed QLI device-task duration from CANN profiler (us).",
            flush=True,
        )
    log = run_dir / "official.log"
    print("Run output:", log, flush=True)
    with log.open("w") as stream:
        process = subprocess.Popen(
            command, cwd=test_dir, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "pid": process.pid,
                    "command": command,
                    "cases": args.cases,
                    "opapi_lib": str(opapi),
                    "plog_dir": str(plog_dir),
                },
                indent=2,
            )
            + "\n"
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        status = process.wait()
    print(f"Official pytest exit={status}; log={log}; plog={plog_dir}", flush=True)
    if status:
        diagnostic = run_dir / "diagnostic.json"
        result = subprocess.run(
            [
                sys.executable,
                str(repo / "tools/collect_qli_timeout.py"),
                "--log",
                str(log),
                "--plog-dir",
                str(plog_dir),
                "--pid",
                str(process.pid),
                "--output",
                str(diagnostic),
            ],
            capture_output=True,
            text=True,
        )
        (run_dir / "collection.log").write_text(result.stdout + result.stderr)
        print(
            f"Failure details: {diagnostic}"
            if diagnostic.is_file()
            else f"Collection log: {run_dir / 'collection.log'}"
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
