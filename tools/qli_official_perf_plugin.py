# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time the exact QLI call after its unmodified upstream pytest check passes."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest


def pytest_addoption(parser):
    parser.addoption("--qli-perf-output")
    parser.addoption("--qli-perf-warmup", type=int, default=5)
    parser.addoption("--qli-perf-iters", type=int, default=20)


def save_report(config):
    report = config._qli_perf_report
    pairs = (("FP8_PA_04", "MXFP4_PA_20"), ("FP8_META_70_002", "MXFP4_META_70_002"))
    comparisons = []
    for fp8_name, fp4_name in pairs:
        fp8, fp4 = (report["cases"].get(name, {}) for name in (fp8_name, fp4_name))
        if fp8.get("status") != "passed" or fp4.get("status") != "passed":
            continue
        if fp8["shape"] != fp4["shape"]:
            continue
        comparisons.append(
            {
                "fp8_case": fp8_name,
                "mxfp4_case": fp4_name,
                "fp8_p50_us": fp8["performance"]["p50_us"],
                "mxfp4_p50_us": fp4["performance"]["p50_us"],
                "fp8_over_mxfp4_p50": fp8["performance"]["p50_us"] / fp4["performance"]["p50_us"],
            }
        )
    report["comparisons"] = comparisons
    path = Path(config.getoption("--qli-perf-output"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def pytest_configure(config):
    if not config.getoption("--qli-perf-output"):
        raise pytest.UsageError("--qli-perf-output is required for QLI profiling")
    config._qli_perf_report = {
        "status": "running",
        "scope": "Official synthetic accuracy; QLI device-task durations from CANN op_summary, in microseconds",
        "note": "Same shapes across quantization modes; independently generated official inputs, not model accuracy",
        "cases": {},
    }
    save_report(config)


def describe_argument(value):
    """Read tensor descriptors only: no device copies, item(), or synchronization."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "stride": list(value.stride()),
            "storage_offset": value.storage_offset(),
        }
    return value


def profile_call(op, args, kwargs, trace_dir, warmup, iterations):
    import torch
    import torch_npu

    for _ in range(warmup):
        outputs = op(*args, **kwargs)
    torch.npu.synchronize()
    profiler = torch_npu.profiler
    experimental = profiler._ExperimentalConfig(
        profiler_level=profiler.ProfilerLevel.Level1,
        export_type=profiler.ExportType.Text,
        data_simplification=False,
    )
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        experimental_config=experimental,
        on_trace_ready=profiler.tensorboard_trace_handler(str(trace_dir), async_mode=False),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ):
        for _ in range(iterations):
            outputs = op(*args, **kwargs)
        torch.npu.synchronize()
    del outputs
    # Load this sibling without adding tools/ (which contains bisect/) to sys.path.
    spec = importlib.util.spec_from_file_location(
        "qli_perf_summary", Path(__file__).with_name("qli_official_perf_summary.py")
    )
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    return summary.summarize_case(trace_dir, iterations)


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if pyfuncitem.module.__name__ != "test_quant_lightning_indexer_v2_single":
        return None
    import torch

    namespace = torch.ops.cann_ops_transformer
    op = namespace.quant_lightning_indexer
    captured = []

    def capture(*args, **kwargs):
        entry["compute_arguments"] = {
            "scope": "Actual arguments before QLI launch; tensor descriptors only, payload values not captured",
            "positional": [describe_argument(value) for value in args],
            "keyword": {key: describe_argument(value) for key, value in kwargs.items()},
        }
        save_report(pyfuncitem.config)
        outputs = op(*args, **kwargs)
        captured.append((args, kwargs))
        return outputs

    params = pyfuncitem.funcargs["param_combinations"]
    name = params["case_name"]
    config = pyfuncitem.config
    shape_keys = (
        "batch_size",
        "q_seq",
        "k_seq",
        "q_t_size",
        "k_t_size",
        "q_head_num",
        "k_head_num",
        "head_dim",
        "block_size",
        "block_num",
        "layout_query",
        "layout_key",
        "sparse_count",
        "sparse_mode",
        "cmp_ratio",
        "max_seqlen_q",
        "return_value",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "seqused_q",
        "seqused_k",
        "cmp_residual_k",
        "output_idx_offset",
    )
    trace_dir = Path(config.getoption("--qli-perf-output")).parent / "profiler" / name
    entry = {
        "status": "accuracy_running",
        "official_accuracy_passed": False,
        "quant_mode": params["quant_mode"],
        "shape": {key: params.get(key) for key in shape_keys},
        "trace_dir": str(trace_dir),
        "warmup": config.getoption("--qli-perf-warmup"),
        "iterations": config.getoption("--qli-perf-iters"),
    }
    config._qli_perf_report["cases"][name] = entry
    save_report(config)
    print(f"\nSTAGE: official accuracy: {name}", flush=True)
    # Upstream still generates inputs, invokes the original C++ bridge, and
    # performs all its original accuracy assertions before profiling can begin.
    try:
        with patch.object(namespace, "quant_lightning_indexer", capture):
            pyfuncitem.obj(param_combinations=params)
        if len(captured) != 1:
            raise RuntimeError(f"Expected one official QLI compute call, captured {len(captured)}")
    except Exception as error:
        entry.update(status="accuracy_failed", error=f"{type(error).__name__}: {error}")
        save_report(config)
        raise
    entry.update(status="accuracy_passed", official_accuracy_passed=True)
    save_report(config)
    print(
        f"\nPERF BEGIN: {name}; official accuracy passed; warmup={entry['warmup']}, samples={entry['iterations']}",
        flush=True,
    )
    try:
        args, kwargs = captured[0]
        entry["performance"] = profile_call(op, args, kwargs, trace_dir, entry["warmup"], entry["iterations"])
        entry["status"] = "passed"
    except Exception as error:
        entry.update(status="performance_failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_report(config)
    perf = entry["performance"]
    print(
        f"PERF {name}: QLI task p50={perf['p50_us']:.3f} us, "
        f"p90={perf['p90_us']:.3f} us, mean={perf['mean_us']:.3f} us",
        flush=True,
    )
    return True


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    config._qli_perf_report["status"] = "passed" if exitstatus == 0 else "failed"
    save_report(config)
    for comparison in config._qli_perf_report["comparisons"]:
        print(
            f"\nCOMPARE {comparison['fp8_case']} / {comparison['mxfp4_case']}: "
            f"FP8/C4 p50 = {comparison['fp8_over_mxfp4_p50']:.3f}x (>1 means C4 is faster)"
        )
    print(f"\nAccuracy/performance report: {config.getoption('--qli-perf-output')}")
