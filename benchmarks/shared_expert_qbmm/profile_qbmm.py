"""Profile one prepared QBMM call using CANN device-task durations.

This module deliberately imports only the standard library until profile_call
is invoked, so the CSV validation can run on machines without Ascend hardware.
It never changes ASCEND_LAUNCH_BLOCKING or other environment settings.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path


QBMM_OP_TYPES = frozenset({"QuantBatchMatmulV3", "QuantBatchMatmulV4"})
# These are memory operations, not additional compute kernels. Keep their counts
# in the report. Do not silently discard unrecognized compute or split kernels.
MEMORY_OP_TYPES = frozenset({"Memcpy", "MemCpy", "Memcopy", "Memset", "MemSet"})


class ProfileValidationError(ValueError):
    """An ambiguous or invalid trace, with structured diagnostics attached."""

    def __init__(self, message: str, diagnostics: dict):
        self.diagnostics = diagnostics
        super().__init__(
            f"{message}; observed Op Type counts={diagnostics.get('observed_op_types', {})}; "
            f"trace retained at {diagnostics.get('trace_dir')}"
        )


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _integer(value: int, name: str, *, positive: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")


def _percentile(ordered: list[float], fraction: float) -> float:
    """Linear interpolation at (n - 1) * fraction, including n=1."""
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_trace(trace_dir: str | Path, expected_calls: int) -> dict:
    """Require one consistent QBMM task type and exactly expected_calls tasks.

No durations are added together. A split/renamed operator, extra compute kernel,
mixed V3/V4 trace, missing duration, and stale/duplicated data all fail closed.
The count establishes the expected one-task-per-call aggregate in this isolated
trace; it is not a correlation-ID-based reconstruction of individual host calls.
"""
    _integer(expected_calls, "expected_calls", positive=True)
    trace = Path(trace_dir).resolve()
    files = sorted(path for path in trace.rglob("op_summary*.csv") if path.is_file())
    diagnostics = {
        "trace_dir": str(trace),
        "source_paths": [str(path) for path in files],
        "expected_calls": expected_calls,
        "observed_op_types": {},
    }
    if not files:
        raise ProfileValidationError("No CANN op_summary*.csv found", diagnostics)

    counts: Counter[str] = Counter()
    samples: list[dict] = []
    other_compute_rows: list[dict] = []
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns = {_normalized(name): name for name in reader.fieldnames or []}
            type_column = columns.get("optype")
            duration_column = columns.get("taskdurationus")
            if type_column is None or duration_column is None:
                raise ProfileValidationError(
                    f"{path} requires Op Type and Task Duration(us) columns; "
                    "operator_details/host duration and Op Name guessing are unsupported",
                    diagnostics,
                )
            for row_number, row in enumerate(reader, start=2):
                op_type = (row.get(type_column) or "").strip()
                counts[op_type or "<missing Op Type>"] += 1
                diagnostics["observed_op_types"] = dict(sorted(counts.items()))
                if op_type not in QBMM_OP_TYPES:
                    if op_type not in MEMORY_OP_TYPES:
                        other_compute_rows.append(
                            {"source_path": str(path), "csv_row": row_number, "op_type": op_type}
                        )
                    continue
                raw = row.get(duration_column)
                try:
                    duration = float(raw)
                except (TypeError, ValueError) as error:
                    raise ProfileValidationError(
                        f"Invalid QBMM Task Duration(us) {raw!r} at {path}:{row_number}",
                        diagnostics,
                    ) from error
                if not math.isfinite(duration) or duration <= 0:
                    raise ProfileValidationError(
                        f"Invalid QBMM Task Duration(us) {raw!r} at {path}:{row_number}",
                        diagnostics,
                    )
                samples.append(
                    {
                        "duration_us": duration,
                        "op_type": op_type,
                        "source_path": str(path),
                        "csv_row": row_number,
                    }
                )

    diagnostics["qbmm_task_count"] = len(samples)
    diagnostics["unrecognized_device_rows"] = other_compute_rows
    if other_compute_rows:
        raise ProfileValidationError(
            "Unrecognized extra device operation(s): cannot establish a single QBMM task per call; "
            "inspect the raw trace for preprocessing, renamed operators, or split kernels",
            diagnostics,
        )
    actual_types = sorted({sample["op_type"] for sample in samples})
    if len(actual_types) != 1:
        raise ProfileValidationError(
            f"Expected exactly one of {sorted(QBMM_OP_TYPES)}, found {actual_types}", diagnostics
        )
    if len(samples) != expected_calls:
        raise ProfileValidationError(
            f"Expected exactly {expected_calls} QBMM tasks for {expected_calls} calls, "
            f"found {len(samples)}; duplicated, missing, or split task data cannot be summed",
            diagnostics,
        )

    values = [sample["duration_us"] for sample in samples]
    ordered = sorted(values)
    return {
        "scope": (
            "CANN QBMM device Task Duration(us), one observed QBMM task per prepared call "
            "by aggregate count; excludes input/weight quantization, format conversion, "
            "host launch/wait time, and explicit memory operations. "
            "Synthetic standalone kernel timing, not end-to-end shared-expert latency."
        ),
        "metric": "CANN op_summary Task Duration(us)",
        "actual_op_types": actual_types,
        "observed_op_types": dict(sorted(counts.items())),
        "excluded_memory_op_types": {
            name: count for name, count in sorted(counts.items()) if name in MEMORY_OP_TYPES
        },
        "trace_dir": str(trace),
        "source_paths": [str(path) for path in files],
        "expected_calls": expected_calls,
        "sample_count": len(values),
        "samples_us": values,
        "raw_samples": samples,
        "p50_us": _percentile(ordered, 0.5),
        "p90_us": _percentile(ordered, 0.9),
        "mean_us": math.fsum(values) / len(values),
        "min_us": ordered[0],
        "max_us": ordered[-1],
        "percentile_method": "linear interpolation at (sample_count - 1) * fraction",
    }


def profile_call(call, trace_dir: str | Path, warmup: int, iterations: int) -> dict:
    """Warm a zero-argument prepared QBMM callable, then profile only its calls.

    The caller must complete correctness checks first and prepare tensors,
    quantization, and format casts before entering this function. Launch blocking
    is preserved. This function does not use wall-clock/Event timing or graphs.
    """
    _integer(warmup, "warmup", positive=False)
    _integer(iterations, "iterations", positive=True)
    if not callable(call):
        raise TypeError("call must be a zero-argument callable")
    trace = Path(trace_dir).resolve()
    if trace.exists() and (not trace.is_dir() or any(trace.iterdir())):
        raise ValueError(f"Profiler trace directory must be new or empty: {trace}")

    import torch
    import torch_npu

    trace.mkdir(parents=True, exist_ok=True)
    for _ in range(warmup):
        output = call()
    torch.npu.synchronize()
    profiler = torch_npu.profiler
    experimental = profiler._ExperimentalConfig(
        profiler_level=profiler.ProfilerLevel.Level0,
        export_type=[profiler.ExportType.Text],
        data_simplification=False,
    )
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        experimental_config=experimental,
        on_trace_ready=profiler.tensorboard_trace_handler(str(trace), async_mode=False),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ):
        for _ in range(iterations):
            output = call()
        torch.npu.synchronize()
    del output
    result = summarize_trace(trace, iterations)
    result["warmup"] = warmup
    result["profiler"] = {
        "level": "Level0",
        "export_type": "Text",
        "data_simplification": False,
        "activities": ["CPU", "NPU"],
        "async_export": False,
    }
    return result
