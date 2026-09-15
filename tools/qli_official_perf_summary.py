# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize QLI V2 compute durations from one official case's profiler trace."""

import csv
import math
import re
from pathlib import Path

COMPUTE_OP = "QuantLightningIndexerV2"
COMPUTE_NAME = "quantlightningindexerv2"


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _is_compute(row: dict, type_column: str | None, name_column: str | None) -> bool:
    if type_column is not None:
        return (row.get(type_column) or "").strip() == COMPUTE_OP
    name = _normalized(row.get(name_column) or "")
    if not name.startswith(COMPUTE_NAME):
        return False
    suffix = name[len(COMPUTE_NAME) :]
    return not suffix.startswith(("metadata", "copy", "memcpy", "viewcopy", "contiguous"))


def _percentile(ordered: list[float], fraction: float) -> float:
    """Linearly interpolate at (sample_count - 1) * fraction."""
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_case(trace_dir: str | Path, expected_calls: int) -> dict:
    """Read every op_summary CSV and require exactly the expected compute calls.

    All failures leave profiler files untouched. Percentiles use linear
    interpolation; the input samples retain CSV file and row order.
    """
    trace = Path(trace_dir).resolve()
    if isinstance(expected_calls, bool) or not isinstance(expected_calls, int) or expected_calls <= 0:
        raise ValueError("expected_calls must be a positive integer")
    files = sorted(path for path in trace.rglob("op_summary*.csv") if path.is_file())
    if not files:
        raise ValueError(f"No op_summary*.csv found in profiler trace: {trace}")

    samples = []
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns = {_normalized(name): name for name in reader.fieldnames or []}
            type_column = columns.get("optype")
            name_column = columns.get("opname")
            duration_column = columns.get("taskdurationus")
            if (type_column is None and name_column is None) or duration_column is None:
                raise ValueError(f"Profiler CSV needs Op Type or Op Name, and Task Duration(us): {path}")
            for row_number, row in enumerate(reader, start=2):
                if not _is_compute(row, type_column, name_column):
                    continue
                raw = row.get(duration_column)
                try:
                    duration = float(raw)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Invalid QLI compute Task Duration(us) {raw!r} at {path}:{row_number}") from error
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError(f"Invalid QLI compute Task Duration(us) {raw!r} at {path}:{row_number}")
                samples.append(duration)

    if len(samples) != expected_calls:
        raise ValueError(
            f"Expected {expected_calls} {COMPUTE_OP} compute samples, found {len(samples)} "
            f"in {trace}; profiler trace retained"
        )
    ordered = sorted(samples)
    return {
        "samples_us": samples,
        "p50_us": _percentile(ordered, 0.5),
        "p90_us": _percentile(ordered, 0.9),
        "mean_us": math.fsum(samples) / len(samples),
        "min_us": ordered[0],
        "max_us": ordered[-1],
        "sample_count": len(samples),
        "expected_calls": expected_calls,
        "scope": "QuantLightningIndexerV2 compute Task Duration(us); metadata and copies excluded",
        "files": [str(path) for path in files],
    }
