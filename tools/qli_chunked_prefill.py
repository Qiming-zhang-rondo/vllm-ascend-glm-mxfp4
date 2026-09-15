# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compute full QLI prefill chunks while sampling only the CPU score reference."""

import json
import math
from functools import partial
from pathlib import Path

import torch

TOPK = 2048
HEAD_DIM = 128
BLOCK_SIZE = 128
MODES = (("MXFP4", 5), ("FP8", 1))


def select_reference_rows(query_start, query_tokens, reference_rows):
    count = min(reference_rows, query_tokens)
    if count < 1:
        raise ValueError("At least one reference row is required")
    candidates = [0, query_tokens - 1]
    candidates += [index - query_start for index in (TOPK - 2, TOPK - 1, TOPK)]
    candidates += torch.linspace(0, query_tokens - 1, steps=count, device="cpu").round().long().tolist()
    selected = []
    for index in candidates:
        if 0 <= index < query_tokens and index not in selected:
            selected.append(index)
            if len(selected) == count:
                break
    return sorted(selected)


def validate_outputs(indices, values, query_start, query_tokens):
    expected_shape = (query_tokens, 1, TOPK)
    if tuple(indices.shape) != expected_shape or indices.dtype != torch.int32:
        raise AssertionError(f"Unexpected indices: {tuple(indices.shape)} {indices.dtype}")
    if values is not None and (tuple(values.shape) != expected_shape or values.dtype != torch.bfloat16):
        raise AssertionError(f"Unexpected values: {tuple(values.shape)} {values.dtype}")
    # Bound CPU temporaries while checking every output row, including padding.
    for first in range(0, query_tokens, 256):
        rows = indices[first : first + 256, 0]
        lengths = torch.arange(query_start + first + 1, query_start + first + 1 + len(rows), device="cpu").unsqueeze(1)
        valid = rows >= 0
        if bool((rows < -1).any()):
            raise AssertionError("Only -1 is permitted as a padding index")
        if not torch.equal(valid.sum(dim=1), lengths[:, 0].clamp_max(TOPK)):
            raise AssertionError("Incorrect sparse index count for the global causal position")
        if bool((valid & (rows >= lengths)).any()):
            raise AssertionError("Sparse index exceeds the global causal length")
        ordered = rows.sort(dim=1).values
        if bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()):
            raise AssertionError("Duplicate valid sparse indices")
        if values is not None:
            row_values = values[first : first + 256, 0]
            if not bool(torch.isfinite(row_values[valid]).all()):
                raise AssertionError("Nonfinite value at a valid sparse index")
            if not bool(torch.isneginf(row_values[~valid]).all()):
                raise AssertionError("Padding sparse values must be negative infinity")


def selection_above_tolerated_cutoff(indices, scores):
    if indices.ndim != 3 or scores.ndim != 2 or indices.shape[0] != scores.shape[0] or not indices.shape[0]:
        raise AssertionError("Sampled indices and reference scores must cover the same nonempty query rows")
    passed = True
    for row_indices, row_scores in zip(indices[:, 0], scores):
        actual = row_indices[row_indices >= 0].long()
        selected = row_scores[actual]
        if not actual.numel() or not bool(torch.isfinite(selected).all()):
            raise AssertionError("Empty selection or nonfinite selected reference score")
        cutoff = row_scores.topk(actual.numel()).values[-1]
        if not bool(torch.isfinite(cutoff)):
            raise AssertionError("Reference contains fewer valid causal scores than returned indices")
        passed = passed and bool((selected >= cutoff - (cutoff.abs() * 0.02 + 0.02)).all())
    return passed


def accuracy_metrics(indices, values, scores):
    """Compare valid top-k entries, averaging scalar metrics over sampled rows."""
    if indices.ndim != 3 or indices.shape != values.shape or scores.ndim != 2 or scores.shape[0] != indices.shape[0]:
        raise AssertionError("Sampled outputs and reference scores must cover the same query rows")
    recalls, cosines, nmaes = [], [], []
    close, above_cutoff = True, True
    for row_indices, row_values, row_scores in zip(indices[:, 0], values[:, 0], scores):
        valid = row_indices >= 0
        actual = row_indices[valid].long()
        actual_values = row_values[valid].float()
        if not actual.numel():
            raise AssertionError("A nonempty causal query must have valid sparse outputs")
        selected = row_scores[actual]
        if not bool(torch.isfinite(actual_values).all()) or not bool(torch.isfinite(selected).all()):
            raise AssertionError("Nonfinite sampled output or selected reference score")
        expected = row_scores.topk(actual.numel())
        if not bool(torch.isfinite(expected.values).all()):
            raise AssertionError("Reference contains fewer valid causal scores than returned indices")
        recalls.append(torch.isin(actual, expected.indices).float().mean().item())
        cosine = torch.nn.functional.cosine_similarity(actual_values, selected, dim=0).item()
        if not bool(actual_values.any()) and not bool(selected.any()):
            cosine = 1.0
        cosines.append(cosine)
        nmaes.append(((actual_values - selected).abs().mean() / selected.abs().mean().clamp_min(1e-12)).item())
        close = close and bool(torch.isclose(actual_values, selected, rtol=0.02, atol=0.02).all())
        cutoff = expected.values[-1]
        above_cutoff = above_cutoff and bool((selected >= cutoff - (cutoff.abs() * 0.02 + 0.02)).all())
    if not recalls or not all(math.isfinite(value) for value in recalls + cosines + nmaes):
        raise AssertionError("Empty or nonfinite sampled accuracy metrics")
    return {
        "topk_recall": sum(recalls) / len(recalls),
        "score_cosine": sum(cosines) / len(cosines),
        "score_nmae": sum(nmaes) / len(nmaes),
        "selected_scores_close": close,
        "selection_above_tolerated_cutoff": above_cutoff,
        "aggregation": "arithmetic mean of per-row metrics over sampled rows; padding excluded",
    }


def _save_report(args, report):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + "\n")


def run_prefill(args, report, backend, device, *, prepare_case, reference_scores, benchmark_compute):
    report.update(
        status="running",
        scope="Full QLI compute for each prefill chunk; sampled CPU score accuracy, not model accuracy",
        timing_scope=(
            "Synchronous QLI call wall latency including ACLNN preparation, allocation, dispatch and synchronization; "
            "input preparation, transfer, quantization and metadata excluded. No cache-write or NPU quantizer test."
        ),
        prefill_tokens=args.prefill_tokens,
        chunk_size=args.chunk_size,
        reference_rows=args.reference_rows,
        chunks=[],
        cases={name: {"sum_chunk_mean_wall_ms": 0.0, "completed_chunks": 0} for name, _ in MODES},
    )
    backend_report = report.setdefault("backend", {})
    backend_report.update(mxfp4_compute_verified=False, mxfp4_numerical_checks_passed=False)
    _save_report(args, report)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    amplitude = torch.tensor([0.125, 0.5, 2.0, 8.0], dtype=torch.float16, device="cpu").repeat_interleave(32)
    key_source = torch.randn(args.prefill_tokens, 1, HEAD_DIM, dtype=torch.float16, generator=generator, device="cpu")
    key_source.mul_(amplitude.flip(0))
    try:
        for query_start in range(0, args.prefill_tokens, args.chunk_size):
            query_tokens = min(args.chunk_size, args.prefill_tokens - query_start)
            key_tokens = query_start + query_tokens
            local_rows = select_reference_rows(query_start, query_tokens, args.reference_rows)
            global_rows = [query_start + row for row in local_rows]
            row_tensor = torch.tensor(local_rows, dtype=torch.long, device="cpu")
            causal_lengths = torch.tensor(global_rows, dtype=torch.long, device="cpu") + 1
            chunk = {
                "index": len(report["chunks"]) + 1,
                "query_start": query_start,
                "query_tokens": query_tokens,
                "key_tokens": key_tokens,
                "reference_local_rows": local_rows,
                "reference_global_rows": global_rows,
                "status": "running",
                "cases": {},
            }
            report["chunks"].append(chunk)
            report["stage"] = f"Chunk {chunk['index']}: CPU source and sampled reference"
            _save_report(args, report)
            print(
                f"PREFILL chunk {chunk['index']}: Q={query_tokens}, K={key_tokens}, sampled rows={global_rows}",
                flush=True,
            )
            query_source = torch.randn(
                query_tokens, args.heads, HEAD_DIM, dtype=torch.float16, generator=generator, device="cpu"
            )
            query_source.mul_(amplitude)
            weights_source = torch.rand(
                query_tokens, args.heads, dtype=torch.float32, generator=generator, device="cpu"
            )
            weights_source.mul_(1.5).add_(0.25)
            prefix_key = key_source[:key_tokens]
            sample_weights = weights_source[row_tensor]
            original_scores = reference_scores(
                query_source[row_tensor], prefix_key, sample_weights, causal_lengths=causal_lengths
            )
            weights = weights_source.to(device)
            cu_q = torch.tensor([0, query_tokens], dtype=torch.int32, device="cpu").to(device)
            seq_k = torch.tensor([key_tokens], dtype=torch.int32, device="cpu").to(device)
            blocks = torch.arange(key_tokens // BLOCK_SIZE, dtype=torch.int32, device="cpu").view(1, -1).to(device)
            for name, quant_mode in MODES:
                entry = chunk["cases"][name] = {"status": "running", "quant_mode": quant_mode}
                report["stage"] = f"Chunk {chunk['index']} {name}: prepare inputs and metadata"
                case = prepare_case(
                    name, quant_mode, query_source, prefix_key, device, backend, cu_q, seq_k, report=report
                )

                invoke = partial(
                    backend.invoke,
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
                    return_value=0,
                )

                report["stage"] = f"Chunk {chunk['index']} {name}: full QLI compute and output validation"
                indices, values = invoke(return_value=1)
                indices, values = indices.cpu(), values.cpu()
                validate_outputs(indices, values, query_start, query_tokens)
                sampled_indices, sampled_values = indices[row_tensor], values[row_tensor]
                decoded_scores = reference_scores(
                    case["decoded_query"][row_tensor],
                    case["decoded_key"],
                    sample_weights,
                    quant_mode=quant_mode,
                    causal_lengths=causal_lengths,
                )
                decoded = accuracy_metrics(sampled_indices, sampled_values, decoded_scores)
                original = accuracy_metrics(sampled_indices, sampled_values, original_scores)
                operator_passed = decoded["selected_scores_close"] and decoded["selection_above_tolerated_cutoff"]
                quantization_passed = (
                    original["topk_recall"] >= args.min_topk_recall
                    and original["score_cosine"] >= args.min_score_cosine
                    and original["score_nmae"] <= args.max_score_nmae
                )
                entry["accuracy"] = {
                    "scope": "All output rows checked structurally; score accuracy sampled at recorded global rows",
                    "decoded_payload_reference": decoded,
                    "original_fp16_input_reference": original,
                    "operator_correctness_passed": operator_passed,
                    "quantization_thresholds_passed": quantization_passed,
                    "full_output_structure_passed": True,
                    "passed": operator_passed and quantization_passed,
                }
                report["stage"] = f"Chunk {chunk['index']} {name}: indices-only structure and sampled selection"
                index_only, empty_values = invoke(return_value=0)
                index_only = index_only.cpu()
                validate_outputs(index_only, None, query_start, query_tokens)
                if empty_values.numel() != 0:
                    raise AssertionError("return_value=0 must return empty sparse values")
                indices_only_passed = selection_above_tolerated_cutoff(index_only[row_tensor], decoded_scores)
                operator_passed = operator_passed and indices_only_passed
                entry["accuracy"].update(
                    indices_only_sampled_selection_passed=indices_only_passed,
                    full_indices_structure=True,
                    operator_correctness_passed=operator_passed,
                    passed=operator_passed and quantization_passed,
                )
                del indices, values, sampled_indices, sampled_values, index_only, empty_values, decoded_scores
                report["stage"] = f"Chunk {chunk['index']} {name}: full QLI wall timing (return_value=0)"
                entry["wall_latency"] = benchmark_compute(invoke, args.warmup, args.iters)
                mean = entry["wall_latency"]["mean_ms"]
                if not math.isfinite(mean) or mean <= 0:
                    raise AssertionError("QLI mean wall latency must be positive and finite")
                report["cases"][name]["sum_chunk_mean_wall_ms"] += mean
                report["cases"][name]["completed_chunks"] += 1
                entry["status"] = "passed" if entry["accuracy"]["passed"] else "accuracy_gate_failed"
                print(
                    f"PREFILL {chunk['index']} {name}: mean={mean:.3f} ms; "
                    f"sampled operator={operator_passed}, quantization={quantization_passed}",
                    flush=True,
                )
                _save_report(args, report)
                del case, invoke
            chunk["status"] = (
                "passed"
                if all(entry["accuracy"]["passed"] for entry in chunk["cases"].values())
                else "accuracy_gate_failed"
            )
            _save_report(args, report)
        for name, _ in MODES:
            mode = report["cases"][name]
            checks = [chunk["cases"][name]["accuracy"] for chunk in report["chunks"]]
            mode["operator_correctness_passed"] = all(check["operator_correctness_passed"] for check in checks)
            mode["quantization_thresholds_passed"] = all(check["quantization_thresholds_passed"] for check in checks)
            mode["passed"] = mode["operator_correctness_passed"] and mode["quantization_thresholds_passed"]
        report["operator_correctness_passed"] = all(
            mode["operator_correctness_passed"] for mode in report["cases"].values()
        )
        report["quantization_thresholds_passed"] = all(
            mode["quantization_thresholds_passed"] for mode in report["cases"].values()
        )
        report["fp8_over_mxfp4_sum_chunk_mean_wall_ratio"] = (
            report["cases"]["FP8"]["sum_chunk_mean_wall_ms"] / report["cases"]["MXFP4"]["sum_chunk_mean_wall_ms"]
        )
        backend_report.update(
            mxfp4_compute_verified=True,
            mxfp4_numerical_checks_passed=report["cases"]["MXFP4"]["operator_correctness_passed"],
            capability_note="Every prefill chunk completed; numerical correctness covers sampled rows only",
        )
        report["status"] = "passed" if all(mode["passed"] for mode in report["cases"].values()) else "failed"
        print(
            "PREFILL sum of chunk mean QLI wall times (not end-to-end or whole-prefill P50): "
            + json.dumps({name: report["cases"][name]["sum_chunk_mean_wall_ms"] for name, _ in MODES}),
            flush=True,
        )
        if report["status"] != "passed":
            raise AssertionError(
                "Prefill accuracy gates failed; all chunk timings were collected without changing thresholds"
            )
    except Exception:
        report["status"] = "failed"
        if report["chunks"] and report["chunks"][-1]["status"] == "running":
            failed_chunk = report["chunks"][-1]
            failed_chunk["status"] = "failed"
            report["failed_chunk"] = failed_chunk["index"]
            for entry in failed_chunk["cases"].values():
                if entry["status"] == "running":
                    entry["status"] = "failed"
        raise
    finally:
        _save_report(args, report)
