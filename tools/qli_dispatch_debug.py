# SPDX-License-Identifier: Apache-2.0
# QLI_DISPATCH_DEBUG_HELPER_V1
"""Opt-in worker diagnostics: Python dispatch evidence, never NPU execution proof."""

import inspect
import json
import os
import sys


def _emit(event, **fields):
    print("[QLI-DISPATCH] " + json.dumps({"event": event, "pid": os.getpid(), **fields}, sort_keys=True), flush=True)


def _error(owner, operation, exc):
    """Diagnostics must not prevent inference, including when output is closed."""
    try:
        seen = getattr(owner, "_qli_dispatch_errors", set())
        if operation not in seen:
            seen.add(operation)
            owner._qli_dispatch_errors = seen
            _emit("diagnostic_error", operation=operation, error_type=type(exc).__name__, message=str(exc)[:240])
    except Exception:
        pass


def _config():
    from vllm_ascend.ascend_config import get_ascend_config

    return get_ascend_config()


def _function(method):
    return getattr(method, "__func__", method)


def _source(method, snippets=None):
    if method is None:
        return None
    fn = _function(method)
    code = getattr(fn, "__code__", None)
    info = {
        "module": getattr(fn, "__module__", type(fn).__module__),
        "qualname": getattr(fn, "__qualname__", type(fn).__qualname__),
        "file": getattr(code, "co_filename", None),
        "line": getattr(code, "co_firstlineno", None),
    }
    wrapped = getattr(fn, "__wrapped__", None)
    if wrapped is not None and wrapped is not fn:
        wrapped_code = getattr(_function(wrapped), "__code__", None)
        info["wrapped"] = {
            "module": getattr(wrapped, "__module__", None),
            "qualname": getattr(wrapped, "__qualname__", None),
            "file": getattr(wrapped_code, "co_filename", None),
            "line": getattr(wrapped_code, "co_firstlineno", None),
        }
    if snippets is not None and code is not None:
        external = not str(info["module"]).startswith("vllm_ascend") or "/vllm_ascend/" not in info["file"]
        key = f"{info['file']}:{info['line']}"
        if external and key not in snippets:
            snippets[key] = {"origin": info}
            try:
                # Passing the code object avoids inspect unwrapping @wraps.
                lines, first = inspect.getsourcelines(code)
                snippets[key].update(
                    lines=[{"line": first + i, "text": line.rstrip()} for i, line in enumerate(lines[:60])],
                    truncated=len(lines) > 60,
                )
            except (OSError, TypeError, IndexError):
                snippets[key]["source_unavailable"] = True
    return info


def _gates(impl):
    return {
        name: getattr(impl, name, None)
        for name in (
            "has_indexer",
            "skip_topk",
            "enable_sparse_li_c8",
            "indexer_quant_mode",
            "use_torch_npu_lightning_indexer",
        )
    }


def _tensor(tensor):
    """Only tensor metadata; no values, copies, synchronization or tensor repr."""
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
    }


def _quant_description(cfg):
    description = getattr(getattr(cfg.vllm_config, "quant_config", None), "quant_description", None)
    if not isinstance(description, dict):
        return None
    groups = {}
    for name, value in description.items():
        if not isinstance(name, str):
            continue
        suffix = next((s for s in (".indexer.quant_type", ".indexer.wq_b_weight") if name.endswith(s)), None)
        if suffix is None:
            continue
        # Quantization descriptions are JSON data. Never repr a non-JSON object.
        key = json.dumps([suffix, value], sort_keys=True)
        group = groups.setdefault(key, {"suffix": suffix, "value": value, "count": 0, "keys": []})
        group["count"] += 1
        if len(group["keys"]) < 3:
            group["keys"].append(name)
    return {"indexer_quant_type": description.get("indexer_quant_type"), "groups": list(groups.values())}


def report_worker_dispatch(builder):
    """Snapshot live worker instances once; retry empty startup contexts at most three times."""
    try:
        state = getattr(builder, "_qli_dispatch_snapshot_state", None)
        if state is None:
            state = {"attempts": 0, "done": False}
            builder._qli_dispatch_snapshot_state = state
        if state["done"] or state["attempts"] >= 3:
            return
        cfg = _config()
        context = cfg.vllm_config.compilation_config.static_forward_context
        instances = {}
        for name, layer in context.items():
            for candidate in (layer, getattr(layer, "mla_attn", None)):
                impl = getattr(candidate, "impl", None)
                if impl is not None and hasattr(impl, "indexer_select_post_process"):
                    instances.setdefault(id(impl), (name, impl))
        if not instances:
            if state["attempts"] < 3:
                state["attempts"] += 1
                _emit("worker_snapshot_empty", attempt=state["attempts"], max_attempts=3)
            return
        groups, snippets = {}, {}
        for name, impl in instances.values():
            method = getattr(impl, "indexer_select_post_process", None)
            globals_ = getattr(_function(method), "__globals__", {})
            adaptor = globals_.get("DeviceOperator")
            adapter_method = getattr(adaptor, "indexer_select_post_process", None)
            adapter_globals = getattr(_function(adapter_method), "__globals__", {})
            signature = {
                "implementation": f"{type(impl).__module__}.{type(impl).__qualname__}",
                "gates": _gates(impl),
                "method_sources": {
                    "forward": _source(getattr(impl, "forward", None)),
                    "forward_mqa": _source(getattr(impl, "forward_mqa", None)),
                    "indexer_select_post_process": _source(method, snippets),
                },
                "device_operator": _source(adapter_method, snippets),
                "select_sfa_topk": _source(adapter_globals.get("select_sfa_topk", globals_.get("select_sfa_topk"))),
            }
            key = json.dumps(signature, sort_keys=True)
            group = groups.setdefault(key, {**signature, "count": 0, "layers": [], "prefixes": []})
            group["count"] += 1
            if len(group["layers"]) < 3:
                group["layers"].append(getattr(impl, "layer_name", None) or name)
                cache = getattr(getattr(impl, "indexer", None), "k_cache", None)
                group["prefixes"].append(getattr(cache, "prefix", None))
        config = {
            name: getattr(cfg, name, None)
            for name in ("enable_sparse_li_c8", "enable_sparse_sfa_c8", "sfa_indexer_quant_mode")
        }
        config.update(
            rank=getattr(getattr(cfg.vllm_config, "parallel_config", None), "rank", None),
            filter_enabled=getattr(cfg, "_sparse_li_c8_layer_filter_enabled", None),
            layer_ids=sorted(getattr(cfg, "_sparse_li_c8_layer_ids", ())),
            layer_names=sorted(getattr(cfg, "_sparse_li_c8_layer_names", ())),
            quant_description=_quant_description(cfg),
        )
        _emit(
            "worker_snapshot",
            snapshot_only=True,
            builder_quant_mode=getattr(builder, "quant_mode", None),
            note="Snapshot is not compute proof; compiled forwards may suppress entry hooks. Entry means attempt only.",
            config=config,
            modules={name: getattr(sys.modules.get(name), "__file__", None) for name in ("vllm_ascend", "ascend_vllm")},
            parser_sources={
                name: _source(getattr(cfg, name, None))
                for name in ("is_sparse_li_c8_layer", "_parse_sparse_li_c8_layers_from_quant_config")
            },
            groups=list(groups.values()),
            external_sources=snippets,
        )
        state["done"] = True
    except Exception as exc:
        _error(builder, "worker_snapshot", exc)


def report_device_entry(sfa_impl, q_li, q_li_scale, enable_sparse_li_c8):
    """Log entry into our device adaptor once per instance, before compute."""
    try:
        if getattr(sfa_impl, "_qli_dispatch_device_reported", False):
            return
        _emit(
            "device_entry",
            attempt_only=True,
            layer=getattr(sfa_impl, "layer_name", None),
            gates=_gates(sfa_impl),
            enable_sparse_li_c8_argument=enable_sparse_li_c8,
            query=_tensor(q_li),
            query_scale=_tensor(q_li_scale),
        )
        sfa_impl._qli_dispatch_device_reported = True
    except Exception as exc:
        _error(sfa_impl, "device_entry", exc)


def report_compute_entry(metadata, query, key, query_scale, key_scale):
    """Log entry into our QLI V2 wrapper once per config/mode, not successful execution."""
    owner = metadata
    try:
        owner = _config()
        seen = getattr(owner, "_qli_dispatch_compute_modes", set())
        mode = metadata.quant_mode
        if mode in seen:
            return
        _emit(
            "compute_entry",
            attempt_only=True,
            quant_mode=mode,
            query=_tensor(query),
            key=_tensor(key),
            query_scale=_tensor(query_scale),
            key_scale=_tensor(key_scale),
        )
        seen.add(mode)
        owner._qli_dispatch_compute_modes = seen
    except Exception as exc:
        _error(owner, "compute_entry", exc)
