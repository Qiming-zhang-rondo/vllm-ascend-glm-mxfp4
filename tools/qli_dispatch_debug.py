# SPDX-License-Identifier: Apache-2.0
# QLI_DISPATCH_DEBUG_HELPER_V1
"""Opt-in worker diagnostics: Python dispatch evidence, never NPU execution proof."""

import hashlib
import inspect
import json
import marshal
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
    return _quant_config_description(getattr(cfg.vllm_config, "quant_config", None))


def _quant_config_description(quant_config):
    description = getattr(quant_config, "quant_description", None)
    if not isinstance(description, dict):
        return None
    groups, entries = {}, []
    for name, value in description.items():
        if not isinstance(name, str):
            continue
        suffix = next((s for s in (".indexer.quant_type", ".indexer.wq_b_weight") if name.endswith(s)), None)
        if suffix is None:
            continue
        entries.append((name, value))
        # Quantization descriptions are JSON data. Never repr a non-JSON object.
        key = json.dumps([suffix, value], sort_keys=True)
        group = groups.setdefault(key, {"suffix": suffix, "value": value, "count": 0, "keys": []})
        group["count"] += 1
        if len(group["keys"]) < 3:
            group["keys"].append(name)
    return {
        "indexer_quant_type": description.get("indexer_quant_type"),
        "groups": list(groups.values()),
        "indexer_entries_sha256": hashlib.sha256(json.dumps(sorted(entries), sort_keys=True).encode()).hexdigest(),
    }


def _selector_cache(cfg):
    return {
        "filter_enabled": getattr(cfg, "_sparse_li_c8_layer_filter_enabled", None),
        "layer_ids": sorted(getattr(cfg, "_sparse_li_c8_layer_ids", ())),
        "layer_names": sorted(getattr(cfg, "_sparse_li_c8_layer_names", ())),
    }


def _parser_details(cfg):
    parser = getattr(cfg, "_parse_sparse_li_c8_layers_from_quant_config", None)
    code = getattr(_function(parser), "__code__", None)
    labels = set()

    def visit(value):
        if isinstance(value, str) and value in ("FP8_DYNAMIC", "W8A8_MXFP8", "INT8_DYNAMIC"):
            labels.add(value)
        elif isinstance(value, (tuple, frozenset)):
            for item in value:
                visit(item)

    if code is not None:
        visit(code.co_consts)
    return {
        "source": _source(parser),
        "code_sha256": hashlib.sha256(marshal.dumps(code)).hexdigest() if code is not None else None,
        # Loaded code constants, not disk source and not proof of acceptance.
        "quant_label_constants": sorted(labels),
    }


def report_selector_initialization(cfg, quant_config):
    """Capture state immediately after the constructor saves its layer filter."""
    try:
        snapshot = {
            "pid": os.getpid(),
            "config_id": id(cfg),
            "quant_config_id": id(quant_config),
            "description_id": id(getattr(quant_config, "quant_description", None)),
            "quant_description": _quant_config_description(quant_config),
            "cached": _selector_cache(cfg),
            "parser": _parser_details(cfg),
        }
        # Store a value snapshot, never references to mutable quantization data.
        snapshot = json.loads(json.dumps(snapshot))
        cfg._qli_selector_initialization = snapshot
        _emit("selector_initialization", **snapshot)
    except Exception as exc:
        _error(cfg, "selector_initialization", exc)


def _selection_diagnosis(cfg, instances):
    """Read-only comparison; never refresh cache allocation or layer switches."""
    result = {
        "config_id": id(cfg),
        "initialization": getattr(cfg, "_qli_selector_initialization", None),
        "cached": _selector_cache(cfg),
        "parser": _parser_details(cfg),
    }
    try:
        quant_config = getattr(cfg.vllm_config, "quant_config", None)
        result.update(
            quant_config_id=id(quant_config),
            description_id=id(getattr(quant_config, "quant_description", None)),
        )
        ids, names = cfg._parse_sparse_li_c8_layers_from_quant_config(quant_config)
        fresh = {
            "filter_enabled": cfg._has_sparse_li_c8_layer_config(quant_config),
            "layer_ids": sorted(ids),
            "layer_names": sorted(names),
        }
        result.update(fresh=fresh, fresh_matches_cached=fresh == result["cached"], layer_matches=[])
        from vllm.model_executor.models.utils import extract_layer_index

        for name, impl in instances:
            if not getattr(impl, "has_indexer", False):
                continue
            cache = getattr(getattr(impl, "indexer", None), "k_cache", None)
            prefix = getattr(cache, "prefix", None)
            match = {
                "layer": getattr(impl, "layer_name", None) or name,
                "prefix": prefix,
                "has_indexer": True,
                "impl_enabled": getattr(impl, "enable_sparse_li_c8", None),
                "layer_id": None,
                "current_layer_enabled": None,
                "cached_name_match": False,
                "cached_id_match": False,
                "fresh_name_match": False,
                "fresh_id_match": False,
            }
            try:
                normalized = (prefix.rstrip(".") or None) if isinstance(prefix, str) else None
                if normalized is None:
                    match["error"] = {"type": "MissingCachePrefix", "message": "No nonempty Indexer cache prefix"}
                    result["layer_matches"].append(match)
                    continue
                for label, selection in (("cached", result["cached"]), ("fresh", fresh)):
                    match[f"{label}_name_match"] = any(
                        normalized == candidate or normalized.startswith(f"{candidate}.")
                        for candidate in selection["layer_names"]
                    )
                match["current_layer_enabled"] = cfg.is_sparse_li_c8_layer(prefix)
                layer_id = extract_layer_index(normalized)
                match["layer_id"] = layer_id
                for label, selection in (("cached", result["cached"]), ("fresh", fresh)):
                    match[f"{label}_id_match"] = layer_id is not None and layer_id in selection["layer_ids"]
            except Exception as exc:
                match["error"] = {"type": type(exc).__name__, "message": str(exc)[:240]}
            result["layer_matches"].append(match)
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:240]}
    return result


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
            selection_diagnosis=_selection_diagnosis(cfg, instances.values()),
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
