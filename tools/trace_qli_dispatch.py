#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install/remove temporary Python-only QLI diagnostics in the deployed VA tree.

Run with Python -I to avoid tools/bisect shadowing the standard library.
This does not import VA/AV, install packages, build operators or change routing.
"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path

DEFAULT_VA_ROOT = Path("/workspace/vllm-ascend-qli-mxfp4-v0.26.0")
BEGIN = "# QLI_DISPATCH_DEBUG_BEGIN:"
END = "# QLI_DISPATCH_DEBUG_END:"
HELPER_MARKER = "QLI_DISPATCH_DEBUG_HELPER_V1"
EXPECTED_PARAMETERS = {
    "worker": {"self", "cumulative_query_lens", "seq_lens", "draft_index"},
    "compute": {"query", "key", "weights", "query_scale", "key_scale", "block_table", "metadata"},
    "device": {
        "sfa_impl",
        "q_li",
        "q_li_scale",
        "q_li_shape_ori",
        "weights",
        "kv_cache",
        "attn_metadata",
        "actual_seq_lengths_query",
        "actual_seq_lengths_key",
        "enable_sparse_li_c8",
        "use_torch_npu_lightning_indexer",
    },
}
HOOKS = (
    (
        "attention/sfa_indexer.py",
        "SFAIndexerMetadataBuilder",
        "build",
        "worker",
        "report_worker_dispatch",
        ("self",),
    ),
    (
        "attention/sfa_indexer.py",
        None,
        "select_sfa_topk",
        "compute",
        "report_compute_entry",
        ("metadata", "query", "key", "query_scale", "key_scale"),
    ),
    (
        "device/device_op.py",
        "A5DeviceAdaptor",
        "indexer_select_post_process",
        "device",
        "report_device_entry",
        ("sfa_impl", "q_li", "q_li_scale", "enable_sparse_li_c8"),
    ),
)


def remove_hooks(source: str) -> str:
    lines = []
    active = None
    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(BEGIN):
            if active is not None:
                raise ValueError("Nested QLI diagnostic markers; refusing to modify")
            active = stripped.removeprefix(BEGIN)
            if active not in {hook[3] for hook in HOOKS}:
                raise ValueError(f"Unknown QLI diagnostic marker: {active}")
        elif stripped.startswith(END):
            if active is None or stripped.removeprefix(END) != active:
                raise ValueError("Mismatched QLI diagnostic markers; refusing to modify")
            active = None
        elif active is None:
            lines.append(line)
    if active is not None:
        raise ValueError("Unclosed QLI diagnostic marker; refusing to modify")
    return "".join(lines)


def add_hooks(source: str, relative: str) -> str:
    source = remove_hooks(source)
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    insertions = []
    for filename, class_name, function_name, event, reporter, arguments in HOOKS:
        if filename != relative:
            continue
        scope = tree
        if class_name is not None:
            classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
            if len(classes) != 1:
                raise ValueError(f"Expected one {class_name} in {relative}; file left unchanged")
            scope = classes[0]
        functions = [node for node in scope.body if isinstance(node, ast.FunctionDef) and node.name == function_name]
        if len(functions) != 1:
            raise ValueError(f"Expected one {class_name or 'module'}.{function_name} in {relative}")
        function = functions[0]
        parameters = {arg.arg for arg in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)}
        if not EXPECTED_PARAMETERS[event] <= parameters:
            raise ValueError(
                f"Unexpected arguments of {function_name}: missing {EXPECTED_PARAMETERS[event] - parameters}"
            )
        statement = function.body[0]
        if statement.lineno == function.lineno:
            raise ValueError(f"Single-line function {function_name} is unsupported; file left unchanged")
        insertion_line = statement.lineno - 1
        indentation = " " * statement.col_offset
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            if isinstance(statement.value.value, str):
                insertion_line = statement.end_lineno
                if len(function.body) > 1 and function.body[1].lineno <= statement.end_lineno:
                    raise ValueError(f"Inline statements after {function_name} docstring are unsupported")
        # The temporary diagnostics must not introduce print/inspection into a
        # torch.compile graph. Metadata snapshots still inspect actual layers.
        block = (
            f"{indentation}{BEGIN}{event}\n"
            f"{indentation}if not torch.compiler.is_compiling():\n"
            f"{indentation}    from vllm_ascend.attention.qli_dispatch_debug import {reporter}\n"
            f"{indentation}    {reporter}({', '.join(arguments)})\n"
            f"{indentation}{END}{event}\n"
        )
        insertions.append((insertion_line, block))
    for line, block in sorted(insertions, reverse=True):
        lines.insert(line, block)
    updated = "".join(lines)
    compile(updated, relative, "exec")
    return updated


def prepare_changes(root: Path, remove: bool) -> dict[Path, str]:
    package = root / "vllm_ascend"
    changes = {}
    for relative in dict.fromkeys(hook[0] for hook in HOOKS):
        path = package / relative
        original = path.read_text()
        updated = remove_hooks(original) if remove else add_hooks(original, relative)
        compile(updated, str(path), "exec")
        if updated != original:
            changes[path] = updated
    if not remove:
        helper_source = Path(__file__).with_name("qli_dispatch_debug.py").read_text()
        if HELPER_MARKER not in helper_source:
            raise ValueError("Unexpected diagnostic helper version; nothing changed")
        compile(helper_source, "qli_dispatch_debug.py", "exec")
        helper_target = package / "attention/qli_dispatch_debug.py"
        existing_helper = helper_target.read_text() if helper_target.exists() else None
        if existing_helper is not None and HELPER_MARKER not in existing_helper:
            raise ValueError(f"Refusing to overwrite an unrelated helper: {helper_target}")
        if existing_helper != helper_source:
            # Install the helper before the hooks. No worker should be running
            # from edited sources; restart after this command completes.
            changes = {helper_target: helper_source, **changes}
    return changes


def apply_changes(changes: dict[Path, str]) -> None:
    for path, content in changes.items():
        temporary = path.with_name(f".{path.name}.qli-trace-{os.getpid()}.tmp")
        try:
            if path.exists():
                backup = path.with_name(path.name + ".before-qli-dispatch-trace")
                if not backup.exists():
                    with backup.open("xb") as output:
                        output.write(path.read_bytes())
            with temporary.open("x", encoding="utf-8") as output:
                output.write(content)
            if path.exists():
                temporary.chmod(path.stat().st_mode & 0o777)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--va-root", type=Path, default=DEFAULT_VA_ROOT)
    parser.add_argument("--remove", action="store_true", help="Remove only marked hooks, preserving other edits")
    parser.add_argument("--check-only", action="store_true", help="Validate targets without changing any files")
    args = parser.parse_args()
    try:
        changes = prepare_changes(args.va_root.resolve(), args.remove)
        for path in changes:
            print(f"{'Would update' if args.check_only else 'Updating'}: {path}")
        if not args.check_only:
            apply_changes(changes)
        action = "removal" if args.remove else "installation"
        if args.check_only:
            print(f"QLI diagnostic {action} check passed; no files changed.")
        else:
            print(f"QLI diagnostic {action} complete. Restart the service to use the updated Python files.")
            if not args.remove:
                print("Look for [QLI-DISPATCH] in worker logs. Snapshot/entry logs do not prove NPU completion.")
                print("Entry hooks are skipped inside torch.compile; their absence alone does not prove bypass.")
            print("No CANN/VA compilation, dependency installation or quantization changes were performed.")
    except (OSError, ValueError, SyntaxError) as error:
        print(f"QLI diagnostic setup failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
