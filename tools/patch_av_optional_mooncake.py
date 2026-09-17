#!/usr/bin/env python3
"""Make the AV Mooncake patch optional for non-Mooncake deployments."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import tempfile
from pathlib import Path

MARKER = "_AV_OPTIONAL_MOONCAKE_PATCH = True"
HEADER_MARKER = "_AV_OPTIONAL_MOONCAKE_GUARD_V2"
GUARD_HEADER = f"""# {MARKER}
# {HEADER_MARKER}
import os as _av_os

_av_false_values = {{"0", "false", "off", "no"}}
_av_mooncake_disabled = (
    _av_os.environ.get("VLLM_ASCEND_ENABLE_MOONCAKE", "").strip().lower()
    in _av_false_values
    or _av_os.environ.get("VLLM_ENABLE_MOONCAKE", "").strip().lower()
    in _av_false_values
)

"""


def find_target() -> Path:
    spec = importlib.util.find_spec("ascend_vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot locate the installed ascend_vllm package")
    return Path(spec.origin).resolve().parent / "patch" / "platform" / "__init__.py"


def is_mooncake_import(node: ast.Import | ast.ImportFrom, alias: ast.alias) -> bool:
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if module == "ascend_vllm.patch.platform" or (node.level and module in ("", "platform")):
            return alias.name.startswith("patch_mooncake")
        return module.startswith("ascend_vllm.patch.platform.patch_mooncake") or (
            bool(node.level) and module.startswith("patch_mooncake")
        )
    return alias.name.startswith("ascend_vllm.patch.platform.patch_mooncake")


def already_guarded(node: ast.AST, parents: dict) -> bool:
    child = node
    while child in parents:
        parent = parents[child]
        if isinstance(parent, ast.If) and child in parent.body:
            if ast.unparse(parent.test) == "not _av_mooncake_disabled":
                return True
        child = parent
    return False


def transform(text: str) -> str:
    tree = ast.parse(text)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    lines = text.splitlines(keepends=True)
    edits = []
    matched = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        flags = [is_mooncake_import(node, alias) for alias in node.names]
        if not any(flags):
            continue
        matched += sum(flags)
        if MARKER in text and already_guarded(node, parents):
            continue
        indent = lines[node.lineno - 1][: node.col_offset]
        suffix = lines[node.end_lineno - 1][node.end_col_offset :].strip()
        if indent.strip() or (suffix and not suffix.startswith("#")):
            raise RuntimeError("Mooncake imports must occupy separate statements; file left unchanged")
        if all(flags):
            original = lines[node.lineno - 1 : node.end_lineno]
            replacement = indent + "if not _av_mooncake_disabled:\n"
            replacement += "".join("    " + line for line in original).rstrip() + "\n"
        else:
            # A mixed import must still execute unrelated AV patches in order.
            statements = []
            for alias, needs_guard in zip(node.names, flags):
                single = (
                    ast.ImportFrom(module=node.module, names=[alias], level=node.level)
                    if isinstance(node, ast.ImportFrom)
                    else ast.Import(names=[alias])
                )
                statement = ast.unparse(single)
                if needs_guard:
                    statements.append(indent + "if not _av_mooncake_disabled:\n" + indent + "    " + statement + "\n")
                else:
                    statements.append(indent + statement + "\n")
            replacement = "".join(statements)
        edits.append((node.lineno - 1, node.end_lineno, replacement))
    if not matched:
        raise RuntimeError("No AV Mooncake patch imports found; file left unchanged")
    if not edits and HEADER_MARKER in text:
        return text
    if HEADER_MARKER not in text:
        if MARKER in text:
            # Upgrade v1: its guard lived at the connector import. Move it to
            # the header so an earlier backend import cannot see an unset flag.
            for node in tree.body:
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in ("_av_false_values", "_av_mooncake_disabled")
                ) or (
                    isinstance(node, ast.Import)
                    and len(node.names) == 1
                    and node.names[0].name == "os"
                    and node.names[0].asname == "_av_os"
                ):
                    edits.append((node.lineno - 1, node.end_lineno, ""))
            for index, line in enumerate(lines):
                if line.strip() == f"# {MARKER}":
                    edits.append((index, index + 1, ""))
        # Insert after the module docstring and future imports, preserving both.
        insert_at = 0
        for node in tree.body:
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "__future__"
                or isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                insert_at = node.end_lineno
            else:
                break
        edits.append((insert_at, insert_at, GUARD_HEADER))
    for start, end, replacement in sorted(edits, reverse=True):
        lines[start:end] = [replacement]
    updated = "".join(lines)
    compile(updated, "AV platform patches", "exec")
    return updated


def patch_target(target: Path) -> bool:
    text = target.read_text(encoding="utf-8")
    updated = transform(text)
    if updated == text:
        return False
    backup = target.with_name(target.name + ".before-optional-mooncake")
    if not backup.exists():
        backup.write_bytes(target.read_bytes())
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as output:
        temp_path = Path(output.name)
        try:
            output.write(updated)
            output.flush()
            os.chmod(temp_path, target.stat().st_mode)
            os.replace(temp_path, target)
        finally:
            temp_path.unlink(missing_ok=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        help="Override ascend_vllm/patch/platform/__init__.py for testing",
    )
    args = parser.parse_args()
    target = args.target.resolve() if args.target else find_target()
    changed = patch_target(target)
    action = "Patched" if changed else "Already patched"
    print(f"{action}: {target}")
    print("All explicit AV patch_mooncake* imports in this file are guarded (connector, backend, etc.).")
    print("Keep VLLM_ASCEND_ENABLE_MOONCAKE=0 in the service process to skip them.")


if __name__ == "__main__":
    main()
