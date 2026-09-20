#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Accept ModelSlim FP8_DYNAMIC layers in an existing VA QLI deployment.

Only edits the layer-selector allowlist. No VA/CANN imports, compilation,
package installation, or changes to model weights and per-layer exclusions.
"""

import argparse
import ast
import json
import os
from pathlib import Path

DEFAULT_VA_ROOT = Path("/workspace/vllm-ascend-qli-mxfp4-v0.26.0")


def transform(source: str) -> str:
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendConfig"]
    if len(classes) != 1:
        raise ValueError("Expected one AscendConfig class; file left unchanged")
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_sparse_li_c8_layers_from_quant_config"
    ]
    if len(methods) != 1:
        raise ValueError("Cannot locate the QLI layer-selector method; file left unchanged")
    assignments = [
        node
        for node in methods[0].body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "VALID_QUANT_TYPES"
    ]
    if len(assignments) != 1:
        raise ValueError("Expected one VALID_QUANT_TYPES assignment; file left unchanged")
    value = assignments[0].value
    labels = ast.literal_eval(value)
    if not isinstance(labels, tuple) or not all(isinstance(label, str) for label in labels):
        raise ValueError("Expected a tuple of quantization labels; file left unchanged")
    if "FP8_DYNAMIC" in labels:
        return source
    if not {"INT8_DYNAMIC", "W8A8_MXFP8"} <= set(labels):
        raise ValueError("Unexpected existing QLI label contract; file left unchanged")
    # AST columns are UTF-8 byte offsets. Preserve every other byte, including
    # user changes and comments elsewhere in the configuration file.
    lines = source.encode().splitlines(keepends=True)
    start = sum(map(len, lines[: value.lineno - 1])) + value.col_offset
    end = sum(map(len, lines[: value.end_lineno - 1])) + value.end_col_offset
    replacement = "(" + ", ".join(json.dumps(label) for label in (*labels, "FP8_DYNAMIC")) + ")"
    updated = (source.encode()[:start] + replacement.encode() + source.encode()[end:]).decode()
    compile(updated, "ascend_config.py", "exec")
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--va-root", type=Path, default=DEFAULT_VA_ROOT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    target = args.va_root.resolve() / "vllm_ascend/ascend_config.py"
    try:
        source = target.read_text()
        updated = transform(source)
        print(f"QLI layer-selector file: {target}")
        if updated == source:
            print("FP8_DYNAMIC is already accepted; no files changed.")
        elif args.check_only:
            print("FP8_DYNAMIC compatibility fix can be applied; no files changed.")
        else:
            backup = target.with_name(target.name + ".before-qli-fp8-dynamic")
            if not backup.exists():
                with backup.open("xb") as output:
                    output.write(target.read_bytes())
            temporary = target.with_name(f".{target.name}.qli-fp8-{os.getpid()}.tmp")
            try:
                with temporary.open("x", encoding="utf-8") as output:
                    output.write(updated)
                temporary.chmod(target.stat().st_mode & 0o777)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            print("Added FP8_DYNAMIC to the existing QLI layer-selector allowlist.")
        if not args.check_only:
            print("Restart service workers to rebuild layer selection and cache allocation; no compilation is needed.")
            print("Per-layer exclusions are preserved; not every Indexer layer is necessarily quantized.")
    except (OSError, ValueError, SyntaxError) as error:
        print(f"QLI FP8_DYNAMIC fix failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
