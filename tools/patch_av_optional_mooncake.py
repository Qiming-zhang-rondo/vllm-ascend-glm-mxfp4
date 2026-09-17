#!/usr/bin/env python3
"""Make the AV Mooncake patch optional for non-Mooncake deployments."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


IMPORT_LINE = (
    "from ascend_vllm.patch.platform import patch_mooncake_connector "
    "as patch_mooncake_connector"
)
MARKER = "_AV_OPTIONAL_MOONCAKE_PATCH = True"
REPLACEMENT = f'''# {MARKER}
import os as _av_os

_av_false_values = {{"0", "false", "off", "no"}}
_av_mooncake_disabled = (
    _av_os.environ.get("VLLM_ASCEND_ENABLE_MOONCAKE", "").lower()
    in _av_false_values
    or _av_os.environ.get("VLLM_ENABLE_MOONCAKE", "").lower()
    in _av_false_values
)
if not _av_mooncake_disabled:
    {IMPORT_LINE}
else:
    patch_mooncake_connector = None
'''


def find_target() -> Path:
    spec = importlib.util.find_spec("ascend_vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot locate the installed ascend_vllm package")
    return Path(spec.origin).resolve().parent / "patch" / "platform" / "__init__.py"


def patch_target(target: Path) -> bool:
    if not target.is_file():
        raise RuntimeError(f"AV platform patch file does not exist: {target}")
    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        return False
    occurrences = text.count(IMPORT_LINE)
    if occurrences != 1:
        raise RuntimeError(
            f"Expected one Mooncake patch import in {target}, found {occurrences}"
        )
    target.write_text(text.replace(IMPORT_LINE, REPLACEMENT), encoding="utf-8")
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
    print("Mooncake imports are now skipped when VLLM_ASCEND_ENABLE_MOONCAKE=0")


if __name__ == "__main__":
    main()
