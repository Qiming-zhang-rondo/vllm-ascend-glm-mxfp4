#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve an existing QLI installation and check ACLNN symbols without torch.

This is a library-loading preflight, not an operator compute/accuracy test.
Run with Python -I -S to avoid site hooks and tools/bisect shadowing the stdlib.
"""

import argparse
import ctypes
import json
import os
import shlex
import sys
from pathlib import Path

REQUIRED_APIS = (
    "aclnnQuantLightningIndexerV2",
    "aclnnQuantLightningIndexerV2GetWorkspaceSize",
    "aclnnQuantLightningIndexerV2Metadata",
    "aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize",
)


def select_private_library(repo, manifest=None, opapi_lib=None):
    if opapi_lib is not None:
        library = Path(opapi_lib).expanduser().resolve()
        vendor = library.parent.parent.parent
    else:
        if manifest is None:
            candidates = (
                repo / ".qli-op-build/install.json",
                repo.parent / "vllm-ascend-glm-mxfp4-optest/.qli-op-build/install.json",
                Path("/workspace/vllm-ascend-glm-mxfp4-optest/.qli-op-build/install.json"),
            )
            manifest = next((path for path in candidates if path.is_file()), None)
        if manifest is None:
            return None
        path = Path(manifest).expanduser().resolve()
        data = json.loads(path.read_text())
        vendor = Path(data["opp_root"]).expanduser().resolve()
        library = Path(data["opapi_lib"]).expanduser().resolve()
    expected = vendor / "op_api/lib/libcust_opapi.so"
    if not library.is_file() or not vendor.is_dir():
        raise ValueError(
            f"Private QLI installation is missing: {library}; specify its current --manifest or --opapi-lib"
        )
    if expected.resolve() != library:
        raise ValueError(f"Expected private library at {expected}, got {library}")
    return vendor, library


def prepend_path(path, current):
    return ":".join(dict.fromkeys([str(path), *([part for part in current.split(":") if part] if current else [])]))


def environment_exports(private):
    values = {
        "FLA_NPU_DISABLE_PTH": "1",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
    }
    if private is not None:
        vendor, library = private
        values["ASCEND_CUSTOM_OPP_PATH"] = prepend_path(vendor, os.environ.get("ASCEND_CUSTOM_OPP_PATH", ""))
        values["LD_LIBRARY_PATH"] = prepend_path(library.parent, os.environ.get("LD_LIBRARY_PATH", ""))
    return "\n".join(f"export {name}={shlex.quote(value)}" for name, value in values.items())


def configured_libraries():
    """Follow VA's op_api_common.h search order before importing its .so."""
    vendors = [Path(value) for value in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if value]
    opp = os.environ.get("ASCEND_OPP_PATH")
    if opp:
        configuration = Path(opp) / "vendors/config.ini"
        if configuration.is_file():
            for line in configuration.read_text().splitlines():
                if line.startswith("load_priority="):
                    vendors.extend(Path(opp) / "vendors" / value for value in line.split("=", 1)[1].split(",") if value)
                    break
    libraries = [str((vendor / "op_api/lib/libcust_opapi.so").resolve()) for vendor in vendors]
    return list(dict.fromkeys([*libraries, "libopapi.so"]))


def dynamic_loader():
    process = ctypes.CDLL(None)
    return process if hasattr(process, "dlopen") else ctypes.CDLL("libdl.so.2")


def open_library(candidate):
    # VA uses RTLD_LAZY. ctypes.CDLL adds RTLD_NOW even when LAZY is supplied,
    # so call the system loader and wrap its handle without reopening the file.
    # This only loads/checks symbols; it never invokes an ACLNN operator.
    loader = dynamic_loader()
    loader.dlopen.argtypes = [ctypes.c_char_p, ctypes.c_int]
    loader.dlopen.restype = ctypes.c_void_p
    loader.dlerror.argtypes = []
    loader.dlerror.restype = ctypes.c_char_p
    loader.dlerror()
    handle = loader.dlopen(os.fsencode(candidate), os.RTLD_LOCAL | os.RTLD_LAZY)
    if not handle:
        reason = loader.dlerror()
        raise OSError(os.fsdecode(reason) if reason else "dlopen failed")
    return ctypes.CDLL(candidate, handle=handle)


def symbol_owner(symbol, fallback):
    class DlInfo(ctypes.Structure):
        _fields_ = [
            ("fname", ctypes.c_char_p),
            ("fbase", ctypes.c_void_p),
            ("sname", ctypes.c_char_p),
            ("saddr", ctypes.c_void_p),
        ]

    process = dynamic_loader()
    try:
        dladdr = process.dladdr
    except AttributeError:
        return f"{fallback} (dladdr unavailable; owner unverified)"
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dladdr.restype = ctypes.c_int
    info = DlInfo()
    if dladdr(ctypes.cast(symbol, ctypes.c_void_p), ctypes.byref(info)) and info.fname:
        return os.fsdecode(info.fname)
    return f"{fallback} (symbol owner unresolved)"


def check_symbols(private):
    # An explicitly selected private installation must be complete. Do not hide
    # missing exports by mixing its compute API with another vendor's metadata.
    candidates = [str(private[1])] if private is not None else configured_libraries()
    resolved = {}
    handles = []  # Keep libraries loaded until ownership has been reported.
    errors = []
    for candidate in candidates:
        try:
            library = open_library(candidate)
            handles.append(library)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        for name in REQUIRED_APIS:
            if name in resolved:
                continue
            try:
                symbol = getattr(library, name)
            except AttributeError:
                continue
            resolved[name] = symbol_owner(symbol, candidate)
        if len(resolved) == len(REQUIRED_APIS):
            break
    for name, owner in resolved.items():
        print(f"QLI ACLNN: {name} -> {owner}", file=sys.stderr)
    missing = [name for name in REQUIRED_APIS if name not in resolved]
    if missing:
        detail = "\n".join(errors)
        raise RuntimeError(
            "Missing QLI CANN exports: " + ", ".join(missing) + "\n"
            "Provide the existing single-operator install with --manifest PATH or --opapi-lib PATH.\n"
            "No build or download was performed.\n" + detail
        )
    print("QLI ACLNN library/symbol preflight passed; compute and kernel ownership are not verified.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("resolve", "check"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path)
    source.add_argument("--opapi-lib", type=Path)
    args = parser.parse_args()
    try:
        private = select_private_library(Path(__file__).resolve().parent.parent, args.manifest, args.opapi_lib)
        if args.action == "resolve":
            if private:
                print(f"Reuse private QLI vendor: {private[0]}", file=sys.stderr)
            else:
                print("No private QLI manifest found; checking the configured CANN libraries.", file=sys.stderr)
            print(environment_exports(private))
        else:
            check_symbols(private)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"QLI runtime preflight failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
