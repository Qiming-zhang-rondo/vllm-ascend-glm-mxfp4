# SPDX-License-Identifier: Apache-2.0
"""Build an isolated QSFA .so with installed CANN/torch; never install or fetch.

The cache key covers sources, interpreter, compiler, CANN version files and
torch/torch_npu identity. This does not touch VA, OPP or site-packages.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _walk_files(root, names):
    """Follow toolkit component symlinks, visiting each real directory once."""
    visited = set()
    for directory, dirs, files in os.walk(root, followlinks=True):
        real = Path(directory).resolve()
        if real in visited:
            dirs[:] = []
            continue
        visited.add(real)
        dirs.sort()
        for name in sorted(set(files).intersection(names)):
            yield Path(directory) / name


def _installed_compilers(root):
    prefixes = ("compiler", "tools", "toolkit/tools", "aarch64-linux", "arm64-linux")
    candidates = [root / prefix / "bisheng_compiler/bin/bisheng" for prefix in prefixes]
    candidates += [root / "bin/bisheng"]
    candidates += [root / prefix / "ccec_compiler/bin/bisheng" for prefix in prefixes]
    found = {}
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            found.setdefault(path.resolve(), path)
    return list(found.values())


def find_cann(candidates=None):
    """Prefer the active toolkit; native bisheng does not require ASCConfig."""
    if candidates is None:
        candidates = _cann_candidates()
    checked = []
    for candidate in candidates:
        candidate = Path(candidate).resolve()
        if str(candidate) in checked:
            continue
        checked.append(str(candidate))
        if not candidate.is_dir():
            continue
        # Native compiler first, not a PATH compiler from another toolkit.
        compiler = next(iter(_installed_compilers(candidate)), None)
        configs = list(_walk_files(candidate, {"ASCConfig.cmake", "FindASC.cmake"}))
        if configs:
            config = next((p for p in configs if p.name == "ASCConfig.cmake"), configs[0])
            return candidate, config, compiler
        if compiler is not None:
            return candidate, None, compiler
    raise RuntimeError(
        "No installed native ASC CMake package or bisheng compiler found. "
        "Source the container's CANN set_env.sh. No toolkit will be downloaded. Checked: " + ", ".join(checked)
    )


def _cann_candidates():
    candidates = [
        Path(value) for name in ("ASCEND_HOME_PATH", "ASCEND_CANN_PACKAGE_PATH") if (value := os.getenv(name))
    ]
    candidates += [Path("/usr/local/Ascend/ascend-toolkit/latest"), Path("/usr/local/Ascend/cann")]
    candidates += sorted(Path("/usr/local/Ascend").glob("cann-*"), reverse=True)
    return candidates


def _toolchain_options(config, compiler):
    if config is None:
        return [f"-DQSFA_BISHENG={compiler}"]
    options = [f"-DCMAKE_ASC_COMPILER={compiler}"] if compiler else []
    if config.name == "FindASC.cmake":
        return [f"-DCMAKE_MODULE_PATH={config.parent}", *options]
    return [f"-DASC_DIR={config.parent}", *options]


def _verify_library_load(torch, library):
    """Resolve host symbols/register schemas only; never dispatch a kernel."""
    print("Checking QSFA shared-library load and registration (no NPU execution)", flush=True)
    torch.ops.load_library(str(library))
    _ = torch.ops.qsfa_q8c4_o8.forward


def build_library(jobs=4, *, _remaining_compilers=None):
    """Return a JSON-safe library manifest after an offline build or cache hit."""
    if platform.system() != "Linux":
        raise RuntimeError("Build this prototype inside the existing A5 Linux container; no container will be created")
    if not 1 <= jobs <= 64:
        raise ValueError("jobs must be in [1,64]")
    import torch
    import torch_npu

    cmake = shutil.which("cmake")
    if not cmake or not shutil.which("make"):
        raise RuntimeError("Container needs installed cmake and make; no dependencies will be installed")
    cann, asc_config, compiler = find_cann()
    compilers = _installed_compilers(cann) if _remaining_compilers is None else _remaining_compilers
    if asc_config is None:
        compiler = compilers[0]
    route = "asc_cmake" if asc_config else "bisheng_direct"
    print(f"CANN build route: {route}; root: {cann}; compiler: {compiler}; ASC package: {asc_config}", flush=True)
    npu_root = Path(torch_npu.__file__).resolve().parent
    required = [
        "torch_npu/csrc/core/npu/NPUStream.h",
        "torch_npu/csrc/core/npu/NPUGuard.h",
        "torch_npu/csrc/core/npu/NPUCachingAllocator.h",
    ]
    missing = [name for name in required if not (npu_root / "include" / name).is_file()]
    if missing:
        raise RuntimeError("Installed torch_npu is missing extension headers: " + ", ".join(missing))
    sources = [ROOT / "CMakeLists.txt", ROOT / "build.py"] + sorted((ROOT / "cmake").rglob("*.*"))
    sources += sorted(file for file in (ROOT / "csrc").rglob("*") if file.is_file())
    for name in ("vector.asc", "matmul.asc", "torch_binding.cpp", "launch.h"):
        if not (ROOT / "csrc" / name).is_file():
            raise RuntimeError(f"Incomplete prototype source: csrc/{name}")
    compiler_paths = []
    for name in ("c++", "bisheng", "ccec"):
        found = shutil.which(name)
        if found:
            compiler_paths.append(_file_identity(found))
    versions = {}
    for pattern in ("version.info", "version.cfg", "*/version.info", "*/version.cfg", "*/version/version.info"):
        for file in cann.glob(pattern):
            if file.is_file():
                versions[str(file.relative_to(cann))] = hashlib.sha256(file.read_bytes()).hexdigest()
    sdk_files = sorted(asc_config.parent.rglob("*.cmake")) if asc_config else []
    if compiler:
        compiler_paths.append(_file_identity(compiler))
    # These API headers may be updated in place without changing version.info.
    sdk_headers = list(_walk_files(cann, {"kernel_operator.h", "asc_simt.h", "device_functions.h"}))
    for prefix in (cann / "bin", cann / "compiler/ccec_compiler/bin", cann / "tools/ccec_compiler/bin"):
        for name in ("bisheng", "ccec"):
            if (prefix / name).is_file():
                compiler_paths.append(_file_identity(prefix / name))
    dependencies = [_file_identity(npu_root / "include" / name) for name in required]
    dependencies += [_file_identity(file) for file in (npu_root / "lib").glob("libtorch_npu.so*")]
    identity = {
        "python": _file_identity(sys.executable),
        "torch": str(torch.__version__),
        "torch_path": str(Path(torch.__file__).resolve()),
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "torch_npu": str(getattr(torch_npu, "__version__", "unknown")),
        "torch_npu_path": str(npu_root),
        "cann": str(cann),
        "cann_versions": versions,
        "build_route": route,
        "asc_config": str(asc_config) if asc_config else None,
        "asc_cmake": {str(file): hashlib.sha256(file.read_bytes()).hexdigest() for file in sdk_files},
        "asc_headers": {str(file): hashlib.sha256(file.read_bytes()).hexdigest() for file in sdk_headers},
        "cmake": _file_identity(cmake),
        "compilers": compiler_paths,
        "dependencies": dependencies,
        "sources": {str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest() for file in sources},
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    build_dir = ROOT / ".build" / fingerprint[:20]
    library = build_dir / "libqsfa_q8c4_o8.so"
    stamp = build_dir / "manifest.json"
    if library.is_file() and stamp.is_file():
        old = json.loads(stamp.read_text())
        if (
            old.get("fingerprint") == fingerprint
            and old.get("library_sha256") == hashlib.sha256(library.read_bytes()).hexdigest()
        ):
            _verify_library_load(torch, library)
            print(f"Reusing standalone QSFA library: {library}", flush=True)
            return {**old, "reused": True}
    build_dir.mkdir(parents=True, exist_ok=True)
    print(f"Build standalone QSFA against existing CANN: {cann}", flush=True)
    configure = [
        cmake,
        "-S",
        str(ROOT),
        "-B",
        str(build_dir),
        "-G",
        "Unix Makefiles",
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DASCEND_HOME_PATH={cann}",
        *_toolchain_options(asc_config, compiler),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    (build_dir / "toolchain_probe.failed").unlink(missing_ok=True)
    try:
        subprocess.run(configure, check=True)
    except subprocess.CalledProcessError:
        # Some SDKs contain both a legacy ccec driver and native bisheng. Only
        # retry a failed native syntax/link probe, never an actual kernel build.
        if asc_config is None and (build_dir / "toolchain_probe.failed").is_file() and len(compilers) > 1:
            print(f"Trying another installed compiler from the same CANN root: {compilers[1]}", flush=True)
            return build_library(jobs, _remaining_compilers=compilers[1:])
        raise
    subprocess.run([cmake, "--build", str(build_dir), "--parallel", str(jobs)], check=True)
    if not library.is_file():
        raise RuntimeError(f"Build returned success but library is absent: {library}")
    _verify_library_load(torch, library)
    result = {
        "library": str(library),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "fingerprint": fingerprint,
        "reused": False,
        "load_verified": True,
        "environment": identity,
    }
    stamp.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--result-file", type=Path)
    args = parser.parse_args()
    result = build_library(args.jobs)
    if args.result_file:
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("library", "fingerprint", "reused")}), flush=True)


if __name__ == "__main__":
    main()
