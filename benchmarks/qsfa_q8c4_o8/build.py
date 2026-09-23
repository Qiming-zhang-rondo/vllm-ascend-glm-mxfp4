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


def find_cann():
    """Find an installed toolkit with the native ASC CMake package."""
    candidates = [
        Path(value) for name in ("ASCEND_HOME_PATH", "ASCEND_CANN_PACKAGE_PATH") if (value := os.getenv(name))
    ]
    candidates += [Path("/usr/local/Ascend/ascend-toolkit/latest"), Path("/usr/local/Ascend/cann")]
    candidates += sorted(Path("/usr/local/Ascend").glob("cann-*"), reverse=True)
    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if str(candidate) in checked:
            continue
        checked.append(str(candidate))
        if not candidate.is_dir():
            continue
        for config in sorted(candidate.rglob("ASCConfig.cmake")):
            return candidate, config.parent
    raise RuntimeError(
        "Installed ASCConfig.cmake not found. Source the container's CANN 9.1 set_env.sh. "
        "No toolkit will be downloaded. Checked: " + ", ".join(checked)
    )


def build_library(jobs=4):
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
    cann, asc_dir = find_cann()
    npu_root = Path(torch_npu.__file__).resolve().parent
    required = [
        "torch_npu/csrc/core/NPUBridge.h",
        "torch_npu/csrc/core/npu/NPUStream.h",
        "torch_npu/csrc/core/npu/NPUGuard.h",
        "torch_npu/csrc/core/npu/NPUCachingAllocator.h",
    ]
    missing = [name for name in required if not (npu_root / "include" / name).is_file()]
    if missing:
        raise RuntimeError("Installed torch_npu is missing extension headers: " + ", ".join(missing))
    sources = [ROOT / "CMakeLists.txt", ROOT / "build.py"]
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
    sdk_files = sorted(asc_dir.rglob("*.cmake"))
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
        "asc_dir": str(asc_dir),
        "asc_cmake": {str(file): hashlib.sha256(file.read_bytes()).hexdigest() for file in sdk_files},
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
            print(f"Reusing standalone QSFA library: {library}", flush=True)
            return {**old, "reused": True}
    build_dir.mkdir(parents=True, exist_ok=True)
    print(f"Build standalone QSFA against existing CANN: {cann}", flush=True)
    subprocess.run(
        [
            cmake,
            "-S",
            str(ROOT),
            "-B",
            str(build_dir),
            "-G",
            "Unix Makefiles",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DASCEND_HOME_PATH={cann}",
            f"-DASC_DIR={asc_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run([cmake, "--build", str(build_dir), "--parallel", str(jobs)], check=True)
    if not library.is_file():
        raise RuntimeError(f"Build returned success but library is absent: {library}")
    result = {
        "library": str(library),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "fingerprint": fingerprint,
        "reused": False,
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
