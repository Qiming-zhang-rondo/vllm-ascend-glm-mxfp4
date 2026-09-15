#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build only QLI V2 + metadata, using existing container dependencies.

No Python packages, source repositories, or dependency archives are fetched.
The install prefix is private to this checkout; the system CANN is read-only.
Exit 78 denotes missing prerequisites; exit 1 denotes a configure/build failure.
"""

import os
import sys

# This repo has tools/bisect, which shadows Python's stdlib bisect when a tool
# is executed by filename. Remove only the entry-point directory, before other
# stdlib modules or container packages are imported.
if sys.path and os.path.realpath(sys.path[0]) == os.path.dirname(os.path.realpath(__file__)):
    sys.path.pop(0)

import argparse
import hashlib
import importlib.util
import json
import platform
import shlex
import shutil
import subprocess
import time
from pathlib import Path

OPS = "quant_lightning_indexer_v2;quant_lightning_indexer_v2_metadata"
VENDOR = "qli_mxfp4"
MISSING_PREREQUISITE = 78


class PrerequisiteError(RuntimeError):
    pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python", default=sys.executable, help="Existing container Python; selected by the shell entry"
    )
    parser.add_argument(
        "--soc", default="ascend950", help="A5 SoC name; ascend950 variants normalize to the CMake family ascend950"
    )
    parser.add_argument(
        "--cann-root", type=Path, help="Existing CANN toolkit root (otherwise discover from environment)"
    )
    parser.add_argument("--build-dir", type=Path, help="Private build directory; default: this checkout/.qli-op-build")
    parser.add_argument("--json-include", type=Path, help="Existing directory containing nlohmann/json.hpp")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument(
        "--check-only", action="store_true", help="Check installed build prerequisites without configuring"
    )
    parser.add_argument(
        "--reuse-only",
        action="store_true",
        help="Return 0 only if a matching private build already exists; never compile",
    )
    args = parser.parse_args(argv)
    args.requested_soc = args.soc
    if not args.soc.lower().startswith("ascend950"):
        parser.error("--soc must be an Ascend A5 / ascend950 variant")
    args.soc = "ascend950"
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.reuse_only and args.check_only:
        parser.error("--reuse-only and --check-only are mutually exclusive")
    return args


def find_cann_root(explicit, environ):
    if explicit is not None:
        candidates = [explicit]
    else:
        candidates = [Path(environ[name]) for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME") if environ.get(name)]
        if environ.get("ASCEND_OPP_PATH"):
            candidates.append(Path(environ["ASCEND_OPP_PATH"]).parent)
        candidates.extend([Path("/usr/local/Ascend/ascend-toolkit/latest"), Path("/usr/local/Ascend/latest")])
    for candidate in candidates:
        if (candidate / "tools/opbuild/op_build").is_file():
            return candidate.resolve()
    locations = "\n  ".join(str(path / "tools/opbuild/op_build") for path in candidates)
    raise PrerequisiteError(
        "The existing CANN runtime may run models but lacks the operator development toolkit. "
        "No toolkit will be downloaded. Missing op_build; checked:\n  " + locations
    )


def container_environment(cann_root, original):
    environ = original.copy()
    # These are existing CANN variables, scoped to child processes, not new VA options.
    environ["ASCEND_HOME_PATH"] = str(cann_root)
    environ["ASCEND_OPP_PATH"] = str(cann_root / "opp")
    setup = next((path for path in (cann_root / "bin/setenv.bash", cann_root / "set_env.sh") if path.is_file()), None)
    if setup:
        result = subprocess.run(
            ["bash", "-c", 'set -e; source "$1" >/dev/null; env -0', "qli-build", str(setup)],
            env=environ,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise PrerequisiteError(
                f"Cannot source existing CANN environment {setup}: {result.stderr.decode(errors='replace')}"
            )
        environ.update(
            entry.decode(errors="surrogateescape").split("=", 1)
            for entry in result.stdout.split(b"\0")
            if b"=" in entry
        )
    environ["ASCEND_HOME_PATH"] = str(cann_root)
    environ["ASCEND_OPP_PATH"] = str(cann_root / "opp")
    return environ


def json_candidates(repo, cann_root, explicit=None):
    if explicit:
        return [explicit]
    candidates = []
    # Locate the already installed torch wheel without importing torch or vLLM.
    spec = importlib.util.find_spec("torch")
    if spec is not None and spec.origin:
        candidates.append(Path(spec.origin).parent / "include")
    ascend_spec = importlib.util.find_spec("vllm_ascend")
    if ascend_spec is not None and ascend_spec.origin:
        candidates.append(Path(ascend_spec.origin).parent.parent / "csrc/third_party/json/include")
    candidates.extend(
        [
            cann_root / "include",
            cann_root / "pkg_inc",
            cann_root / f"{platform.machine()}-linux/include",
            cann_root / "include/third_party",
            cann_root / "include/external",
            cann_root / "ops_base/include",
            cann_root / "ops_base/pkg_inc",
            cann_root / "ops_base/include/third_party/json/include",
            repo / "csrc/third_party/json/include",
            Path("/usr/include"),
            Path("/usr/local/include"),
        ]
    )
    # Some wheels/toolkits bundle nlohmann under a deeper third-party directory.
    # Search only known installed include trees, never fetch an archive.
    for root in list(candidates):
        if root.is_dir() and not (root / "nlohmann/json.hpp").is_file():
            candidates.extend(path.parent.parent for path in sorted(root.glob("**/nlohmann/json.hpp")))
    if not any((path / "nlohmann/json.hpp").is_file() for path in candidates):
        candidates.extend(path.parent.parent for path in sorted(cann_root.glob("**/nlohmann/json.hpp")))
    return candidates


def check_prerequisites(repo, cann_root, environ, explicit_json=None):
    missing = []
    paths = json_candidates(repo, cann_root, explicit_json)
    json_include = next((path.resolve() for path in paths if (path / "nlohmann/json.hpp").is_file()), None)
    if json_include is None:
        missing.append(
            "nlohmann/json.hpp (pass --json-include to reuse another existing include directory); checked "
            + ", ".join(map(str, paths))
        )
    commands = {
        name: shutil.which(name, path=environ.get("PATH")) for name in ("cmake", "gcc", "g++", "bisheng", "opc")
    }
    for name, command in commands.items():
        if not command:
            missing.append(f"{name} in the container PATH after sourcing CANN")
    generator = "Ninja" if shutil.which("ninja", path=environ.get("PATH")) else "Unix Makefiles"
    if generator == "Unix Makefiles" and not shutil.which("make", path=environ.get("PATH")):
        missing.append("ninja or make in the container PATH")
    required = [
        cann_root / "tools/opbuild/op_build",
        cann_root / "toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc",
        cann_root / "toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++",
    ]
    for path in required:
        if not path.is_file() or not os.access(path, os.X_OK):
            missing.append(f"executable {path}")
    for filename in ("libaicpu_context.a", "libbase_ascend_protobuf.a"):
        library_paths = [cann_root / "ops_base/lib64" / filename, cann_root / "lib64" / filename]
        if not any(path.is_file() for path in library_paths):
            missing.append("CANN's existing AICPU static library: " + " or ".join(map(str, library_paths)))
    for filename in ("libopapi_math.so", "libnnopbase.so"):
        library_paths = [cann_root / "lib64" / filename, cann_root / f"{platform.machine()}-linux/lib64" / filename]
        if not any(path.is_file() for path in library_paths):
            missing.append(
                "CANN's real runtime library (a build stub is not sufficient): " + " or ".join(map(str, library_paths))
            )
    acl_headers = [
        cann_root / "include/acl/acl_base.h",
        cann_root / f"{platform.machine()}-linux/include/acl/acl_base.h",
    ]
    acl_header = next((path for path in acl_headers if path.is_file()), None)
    if acl_header is None:
        missing.append("CANN development acl/acl_base.h under " + ", ".join(map(str, acl_headers)))
    # CANN 9.x may expose types through includes in acl_base.h. Do not reject a
    # forwarding header by searching its raw text; compilation resolves those
    # declarations and the runtime probe checks the actual FP4 enum and compute.
    # Compilation uses the selected, already installed Python and CANN Python modules.
    python_check = subprocess.run(
        [sys.executable, "-c", "import numpy; import tbe"], env=environ, capture_output=True, text=True, check=False
    )
    if python_check.returncode:
        missing.append(
            f"existing numpy and CANN tbe Python modules for {sys.executable}: {python_check.stderr.strip()}"
        )
    if missing:
        raise PrerequisiteError(
            "Cannot build QLI V2 offline; nothing will be installed or downloaded. Missing prerequisites:\n  - "
            + "\n  - ".join(missing)
        )
    return {"commands": commands, "generator": generator, "json_include": json_include, "acl_header": acl_header}


def source_identity(repo):
    ref = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    digest = hashlib.sha256(ref.encode())
    digest.update(subprocess.check_output(["git", "diff", "HEAD", "--binary", "--", "csrc"], cwd=repo))
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", "csrc"], cwd=repo
    )
    for relative in sorted(part for part in untracked.split(b"\0") if part):
        digest.update(relative)
        digest.update((repo / os.fsdecode(relative)).read_bytes())
    return ref, digest.hexdigest()


def cann_identity(cann_root, prerequisites):
    digest = hashlib.sha256(str(cann_root).encode())
    digest.update((prerequisites["json_include"] / "nlohmann/json.hpp").read_bytes())
    for path in [prerequisites["acl_header"], cann_root / "version.info", cann_root / "ascend_toolkit_install.info"]:
        if path.is_file():
            digest.update(path.read_bytes())
    for name in ("libnnopbase.so", "libopapi_math.so", "libascendcl.so"):
        for parent in (cann_root / "lib64", cann_root / f"{platform.machine()}-linux/lib64"):
            path = parent / name
            if path.is_file():
                stat = path.stat()
                digest.update(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def build_commands(repo, build_dir, install_dir, cann_root, prerequisites, args):
    cmake = prerequisites["commands"]["cmake"]
    configure = [
        cmake,
        "-S",
        str(repo / "csrc"),
        "-B",
        str(build_dir),
        "-G",
        prerequisites["generator"],
        "-DQLI_STANDALONE_OFFLINE=ON",
        "-DBUILD_OPEN_PROJECT=ON",
        "-DBUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG=ON",
        "-DBUILD_OPS_RTY_KERNEL=OFF",
        "-DENABLE_BUILT_IN=OFF",
        "-DENABLE_BUILD_PKG=OFF",
        "-DENABLE_TEST=OFF",
        "-DENABLE_CCACHE=OFF",
        "-DENABLE_OPS_HOST=ON",
        "-DENABLE_OPS_KERNEL=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_BUILD_MODE=-O2",
        f"-DASCEND_OP_NAME={OPS}",
        f"-DASCEND_COMPUTE_UNIT={args.soc}",
        f"-DVENDOR_NAME={VENDOR}",
        f"-DCUSTOM_ASCEND_CANN_PACKAGE_PATH={cann_root}",
        f"-DQLI_JSON_INCLUDE_DIR={prerequisites['json_include']}",
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DHI_PYTHON={sys.executable}",
        f"-DASCEND_PYTHON_EXECUTABLE={sys.executable}",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        f"-DCMAKE_C_COMPILER={prerequisites['commands']['gcc']}",
        f"-DCMAKE_CXX_COMPILER={prerequisites['commands']['g++']}",
        "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
        "-DFETCHCONTENT_UPDATES_DISCONNECTED=ON",
    ]
    # Kernel compile scripts only exist after prepare_build. Reconfigure so
    # add_bin_compile_target sees them, without invoking the recursive prepare.sh.
    return [
        configure,
        [cmake, "--build", str(build_dir), "--target", "prepare_build", "--parallel", str(args.jobs)],
        configure.copy(),
        [cmake, "--build", str(build_dir), "--parallel", str(args.jobs)],
        [cmake, "--install", str(build_dir), "--prefix", str(install_dir)],
    ]


def installed_artifacts(install_dir):
    vendor = install_dir / f"packages/vendors/{VENDOR}_transformer"
    opapi = vendor / "op_api/lib/libcust_opapi.so"
    if not opapi.is_file():
        raise RuntimeError(f"Build did not install the QLI ACLNN library: {opapi}")
    binary_root = vendor / "op_impl/ai_core/tbe/kernel"
    binaries = list((binary_root / "ascend950/quant_lightning_indexer_v2").glob("*.o"))
    if not binaries:
        raise RuntimeError(f"No compiled device kernels in {binary_root}; refusing to publish an API-only manifest")
    aicpu = vendor / "op_impl/cpu/aicpu_kernel/impl/libtransformer_aicpu_kernels.so"
    if not aicpu.is_file():
        raise RuntimeError(f"No QLI metadata AICPU kernel library: {aicpu}")
    configs = [
        vendor / "op_impl/cpu/config/cust_aicpu_kernel.json",
        binary_root / "config/ascend950/binary_info_config.json",
        binary_root / "config/ascend950/quant_lightning_indexer_v2.json",
    ]
    for config in configs:
        if not config.is_file():
            raise RuntimeError(f"Missing QLI kernel registration/configuration file: {config}")
    # The backend must smoke-test MXFP4 after loading this complete private OPP.
    return opapi.resolve(), vendor.resolve()


def unchanged_clean_build_source(data, repo):
    """Allow a clean schema-1 build to survive changes outside csrc."""
    ref = data.get("ref")
    if (
        repo is None
        or data.get("schema_version") != 1
        or not isinstance(ref, str)
        or len(ref) not in (40, 64)
        or any(char not in "0123456789abcdef" for char in ref)
        or data.get("source_digest") != hashlib.sha256(ref.encode()).hexdigest()
    ):
        return False
    try:
        # Explicitly require the original commit to be available locally. A
        # missing shallow-history object must never be treated as an empty diff.
        subprocess.run(["git", "cat-file", "-e", f"{ref}^{{commit}}"], cwd=repo, check=True, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "diff", "--quiet", ref, "--", "csrc"], cwd=repo, check=True)
        return not subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", "csrc"], cwd=repo
        )
    except (OSError, subprocess.CalledProcessError):
        return False


def reusable_manifest(manifest, source_digest, cann_digest, soc, repo=None):
    try:
        data = json.loads(manifest.read_text())
        if any(data.get(key) != value for key, value in (("cann_digest", cann_digest), ("soc", soc))):
            return False
        if data.get("source_digest") != source_digest and not unchanged_clean_build_source(data, repo):
            return False
        opapi = Path(data["opapi_lib"])
        vendor = Path(data["opp_root"])
        actual_opapi, actual_vendor = installed_artifacts(vendor.parents[2])
        return actual_opapi == opapi and actual_vendor == vendor
    except (OSError, ValueError, KeyError, IndexError, RuntimeError):
        return False


def main(argv=None):
    args = parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    manifest = repo / ".qli-op-build/install.json"
    if args.reuse_only and not manifest.is_file():
        print("No private QLI install manifest; --reuse-only performed no build or installation.")
        return MISSING_PREREQUISITE
    if not args.check_only and not args.reuse_only:
        manifest.unlink(missing_ok=True)
    try:
        if platform.system() != "Linux":
            raise PrerequisiteError(
                "QLI compilation requires the existing Linux A5 container; no container will be downloaded or created."
            )
        cann_root = find_cann_root(args.cann_root, os.environ)
        environ = container_environment(cann_root, os.environ)
        prereqs = check_prerequisites(repo, cann_root, environ, args.json_include)
        print(
            f"Reuse CANN: {cann_root}\nReuse Python: {sys.executable}\nReuse JSON: {prereqs['json_include']}",
            flush=True,
        )
        if args.check_only:
            print(
                "Offline build prerequisites are present. CANN compilation and device execution have not been tested."
            )
            return 0
        ref, source_digest = source_identity(repo)
        cann_digest = cann_identity(cann_root, prereqs)
        if args.reuse_only:
            if reusable_manifest(manifest, source_digest, cann_digest, args.soc, repo):
                print(f"Reuse matching private QLI build: {manifest}")
                return 0
            print("No matching private QLI build; --reuse-only performed no build or installation.")
            return MISSING_PREREQUISITE
        # A fresh directory per build avoids stale binaries after code/CANN changes
        # and failed rebuilds. Completed installs remain usable until explicitly removed.
        base_dir = (args.build_dir or repo / ".qli-op-build").resolve()
        session = base_dir / f"{source_digest[:12]}-{cann_digest[:12]}-{time.time_ns()}"
        build_dir, install_dir = session / "build", session / "install"
        build_dir.mkdir(parents=True)
        for command in build_commands(repo, build_dir, install_dir, cann_root, prereqs, args):
            print("+ " + shlex.join(command), flush=True)
            subprocess.run(command, cwd=repo, env=environ, check=True)
        opapi, vendor = installed_artifacts(install_dir)
        data = {
            "schema_version": 1,
            "opapi_lib": str(opapi),
            "opp_root": str(vendor),
            "ref": ref,
            "source_digest": source_digest,
            "cann_digest": cann_digest,
            "cann_root": str(cann_root),
            "soc": args.soc,
            "requested_soc": args.requested_soc,
            "python": sys.executable,
            "operators": OPS.split(";"),
            "build_dir": str(build_dir),
            "device_tested": False,
        }
        manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(manifest)
        print(f"Private QLI operator installed. Manifest: {manifest}\nA5 MXFP4 smoke/accuracy tests must run next.")
        return 0
    except PrerequisiteError as error:
        print(str(error), file=sys.stderr)
        return MISSING_PREREQUISITE
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(
            f"Offline QLI build failed: {error}\n"
            "No valid install manifest was written. The system CANN was not replaced.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
