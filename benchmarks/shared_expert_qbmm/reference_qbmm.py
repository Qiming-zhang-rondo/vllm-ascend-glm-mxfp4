"""Thin adapters around byte-for-byte vendored CANN QBMM golden sources.

No matmul/dequantization golden is implemented here.  Input arrays are decoded
quantized values, not the pre-quantization BF16 model inputs.  All arithmetic and
output rounding are performed by the original CANN TestSpec.golden methods.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys

_BASE = Path(__file__).resolve().parent
_VENDOR = _BASE / "vendor" / "cann-ops-nn"
_MODULES = None


def verify_vendor() -> dict:
    """Fail if any vendored upstream source differs from the pinned Git blob."""
    manifest = json.loads((_VENDOR / "MANIFEST.json").read_text())
    for item in manifest["files"]:
        path = _VENDOR / item["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise RuntimeError(f"Vendored source checksum mismatch: {path}")
    return {"commit": manifest["commit"], "verified_files": len(manifest["files"])}


def _load():
    global _MODULES
    if _MODULES is not None:
        return _MODULES
    verify_vendor()
    dependencies = ("numpy", "torch", "ml_dtypes", "en_dtypes")
    missing = []
    for name in dependencies:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            missing.append(f"{name}: {exc}")
    if missing:
        raise RuntimeError(
            "Original CANN golden dependencies unavailable. No substitute golden "
            "or dtype shim is used. Supply these in the A5 Python environment: "
            + "; ".join(missing)
        )
    common = _VENDOR / "matmul/common/tests/st/arch35"
    v3path = _VENDOR / "matmul/quant_batch_matmul_v3/tests/assets"
    v4path = _VENDOR / "matmul/quant_batch_matmul_v4/tests/assets"
    for path in (common, v3path, v4path):
        sys.path.insert(0, str(path))
    np = importlib.import_module("numpy")
    util = importlib.import_module("matmul_golden_util")
    v3 = importlib.import_module("qbmmv3_kernel_golden")
    v4 = importlib.import_module("qbmmv4_kernel_golden")
    _MODULES = np, util, v3, v4
    return _MODULES


def _exact_cast(values, dtype, name):
    np, _, _, _ = _load()
    values = np.asarray(values, dtype=np.float32)
    typed = values.astype(dtype)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.array_equal(typed.astype(np.float32), values):
        raise ValueError(f"{name} is not already exactly representable as {dtype}")
    return typed


def official_reference(kind, a_values_mk, w_values_nk, a_scale_mg, w_scale_ng):
    """Return float32 view of BF16-rounded *original CANN golden* output [M,N].

    Scales are the decoded E8M0 powers of two per 32 K elements.  Inputs must
    already be decoded E4M3FN (A and W8) / E2M1 (W4) values.  Canonical W is
    [N,K], so transpose_x2=True.  Scale storage is [rows,ceil(K/64),2].
    """
    np, util, v3, v4 = _load()
    if kind not in ("mxa8w4", "mxa8w8"):
        raise ValueError(f"Unsupported kind: {kind}")
    a = _exact_cast(a_values_mk, util.np_fp8_e4m3, "a_values")
    w_dtype = util.np_fp4_e2m1 if kind == "mxa8w4" else util.np_fp8_e4m3
    w = _exact_cast(w_values_nk, w_dtype, "w_values")
    if a.ndim != 2 or w.ndim != 2 or a.shape[1] != w.shape[1]:
        raise ValueError("Expected A[M,K] and W[N,K] with matching K")
    m, k = a.shape
    n = w.shape[0]
    g = (k + 31) // 32
    padded_g = 2 * ((g + 1) // 2)

    def prepare_scale(values, rows, name):
        s = np.asarray(values, dtype=np.float32)
        if s.shape not in ((rows, g), (rows, padded_g)):
            raise ValueError(f"{name} shape {s.shape}, expected {(rows, g)}")
        if np.any(s <= 0):
            raise ValueError(f"{name} must contain positive E8M0 scales")
        if s.shape[1] < padded_g:
            s = np.pad(s, ((0, 0), (0, padded_g - g)), constant_values=1)
        return _exact_cast(s, util.np_mx_scale, name).reshape(rows, -1, 2)

    a_scale = prepare_scale(a_scale_mg, m, "a_scale")
    w_scale = prepare_scale(w_scale_ng, n, "w_scale")
    if kind == "mxa8w4":
        result = v4.QuantBatchMatmulV4TestSpec.golden(
            a, w, x1_scale=a_scale, x2_scale=w_scale,
            transpose_x1=False, transpose_x2=True,
            dtype=27, group_size=32, output_dtypes=["bfloat16"],
        )[0]
    else:
        result = v3.QuantBatchMatmulV3TestSpec.golden(
            a, w, w_scale, pertoken_scale=a_scale,
            transpose_x1=False, transpose_x2=True,
            dtype=27, group_size=32, output_dtypes=["bfloat16"],
        )[0]
    return result.astype(np.float32)


def make_official_inputs(case, seed=2026):
    """Allocate official CSV parameters/ranges and run original input sanitation.

    The upstream TTK infrastructure is not vendored.  This harness supplies
    deterministic allocation, then delegates E8M0 sanitation to the unchanged
    upstream TestSpec.  Returned values are for storage packing, not a new
    quantization step.  Only the three no-bias MXA8W4 rows in official_cases.json
    are supported; do not label derived W8 controls as original upstream cases.
    """
    np, util, _, v4 = _load()
    if case.get("stage") != "official" or case["kind"] != "mxa8w4":
        raise ValueError("make_official_inputs requires an original MXA8W4 CSV case")
    row = case["original_csv_fields"]
    api = case["source_type"] == "aclnn_st"
    ranges = ast.literal_eval(row["input_data_ranges"])
    shapes = ast.literal_eval(row["tensor_view_shapes"] if api else row["input_ori_shapes"])
    a_scale_index, w_scale_index = (2, 3) if api else (3, 4)
    dtypes = (util.np_fp8_e4m3, util.np_fp4_e2m1, util.np_mx_scale, util.np_mx_scale)
    indexes = (0, 1, a_scale_index, w_scale_index)
    saved_state = np.random.get_state()
    try:
        np.random.seed(seed)
        arrays = []
        for index, dtype in zip(indexes, dtypes):
            # The singleton CSV range applies to every allocated input.
            low, high = ranges[0 if len(ranges) == 1 else index]
            arrays.append(np.random.uniform(low, high, shapes[index]).astype(dtype))
        a, w, a_scale, w_scale = arrays
        cleaned = v4.QuantBatchMatmulV4TestSpec.customize_inputs(
            a, w, x1_scale=a_scale, x2_scale=w_scale,
            testcase_name=case["source_testcase"],
        )
        a, w, a_scale, w_scale = cleaned[0], cleaned[1], cleaned[3], cleaned[4]
    finally:
        np.random.set_state(saved_state)
    return {
        "a_values": a.astype(np.float32),
        "w_values": w.astype(np.float32),
        "a_scale": a_scale.astype(np.float32).reshape(case["m"], -1),
        "w_scale": w_scale.astype(np.float32).reshape(case["n"], -1),
    }


def official_compare(actual, expected, case):
    """Use upstream's comparator, retaining CSV rtol/ptol/atol semantics."""
    _, util, _, _ = _load()

    class Context:
        csv_fields = case["original_csv_fields"]

    return util.isclose_compare(actual, expected, compare_context=Context())


def reference_metadata():
    result = verify_vendor()
    result.update({
        "mxa8w4": "qbmmv4_kernel_golden.QuantBatchMatmulV4TestSpec.golden",
        "mxa8w8": "qbmmv3_kernel_golden.QuantBatchMatmulV3TestSpec.golden",
        "input_semantics": "decoded quantized values and decoded E8M0 scales",
        "output_semantics": "upstream BF16-rounded result returned as float32",
        "golden_modified": False,
    })
    return result


if __name__ == "__main__":
    print(json.dumps(reference_metadata(), indent=2))
