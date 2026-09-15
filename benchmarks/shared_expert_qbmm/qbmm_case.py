# SPDX-License-Identifier: Apache-2.0
"""One isolated NPU case. Invoke through run_qbmm.sh, not directly."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

from run_qbmm import write_json


def decode_fp4(packed):
    import numpy as np
    packed = np.asarray(packed, dtype=np.uint8)
    lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    out = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), np.float32)
    out[..., 0::2], out[..., 1::2] = lut[packed & 15], lut[packed >> 4]
    return out


def pack_fp4(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] % 2:
        raise ValueError("FP4 packing requires even K")
    lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
    code = np.searchsorted(lut, np.abs(values))
    if np.any(code > 7) or not np.array_equal(lut[np.minimum(code, 7)], np.abs(values)):
        raise ValueError("Input contains a value not exactly representable in E2M1")
    code = code.astype(np.uint8) | (np.signbit(values).astype(np.uint8) << 3)
    return code[..., 0::2] | (code[..., 1::2] << 4)


def decode_scale(raw):
    import numpy as np
    raw = np.asarray(raw, dtype=np.uint8)
    if np.any(raw == 255):
        raise ValueError("Non-finite E8M0 scale (0xff)")
    return np.exp2(raw.astype(np.float32) - 127)


def encode_scale(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("E8M0 requires finite positive scales")
    code = np.rint(np.log2(values) + 127)
    if np.any(code < 0) or np.any(code > 254):
        raise ValueError("Scale outside E8M0 range")
    raw = code.astype(np.uint8)
    if not np.array_equal(decode_scale(raw), values):
        raise ValueError("Scale is not an exact power of two")
    return raw


def error_metrics(actual, reference):
    import numpy as np
    a, r = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    if a.shape != r.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(r)):
        raise ValueError("Accuracy requires matching shapes and all-finite outputs")
    diff = a - r
    norm_a, norm_r = np.linalg.norm(a), np.linalg.norm(r)
    return {"max_abs": float(np.max(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "relative_l2": float(np.linalg.norm(diff) / norm_r) if norm_r else None,
            "cosine": float(np.sum(a * r) / (norm_a * norm_r)) if norm_a and norm_r else None,
            "elements": int(a.size)}


def prepare_call(torch, npu, kind, a, w_nk, a_scale, w_scale_ng, weight_format=None):
    """Use the v0.26.0rc1 Linear layout and exact npu_quant_matmul arguments."""
    n = w_nk.shape[0]
    ws = w_scale_ng.reshape(n, -1, 2).transpose(0, 1)
    if kind == "mxa8w4":
        fmt = weight_format or "NZ"
        if fmt == "NZ":
            w_nk = npu.npu_format_cast(w_nk, 29, customize_dtype=torch.float8_e4m3fn,
                                     input_dtype=npu.float4_e2m1fn_x2)
        elif fmt != "ND":
            raise ValueError(f"Unsupported weight format: {fmt}")
        w = w_nk.transpose(-1, -2)
        extra = {"x2_dtype": npu.float4_e2m1fn_x2, "group_sizes": [0, 0, 32]}
    elif kind == "mxa8w8":
        fmt = "ND"
        w = w_nk.transpose(0, 1).contiguous()
        ws = ws.contiguous()
        extra = {"group_sizes": [1, 1, 32]}
    else:
        raise ValueError(f"Unknown kind {kind}")
    kwargs = dict(scale_dtype=npu.float8_e8m0fnu, pertoken_scale=a_scale,
                  pertoken_scale_dtype=npu.float8_e8m0fnu, bias=None,
                  output_dtype=torch.bfloat16, **extra)

    def describe(tensor):
        return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype), "contiguous": tensor.is_contiguous()}

    layout = {"weight_format": fmt, "a": describe(a), "weight": describe(w),
              "a_scale": describe(a_scale), "w_scale": describe(ws),
              "group_sizes": extra["group_sizes"], "scale_encoding": "E8M0, block32",
              "output_dtype": "bfloat16", "bridge": "torch_npu.npu_quant_matmul"}
    if hasattr(npu, "get_npu_format"):
        layout["observed_weight_format_code"] = int(npu.get_npu_format(w))
    return lambda: npu.npu_quant_matmul(a, w, ws, **kwargs), layout


def execute(spec, directory, report):
    import numpy as np
    import torch
    import torch_npu
    from reference_qbmm import (make_official_inputs, official_compare,
                                official_reference, reference_metadata)
    from profile_qbmm import profile_call

    for symbol in ("npu_quant_matmul", "npu_dynamic_mx_quant", "npu_format_cast",
                   "float8_e8m0fnu", "float4_e2m1fn_x2"):
        if not hasattr(torch_npu, symbol):
            raise RuntimeError(f"Installed torch_npu lacks {symbol}; this environment cannot run the A5 MX path")
    if not torch.npu.is_available():
        raise RuntimeError("No NPU available to installed torch_npu")
    torch.npu.set_device(spec["device"])
    device = f"npu:{spec['device']}"
    report["environment"] = {"python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "device": device, "device_name": torch.npu.get_device_name(spec["device"]),
        "variables": {key: os.environ.get(key) for key in
                      ("ASCEND_LAUNCH_BLOCKING", "FLA_NPU_DISABLE_PTH", "TORCH_DEVICE_BACKEND_AUTOLOAD",
                       "ASCEND_PROCESS_LOG_PATH", "ASCEND_HOME_PATH", "ASCEND_RT_VISIBLE_DEVICES")}}
    report["reference"] = reference_metadata()
    cann_versions = {}
    candidates = [Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
                  Path("/usr/local/Ascend/ascend-toolkit/latest/version.info")]
    if os.environ.get("ASCEND_HOME_PATH"):
        candidates += [Path(os.environ["ASCEND_HOME_PATH"]) / name for name in ("version.cfg", "version.info")]
    for path in candidates:
        if path.is_file():
            cann_versions[str(path)] = path.read_text(errors="replace")[:10000]
    report["environment"]["cann_version_files"] = cann_versions
    if shutil.which("npu-smi"):
        try:
            info = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=10, check=False)
            (directory / "npu-smi.txt").write_text(info.stdout + info.stderr)
        except subprocess.TimeoutExpired:
            (directory / "npu-smi.txt").write_text("npu-smi info timed out after 10s")
    for package in ("numpy", "ml_dtypes", "en_dtypes"):
        report["environment"][package] = importlib.metadata.version(package)
    report["status"] = "preparing"
    write_json(directory / "result.json", report)
    case, kind = spec["case"], spec["kind"]
    m, k, n = (case[key] for key in ("m", "k", "n"))
    original = None
    with torch.inference_mode():
        if case["stage"] in ("official", "derived_control"):
            inputs = make_official_inputs(dict(case, stage="official", kind="mxa8w4"), spec["seed"])
            a = torch.from_numpy(inputs["a_values"]).to(torch.float8_e4m3fn).to(device)
            w = (torch.from_numpy(pack_fp4(inputs["w_values"])) if kind == "mxa8w4" else
                 torch.from_numpy(inputs["w_values"]).to(torch.float8_e4m3fn)).to(device)
            a_scale = torch.from_numpy(encode_scale(inputs["a_scale"])).reshape(m, k // 64, 2).to(device)
            w_scale = torch.from_numpy(encode_scale(inputs["w_scale"])).to(device)
        else:
            generator = torch.Generator(device="cpu").manual_seed(spec["seed"])
            x = torch.randn(m, k, generator=generator, dtype=torch.float32).to(torch.bfloat16)
            weight = (torch.randn(n, k, generator=generator, dtype=torch.float32) / k ** .5).to(torch.bfloat16)
            original = (x.float().numpy(), weight.float().numpy())
            report["input_policy"] = "Seeded Gaussian BF16 A, W/sqrt(K); same seed and source tensors for W4/W8"
            report["source_input_sha256"] = [hashlib.sha256(t.tobytes()).hexdigest() for t in original]
            a, a_scale = torch_npu.npu_dynamic_mx_quant(x.to(device), dst_type=torch.float8_e4m3fn)
            w, w_scale = torch_npu.npu_dynamic_mx_quant(weight.to(device),
                dst_type=torch_npu.float4_e2m1fn_x2 if kind == "mxa8w4" else torch.float8_e4m3fn)
            torch.npu.synchronize()
            inputs = {"a_values": a.cpu().float().numpy(),
                      "w_values": decode_fp4(w.cpu().numpy()) if kind == "mxa8w4" else w.cpu().float().numpy(),
                      "a_scale": decode_scale(a_scale.cpu().numpy()).reshape(m, -1),
                      "w_scale": decode_scale(w_scale.cpu().numpy()).reshape(n, -1)}
        # Golden is CPU work. Quantization, transfers and format cast never enter profiling.
        expected = official_reference(kind, inputs["a_values"], inputs["w_values"],
                                      inputs["a_scale"], inputs["w_scale"])
        call, layout = prepare_call(torch, torch_npu, kind, a, w, a_scale, w_scale,
                                    case.get("weight_format"))
        report["layout"] = layout
        actual = call().cpu().float().numpy()
        torch.npu.synchronize()
        report["operator_error"] = error_metrics(actual, expected)
        # Derived GLM cases retain the stricter official NZ case tolerance.
        comparison_case = case if case["stage"] != "glm" else {
            "original_csv_fields": {"precision_tolerances": "((0.001,0.001),)", "absolute_precision": ""}}
        comparison = official_compare(actual, expected, comparison_case)
        report["operator_check"] = {"pass": bool(comparison["pass"]), "precision": float(comparison["precision"])}
        report["operator_tolerance"] = comparison_case["original_csv_fields"]["precision_tolerances"]
        if original is not None:
            floating_reference = original[0] @ original[1].T
            report["quantization_loss"] = error_metrics(expected, floating_reference)
            report["total_error_vs_bf16_inputs"] = error_metrics(actual, floating_reference)
            report["quantization_loss_reference"] = "FP32 matmul of original BF16 inputs; diagnostic only, no model accuracy claim"
        if not comparison["pass"]:
            np.savez_compressed(directory / "accuracy_failure.npz", actual=actual, expected=expected)
            raise RuntimeError(f"Official operator accuracy gate failed: {comparison}")
        report["status"] = "accuracy_passed"
        write_json(directory / "result.json", report)
        if not spec["accuracy_only"]:
            report["performance"] = profile_call(call, directory / "trace", spec["warmup"], spec["iterations"])
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    directory = args.spec.resolve().parent
    report = {"case": spec["case"], "kind": spec["kind"], "status": "starting"}
    try:
        execute(spec, directory, report)
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        if hasattr(error, "diagnostics"):
            report["profile_diagnostics"] = error.diagnostics
        traceback.print_exc()
    write_json(directory / "result.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
