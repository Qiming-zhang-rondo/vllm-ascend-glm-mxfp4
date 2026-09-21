# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU MX fake quantization for the QSFA input sensitivity experiment.

Contract: CANN ops-nn@2a77283db46e6648ff47bc8277442cf9c721e3c2,
quant/dynamic_mx_quant/tests/assets/golden.py::_mx_calculate_share_exp,
_mx_round_mantissa, _mx_quantize_to_element_format and _mx_quantize.
Both formats use axis=-1, block_size=32, scale_alg=0 and E8M0 scales.
MXFP4 uses round_mode='round' (ties away); MXFP8 uses 'rint' (ties even).

The result is dequantized float32, not an NPU low-bit tensor or a test of the
hardware quantizer. A following cast to BF16 introduces separate input rounding.
"""

import torch

GROUP_SIZE = 32
E8M0_MIN_EXPONENT = -127
FP4_MAX_EXPONENT = 2
FP8_MAX_EXPONENT = 8
FP8_MAX_VALUE = 448.0


def _prepare_groups(x: torch.Tensor, max_exponent: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(x, torch.Tensor):
        raise TypeError("MX roundtrip input must be a torch.Tensor")
    if x.device.type != "cpu":
        raise ValueError("MX roundtrip input must be on CPU")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("MX roundtrip input must be float16, bfloat16 or float32")
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("MX roundtrip input must have a nonempty last dimension")
    values = x.detach().float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("MX roundtrip input must be finite")
    padding = (-values.shape[-1]) % GROUP_SIZE
    if padding:
        values = torch.nn.functional.pad(values, (0, padding))
    groups = values.reshape(*values.shape[:-1], -1, GROUP_SIZE)
    maxima = groups.abs().amax(dim=-1, keepdim=True)
    _, exponents = torch.frexp(maxima)
    # frexp(a).exponent - 1 == floor(log2(a)) without logarithm boundary error.
    exponents = (exponents - 1 - max_exponent).clamp_min(E8M0_MIN_EXPONENT)
    exponents = torch.where(maxima == 0, E8M0_MIN_EXPONENT, exponents)
    scales = torch.ldexp(torch.ones_like(maxima), exponents)
    return groups / scales, scales


def mxfp4_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Roundtrip through E2M1/group32/E8M0, with ties away and saturation."""
    normalized, scales = _prepare_groups(x, FP4_MAX_EXPONENT)
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device="cpu", dtype=torch.float32)
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cpu", dtype=torch.float32)
    codes = torch.bucketize(normalized.abs().contiguous(), midpoints, right=True)
    quantized = torch.copysign(magnitudes[codes], normalized)
    return (quantized * scales).flatten(-2)[..., : x.shape[-1]].contiguous()


def mxfp8_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Roundtrip through E4M3FN/group32/E8M0, scale_alg=0 and nearest-even.

    Explicit saturation is required: casting out-of-range values directly to
    torch.float8_e4m3fn can produce NaN instead of the CANN golden's ±448.
    """
    normalized, scales = _prepare_groups(x, FP8_MAX_EXPONENT)
    quantized = normalized.clamp(-FP8_MAX_VALUE, FP8_MAX_VALUE).to(torch.float8_e4m3fn).float()
    return (quantized * scales).flatten(-2)[..., : x.shape[-1]].contiguous()
