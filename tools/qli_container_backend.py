# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Call the container's QLI V2 ACLNN APIs without installing vLLM or an extension.

This intentionally synchronous, single-device benchmark adapter is not a model
runtime integration. Every call drains torch_npu's task queue before raw ACLNN
submission and synchronizes before releasing descriptors or workspace. Report
its timing as synchronous wall latency including ACLNN preparation, not kernel
latency. API presence alone does not establish MXFP4 compute support.

ABI references: CANN ops-transformer 632dddba712a4e6cace3f8b44f198aef8a82ce3e,
attention/quant_lightning_indexer_v2/docs/aclnnQuantLightningIndexerV2.md;
attention/quant_lightning_indexer_v2_metadata/op_host/op_api/
aclnn_quant_lightning_indexer_v2_metadata.h. The aclCreateTensor ABI matches
csrc/aclnn_torch_adapter/op_api_common.h in this repository.
"""

import ast
import ctypes
import ctypes.util
import importlib
import os
import re
from pathlib import Path

I64 = ctypes.c_int64
U64 = ctypes.c_uint64
PTR = ctypes.c_void_p
I64_PTR = ctypes.POINTER(I64)
API_NAMES = (
    "aclnnQuantLightningIndexerV2GetWorkspaceSize",
    "aclnnQuantLightningIndexerV2",
    "aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize",
    "aclnnQuantLightningIndexerV2Metadata",
)


class BackendUnavailable(RuntimeError):
    """The container has no loadable matching QLI V2 API pair."""


class QLICallError(RuntimeError):
    """An ACLNN call failed; the numeric status alone does not identify why."""

    def __init__(self, api_name, status, phase):
        self.api_name, self.status, self.phase = api_name, status, phase
        super().__init__(
            f"{api_name} {phase} failed: status={status}. API presence does not establish "
            "support for this quant_mode/layout; see the CANN operator log for the cause."
        )


def _integer_expression(expression, names):
    """Evaluate only integer enum expressions; never execute header text."""
    expression = re.sub(r"\b(0x[0-9a-fA-F]+|\d+)[uUlL]+\b", r"\1", expression.strip())
    tree = ast.parse(expression, mode="eval").body

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name):
            return names[node.id]
        if isinstance(node, ast.UnaryOp):
            value = evaluate(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.Invert):
                return ~value
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.LShift):
                return left << right
            if isinstance(node.op, ast.BitOr):
                return left | right
        raise ValueError("unsupported C enum expression")

    return evaluate(tree)


def parse_acl_enums(header_text):
    """Read ACL datatype/format enums from the installed CANN header."""
    clean = re.sub(r"/\*.*?\*/|//[^\n]*", "", header_text, flags=re.S)
    values = {}
    for body in re.findall(r"\benum(?:\s+\w+)?\s*\{([^{}]*)\}", clean, flags=re.S):
        previous = -1
        for item in body.split(","):
            match = re.fullmatch(r"\s*(\w+)\s*(?:=\s*(.*?))?\s*", item, flags=re.S)
            if match is None:
                continue
            name, expression = match.groups()
            try:
                value = previous + 1 if expression is None else _integer_expression(expression, values)
            except (KeyError, ValueError, SyntaxError, TypeError):
                # Ignore unrelated enum expressions; required names are checked
                # by the caller. Do not infer values after an unknown member.
                previous = None
                continue
            values[name] = value
            previous = value
    return values


def packed_layout(shape, strides, storage_offset, storage_bytes):
    """Convert packed byte units to ACL logical nibble units, including views."""
    shape, strides = list(shape), list(strides)
    if not shape or len(shape) != len(strides) or strides[-1] != 1:
        raise ValueError("packed FP4 requires a contiguous innermost byte axis")
    integers = [*shape, *strides, storage_offset, storage_bytes]
    if any(value < 0 or value > (2**63 - 1) // 2 for value in integers):
        raise ValueError("packed FP4 layout exceeds the logical element range")
    shape[-1] *= 2
    strides[:-1] = [stride * 2 for stride in strides[:-1]]
    return shape, strides, storage_offset * 2, storage_bytes * 2


def _toolkit_roots(cann_root=None):
    roots = [Path(cann_root).expanduser()] if cann_root else []
    for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_HOME"):
        if os.environ.get(name):
            roots.append(Path(os.environ[name]).expanduser())
    roots.extend((Path("/usr/local/Ascend/ascend-toolkit/latest"), Path("/usr/local/Ascend/cann")))
    return list(dict.fromkeys(roots))


def _load_enums(explicit_header, cann_root=None):
    candidates = [Path(explicit_header).expanduser()] if explicit_header else []
    if not explicit_header:
        for root in _toolkit_roots(cann_root):
            candidates.extend(root.glob("*/include/acl/acl_base.h"))
            candidates.extend((root / "include/acl/acl_base.h", root / "include/acl_base.h"))
    required = (
        "ACL_FLOAT",
        "ACL_INT32",
        "ACL_INT64",
        "ACL_UINT8",
        "ACL_BF16",
        "ACL_FLOAT8_E4M3FN",
        "ACL_FLOAT8_E8M0",
        "ACL_FLOAT4_E2M1",
        "ACL_FORMAT_ND",
    )
    errors = []
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        enums = parse_acl_enums(candidate.read_text())
        missing = [name for name in required if name not in enums]
        if not missing:
            return candidate.resolve(), enums
        errors.append(f"{candidate}: missing {', '.join(missing)}")
    detail = "; ".join(errors) or "no readable acl/acl_base.h found"
    raise RuntimeError(
        "Cannot verify ACL datatype/format constants from the container's CANN headers: "
        f"{detail}. Source the installed CANN set_env.sh or provide acl_header. "
        "No dependencies will be downloaded."
    )


def _library_candidates(explicit_library, cann_root=None):
    if explicit_library:
        path = Path(explicit_library).expanduser()
        if not path.is_file():
            raise RuntimeError(f"The specified op API library does not exist: {path}")
        return [str(path.resolve())]
    candidates = []
    for entry in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":"):
        if entry:
            root = Path(entry)
            candidates.extend((root / "op_api/lib/libcust_opapi.so", root / "lib/libcust_opapi.so"))
    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if entry:
            candidates.append(Path(entry) / "libcust_opapi.so")
    for root in _toolkit_roots(cann_root):
        candidates.extend(root.glob("opp/vendors/*/op_api/lib/libcust_opapi.so"))
        candidates.extend((root / "lib64/libopapi.so", root / "aarch64-linux/lib64/libopapi.so"))
    paths = [str(path.resolve()) for path in candidates if path.is_file()]
    paths.append("libopapi.so")
    return list(dict.fromkeys(paths))


class _DlInfo(ctypes.Structure):
    _fields_ = [("filename", ctypes.c_char_p), ("base", PTR), ("symbol", ctypes.c_char_p), ("address", PTR)]


def _symbol_path(function, fallback):
    try:
        library = ctypes.CDLL(ctypes.util.find_library("dl") or None)
        dladdr = library.dladdr
        dladdr.argtypes = [PTR, ctypes.POINTER(_DlInfo)]
        dladdr.restype = ctypes.c_int
        info = _DlInfo()
        if dladdr(ctypes.cast(function, PTR), ctypes.byref(info)) and info.filename:
            return str(Path(os.fsdecode(info.filename)).resolve())
    except (OSError, AttributeError):
        pass
    return str(Path(fallback).resolve()) if Path(fallback).is_file() else fallback


class QLIBackend:
    """Small synchronous adapter to installed QLI V2 compute and metadata APIs."""

    def __init__(self, opapi_lib=None, acl_header=None, cann_root=None):
        self.torch = importlib.import_module("torch")
        self.torch_npu = importlib.import_module("torch_npu")
        if not self.torch.npu.is_available():
            raise RuntimeError("An available Ascend NPU is required; no CPU/emulated compute fallback is used")
        self.header, self.enums = _load_enums(acl_header, cann_root)
        self._libraries = []
        errors = []
        for candidate in _library_candidates(opapi_lib, cann_root):
            try:
                library = ctypes.CDLL(candidate, mode=ctypes.RTLD_LOCAL)
                functions = {name: getattr(library, name) for name in API_NAMES}
            except (OSError, AttributeError) as error:
                errors.append(f"{candidate}: {error}")
                continue
            api_paths = {name: _symbol_path(function, candidate) for name, function in functions.items()}
            if len(set(api_paths.values())) != 1:
                errors.append(f"{candidate}: compute/metadata symbols resolve to different libraries: {api_paths}")
                continue
            if opapi_lib and next(iter(api_paths.values())) != str(Path(candidate).resolve()):
                errors.append(f"{candidate}: requested API symbols are supplied by a dependency instead: {api_paths}")
                continue
            self._libraries.append(library)
            self._api = functions
            self._api_paths = api_paths
            break
        else:
            raise BackendUnavailable(
                "No installed library exports both QLI V2 compute and metadata two-phase APIs. "
                "Install the selective QLI operator package, or specify its libcust_opapi.so. "
                "Installing torch/vLLM will not supply a missing CANN kernel.\n" + "\n".join(errors)
            )

        # aclCreateTensor/aclDestroyTensor normally come from libnnopbase through
        # libopapi's dependency chain. Resolve both from the same handle.
        handles = [self._libraries[0]]
        for candidate in ("libnnopbase.so", "libopapi.so"):
            try:
                library = ctypes.CDLL(candidate, mode=ctypes.RTLD_LOCAL)
                self._libraries.append(library)
                handles.append(library)
            except OSError:
                continue
        for library in handles:
            if hasattr(library, "aclCreateTensor") and hasattr(library, "aclDestroyTensor"):
                self._create_tensor, self._destroy_tensor = library.aclCreateTensor, library.aclDestroyTensor
                self._api_paths["aclCreateTensor"] = _symbol_path(self._create_tensor, library._name)
                break
        else:
            raise RuntimeError("The installed CANN libraries do not export aclCreateTensor/aclDestroyTensor")
        self._create_tensor.argtypes = [I64_PTR, U64, ctypes.c_int, I64_PTR, I64, ctypes.c_int, I64_PTR, U64, PTR]
        self._create_tensor.restype = PTR
        self._destroy_tensor.argtypes, self._destroy_tensor.restype = [PTR], ctypes.c_int
        self._api[API_NAMES[0]].argtypes = (
            [PTR] * 13
            + [I64] * 3
            + [ctypes.c_char_p] * 2
            + [I64] * 3
            + [PTR] * 2
            + [ctypes.POINTER(U64), ctypes.POINTER(PTR)]
        )
        self._api[API_NAMES[2]].argtypes = (
            [PTR] * 5 + [I64] * 8 + [ctypes.c_char_p] * 2 + [I64] * 2 + [PTR, ctypes.POINTER(U64), ctypes.POINTER(PTR)]
        )
        for name in (API_NAMES[1], API_NAMES[3]):
            self._api[name].argtypes = [PTR, U64, PTR, PTR]
        for function in self._api.values():
            function.restype = ctypes.c_int

    def describe(self):
        return {
            "backend": "container_acl_ctypes",
            "acl_header": str(self.header),
            "api_libraries": dict(self._api_paths),
            "acl_fp4_dtype": self.enums["ACL_FLOAT4_E2M1"],
            "acl_e8m0_dtype": self.enums["ACL_FLOAT8_E8M0"],
            "mxfp4_compute_verified": False,
            "capability_note": "API presence only; mode=5 compute must complete and pass numerical checks",
            "timing": "synchronous wall latency including descriptor/workspace preparation and queue synchronization",
        }

    def _dtype_name(self, tensor):
        types = {
            self.torch.float32: "ACL_FLOAT",
            self.torch.bfloat16: "ACL_BF16",
            self.torch.int32: "ACL_INT32",
            self.torch.int64: "ACL_INT64",
            self.torch.uint8: "ACL_UINT8",
        }
        fp8 = getattr(self.torch, "float8_e4m3fn", None)
        if fp8 is not None:
            types[fp8] = "ACL_FLOAT8_E4M3FN"
        try:
            return types[tensor.dtype]
        except KeyError as error:
            raise ValueError(f"Unsupported ACL adapter tensor dtype: {tensor.dtype}") from error

    def _descriptor(self, tensor, handles, dtype_name=None):
        if tensor is None:
            return None
        if tensor.device.type not in ("npu", "privateuseone"):
            raise ValueError("QLI benchmark ACL descriptors require NPU tensors")
        get_format = getattr(self.torch_npu, "get_npu_format", None)
        if get_format is None:
            raise RuntimeError("torch_npu.get_npu_format is required to verify base-format tensor storage")
        allowed_formats = [self.enums["ACL_FORMAT_ND"]]
        if "ACL_FORMAT_NCHW" in self.enums:
            allowed_formats.append(self.enums["ACL_FORMAT_NCHW"])
        if get_format(tensor) not in allowed_formats:
            raise ValueError("QLI benchmark supports only base-format NPU storage, not opaque NZ/FRACTAL storage")
        dtype_name = dtype_name or self._dtype_name(tensor)
        shape, strides = list(tensor.shape), list(tensor.stride())
        offset = tensor.storage_offset()
        storage = tensor.untyped_storage()
        storage_size = storage.nbytes() // tensor.element_size()
        if dtype_name in ("ACL_FLOAT4_E2M1", "ACL_FLOAT8_E8M0") and tensor.element_size() != 1:
            raise ValueError("MXFP4/E8M0 require byte-addressed storage")
        if dtype_name == "ACL_FLOAT4_E2M1":
            shape, strides, offset, storage_size = packed_layout(shape, strides, offset, storage.nbytes())
        shape_array, stride_array = (I64 * len(shape))(*shape), (I64 * len(strides))(*strides)
        storage_array = (I64 * 1)(storage_size)
        handle = self._create_tensor(
            shape_array,
            len(shape),
            self.enums[dtype_name],
            stride_array,
            offset,
            self.enums["ACL_FORMAT_ND"],
            storage_array,
            1,
            storage.data_ptr(),
        )
        if not handle:
            raise RuntimeError(f"aclCreateTensor failed for {dtype_name} {shape}")
        # Keep all Python owners and shape buffers through ACL execution, even
        # though aclCreateTensor normally copies the descriptor arrays.
        handles.append((handle, tensor, shape_array, stride_array, storage_array))
        return handle

    def _execute(self, api_name, arguments, handles, device):
        with self.torch.npu.device(device):
            self._execute_current_device(api_name, arguments, handles, device)

    def _execute_current_device(self, api_name, arguments, handles, device):
        workspace = None
        submitted = False
        try:
            self.torch.npu.synchronize(device)
            workspace_size, executor = U64(), PTR()
            status = self._api[api_name + "GetWorkspaceSize"](
                *arguments,
                ctypes.byref(workspace_size),
                ctypes.byref(executor),
            )
            if status != 0:
                raise QLICallError(api_name, status, "GetWorkspaceSize")
            if not executor.value:
                raise RuntimeError(f"{api_name}GetWorkspaceSize returned success without an executor")
            workspace = self.torch.empty(workspace_size.value, dtype=self.torch.uint8, device=device)
            stream = self.torch.npu.current_stream(device).npu_stream
            submitted = True
            status = self._api[api_name](workspace.data_ptr(), workspace_size.value, executor, stream)
            if status != 0:
                raise QLICallError(api_name, status, "execute")
        finally:
            # Raw ACL launch bypasses torch_npu's task queue. Synchronization is
            # necessary before dropping workspace and input descriptor owners.
            try:
                if submitted:
                    self.torch.npu.synchronize(device)
            finally:
                for handle, *_ in reversed(handles):
                    self._destroy_tensor(handle)
                handles.clear()

    def create_metadata(self, *, num_heads_q, head_dim, topk, quant_mode, cu_seqlens_q, seqused_k):
        device = cu_seqlens_q.device
        if seqused_k.device != device:
            raise ValueError("Metadata sequence tensors must be on the same NPU device")
        if cu_seqlens_q.dtype != self.torch.int32 or seqused_k.dtype != self.torch.int32:
            raise ValueError("Metadata requires INT32 cumulative Q lengths and per-request K lengths")
        if cu_seqlens_q.ndim != 1 or seqused_k.ndim != 1 or cu_seqlens_q.numel() != seqused_k.numel() + 1:
            raise ValueError("Metadata requires B+1 cumulative Q lengths and B per-request K lengths")
        output = self.torch.empty(1024, dtype=self.torch.int32, device=device)
        handles = []
        try:
            arguments = [
                self._descriptor(cu_seqlens_q, handles),
                None,
                None,
                self._descriptor(seqused_k, handles),
                None,
                num_heads_q,
                1,
                head_dim,
                topk,
                quant_mode,
                seqused_k.numel(),
                -1,
                -1,
                b"TND",
                b"PA_BBND",
                3,
                1,
                self._descriptor(output, handles),
            ]
            self._execute("aclnnQuantLightningIndexerV2Metadata", arguments, handles, device)
        finally:
            for handle, *_ in reversed(handles):
                self._destroy_tensor(handle)
        return output

    def invoke(
        self,
        *,
        query,
        key,
        weights,
        query_scale,
        key_scale,
        block_table,
        metadata,
        cu_seqlens_q,
        seqused_k,
        quant_mode,
        topk,
        return_value=0,
    ):
        if quant_mode not in (1, 5):
            raise ValueError("This benchmark adapter supports FP8 per-token mode=1 and MXFP4 mode=5")
        if return_value not in (0, 1):
            raise ValueError("return_value must be 0 or 1")
        if topk <= 0:
            raise ValueError("topk must be positive")
        device = query.device
        tensors = (query, key, weights, query_scale, key_scale, block_table, metadata, cu_seqlens_q, seqused_k)
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("All QLI tensors must reside on the same NPU device")
        if len(query.shape) != 3 or len(key.shape) != 4 or key.shape[2] != 1:
            raise ValueError("QLI adapter requires TND query and PA_BBND key with one K head")
        if any(size <= 0 for tensor in (query, key) for size in tensor.shape):
            raise ValueError("QLI query and key shapes must be nonempty")
        if not query.is_contiguous() or not query_scale.is_contiguous() or not weights.is_contiguous():
            raise ValueError("Query, query scale and weights must be contiguous")
        if tuple(weights.shape) != tuple(query.shape[:2]) or weights.dtype != self.torch.float32:
            raise ValueError("QLI weights must be contiguous FP32 [T,N]")
        for tensor in (key, key_scale):
            expected_stride = 1
            for axis in range(tensor.ndim - 1, 0, -1):
                if tensor.shape[axis] != 1 and tensor.stride(axis) != expected_stride:
                    raise ValueError("QLI cache may be noncontiguous only on axis 0")
                expected_stride *= tensor.shape[axis]
            if tensor.stride(0) < expected_stride:
                raise ValueError("QLI cache block views must not overlap")
        if quant_mode == 5:
            fp4_types = (self.torch.uint8, getattr(self.torch, "float4_e2m1fn_x2", None))
            e8m0_types = (self.torch.uint8, getattr(self.torch, "float8_e8m0fnu", None))
            if query.dtype not in fp4_types or key.dtype not in fp4_types:
                raise ValueError("MXFP4 Q/K must be packed UINT8 or float4_e2m1fn_x2 tensors")
            if query_scale.dtype not in e8m0_types or key_scale.dtype not in e8m0_types:
                raise ValueError("MXFP4 scales must be UINT8 E8M0 bits or float8_e8m0fnu tensors")
            if query.shape[-1] != 64 or key.shape[-1] != 64 or not 1 <= query.shape[1] <= 64:
                raise ValueError("MXFP4 requires packed D64 bytes (logical D128) and query heads in [1,64]")
            if tuple(query_scale.shape) != (*query.shape[:2], 2, 2):
                raise ValueError("MXFP4 query scale must be [T,N,2,2]")
            if tuple(key_scale.shape) != (*key.shape[:3], 2, 2):
                raise ValueError("MXFP4 key scale must be [blocks,block_size,1,2,2]")
            if any(tensor.storage_offset() % 2 for tensor in (query_scale, key_scale)):
                raise ValueError("MXFP4 E8M0 storage offsets must be aligned to scale pairs")
            if any(tensor.stride(0) % 2 for tensor in (key, key_scale)):
                raise ValueError("MXFP4 cache block strides must be pair-aligned")
        else:
            if query.dtype != self.torch.float8_e4m3fn or key.dtype != self.torch.float8_e4m3fn:
                raise ValueError("FP8 mode=1 requires E4M3 query and key tensors")
            if query.shape[-1] != 128 or key.shape[-1] != 128:
                raise ValueError("This FP8 comparison requires logical D128")
            if tuple(query_scale.shape) != tuple(query.shape[:2]) or tuple(key_scale.shape) != tuple(key.shape[:3]):
                raise ValueError("FP8 mode=1 requires per-token-head scales [T,N] and [blocks,block_size,1]")
            if query_scale.dtype != self.torch.float32 or key_scale.dtype != self.torch.float32:
                raise ValueError("FP8 mode=1 requires FP32 scales")
        output_shape = (query.shape[0], key.shape[2], topk)
        indices = self.torch.empty(output_shape, dtype=self.torch.int32, device=device)
        values = self.torch.empty(output_shape if return_value else (0,), dtype=self.torch.bfloat16, device=device)
        handles = []
        try:
            qk_dtype = "ACL_FLOAT4_E2M1" if quant_mode == 5 else None
            scale_dtype = "ACL_FLOAT8_E8M0" if quant_mode == 5 else None
            arguments = [
                self._descriptor(query, handles, qk_dtype),
                self._descriptor(key, handles, qk_dtype),
                self._descriptor(weights, handles),
                self._descriptor(query_scale, handles, scale_dtype),
                self._descriptor(key_scale, handles, scale_dtype),
                self._descriptor(cu_seqlens_q, handles),
                None,
                None,
                self._descriptor(seqused_k, handles),
                None,
                self._descriptor(block_table, handles),
                None,
                self._descriptor(metadata, handles),
                topk,
                quant_mode,
                -1,
                b"TND",
                b"PA_BBND",
                3,
                1,
                return_value,
                self._descriptor(indices, handles),
                self._descriptor(values, handles),
            ]
            self._execute("aclnnQuantLightningIndexerV2", arguments, handles, device)
        finally:
            for handle, *_ in reversed(handles):
                self._destroy_tensor(handle)
        return indices, values
