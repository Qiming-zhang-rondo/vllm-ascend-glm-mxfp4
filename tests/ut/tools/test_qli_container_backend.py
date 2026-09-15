# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests of raw ACL descriptor ABI and ownership; no NPU claims."""

import contextlib
import ctypes
import importlib.util
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[3] / "tools/qli_container_backend.py"
SPEC = importlib.util.spec_from_file_location("qli_container_backend", MODULE_PATH)
backend_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend_module)

# Deliberately not production enum numbers: the adapter must use the header.
HEADER = """
typedef enum {
    ACL_FLOAT = 90, ACL_INT32, ACL_INT64, ACL_UINT8, ACL_BF16,
    ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0 = 0xB1U,
    ACL_FLOAT4_E2M1 = (ACL_FLOAT8_E8M0 + 4)
} aclDataType;
typedef enum { ACL_FORMAT_NCHW = 0, ACL_FORMAT_NHWC = 1, ACL_FORMAT_ND = 2, ACL_FORMAT_NCDHW = 30 } aclFormat;
"""


class FakeDevice:
    type = "npu"
    index = 0


class FakeTensor:
    def __init__(self, shape, dtype="uint8", device=None, strides=None, offset=0, storage_bytes=None):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype = dtype
        self.device = device or FakeDevice()
        self.ndim = len(self.shape)
        self._offset = offset
        contiguous = []
        stride = 1
        for length in reversed(self.shape):
            contiguous.insert(0, stride)
            stride *= length
        self._contiguous = tuple(contiguous)
        self._strides = tuple(strides) if strides else self._contiguous
        elements = offset + 1 + sum(max(0, size - 1) * value for size, value in zip(self.shape, self._strides))
        self._storage_bytes = storage_bytes if storage_bytes is not None else elements * self.element_size()

    def stride(self, axis=None):
        return self._strides if axis is None else self._strides[axis]

    def storage_offset(self):
        return self._offset

    def element_size(self):
        return {"float32": 4, "int32": 4, "int64": 8, "bfloat16": 2}.get(self.dtype, 1)

    def untyped_storage(self):
        return types.SimpleNamespace(nbytes=lambda: self._storage_bytes, data_ptr=lambda: 0x100000)

    def data_ptr(self):
        return 0x100000 + self._offset * self.element_size()

    def is_contiguous(self):
        return self._strides == self._contiguous

    def numel(self):
        product = 1
        for size in self.shape:
            product *= size
        return product


class FakeFunction:
    def __init__(self, call):
        self.call = call

    def __call__(self, *args):
        return self.call(*args)


class Harness:
    def __init__(self):
        self.events = []
        self.descriptors = {}
        self.next_handle = 1
        self.workspace_status = 0
        self.execute_status = 0
        self.workspace_arguments = None
        self.library = types.SimpleNamespace(_name="/container/libopapi.so")
        self.library.aclCreateTensor = FakeFunction(self.create_tensor)
        self.library.aclDestroyTensor = FakeFunction(self.destroy_tensor)
        for name in backend_module.API_NAMES:
            callback = self.get_workspace if name.endswith("GetWorkspaceSize") else self.execute
            setattr(self.library, name, FakeFunction(callback))
        self.torch = types.SimpleNamespace(
            float32="float32",
            bfloat16="bfloat16",
            int32="int32",
            int64="int64",
            uint8="uint8",
            float8_e4m3fn="float8_e4m3fn",
            float8_e8m0fnu="float8_e8m0fnu",
            float4_e2m1fn_x2="float4_e2m1fn_x2",
            empty=lambda shape, dtype, device: FakeTensor(shape, dtype=dtype, device=device),
            npu=types.SimpleNamespace(
                is_available=lambda: True,
                synchronize=lambda device: self.events.append("synchronize"),
                current_stream=lambda device: types.SimpleNamespace(npu_stream=0xABCDEF),
                device=lambda device: contextlib.nullcontext(),
            ),
        )
        self.torch_npu = types.SimpleNamespace(get_npu_format=lambda tensor: 2)

    def create_tensor(self, shape, rank, dtype, strides, offset, fmt, storage_shape, storage_rank, data):
        handle = self.next_handle
        self.next_handle += 1
        self.descriptors[handle] = {
            "shape": tuple(shape[:rank]),
            "strides": tuple(strides[:rank]),
            "offset": offset,
            "dtype": dtype,
            "format": fmt,
            "storage": tuple(storage_shape[:storage_rank]),
            "data": data,
        }
        self.events.append(("create", handle))
        return handle

    def destroy_tensor(self, handle):
        self.events.append(("destroy", handle))
        return 0

    def get_workspace(self, *arguments):
        self.events.append("workspace")
        self.workspace_arguments = arguments
        ctypes.cast(arguments[-2], ctypes.POINTER(ctypes.c_uint64))[0] = 512
        ctypes.cast(arguments[-1], ctypes.POINTER(ctypes.c_void_p))[0] = 0x12345
        return self.workspace_status

    def execute(self, workspace, size, executor, stream):
        self.events.append("execute")
        assert size == 512 and stream == 0xABCDEF
        return self.execute_status

    def make_backend(self, symbol_paths=None, **options):
        enums = backend_module.parse_acl_enums(HEADER)
        with (
            patch.object(backend_module.importlib, "import_module", side_effect=lambda name: getattr(self, name)),
            patch.object(backend_module, "_load_enums", return_value=(Path("/container/acl_base.h"), enums)),
            patch.object(backend_module, "_library_candidates", return_value=["/container/libopapi.so"]),
            patch.object(backend_module.ctypes, "CDLL", return_value=self.library),
            patch.object(
                backend_module, "_symbol_path", side_effect=symbol_paths, return_value="/container/libopapi.so"
            ),
        ):
            return backend_module.QLIBackend(**options)

    def inputs(self):
        device = FakeDevice()
        return {
            "query": FakeTensor((1, 32, 64), device=device),
            "key": FakeTensor((16, 128, 1, 64), device=device, strides=(128 * 80, 64, 64, 1), offset=2),
            "weights": FakeTensor((1, 32), dtype="float32", device=device),
            "query_scale": FakeTensor((1, 32, 2, 2), device=device),
            "key_scale": FakeTensor((16, 128, 1, 2, 2), device=device, strides=(128 * 8, 4, 4, 2, 1)),
            "block_table": FakeTensor((1, 16), dtype="int32", device=device),
            "metadata": FakeTensor((1024,), dtype="int32", device=device),
            "cu_seqlens_q": FakeTensor((2,), dtype="int32", device=device),
            "seqused_k": FakeTensor((1,), dtype="int32", device=device),
            "quant_mode": 5,
            "topk": 2048,
        }


class EnumAndLayoutTests(unittest.TestCase):
    @unittest.skipUnless(
        any(shutil.which(name) for name in ("c++", "g++", "clang++", "cc")), "needs a local preprocessor"
    )
    def test_wrapper_header_follows_nested_includes_and_active_preprocessor_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            include = Path(directory) / "include"
            acl_dir = include / "acl"
            nested = acl_dir / "details"
            nested.mkdir(parents=True)
            wrapper = acl_dir / "acl_base.h"
            wrapper.write_text('#include "acl/acl_base_rt.h"\n')
            (acl_dir / "acl_base_rt.h").write_text('#include "details/types.h"\n')
            # The inactive declarations deliberately come last: concatenating
            # header text instead of preprocessing would overwrite live values.
            (nested / "types.h").write_text(
                HEADER + "\n#if 0\nenum { ACL_FLOAT = 999, ACL_FLOAT4_E2M1 = 999, ACL_FORMAT_ND = 999 };\n#endif\n"
            )
            header, values = backend_module._load_enums(wrapper)
            self.assertEqual(header, wrapper.resolve())
            self.assertEqual(values["ACL_FLOAT"], 90)
            self.assertEqual(values["ACL_FLOAT4_E2M1"], 181)
            self.assertEqual(values["ACL_FORMAT_ND"], 2)

    @unittest.skipUnless(
        any(shutil.which(name) for name in ("c++", "g++", "clang++", "cc")), "needs a local preprocessor"
    )
    def test_wrapper_header_still_rejects_missing_fp4_after_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            wrapper = path / "acl_base.h"
            wrapper.write_text('#include "types.h"\n')
            (path / "types.h").write_text(HEADER.replace("ACL_FLOAT4_E2M1", "OLD_TYPE"))
            with self.assertRaisesRegex(RuntimeError, "ACL_FLOAT4_E2M1"):
                backend_module._load_enums(wrapper)

    def test_wrapper_header_does_not_invent_constants_when_preprocessor_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            wrapper = path / "acl_base.h"
            wrapper.write_text('#include "types.h"\n')
            (path / "types.h").write_text(HEADER)
            with (
                patch.object(backend_module.shutil, "which", return_value=None),
                self.assertRaisesRegex(RuntimeError, "no existing C/C\\+\\+ compiler"),
            ):
                backend_module._load_enums(wrapper)

    def test_enums_are_read_and_aliases_resolved(self):
        values = backend_module.parse_acl_enums(HEADER)
        self.assertEqual(values["ACL_FLOAT"], 90)
        self.assertEqual(values["ACL_INT32"], 91)
        self.assertEqual(values["ACL_FLOAT4_E2M1"], 181)
        self.assertEqual(values["ACL_FORMAT_ND"], 2)

    def test_unknown_enum_values_are_not_inferred_or_executed(self):
        values = backend_module.parse_acl_enums("enum { A=__import__('os').system('false'), B, C=7, D };")
        self.assertNotIn("A", values)
        self.assertNotIn("B", values)
        self.assertEqual(values["D"], 8)

    def test_explicit_header_missing_fp4_is_not_installable_api_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "acl_base.h"
            path.write_text(HEADER.replace("ACL_FLOAT4_E2M1", "OLD_TYPE"))
            with self.assertRaisesRegex(RuntimeError, "missing ACL_FLOAT4_E2M1") as caught:
                backend_module._load_enums(path)
            self.assertNotIsInstance(caught.exception, backend_module.BackendUnavailable)

    def test_packed_layout_preserves_view_offset_and_block_padding(self):
        self.assertEqual(
            backend_module.packed_layout((4, 128, 1, 64), (10240, 64, 64, 1), 2, 50000),
            ([4, 128, 1, 128], [20480, 128, 128, 1], 4, 100000),
        )

    def test_bad_packed_layouts_fail(self):
        for shape, strides, offset, size in [
            ((4, 64), (128, 2), 0, 1024),
            ((4, 64), (64, 1), -1, 1024),
            ((4, 64), (64, 1), 0, 2**63),
        ]:
            with self.subTest(shape=shape, strides=strides, offset=offset, size=size), self.assertRaises(ValueError):
                backend_module.packed_layout(shape, strides, offset, size)


class AclCallTests(unittest.TestCase):
    def test_five_dimensional_ncdhw_scale_cache_preserves_logical_nd_view(self):
        harness = Harness()
        backend = harness.make_backend()
        harness.torch_npu.get_npu_format = lambda tensor: backend.enums["ACL_FORMAT_NCDHW"] if tensor.ndim == 5 else 2
        inputs = harness.inputs()
        inputs["key_scale"]._offset = 4
        backend.invoke(**inputs)
        scale = harness.descriptors[harness.workspace_arguments[4]]
        self.assertEqual(scale["format"], backend.enums["ACL_FORMAT_ND"])
        self.assertEqual(scale["shape"], inputs["key_scale"].shape)
        self.assertEqual(scale["strides"], inputs["key_scale"].stride())
        self.assertEqual(scale["offset"], 4)
        self.assertEqual(scale["data"], 0x100000)

    def test_nhwc_base_storage_is_accepted_without_repacking(self):
        harness = Harness()
        backend = harness.make_backend()
        harness.torch_npu.get_npu_format = lambda tensor: backend.enums["ACL_FORMAT_NHWC"]
        backend.invoke(**harness.inputs())
        self.assertIn("execute", harness.events)
        self.assertTrue(all(item["format"] == backend.enums["ACL_FORMAT_ND"] for item in harness.descriptors.values()))

    def test_opaque_storage_is_rejected_with_tensor_details(self):
        harness = Harness()
        backend = harness.make_backend()
        harness.torch_npu.get_npu_format = lambda tensor: 29
        with self.assertRaisesRegex(ValueError, r"format=29, shape=\(1, 32, 64\), dtype=uint8"):
            backend.invoke(**harness.inputs())
        self.assertEqual(harness.events, [])

    def test_fp8_uses_mode_one_e4m3_and_fp32_scales(self):
        harness = Harness()
        backend = harness.make_backend()
        inputs = harness.inputs()
        device = inputs["query"].device
        inputs.update(
            query=FakeTensor((1, 32, 128), dtype="float8_e4m3fn", device=device),
            key=FakeTensor((16, 128, 1, 128), dtype="float8_e4m3fn", device=device),
            query_scale=FakeTensor((1, 32), dtype="float32", device=device),
            key_scale=FakeTensor((16, 128, 1), dtype="float32", device=device),
            quant_mode=1,
        )
        backend.invoke(**inputs)
        arguments = harness.workspace_arguments
        self.assertEqual(arguments[14], 1)
        self.assertEqual(harness.descriptors[arguments[0]]["dtype"], 95)
        self.assertEqual(harness.descriptors[arguments[3]]["dtype"], 90)
        inputs["quant_mode"] = 2
        with self.assertRaisesRegex(ValueError, "mode=1"):
            backend.invoke(**inputs)

    def test_missing_api_pair_is_classified_as_unavailable(self):
        harness = Harness()
        delattr(harness.library, backend_module.API_NAMES[1])
        with self.assertRaises(backend_module.BackendUnavailable):
            harness.make_backend()

    def test_compute_metadata_cannot_silently_mix_dependency_libraries(self):
        harness = Harness()
        paths = ["/custom/libcust_opapi.so"] * 2 + ["/container/libopapi.so"] * 2
        with self.assertRaisesRegex(backend_module.BackendUnavailable, "different libraries"):
            harness.make_backend(symbol_paths=paths)

    def test_explicit_library_must_own_its_qli_symbols(self):
        harness = Harness()
        paths = ["/dependency/libopapi.so"] * 4
        with self.assertRaisesRegex(backend_module.BackendUnavailable, "supplied by a dependency"):
            harness.make_backend(symbol_paths=paths, opapi_lib="/container/libopapi.so")

    def test_compute_abi_and_fp4_descriptor_match_byte_storage(self):
        harness = Harness()
        backend = harness.make_backend()
        inputs = harness.inputs()
        indices, values = backend.invoke(**inputs, return_value=1)
        self.assertEqual(len(backend._api[backend_module.API_NAMES[0]].argtypes), 25)
        self.assertEqual(len(harness.workspace_arguments), 25)
        arguments = harness.workspace_arguments
        self.assertEqual(arguments[13:21], (2048, 5, -1, b"TND", b"PA_BBND", 3, 1, 1))
        q, k, qs = (harness.descriptors[arguments[index]] for index in (0, 1, 3))
        self.assertEqual(q["shape"], (1, 32, 128))
        self.assertEqual(q["dtype"], 181)
        self.assertEqual(k["strides"], (20480, 128, 128, 1))
        self.assertEqual(k["offset"], 4)
        self.assertEqual(k["data"], 0x100000)  # base pointer, not offset data_ptr
        self.assertEqual(k["storage"], (inputs["key"]._storage_bytes * 2,))
        self.assertEqual(qs["shape"], (1, 32, 2, 2))
        self.assertEqual(qs["dtype"], 177)
        self.assertEqual(indices.shape, (1, 1, 2048))
        self.assertEqual(values.dtype, "bfloat16")

    def test_metadata_abi_optional_nulls_and_sequence_contract(self):
        harness = Harness()
        backend = harness.make_backend()
        inputs = harness.inputs()
        output = backend.create_metadata(
            num_heads_q=32,
            head_dim=128,
            topk=2048,
            quant_mode=5,
            cu_seqlens_q=inputs["cu_seqlens_q"],
            seqused_k=inputs["seqused_k"],
        )
        arguments = harness.workspace_arguments
        self.assertEqual(len(arguments), 20)
        self.assertEqual(len(backend._api[backend_module.API_NAMES[2]].argtypes), 20)
        self.assertEqual(arguments[1:3], (None, None))
        self.assertEqual(arguments[4], None)
        self.assertEqual(arguments[5:17], (32, 1, 128, 2048, 5, 1, -1, -1, b"TND", b"PA_BBND", 3, 1))
        self.assertEqual(output.shape, (1024,))

    def test_owner_lifetime_is_bracketed_by_queue_synchronization(self):
        harness = Harness()
        backend = harness.make_backend()
        backend.invoke(**harness.inputs())
        execute_index = harness.events.index("execute")
        self.assertEqual(
            harness.events[execute_index - 2 : execute_index + 2],
            ["synchronize", "workspace", "execute", "synchronize"],
        )
        destroys = [
            index for index, event in enumerate(harness.events) if isinstance(event, tuple) and event[0] == "destroy"
        ]
        self.assertTrue(all(index > execute_index + 1 for index in destroys))
        handles = [harness.events[index][1] for index in destroys]
        self.assertEqual(len(handles), len(set(handles)))
        self.assertEqual(len(handles), len(harness.descriptors))

    def test_workspace_failure_carries_phase_status_and_frees_once(self):
        harness = Harness()
        harness.workspace_status = 161002
        backend = harness.make_backend()
        with self.assertRaises(backend_module.QLICallError) as caught:
            backend.invoke(**harness.inputs())
        self.assertEqual(caught.exception.phase, "GetWorkspaceSize")
        self.assertEqual(caught.exception.status, 161002)
        self.assertNotIn("execute", harness.events)
        destroys = [event[1] for event in harness.events if isinstance(event, tuple) and event[0] == "destroy"]
        self.assertEqual(len(destroys), len(set(destroys)))
        self.assertEqual(len(destroys), len(harness.descriptors))

    def test_execute_failure_still_synchronizes_before_destroy(self):
        harness = Harness()
        harness.execute_status = 500000
        backend = harness.make_backend()
        with self.assertRaises(backend_module.QLICallError) as caught:
            backend.invoke(**harness.inputs())
        self.assertEqual(caught.exception.phase, "execute")
        self.assertEqual(harness.events[harness.events.index("execute") + 1], "synchronize")

    def test_sync_failures_identify_before_and_after_aclnn_launch(self):
        for failing_call, label in ((1, "BEFORE"), (2, "AFTER")):
            with self.subTest(label=label):
                harness = Harness()
                backend = harness.make_backend()
                sync_calls = 0

                def sync(device, failing_call=failing_call):
                    nonlocal sync_calls
                    sync_calls += 1
                    if sync_calls == failing_call:
                        raise RuntimeError("507014 test timeout")

                harness.torch.npu.synchronize = sync
                with self.assertRaisesRegex(RuntimeError, f"aclnnQuantLightningIndexerV2: .*{label} ACLNN launch"):
                    backend.invoke(**harness.inputs())
                self.assertEqual("execute" in harness.events, failing_call == 2)
                destroyed = [event[1] for event in harness.events if isinstance(event, tuple) and event[0] == "destroy"]
                self.assertEqual(len(destroyed), len(harness.descriptors))
                self.assertEqual(len(set(destroyed)), len(destroyed))

    def test_unaligned_scale_rejected_before_acl_submission(self):
        harness = Harness()
        backend = harness.make_backend()
        inputs = harness.inputs()
        inputs["query_scale"]._offset = 1
        with self.assertRaisesRegex(ValueError, "aligned to scale pairs"):
            backend.invoke(**inputs)
        self.assertNotIn("workspace", harness.events)

    def test_descriptor_creation_failure_releases_previous_descriptors(self):
        harness = Harness()
        backend = harness.make_backend()
        create = backend._create_tensor
        calls = 0

        def fail_third(*args):
            nonlocal calls
            calls += 1
            return None if calls == 3 else create(*args)

        backend._create_tensor = fail_third
        with self.assertRaisesRegex(RuntimeError, "aclCreateTensor failed"):
            backend.invoke(**harness.inputs())
        self.assertEqual([event for event in harness.events if event[0] == "destroy"], [("destroy", 2), ("destroy", 1)])

    def test_symbol_presence_is_never_reported_as_compute_verified(self):
        harness = Harness()
        backend = harness.make_backend()
        backend.invoke(**harness.inputs())
        self.assertFalse(backend.describe()["mxfp4_compute_verified"])
        self.assertEqual(backend.describe()["api_libraries"][backend_module.API_NAMES[0]], "/container/libopapi.so")


if __name__ == "__main__":
    unittest.main()
