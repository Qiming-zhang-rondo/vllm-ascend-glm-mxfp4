# SPDX-License-Identifier: Apache-2.0
# This mode is deliberately limited to the standalone ACLNN benchmark. QLI has
# no ONNX plugin or FIA device-tiling dependency, so protobuf/absl and makeself
# are unnecessary. The ordinary vLLM-Ascend build retains its existing behavior.
include_guard(GLOBAL)

if(NOT BUILD_OPEN_PROJECT OR BUILD_OPS_RTY_KERNEL OR ENABLE_BUILT_IN OR
   ENABLE_TEST OR ENABLE_EXPERIMENTAL OR ENABLE_BUILD_PKG OR ENABLE_STATIC)
    message(FATAL_ERROR "QLI_STANDALONE_OFFLINE requires a custom, non-test, non-package build")
endif()
if(NOT "${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    message(FATAL_ERROR "QLI_STANDALONE_OFFLINE supports only ascend950 (A5)")
endif()
set(_qli_required_ops quant_lightning_indexer_v2 quant_lightning_indexer_v2_metadata)
foreach(_qli_op IN LISTS _qli_required_ops)
    if(NOT _qli_op IN_LIST ASCEND_OP_NAME)
        message(FATAL_ERROR "QLI_STANDALONE_OFFLINE requires ${_qli_op}")
    endif()
endforeach()
foreach(_qli_op IN LISTS ASCEND_OP_NAME)
    # lightning_indexer_v2 is added by the existing compute-op dependency graph.
    if(NOT _qli_op IN_LIST _qli_required_ops AND NOT "${_qli_op}" STREQUAL "lightning_indexer_v2")
        message(FATAL_ERROR "QLI_STANDALONE_OFFLINE does not build unrelated operator ${_qli_op}")
    endif()
endforeach()

# The quantized kernel includes LI V2's shared vector code by relative path.
# A selective build must copy that sibling into the generated kernel source tree.
set(quant_lightning_indexer_v2_depends attention/lightning_indexer_v2)

# Do not use the normal json.cmake download fallback, including ExternalProject
# URLs (which FETCHCONTENT_FULLY_DISCONNECTED alone does not disable).
if(NOT EXISTS "${QLI_JSON_INCLUDE_DIR}/nlohmann/json.hpp")
    message(FATAL_ERROR "Offline QLI build needs an existing nlohmann/json.hpp; set QLI_JSON_INCLUDE_DIR to its include directory. No dependency will be downloaded.")
endif()
set(JSON_INCLUDE_DIR "${QLI_JSON_INCLUDE_DIR}")
set(JSON_INCLUDE "${QLI_JSON_INCLUDE_DIR}")
set(json_FOUND TRUE)
add_library(json INTERFACE IMPORTED)
set_target_properties(json PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${JSON_INCLUDE_DIR}")
set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL "No FetchContent downloads" FORCE)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "No FetchContent updates" FORCE)
message(STATUS "Offline QLI: reuse JSON headers at ${JSON_INCLUDE_DIR}; omit ONNX, protobuf, absl, makeself and FIA tiling-sink targets")
