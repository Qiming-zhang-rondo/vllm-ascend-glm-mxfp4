# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED ASCEND_HOME_PATH)
    message(FATAL_ERROR "Pass the existing CANN root as -DASCEND_HOME_PATH=...; no downloads are performed")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${ASCEND_HOME_PATH}")
# These kernels use SIMD Cube + SIMD/SIMT VF_CALL, not pure SIMT kernels.
# --enable-simt forbids the SIMD headers used by this project.
set(CMAKE_ASC_ENABLE_SIMT OFF)
set(CMAKE_ASC_ARCHITECTURES "dav-3510")
if(QSFA_BISHENG)
    set(CMAKE_CXX_COMPILER "${QSFA_BISHENG}")
    project(qsfa_q8c4_o8 LANGUAGES CXX)
else()
    find_package(ASC REQUIRED)
    project(qsfa_q8c4_o8 LANGUAGES ASC CXX)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import torch; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE TORCH_QUERY_RESULT OUTPUT_VARIABLE TORCH_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT TORCH_QUERY_RESULT EQUAL 0)
    message(FATAL_ERROR "Cannot import the container's torch")
endif()
find_package(Torch REQUIRED HINTS "${TORCH_PREFIX}")
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import pathlib, torch_npu; print(pathlib.Path(torch_npu.__file__).resolve().parent)"
    RESULT_VARIABLE NPU_QUERY_RESULT OUTPUT_VARIABLE TORCH_NPU_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT NPU_QUERY_RESULT EQUAL 0)
    message(FATAL_ERROR "Cannot import the container's torch_npu")
endif()

set(CANN_LIBRARY_DIRS "${ASCEND_HOME_PATH}/lib64"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64"
    "${ASCEND_HOME_PATH}/aarch64-linux/lib64" "${ASCEND_HOME_PATH}/arm64-linux/lib64"
    "${ASCEND_HOME_PATH}/toolkit/lib64")
find_library(TORCH_NPU_LIBRARY torch_npu PATHS "${TORCH_NPU_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(ASCENDCL_LIBRARY ascendcl PATHS ${CANN_LIBRARY_DIRS} NO_DEFAULT_PATH REQUIRED)
find_library(ASCENDC_RUNTIME_LIBRARY ascendc_runtime PATHS ${CANN_LIBRARY_DIRS} NO_DEFAULT_PATH REQUIRED)

set(QSFA_CANN_INCLUDES
    "${ASCEND_HOME_PATH}/include"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
    "${ASCEND_HOME_PATH}/aarch64-linux/include" "${ASCEND_HOME_PATH}/arm64-linux/include"
    "${ASCEND_HOME_PATH}/asc/include" "${ASCEND_HOME_PATH}/asc"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc")
foreach(prefix "" "${CMAKE_SYSTEM_PROCESSOR}-linux" "aarch64-linux" "arm64-linux" "toolkit")
    foreach(suffix "asc/include" "asc/include/basic_api" "asc/include/simt_api"
            "compiler/tikcpp/tikcfw" "tools/tikcpp/tikcfw" "tikcpp/tikcfw")
        set(inc "${ASCEND_HOME_PATH}/${prefix}/${suffix}")
        if(IS_DIRECTORY "${inc}")
            list(APPEND QSFA_CANN_INCLUDES "${inc}" "${inc}/interface" "${inc}/impl")
        endif()
    endforeach()
endforeach()
list(REMOVE_DUPLICATES QSFA_CANN_INCLUDES)
separate_arguments(TORCH_ABI_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

if(QSFA_BISHENG)
    include(cmake/direct_bisheng.cmake)
    qsfa_direct_objects(QSFA_KERNEL_OBJECTS)
    add_library(qsfa_q8c4_o8 SHARED csrc/torch_binding.cpp ${QSFA_KERNEL_OBJECTS})
    # bisheng performs the final device/host link, using the same installation.
    target_link_options(qsfa_q8c4_o8 PRIVATE --npu-arch=dav-3510)
else()
    add_library(qsfa_q8c4_o8 SHARED csrc/torch_binding.cpp csrc/vector.asc csrc/matmul.asc)
endif()
target_include_directories(qsfa_q8c4_o8 PRIVATE
    csrc ${TORCH_INCLUDE_DIRS} "${TORCH_NPU_ROOT}/include" ${QSFA_CANN_INCLUDES})
target_compile_options(qsfa_q8c4_o8 PRIVATE ${TORCH_ABI_FLAGS}
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
    $<$<COMPILE_LANGUAGE:ASC>:-O2>)
target_link_libraries(qsfa_q8c4_o8 PRIVATE ${TORCH_LIBRARIES} "${TORCH_NPU_LIBRARY}"
    "${ASCENDCL_LIBRARY}" "${ASCENDC_RUNTIME_LIBRARY}" dl)
target_link_options(qsfa_q8c4_o8 PRIVATE -Wl,--no-undefined)
set_target_properties(qsfa_q8c4_o8 PROPERTIES BUILD_RPATH
    "${TORCH_NPU_ROOT}/lib;${TORCH_INSTALL_PREFIX}/lib;${CANN_LIBRARY_DIRS}")
