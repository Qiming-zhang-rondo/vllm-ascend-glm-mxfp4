# SPDX-License-Identifier: Apache-2.0
# Native bisheng heterogeneous compilation. Do not use legacy ascendc_library:
# its generated kernel renaming is not interchangeable with native .asc launch.
function(qsfa_direct_objects output)
    set(flags --npu-arch=dav-3510 -O2 -std=c++17 -fPIC ${TORCH_ABI_FLAGS})
    foreach(inc IN LISTS QSFA_CANN_INCLUDES)
        list(APPEND flags "-I${inc}")
    endforeach()
    set(link_flags)
    foreach(libdir IN LISTS CANN_LIBRARY_DIRS)
        list(APPEND link_flags "-L${libdir}")
    endforeach()
    message(STATUS "Checking native bisheng Cube/VF_CALL compilation and link (no device execution)")
    file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/toolchain_probe.failed"
        "${CMAKE_CURRENT_BINARY_DIR}/toolchain_probe.so")
    execute_process(
        COMMAND "${QSFA_BISHENG}" -shared "${CMAKE_CURRENT_SOURCE_DIR}/cmake/toolchain_probe.asc"
            -o "${CMAKE_CURRENT_BINARY_DIR}/toolchain_probe.so" ${flags} ${link_flags}
            ${QSFA_CANN_RUNTIME_LIBRARIES} -Wl,--no-undefined
        RESULT_VARIABLE probe_result OUTPUT_VARIABLE probe_stdout ERROR_VARIABLE probe_stderr)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/toolchain_probe.log" "${probe_stdout}\n${probe_stderr}")
    if(NOT probe_result EQUAL 0 OR NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/toolchain_probe.so")
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/toolchain_probe.failed" "${probe_result}\n")
        message(FATAL_ERROR
            "Existing bisheng could not compile/link the required native Cube/VF_CALL example.\n"
            "Compiler: ${QSFA_BISHENG}\nCANN: ${ASCEND_HOME_PATH}\n"
            "This is a build-toolchain/header failure; no NPU kernel has run.\n"
            "Full log: ${CMAKE_CURRENT_BINARY_DIR}/toolchain_probe.log\n${probe_stdout}\n${probe_stderr}")
    endif()
    set(objects)
    file(GLOB kernel_headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.h")
    set(kernel_names vector matmul tiled)
    if(QSFA_BUILD_OFFICIAL)
        list(APPEND kernel_names official_baseline official_candidate)
        file(GLOB_RECURSE official_headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/official/*.h")
        list(APPEND kernel_headers ${official_headers})
    endif()
    foreach(name IN LISTS kernel_names)
        if(name MATCHES "^official_(.*)$")
            set(kernel_source "official/${CMAKE_MATCH_1}.asc")
        else()
            set(kernel_source "csrc/${name}.asc")
        endif()
        set(obj "${CMAKE_CURRENT_BINARY_DIR}/${name}.asc.o")
        # Each translation unit contains all its device callees; -c embeds the
        # complete kernel and host launch stub, no cross-file device calls.
        add_custom_command(OUTPUT "${obj}"
            COMMAND "${QSFA_BISHENG}" -c "${CMAKE_CURRENT_SOURCE_DIR}/${kernel_source}"
                -o "${obj}" ${flags} "-I${CMAKE_CURRENT_SOURCE_DIR}/csrc"
            DEPENDS "${kernel_source}" ${kernel_headers}
            COMMENT "Compiling ${name}.asc with installed native bisheng"
            VERBATIM COMMAND_EXPAND_LISTS)
        set_source_files_properties("${obj}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        list(APPEND objects "${obj}")
    endforeach()
    set(${output} ${objects} PARENT_SCOPE)
endfunction()
