# SPDX-License-Identifier: Apache-2.0
# CANN 9.1 AI Core Compilation Basics / Built-in Libraries. A final CXX
# link of precompiled .asc.o files does not guarantee the ASC driver's
# implicit runtime dependencies, even when CMAKE_CXX_COMPILER is bisheng.
# Keep the static archive before its providers (including with --as-needed).
set(QSFA_CANN_RUNTIME_LIBRARIES)
foreach(name ascendc_runtime runtime profapi unified_dlog mmpa ascend_dump c_sec error_manager ascendcl)
    if(name STREQUAL "ascendc_runtime")
        set(filename "lib${name}.a")
    else()
        # Several CANN dependencies ship both .a and .so. Use the documented
        # shared providers instead of pulling in another static dependency tree.
        set(filename "lib${name}.so")
    endif()
    find_library(QSFA_CANN_${name}_LIBRARY NAMES "${filename}"
        PATHS ${CANN_LIBRARY_DIRS} PATH_SUFFIXES common platform NO_DEFAULT_PATH REQUIRED)
    list(APPEND QSFA_CANN_RUNTIME_LIBRARIES "${QSFA_CANN_${name}_LIBRARY}")
    get_filename_component(libdir "${QSFA_CANN_${name}_LIBRARY}" DIRECTORY)
    list(APPEND CANN_LIBRARY_DIRS "${libdir}")
    message(STATUS "QSFA CANN runtime ${name}: ${QSFA_CANN_${name}_LIBRARY}")
endforeach()
list(REMOVE_DUPLICATES CANN_LIBRARY_DIRS)
