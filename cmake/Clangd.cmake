# clangd does not always discover the target C++ standard-library headers from
# a cross-compilation database. Generate its configuration from the active
# Yocto SDK environment instead of baking a developer-specific SDK path into
# the repository.
if(DEFINED ENV{OECORE_TARGET_SYSROOT} AND NOT "$ENV{OECORE_TARGET_SYSROOT}" STREQUAL "")
    file(TO_CMAKE_PATH "$ENV{OECORE_TARGET_SYSROOT}" CLANGD_TARGET_SYSROOT)

    file(GLOB CLANGD_CXX_INCLUDE_DIRS LIST_DIRECTORIES true
        "${CLANGD_TARGET_SYSROOT}/usr/include/c++/*")
    list(SORT CLANGD_CXX_INCLUDE_DIRS COMPARE NATURAL ORDER DESCENDING)

    unset(CLANGD_CXX_INCLUDE_DIR)
    foreach(CANDIDATE IN LISTS CLANGD_CXX_INCLUDE_DIRS)
        if(IS_DIRECTORY "${CANDIDATE}")
            set(CLANGD_CXX_INCLUDE_DIR "${CANDIDATE}")
            break()
        endif()
    endforeach()

    if(NOT CLANGD_CXX_INCLUDE_DIR)
        message(FATAL_ERROR
            "No C++ standard-library headers found below "
            "${CLANGD_TARGET_SYSROOT}/usr/include/c++")
    endif()

    # GCC keeps target-specific headers (notably bits/c++config.h) in a child
    # directory named after the target tuple.
    unset(CLANGD_CXX_TARGET_INCLUDE_DIR)
    file(GLOB CLANGD_CXX_INCLUDE_CHILDREN LIST_DIRECTORIES true
        "${CLANGD_CXX_INCLUDE_DIR}/*")
    foreach(CANDIDATE IN LISTS CLANGD_CXX_INCLUDE_CHILDREN)
        if(EXISTS "${CANDIDATE}/bits/c++config.h")
            set(CLANGD_CXX_TARGET_INCLUDE_DIR "${CANDIDATE}")
            break()
        endif()
    endforeach()

    if(NOT CLANGD_CXX_TARGET_INCLUDE_DIR)
        message(FATAL_ERROR
            "No target-specific C++ headers found below ${CLANGD_CXX_INCLUDE_DIR}")
    endif()

    configure_file(
        "${CMAKE_CURRENT_LIST_DIR}/clangd.in"
        "${CMAKE_SOURCE_DIR}/.clangd"
        @ONLY
    )
    message(STATUS "Generated .clangd using ${CLANGD_TARGET_SYSROOT}")
else()
    message(WARNING
        "OECORE_TARGET_SYSROOT is not set; .clangd was not generated. "
        "Source the Yocto SDK environment before configuring CMake.")
endif()
