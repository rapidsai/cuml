#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

function(find_and_configure_gputreeshap)

    set(oneValueArgs VERSION PINNED_TAG)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    rapids_cpm_find(GPUTreeShap 0.0.1
        GLOBAL_TARGETS  GPUTreeShap::GPUTreeShap GPUTreeShap
        CPM_ARGS
            GIT_REPOSITORY  https://github.com/rapidsai/gputreeshap.git
            GIT_TAG         ${PKG_PINNED_TAG}
    )

    if (GPUTreeShap_ADDED)
        include(GNUInstallDirs)
        install(TARGETS GPUTreeShap
                EXPORT  gputreeshap-exports)

        # clear out incorrect INTERFACE_SOURCES
        set_target_properties(GPUTreeShap PROPERTIES INTERFACE_SOURCES "")
        get_target_property(all_includes GPUTreeShap INTERFACE_INCLUDE_DIRECTORIES)
        # clear out incorrect INTERFACE_INCLUDE_DIRECTORIES
        set_target_properties(GPUTreeShap PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
        # set INTERFACE_INCLUDE_DIRECTORIES appropriately
        target_include_directories(GPUTreeShap INTERFACE
            $<BUILD_INTERFACE:${all_includes}>
            $<INSTALL_INTERFACE:${CMAKE_INCLUDE_DIR}>)

        # generate gputreeshap-targets.cmake for install dir
        rapids_export(INSTALL GPUTreeShap
            EXPORT_SET gputreeshap-exports
            GLOBAL_TARGETS GPUTreeShap
            NAMESPACE GPUTreeShap::)

        # generate gputreeshap-targets.cmake for binary dir
        rapids_export(BUILD GPUTreeShap
            EXPORT_SET gputreeshap-exports
            GLOBAL_TARGETS GPUTreeShap
            NAMESPACE GPUTreeShap::)

    endif()

    # do `find_dependency(GPUTreeShap) in build and install`
    rapids_export_package(BUILD GPUTreeShap cuml-exports)
    rapids_export_package(INSTALL GPUTreeShap cuml-exports)

    # Tell cmake where it can find the generated gputreeshap-config.cmake we wrote.
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(BUILD GPUTreeShap [=[${CMAKE_CURRENT_LIST_DIR}]=] EXPORT_SET cuml-exports)

    set(GPUTreeShap_ADDED ${GPUTreeShap_ADDED} PARENT_SCOPE)

endfunction()

find_and_configure_gputreeshap(PINNED_TAG 93292317b23ef733f881c881865f5d5728dc2fea)
