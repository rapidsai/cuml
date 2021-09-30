#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

function(find_and_configure_treelite)

    set(oneValueArgs VERSION PINNED_TAG BUILD_STATIC_LIBS)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    rapids_cpm_find(Treelite ${PKG_VERSION}
        GLOBAL_TARGETS   treelite::treelite treelite
        BUILD_EXPORT_SET cuml-exports
        CPM_ARGS
            GIT_REPOSITORY  https://github.com/dmlc/treelite.git
            GIT_TAG         ${PKG_PINNED_TAG}
            OPTIONS
              "USE_OPENMP ON"
              "BUILD_STATIC_LIBS ${BUILD_STATIC_LIBS}"
    )

    set(Treelite_ADDED ${Treelite_ADDED} PARENT_SCOPE)

    if(Treelite_ADDED)
        target_include_directories(treelite
            PUBLIC $<BUILD_INTERFACE:${Treelite_SOURCE_DIR}/include>
                   $<BUILD_INTERFACE:${Treelite_BINARY_DIR}/include>)
        target_include_directories(treelite_static
            PUBLIC $<BUILD_INTERFACE:${Treelite_SOURCE_DIR}/include>
                   $<BUILD_INTERFACE:${Treelite_BINARY_DIR}/include>)
        target_include_directories(treelite_runtime
            PUBLIC $<BUILD_INTERFACE:${Treelite_SOURCE_DIR}/include>
                   $<BUILD_INTERFACE:${Treelite_BINARY_DIR}/include>)
        target_include_directories(treelite_runtime_static
            PUBLIC $<BUILD_INTERFACE:${Treelite_SOURCE_DIR}/include>
                   $<BUILD_INTERFACE:${Treelite_BINARY_DIR}/include>)

        if(NOT TARGET treelite::treelite_static)
            add_library(treelite::treelite_static ALIAS treelite_static)
            add_library(treelite::treelite_runtime_static ALIAS treelite_runtime_static)
        endif()

        rapids_export(BUILD treelite EXPORT_SET TreeliteTargets)
    endif()

    if (Treelite_ADDED AND BUILD_STATIC_LIBS)
        list(APPEND TREELITE_LIBS treelite::treelite_static treelite::treelite_runtime_static)
    else()
        list(APPEND TREELITE_LIBS treelite::treelite treelite::treelite_runtime)
    endif()

    set(TREELITE_LIBS ${TREELITE_LIBS} PARENT_SCOPE)

endfunction()

find_and_configure_treelite(VERSION     2.1.0
                        PINNED_TAG  e5248931c62e3807248e0b150e27b2530a510634
                        BUILD_STATIC_LIBS ${CUML_USE_TREELITE_STATIC})
