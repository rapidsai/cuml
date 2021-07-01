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

    if(TARGET treelite::treelite)
        return()
    endif()

    set(oneValueArgs VERSION PINNED_TAG)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    rapids_cpm_find(Treelite ${PKG_VERSION}
        GLOBAL_TARGETS  treelite
        CPM_ARGS
            GIT_REPOSITORY  https://github.com/dmlc/treelite.git
            GIT_TAG         ${PKG_PINNED_TAG}
            OPTIONS
              "USE_OPENMP ON"
              "BUILD_STATIC_LIBS ON"
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
    endif()

endfunction()

find_and_configure_treelite(VERSION     1.3.0
                        PINNED_TAG  ae5436c45563a5c2ed7aa8f0f41f88fff48f49e4)
