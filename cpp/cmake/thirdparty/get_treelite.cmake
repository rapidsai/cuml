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

function(find_and_configure_nccl)

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

    # if (Treelite_ADDED)
    #     # todo (DD): Add building nccl from source, works fine for conda installed

    # endif()
endfunction()

find_and_configure_nccl(VERSION     1.1.0
                        PINNED_TAG  342be01cde4fd99f81e529c1a0c85c496b2a3226)
