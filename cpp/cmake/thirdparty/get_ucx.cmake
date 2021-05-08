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

function(find_and_configure_ucx)
    set(oneValueArgs VERSION PINNED_TAG)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    rapids_find_generate_module(ucx
        HEADER_NAMES  ucp/api/ucp.h
        LIBRARY_NAMES ucp
    )

    rapids_cpm_find(ucx ${PKG_VERSION}
        CPM_ARGS
            GIT_REPOSITORY  https://github.com/openucx/ucx
            GIT_TAG         ${PKG_PINNED_TAG}
            GIT_SHALLOW     TRUE
            DOWNLOAD_ONLY   YES
    )

    set(ucx_SOURCE_DIR "${ucx_SOURCE_DIR}" PARENT_SCOPE)

    if (ucx_ADDED)
        # todo (DD): Add building ucx from source, works fine for conda installed


    endif()
endfunction()

find_and_configure_ucx(VERSION     0
                       PINNED_TAG  v1.9.0)
