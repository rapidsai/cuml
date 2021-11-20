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

    set(GPUTreeShap_ADDED ${GPUTreeShap_ADDED} PARENT_SCOPE)

endfunction()

find_and_configure_gputreeshap(PINNED_TAG c78fe621e429117cbca45e7b23eb5c3b6280fa3a)
