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

function(find_and_configure_raft)

    set(oneValueArgs VERSION FORK PINNED_TAG)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    rapids_cpm_find(raft ${PKG_VERSION}
      GLOBAL_TARGETS      raft::raft
      BUILD_EXPORT_SET    cuml-exports
      INSTALL_EXPORT_SET  cuml-exports
        CPM_ARGS
            GIT_REPOSITORY https://github.com/cjnolet/raft.git
            GIT_TAG        fea-020-hdbscan
            SOURCE_SUBDIR  cpp
            OPTIONS
              "BUILD_TESTS OFF"

    )

    message(VERBOSE "CUML: Using RAFT located in ${raft_SOURCE_DIR}")

endfunction()

set(CUML_MIN_VERSION_raft "${CUML_VERSION_MAJOR}.${CUML_VERSION_MINOR}.00")
set(CUML_BRANCH_VERSION_raft "${CUML_VERSION_MAJOR}.${CUML_VERSION_MINOR}")

# Change pinned tag here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_raft(VERSION    ${CUML_MIN_VERSION_raft}
                        FORK       rapidsai
                        PINNED_TAG branch-${CUML_BRANCH_VERSION_raft}
                        )
