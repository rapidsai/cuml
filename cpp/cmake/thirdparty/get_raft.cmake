#=============================================================================
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

    set(oneValueArgs VERSION FORK PINNED_TAG USE_RAFT_NN USE_FAISS_STATIC)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    string(APPEND RAFT_COMPONENTS "distance")
    if(PKG_USE_RAFT_NN)
        string(APPEND RAFT_COMPONENTS " nn")
    endif()

    message("CUML: raft FIND_PACKAGE_ARGUMENTS COMPONENTS ${RAFT_COMPONENTS}")

    rapids_cpm_find(raft ${PKG_VERSION}
            GLOBAL_TARGETS      raft::raft
            BUILD_EXPORT_SET    cuml-exports
            INSTALL_EXPORT_SET  cuml-exports
            CPM_ARGS
            GIT_REPOSITORY https://github.com/${PKG_FORK}/raft.git
            GIT_TAG        ${PKG_PINNED_TAG}
            SOURCE_SUBDIR  cpp
            FIND_PACKAGE_ARGUMENTS "COMPONENTS ${RAFT_COMPONENTS}"
            OPTIONS
              "BUILD_TESTS OFF"
              "RAFT_USE_FAISS_STATIC ${PKG_USE_FAISS_STATIC}"
              "NVTX ${NVTX}"
    )

    if(raft_ADDED)
        message(VERBOSE "CUML: Using RAFT located in ${raft_SOURCE_DIR}")
    else()
        message(VERBOSE "CUML: Using RAFT located in ${raft_DIR}")
    endif()

endfunction()

set(CUML_MIN_VERSION_raft "${CUML_VERSION_MAJOR}.${CUML_VERSION_MINOR}.00")
set(CUML_BRANCH_VERSION_raft "${CUML_VERSION_MAJOR}.${CUML_VERSION_MINOR}")

# Change pinned tag here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_raft(VERSION          ${CUML_MIN_VERSION_raft}
                        FORK             rapidsai
                        PINNED_TAG       branch-${CUML_BRANCH_VERSION_raft}
                        USE_RAFT_NN      ${CUML_USE_RAFT_NN}
                        USE_FAISS_STATIC ${CUML_USE_FAISS_STATIC}
                        )
