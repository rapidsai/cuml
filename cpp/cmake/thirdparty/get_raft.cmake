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

set(CUML_MIN_VERSION_raft "${CUML_VERSION_MAJOR}.${CUML_VERSION_MINOR}.00")
set(CUML_BRANCH_VERSION_raft "${CUML_VERSION_MAJOR}.${CUML_VERSION_MINOR}")

function(find_and_configure_raft)

    set(oneValueArgs VERSION FORK PINNED_TAG USE_RAFT_NN USE_FAISS_STATIC CLONE_ON_PIN)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    set(RAFT_STATIC_LINK_LIBRARIES OFF)
    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "branch-${CUML_BRANCH_VERSION_raft}")
        message("Pinned tag found: ${PKG_PINNED_TAG}. Cloning raft locally.")
        set(CPM_DOWNLOAD_raft ON)
        set(RAFT_CXX_FLAGS ${RAFT_CXX_FLAGS} -fPIC)
        set(RAFT_CUDA_FLAGS ${RAFT_CUDA_FLAGS} -fPIC)
        set(CUML_CXX_FLAGS ${CUML_CXX_FLAGS} -fPIC)
        set(CUML_CUDA_FLAGS ${CUML_CUDA_FLAGS} -fPIC)
        set(RAFT_STATIC_LINK_LIBRARIES ON)
    endif()

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
              "RAFT_STATIC_LINK_LIBRARIES ${RAFT_STATIC_LINK_LIBRARIES}"
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

# Change pinned tag here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_raft(VERSION          ${CUML_MIN_VERSION_raft}
                        FORK             cjnolet
                        PINNED_TAG       fea-2204-rbc_3d

                        # When PINNED_TAG above doesn't match cuml,
                        # force local raft clone in build directory
                        # even if it's already installed.
                        CLONE_ON_PIN     ON
                        USE_RAFT_NN      ${CUML_USE_RAFT_NN}
                        USE_FAISS_STATIC ${CUML_USE_FAISS_STATIC}
                        )
