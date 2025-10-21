#=============================================================================
# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

function(find_and_configure_raft)
    set(oneValueArgs VERSION FORK PINNED_TAG EXCLUDE_FROM_ALL CLONE_ON_PIN NVTX)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "${rapids-cmake-checkout-tag}")
        message(STATUS "CUML: RAFT pinned tag found: ${PKG_PINNED_TAG}. Cloning raft locally.")
        set(CPM_DOWNLOAD_raft ON)
    endif()

    # We need RAFT::distributed for MG tests
    if(BUILD_CUML_MG_TESTS)
        string(APPEND RAFT_COMPONENTS " distributed")
    endif()

    # We need to set this each time so that on subsequent calls to cmake
    # the raft-config.cmake re-evaluates the RAFT_NVTX value
    set(RAFT_NVTX ${PKG_NVTX})

    message(VERBOSE "CUML: raft FIND_PACKAGE_ARGUMENTS COMPONENTS ${RAFT_COMPONENTS}")

    rapids_cpm_find(raft ${PKG_VERSION}
      GLOBAL_TARGETS      raft::raft
      BUILD_EXPORT_SET    cuml-exports
      INSTALL_EXPORT_SET  cuml-exports
      COMPONENTS          ${RAFT_COMPONENTS}
      CPM_ARGS
        GIT_REPOSITORY         https://github.com/${PKG_FORK}/raft.git
        GIT_TAG                ${PKG_PINNED_TAG}
        SOURCE_SUBDIR          cpp
        EXCLUDE_FROM_ALL       ${PKG_EXCLUDE_FROM_ALL}
        OPTIONS
          "BUILD_TESTS OFF"
          "BUILD_PRIMS_BENCH OFF"
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
      FORK             rapidsai
      PINNED_TAG       ${rapids-cmake-checkout-tag}
      EXCLUDE_FROM_ALL ${CUML_EXCLUDE_RAFT_FROM_ALL}
      # When PINNED_TAG above doesn't match cuml,
      # force local raft clone in build directory
      # even if it's already installed.
      CLONE_ON_PIN     ${CUML_RAFT_CLONE_ON_PIN}
      NVTX             ${NVTX}
      )
