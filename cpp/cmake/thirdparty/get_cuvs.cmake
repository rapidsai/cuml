#=============================================================================
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

set(CUML_MIN_VERSION_cuvs "${CUML_VERSION_MAJOR}.${CUML_VERSION_MINOR}.00")

function(find_and_configure_cuvs)
    set(oneValueArgs VERSION FORK PINNED_TAG EXCLUDE_FROM_ALL USE_CUVS_STATIC COMPILE_LIBRARY CLONE_ON_PIN)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "${rapids-cmake-checkout-tag}")
        message(STATUS "CUML: CUVS pinned tag found: ${PKG_PINNED_TAG}. Cloning cuvs locally.")
        set(CPM_DOWNLOAD_cuvs ON)
    elseif(PKG_USE_CUVS_STATIC AND (NOT CPM_cuvs_SOURCE))
        message(STATUS "CUML: Cloning cuvs locally to build static libraries.")
        set(CPM_DOWNLOAD_cuvs ON)
    else()
        message(STATUS "Not cloning cuvs locally")
    endif()

    if(PKG_USE_CUVS_STATIC)
      set(CUVS_LIB cuvs::cuvs_static PARENT_SCOPE)
    else()
      set(CUVS_LIB cuvs::cuvs PARENT_SCOPE)
    endif()

    set(CUVS_BUILD_MG_ALGOS ON)
    if(SINGLEGPU)
      set(CUVS_BUILD_MG_ALGOS OFF)
    endif()

    rapids_cpm_find(cuvs ${PKG_VERSION}
      GLOBAL_TARGETS      cuvs::cuvs
      CPM_ARGS
        GIT_REPOSITORY         https://github.com/${PKG_FORK}/cuvs.git
        GIT_TAG                ${PKG_PINNED_TAG}
        SOURCE_SUBDIR          cpp
        EXCLUDE_FROM_ALL       ${PKG_EXCLUDE_FROM_ALL}
        OPTIONS
          "BUILD_TESTS OFF"
          "BUILD_CAGRA_HNSWLIB OFF"
          "BUILD_CUVS_BENCH OFF"
          "BUILD_MG_ALGOS ${CUVS_BUILD_MG_ALGOS}"

    )

    if(cuvs_ADDED)
        message(VERBOSE "CUML: Using CUVS located in ${cuvs_SOURCE_DIR}")
    else()
        message(VERBOSE "CUML: Using CUVS located in ${cuvs_DIR}")
    endif()


endfunction()

# Change pinned tag here to test a commit in CI
# To use a different CUVS locally, set the CMake variable
# CPM_cuvs_SOURCE=/path/to/local/cuvs
find_and_configure_cuvs(VERSION          ${CUML_MIN_VERSION_cuvs}
      FORK             rapidsai
      PINNED_TAG       ${rapids-cmake-checkout-tag}
      EXCLUDE_FROM_ALL ${CUML_EXCLUDE_CUVS_FROM_ALL}
      # When PINNED_TAG above doesn't match cuml,
      # force local cuvs clone in build directory
      # even if it's already installed.
      CLONE_ON_PIN     ${CUML_CUVS_CLONE_ON_PIN}
      COMPILE_LIBRARY  ${CUML_CUVS_COMPILED}
      USE_CUVS_STATIC  ${CUML_USE_CUVS_STATIC}
      )
