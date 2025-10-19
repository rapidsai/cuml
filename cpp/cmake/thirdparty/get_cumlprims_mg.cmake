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

set(CUML_MIN_VERSION_cumlprims_mg "${CUML_VERSION_MAJOR}.${CUML_VERSION_MINOR}.00")

function(find_and_configure_cumlprims_mg)

    set(oneValueArgs VERSION FORK PINNED_TAG EXCLUDE_FROM_ALL CLONE_ON_PIN)
    cmake_parse_arguments(PKG "" "${oneValueArgs}" "" ${ARGN} )

    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "${rapids-cmake-checkout-tag}")
        message("Pinned tag found: ${PKG_PINNED_TAG}. Cloning cumlprims locally.")
        set(CPM_DOWNLOAD_cumlprims_mg ON)
    endif()

    rapids_cpm_find(cumlprims_mg ${PKG_VERSION}
      GLOBAL_TARGETS      cumlprims_mg::cumlprims_mg
      BUILD_EXPORT_SET    cuml-exports
      INSTALL_EXPORT_SET  cuml-exports
        CPM_ARGS
          GIT_REPOSITORY https://github.com/${PKG_FORK}/cumlprims_mg.git
          GIT_TAG        ${PKG_PINNED_TAG}
          EXCLUDE_FROM_ALL ${PKG_EXCLUDE_FROM_ALL}
          SOURCE_SUBDIR    cpp
          OPTIONS
            "BUILD_TESTS OFF"
            "BUILD_BENCHMARKS OFF"
    )

endfunction()

###
# Change pinned tag and fork here to test a commit in CI
#
# To use a locally-built cumlprims_mg package, set the CMake variable
# `-D cumlprims_mg_ROOT=/path/to/cumlprims_mg/build`
#
# To use a local clone of cumlprims_mg source and allow CMake to build
# cumlprims_mg as part of building cuml itself, set the CMake variable
# `-D CPM_cumlprims_mg_SOURCE=/path/to/cumlprims_mg`
###
find_and_configure_cumlprims_mg(VERSION          ${CUML_MIN_VERSION_cumlprims_mg}
                                FORK       rapidsai
                                PINNED_TAG ${rapids-cmake-checkout-tag}
                                EXCLUDE_FROM_ALL ${CUML_EXCLUDE_CUMLPRIMS_MG_FROM_ALL}
                                # When PINNED_TAG above doesn't match cuml,
                                # force local cumlprims_mg clone in build directory
                                # even if it's already installed.
                                CLONE_ON_PIN     ON
                                )
