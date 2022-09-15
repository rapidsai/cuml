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

set(CUML_MIN_VERSION_cumlprims_mg "${CUML_VERSION_MAJOR}.${CUML_VERSION_MINOR}.00")

if(NOT DEFINED CUML_CUMLPRIMS_MG_VERSION)
  set(CUML_CUMLPRIMS_MG_VERSION "${CUML_VERSION_MAJOR}.${CUML_VERSION_MINOR}")
endif()

if(NOT DEFINED CUML_CUMLPRIMS_MG_BRANCH)
  set(CUML_CUMLPRIMS_MG_BRANCH "branch-${CUML_CUMLPRIMS_MG_VERSION}")
endif()

if(NOT DEFINED CUML_CUMLPRIMS_MG_REPOSITORY)
  set(CUML_CUMLPRIMS_MG_REPOSITORY "git@github.com:rapidsai/cumlprims_mg.git")
endif()

function(find_and_configure_cumlprims_mg)

    set(oneValueArgs VERSION REPO PINNED_TAG BUILD_STATIC EXCLUDE_FROM_ALL CLONE_ON_PIN)
    cmake_parse_arguments(PKG "" "${oneValueArgs}" "" ${ARGN})

    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "branch-${CUML_CUMLPRIMS_MG_VERSION}")
      message(STATUS "Pinned tag found: ${PKG_PINNED_TAG}. Cloning cumlprims locally.")
      set(CPM_DOWNLOAD_cumlprims_mg ON)
    elseif(PKG_BUILD_STATIC AND (NOT CPM_cumlprims_mg_SOURCE))
      message(STATUS "CUML: Cloning cumlprims_mg locally to build static libraries.")
      set(CPM_DOWNLOAD_cumlprims_mg ON)
    endif()

    set(CUMLPRIMS_MG_BUILD_SHARED_LIBS ON)
    if(PKG_BUILD_STATIC)
      set(CUMLPRIMS_MG_BUILD_SHARED_LIBS OFF)
    endif()

    rapids_cpm_find(cumlprims_mg ${PKG_VERSION}
      GLOBAL_TARGETS      cumlprims_mg::cumlprims_mg
      BUILD_EXPORT_SET    cuml-exports
      INSTALL_EXPORT_SET  cuml-exports
      CPM_ARGS
        SOURCE_SUBDIR    cpp
        GIT_REPOSITORY   ${PKG_REPO}
        GIT_TAG          ${PKG_PINNED_TAG}
        EXCLUDE_FROM_ALL ${PKG_EXCLUDE_FROM_ALL}
        OPTIONS
          "BUILD_TESTS OFF"
          "BUILD_BENCHMARKS OFF"
          "BUILD_SHARED_LIBS ${CUMLPRIMS_MG_BUILD_SHARED_LIBS}"
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
                                REPO             ${CUML_CUMLPRIMS_MG_REPOSITORY}
                                PINNED_TAG       ${CUML_CUMLPRIMS_MG_BRANCH}
                                BUILD_STATIC     ${CUML_USE_CUMLPRIMS_MG_STATIC}
                                EXCLUDE_FROM_ALL ${CUML_EXCLUDE_CUMLPRIMS_MG_FROM_ALL}
                                # When PINNED_TAG above doesn't match cuml,
                                # force local cumlprims_mg clone in build directory
                                # even if it's already installed.
                                CLONE_ON_PIN     ON
                                )
