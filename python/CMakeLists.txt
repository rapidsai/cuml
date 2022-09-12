# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)

if(POLICY CMP0094)
  # Make FindPython{,2,3} modules use LOCATION for lookup strategy.
  # https://cmake.org/cmake/help/latest/policy/CMP0094.html#policy:CMP0094
  cmake_policy(SET CMP0094 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0094 NEW)
endif()

include(../fetch_rapids.cmake)

include(rapids-cuda)
# TODO: This will not be necessary once we upgrade to CMake 3.22, which will pull in the required
# languages for the C++ project even if this project does not require those languges.
rapids_cuda_init_architectures(cuml-python)

set(CUML_VERSION 22.10.00)

project(
  cuml-python
  VERSION ${CUML_VERSION}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX
            # TODO: This will not be necessary once we upgrade to CMake 3.22, which will pull in the required
            # languages for the C++ project even if this project does not require those languges.
            CUDA
)

################################################################################
# - User Options  --------------------------------------------------------------
option(FIND_CUML_CPP "Search for existing CUML C++ installations before defaulting to local files" OFF)
option(SINGLEGPU "Disable all mnmg components and comms libraries" OFF)
option(CUML_BUILD_WHEELS "Whether we're building a wheel for pypi" OFF)

# todo: use CMAKE_MESSAGE_CONTEXT for prefix for logging.
# https://github.com/rapidsai/cuml/issues/4843
message(VERBOSE "CUML_PY: Searching for existing CUML C++ installations before defaulting to local files: ${FIND_CUML_CPP}")
message(VERBOSE "CUML_PY: Disabling all mnmg components and comms libraries: ${SINGLEGPU}")

set(CUML_CPP_TARGET "cuml++")

################################################################################
# - Process User Options  ------------------------------------------------------

# If the user requested it,  we attempt to find cuml.
if(FIND_CUML_CPP)
  find_package(cuml ${CUML_VERSION} REQUIRED)
else()

  set(BUILD_CUML_TESTS OFF)
  set(BUILD_PRIMS_TESTS OFF)
  set(BUILD_CUML_C_LIBRARY OFF)
  set(BUILD_CUML_EXAMPLES OFF)
  set(BUILD_CUML_BENCH OFF)
  set(BUILD_CUML_PRIMS_BENCH OFF)
  set(CUML_EXPORT_TREELITE_LINKAGE ON)
  message(STATUS "installing packages")

  # Statically link dependencies if building wheels
  set(CUDA_STATIC_RUNTIME ${CUML_BUILD_WHEELS})
  set(CUML_USE_RAFT_STATIC ${CUML_BUILD_WHEELS})
  set(CUML_USE_FAISS_STATIC ${CUML_BUILD_WHEELS})
  set(CUML_USE_TREELITE_STATIC ${CUML_BUILD_WHEELS})
  set(CUML_USE_CUMLPRIMS_MG_STATIC ${CUML_BUILD_WHEELS})
  # Don't install the static libs into wheels
  set(CUML_EXCLUDE_RAFT_FROM_ALL ${CUML_BUILD_WHEELS})
  set(RAFT_EXCLUDE_FAISS_FROM_ALL ${CUML_BUILD_WHEELS})
  set(CUML_EXCLUDE_TREELITE_FROM_ALL ${CUML_BUILD_WHEELS})
  set(CUML_EXCLUDE_CUMLPRIMS_MG_FROM_ALL ${CUML_BUILD_WHEELS})

  set(_exclude_from_all "")
  if(CUML_BUILD_WHEELS)
    # Don't install the cuML C++ targets into wheels
    set(_exclude_from_all EXCLUDE_FROM_ALL)
  endif()

  add_subdirectory(../cpp cuml-cpp ${_exclude_from_all})

  # Install libcuml++.so into the wheel dir
  install(TARGETS ${CUML_CPP_TARGET} DESTINATION cuml)
endif()

include(rapids-cython)
rapids_cython_init()

add_subdirectory(cuml/common)
add_subdirectory(cuml/internals)

add_subdirectory(cuml/cluster)
add_subdirectory(cuml/datasets)
add_subdirectory(cuml/decomposition)
add_subdirectory(cuml/ensemble)
add_subdirectory(cuml/explainer)
add_subdirectory(cuml/fil)
add_subdirectory(cuml/kernel_ridge)
add_subdirectory(cuml/linear_model)
add_subdirectory(cuml/manifold)
add_subdirectory(cuml/metrics)
add_subdirectory(cuml/metrics/cluster)
add_subdirectory(cuml/neighbors)
add_subdirectory(cuml/random_projection)
add_subdirectory(cuml/solvers)
add_subdirectory(cuml/svm)
add_subdirectory(cuml/tsa)

add_subdirectory(cuml/experimental/linear_model)
