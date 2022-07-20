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

file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-22.08/RAPIDS.cmake
    ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-find)

set(CUML_VERSION 22.08.00)

project(
  cuml-python
  VERSION ${CUML_VERSION}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX
)

################################################################################
# - User Options  --------------------------------------------------------------
option(FIND_CUML_CPP "Search for existing CUML C++ installations before defaulting to local files" OFF)
option(SINGLEGPU "Disable all mnmg components and comms libraries" OFF)

message(VERBOSE "CUML_PY: Searching for existing CUML C++ installations before defaulting to local files: ${FIND_CUML_CPP}")
message(VERBOSE "CUML_PY: Disabling all mnmg components and comms libraries: ${SINGLEGPU}")

set(CUML_CPP_TARGET "cuml++")

################################################################################
# - Process User Options  ------------------------------------------------------


# If the user requested it,  we attempt to find cuml.
if(FIND_CUML_CPP)
  rapids_cpm_init()
	include(rapids-cuda)
	rapids_cuda_init_architectures(CUML)
  enable_language(CUDA)

  # FIXME: check of this is necessary
  # Since cugraph only enables CUDA optionally, we need to manually include the file that
  # rapids_cuda_init_architectures relies on `project` including.

  include("${CMAKE_PROJECT_cuml-python_INCLUDE}")
  include(../cpp/cmake/thirdparty/get_treelite.cmake)
    rapids_cpm_init()
  include(rapids-cuda)
  rapids_cuda_init_architectures(CUML)
  enable_language(CUDA)

  include("${CMAKE_PROJECT_cuml-python_INCLUDE}")
  include(../cpp/cmake/thirdparty/get_treelite.cmake)

  set(CUML_VERSION_MAJOR "22")
  set(CUML_VERSION_MINOR "08")
  set(CUML_USE_RAFT_NN ON)
  set(CUML_USE_RAFT_DIST ON)

  include(../cpp/cmake/thirdparty/get_raft.cmake)

  find_package(cuml ${CUML_VERSION} REQUIRED)
else()
  set(cuml_FOUND OFF)
endif()


if(NOT cuml_FOUND)
  # TODO: This will not be necessary once we upgrade to CMake 3.22, which will pull in the required
  # languages for the C++ project even if this project does not require those languges.
  include(rapids-cuda)
  rapids_cuda_init_architectures(cuml-python)
  enable_language(CUDA)

  # Since cuml only enables CUDA optionally, we need to manually include the file that
  # rapids_cuda_init_architectures relies on `project` including.
  include("${CMAKE_PROJECT_cuml-python_INCLUDE}")

  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)
  message(STATUS "installing packages")
  add_subdirectory(../../cpp cuml-cpp)

  # Since there are multiple subpackages of cudf._lib that require access to libcudf, we place the
  # library in the _lib/cpp directory as a single source of truth and modify the other rpaths
  # appropriately.
  install(TARGETS cuml DESTINATION cuml/library)
endif()


include(rapids-cython)
rapids_cython_init()

# FIXME: Update build directory
# add_subdirectory(cuml)

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

