# =============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

include(rapids-cuda)
rapids_cuda_init_architectures(libcuml-python)

project(
  libcuml-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# Check if cuml is already available. If so, it is the user's responsibility to ensure that the
# CMake package is also available at build time of the Python cuml package.
find_package(cuml "${RAPIDS_VERSION}")

if(cuml_FOUND)
  return()
endif()

unset(cuml_FOUND)

# --- cuML --- #
set(BUILD_CUML_TESTS OFF)
set(BUILD_PRIMS_TESTS OFF)
set(BUILD_CUML_C_LIBRARY OFF)
set(BUILD_CUML_EXAMPLES OFF)
set(BUILD_CUML_BENCH OFF)

# In libcuml wheels, we always want to build in all cuML algorithms. This is the default in
# cpp/CMakeLists.txt, but just making that choice for wheels explicit here.
set(CUML_ALGORITHMS
    "ALL"
    CACHE
      STRING
      "Choose which algorithms are built cuML. Can specify individual algorithms or groups in a semicolon-separated list."
)

# for libcuml wheels, always compile in the multi-node, multi-GPU stuff from cumlprims_mg
set(SINGLEGPU OFF)

set(CUML_CPP_TARGET "cuml++")
set(CUML_CPP_SRC "../../cpp")

# ship cumlprims_mg in the 'libcuml' wheel (for re-use by 'cuml' wheels)
set(CUML_EXCLUDE_CUMLPRIMS_MG_FROM_ALL OFF)

# --- cuVS --- #
set(CUML_USE_CUVS_STATIC ON)
set(CUML_EXCLUDE_CUVS_FROM_ALL ON)

# --- raft --- #
set(CUML_RAFT_CLONE_ON_PIN OFF)
set(CUML_EXCLUDE_RAFT_FROM_ALL ON)

# --- treelite --- #
set(CUML_PYTHON_TREELITE_TARGET treelite::treelite_static)
set(CUML_USE_TREELITE_STATIC ON)
set(CUML_EXCLUDE_TREELITE_FROM_ALL ON)

# --- CUDA --- #
set(CUDA_STATIC_RUNTIME ON)
set(CUDA_STATIC_MATH_LIBRARIES OFF)

add_subdirectory(../../cpp cuml-cpp)

# assumes libcuml++ is installed 2 levels deep, e.g. site-packages/cuml/lib64/libcuml++.so
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  # starting with CTK 13 wheels, all libraries except NCCL are grouped in this 'nvidia/cu13/lib'
  # directory
  set(rpaths "$ORIGIN/../../nvidia/cu13/lib")
else()
  set(rpaths
      "$ORIGIN/../../nvidia/cublas/lib"
      "$ORIGIN/../../nvidia/cufft/lib"
      "$ORIGIN/../../nvidia/curand/lib"
      "$ORIGIN/../../nvidia/cusolver/lib"
      "$ORIGIN/../../nvidia/cusparse/lib"
      "$ORIGIN/../../nvidia/nvjitlink/lib"
  )
endif()
list(APPEND rpaths "$ORIGIN/../../nvidia/nccl/lib")

set_property(
  TARGET ${CUML_CPP_TARGET}
  PROPERTY INSTALL_RPATH ${rpaths}
  APPEND
)
