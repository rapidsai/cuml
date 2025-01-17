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

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

include(../../rapids_config.cmake)

include(rapids-cuda)
rapids_cuda_init_architectures(libcuml-python)

project(
  libcuml-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

################################################################################
# - User Options  --------------------------------------------------------------
option(USE_CUDA_MATH_WHEELS "Use the CUDA math wheels instead of the system libraries" OFF)

################################################################################
# - Process User Options  ------------------------------------------------------

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

# In libcuml wheels, we always want to build in all cuML algorithms.
# This is the default in cpp/CMakeLists.txt, but just making that choice for wheels explicit here.
set(CUML_ALGORITHMS "ALL" CACHE STRING "Choose which algorithms are built cuML. Can specify individual algorithms or groups in a semicolon-separated list.")

# for libcuml wheels, always compile in the multi-node, multi-GPU stuff from cumlprims_mg
set(SINGLEGPU OFF)

set(CUML_CPP_TARGET "cuml++")
set(CUML_CPP_SRC "../../cpp")

# --- cumlprims_mg --- #
# ship cumlprims_mg in the 'libcuml' wheel (for re-use by 'cuml' wheels)
set(CUML_USE_CUMLPRIMS_MG_STATIC OFF)
set(CUML_EXCLUDE_CUMLPRIMS_MG_FROM_ALL OFF)

# --- cuVS --- #
set(CUML_USE_CUVS_STATIC OFF)

# don't unclude any cuVS stuff in libcuml wheel
# (expect anything building against libcuml to provide cuvs headers externally)
set(CUML_EXCLUDE_CUVS_FROM_ALL ON)

# --- raft --- #
set(CUML_RAFT_CLONE_ON_PIN OFF)
set(CUML_EXCLUDE_RAFT_FROM_ALL ON)

# --- treelite --- #
set(CUML_EXPORT_TREELITE_LINKAGE ON)
set(CUML_PYTHON_TREELITE_TARGET treelite::treelite_static)
set(CUML_USE_TREELITE_STATIC ON)
set(CUML_EXCLUDE_TREELITE_FROM_ALL ON)

# --- CUDA --- #
set(CUDA_STATIC_RUNTIME ON)

# Link to the CUDA wheels with shared libraries for CUDA 12+
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.0)
  set(CUDA_STATIC_MATH_LIBRARIES OFF)
else()
  if(USE_CUDA_MATH_WHEELS)
    message(FATAL_ERROR "Cannot use CUDA math wheels with CUDA < 12.0")
  endif()
  set(CUDA_STATIC_MATH_LIBRARIES ON)
endif()

add_subdirectory(../../cpp cuml-cpp)

if(NOT CUDA_STATIC_MATH_LIBRARIES AND USE_CUDA_MATH_WHEELS)
  # assumes libcuml++ is installed 2 levels deep, e.g. site-packages/cuml/lib64/libcuml++.so
  set(rpaths
    "$ORIGIN/../../nvidia/cublas/lib"
    "$ORIGIN/../../nvidia/cufft/lib"
    "$ORIGIN/../../nvidia/curand/lib"
    "$ORIGIN/../../nvidia/cusolver/lib"
    "$ORIGIN/../../nvidia/cusparse/lib"
    "$ORIGIN/../../nvidia/nvjitlink/lib"
  )
  set_property(TARGET ${CUML_CPP_TARGET} PROPERTY INSTALL_RPATH ${rpaths} APPEND)
endif()

# if dynamically linking to cuVS, set up an RPATH to find the libcuvs.so installed via the cuvs wheel
if(NOT CUML_USE_CUVS_STATIC)
  # assumes libcuml++ is installed 2 levels deep, e.g. site-packages/cuml/lib64/libcuml++.so
  set(rpaths "$ORIGIN/../../cuvs")
  set_property(TARGET ${CUML_CPP_TARGET} PROPERTY INSTALL_RPATH ${rpaths} APPEND)
endif()
