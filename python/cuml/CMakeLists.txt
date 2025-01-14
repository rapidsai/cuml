# =============================================================================
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

option(CUML_CPU "Build only cuML CPU Python components." OFF)
set(language_list "CXX")

if(NOT CUML_CPU)
  # We always need CUDA for cuML GPU because the raft dependency brings in a
  # header-only cuco dependency that enables CUDA unconditionally.
  include(rapids-cuda)
  rapids_cuda_init_architectures(cuml-python)
  list(APPEND language_list "CUDA")
endif()

project(
  cuml-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES ${language_list}
)

################################################################################
# - User Options  --------------------------------------------------------------
option(CUML_UNIVERSAL "Build all cuML Python components." ON)
option(SINGLEGPU "Disable all mnmg components and comms libraries" OFF)
option(USE_CUDA_MATH_WHEELS "Use the CUDA math wheels instead of the system libraries" OFF)
option(USE_CUVS_WHEEL "Use the cuVS wheel" OFF)
option(USE_LIBCUML_WHEEL "Use libcuml wheel to provide some dependencies" OFF)

# todo: use CMAKE_MESSAGE_CONTEXT for prefix for logging.
# https://github.com/rapidsai/cuml/issues/4843
message(VERBOSE "CUML_PY: Build only cuML CPU Python components.: ${CUML_CPU}")
message(VERBOSE "CUML_PY: Use CUDA math wheels instead of system libraries: ${USE_CUDA_MATH_WHEELS}")
message(VERBOSE "CUML_PY: Disabling all mnmg components and comms libraries: ${SINGLEGPU}")

set(CUML_ALGORITHMS "ALL" CACHE STRING "Choose which algorithms are built cuML. Can specify individual algorithms or groups in a semicolon-separated list.")

set(CUML_CPP_TARGET "cuml++")
set(CUML_CPP_SRC "../../cpp")

################################################################################
# - Process User Options  ------------------------------------------------------

if(CUML_CPU)
  set(CUML_UNIVERSAL OFF)
  set(SINGLEGPU ON)

  # only a subset of algorithms are supported in CPU-only cuML
  set(CUML_ALGORITHMS "linearregression")
  list(APPEND CUML_ALGORITHMS "pca")
  list(APPEND CUML_ALGORITHMS "tsvd")
  list(APPEND CUML_ALGORITHMS "elasticnet")
  list(APPEND CUML_ALGORITHMS "logisticregression")
  list(APPEND CUML_ALGORITHMS "ridge")
  list(APPEND CUML_ALGORITHMS "lasso")
  list(APPEND CUML_ALGORITHMS "umap")
  list(APPEND CUML_ALGORITHMS "knn")
  list(APPEND CUML_ALGORITHMS "hdbscan")
  list(APPEND CUML_ALGORITHMS "dbscan")
  list(APPEND CUML_ALGORITHMS "kmeans")

  # this won't be needed when we add CPU libcuml++ (FIL)
  set(cuml_sg_libraries "")

  list(APPEND CYTHON_FLAGS
  "--compile-time-env GPUBUILD=0")

# cuml-cpu does not need libcuml++.so
else()

  include(rapids-cpm)
  include(rapids-export)
  rapids_cpm_init()

  # --- CUDA --- #
  set(CUDA_STATIC_RUNTIME ON)

  # Link to the CUDA wheels with shared libraries for CUDA 12+
  #
  # This is here because we're rebuilding cuVS below...without it, cuVS on CUDA 11
  # dynamically links to CUDA math libs (cuBLAS, cuFFT, etc.), and then
  # that linkage results in those libraries being vendored in wheels by auditwheel.
  find_package(CUDAToolkit REQUIRED)
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.0)
    set(CUDA_STATIC_MATH_LIBRARIES OFF)
  else()
    if(USE_CUDA_MATH_WHEELS)
      message(FATAL_ERROR "Cannot use CUDA math wheels with CUDA < 12.0")
    endif()
    set(CUDA_STATIC_MATH_LIBRARIES ON)
  endif()

  # --- CCCL, RAFT, RMM ---#
  # find CCCL, RAFT, and RMM before cuVS, to avoid
  # cuVS CMake defining conflicting versions of targets like 'nvidia::cutlass::cutlass'
  include(${CUML_CPP_SRC}/cmake/thirdparty/get_cccl.cmake)
  include(${CUML_CPP_SRC}/cmake/thirdparty/get_rmm.cmake)
  include(${CUML_CPP_SRC}/cmake/thirdparty/get_raft.cmake)

  # --- cuVS --- #
  # Once there are 'libcuvs' wheels, it should be possible to remove this CPM build of cuvs.
  #
  #   * conda builds will find 'libcuvs' in the build environment
  #   * wheel builds will find 'libcuvs' wherever that wheel is installed
  #
  # Until then, this is necessary because cuVS is included in the public headers of both
  # libcuml (C++) and cuml (Cython).
  message(STATUS "--- [debug] finding cuVS for cuML")
  set(CUML_USE_CUVS_STATIC OFF)
  set(CUML_EXCLUDE_CUVS_FROM_ALL ON)
  include(${CUML_CPP_SRC}/cmake/thirdparty/get_cuvs.cmake)
  message(STATUS "--- [debug] done finding cuVS for cuML")

  # --- treelite --- #
  # We need to call get_treelite explicitly because we need the correct
  # ${TREELITE_LIBS} definition for RF.
  #
  # And because cuml Cython code needs to headers to satisfy calls like 'cdef extern from "treelite/c_api.h"'
  #
  # and it needs to come before find_package(cuml), because it's a PUBLIC
  # dependency of cuml::cuml.
  #
  # TODO(jameslamb): clean up these comments

  # wheel builds use a static treelite, because the 'libtreelite.so' in 'treelite' wheels
  # isn't intended for dynamic linking by third-party projects (e.g. hides its symbols)
  if(USE_LIBCUML_WHEEL)
    # TODO(jameslamb): is it safe for libcuml++ libcuml wheels and Cython extensions in 'cuml' wheels
    #                  to use separate static treelite?
    set(CUML_PYTHON_TREELITE_TARGET treelite::treelite_static)
    set(CUML_USE_TREELITE_STATIC ON)
  else()
    set(CUML_PYTHON_TREELITE_TARGET treelite::treelite)
    set(CUML_USE_TREELITE_STATIC OFF)
  endif()

  set(CUML_EXCLUDE_TREELITE_FROM_ALL ON)

  message(STATUS "--- [debug] finding treelite for cuML")
  include(${CUML_CPP_SRC}/cmake/thirdparty/get_treelite.cmake)

  # --- libcuml --- #
  message(STATUS "--- [debug] finding cuML")
  find_package(cuml "${RAPIDS_VERSION}" REQUIRED)
  message(STATUS "--- [debug] found cuML")

  set(cuml_sg_libraries cuml::${CUML_CPP_TARGET})
  set(cuml_mg_libraries cuml::${CUML_CPP_TARGET})

  list(APPEND CYTHON_FLAGS
  "--compile-time-env GPUBUILD=1")
endif()

 ################################################################################
 # - Build Cython artifacts -----------------------------------------------------

include("${CUML_CPP_SRC}/cmake/modules/ConfigureAlgorithms.cmake")
include(cmake/ConfigureCythonAlgorithms.cmake)

if(CUML_CPU)
  # libcuml requires metrics built if HDSCAN is built, which is not the case
  # for cuml-cpu
  unset(metrics_algo)
endif()

message(VERBOSE "CUML_PY: Building cuML with algorithms: '${CUML_ALGORITHMS}'.")

include(rapids-cython-core)
rapids_cython_init()

add_subdirectory(cuml/common)
add_subdirectory(cuml/internals)

add_subdirectory(cuml/cluster)
add_subdirectory(cuml/datasets)
add_subdirectory(cuml/decomposition)
add_subdirectory(cuml/ensemble)
add_subdirectory(cuml/explainer)
add_subdirectory(cuml/experimental/fil)
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
