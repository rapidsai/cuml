# =============================================================================
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
option(FIND_CUML_CPP "Search for existing CUML C++ installations before defaulting to local files" OFF)
option(SINGLEGPU "Disable all mnmg components and comms libraries" OFF)
option(USE_CUDA_MATH_WHEELS "Use the CUDA math wheels instead of the system libraries" OFF)
option(USE_CUVS_WHEEL "Use the cuVS wheel" OFF)
set(CUML_RAFT_CLONE_ON_PIN OFF)

# todo: use CMAKE_MESSAGE_CONTEXT for prefix for logging.
# https://github.com/rapidsai/cuml/issues/4843
message(VERBOSE "CUML_PY: Build only cuML CPU Python components.: ${CUML_CPU}")
message(VERBOSE "CUML_PY: Searching for existing CUML C++ installations before defaulting to local files: ${FIND_CUML_CPP}")
message(VERBOSE "CUML_PY: Disabling all mnmg components and comms libraries: ${SINGLEGPU}")

set(CUML_ALGORITHMS "ALL" CACHE STRING "Choose which algorithms are built cuML. Can specify individual algorithms or groups in a semicolon-separated list.")

set(CUML_CPP_TARGET "cuml++")
set(CUML_CPP_SRC "../../cpp")

################################################################################
# - Process User Options  ------------------------------------------------------

# We need to call get_treelite explicitly because we need the correct
# ${TREELITE_LIBS} definition for RF
#
# and it needs to come before find_package(cuml), because it's a PUBLIC
# dependency of cuml::cuml
include(rapids-cpm)
include(rapids-export)
rapids_cpm_init()

set(CUML_EXPORT_TREELITE_LINKAGE ON)
set(CUML_PYTHON_TREELITE_TARGET treelite::treelite)
# use dynamic treelite (provided by treelite wheels)
# TODO(jameslamb): does this make sense?
set(CUML_USE_TREELITE_STATIC OFF)
# TODO(jameslamb): should treelite just be included here?
set(CUML_EXCLUDE_TREELITE_FROM_ALL ON)
include(${CUML_CPP_SRC}/cmake/thirdparty/get_treelite.cmake)

find_package(cuml "${RAPIDS_VERSION}" REQUIRED)

# # If the user requested it, we attempt to find cuml.
# if(FIND_CUML_CPP)
#   # We need to call get_treelite explicitly because we need the correct
#   # ${TREELITE_LIBS} definition for RF
#   include(rapids-cpm)
#   include(rapids-export)
#   rapids_cpm_init()
#   find_package(cuml ${CUML_VERSION} REQUIRED)
#   include(${CUML_CPP_SRC}/cmake/thirdparty/get_treelite.cmake)
# else()
#   set(cuml_FOUND OFF)
# endif()

include(rapids-cython-core)

# cuml-cpu does not need libcuml++.so
if(NOT CUML_CPU)
  # (moved to libcuml/CMakeLists.txt)
endif()

if(CUML_CPU)
  set(CUML_UNIVERSAL OFF)
  set(SINGLEGPU ON)

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
else()
  set(cuml_sg_libraries cuml::${CUML_CPP_TARGET})
  set(cuml_mg_libraries cuml::${CUML_CPP_TARGET})

  list(APPEND CYTHON_FLAGS
  "--compile-time-env GPUBUILD=1")
endif()

if(NOT SINGLEGPU)
  include("${CUML_CPP_SRC}/cmake/thirdparty/get_cumlprims_mg.cmake")
  set(cuml_mg_libraries
    cuml::${CUML_CPP_TARGET}
    cumlprims_mg::cumlprims_mg
  )
endif()

 ################################################################################
 # - Build Cython artifacts -----------------------------------------------------

include("${CUML_CPP_SRC}/cmake/modules/ConfigureAlgorithms.cmake")
include(cmake/ConfigureCythonAlgorithms.cmake)

if(${CUML_CPU})
  # libcuml requires metrics built if HDSCAN is built, which is not the case
  # for cuml-cpu
  unset(metrics_algo)
endif()

message(VERBOSE "CUML_PY: Building cuML with algorithms: '${CUML_ALGORITHMS}'.")

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

if(USE_CUVS_WHEEL)
  rapids_cython_add_rpath_entries(TARGET cuml PATHS cuvs)
endif()
