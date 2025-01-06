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
option(USE_CUVS_WHEEL "Use the cuVS wheel" OFF)
set(CUML_RAFT_CLONE_ON_PIN OFF)

# todo: use CMAKE_MESSAGE_CONTEXT for prefix for logging.
# https://github.com/rapidsai/cuml/issues/4843
message(VERBOSE "CUML_PY: Build only cuML CPU Python components.: ${CUML_CPU}")
message(VERBOSE "CUML_PY: Disabling all mnmg components and comms libraries: ${SINGLEGPU}")

set(CUML_ALGORITHMS "ALL" CACHE STRING "Choose which algorithms are built cuML. Can specify individual algorithms or groups in a semicolon-separated list.")

set(CUML_CPP_TARGET "cuml++")
set(CUML_CPP_SRC "../../cpp")

################################################################################
# - Process User Options  ------------------------------------------------------

include(rapids-cpm)
include(rapids-export)
rapids_cpm_init()

# --- CCCL, RAFT, and RMM --- #
# CCCL before RMM, and RMM before RAFT
# TODO(jameslamb): skipping CCCL because we're already getting it from libraft... is that ok?
#   -- Found CCCL: /pyenv/versions/3.12.8/lib/python3.12/site-packages/libraft/lib64/rapids/cmake/cccl/cccl-config.cmake (found version "2.7.0.0")
#     CMake Error at /pyenv/versions/3.12.8/lib/python3.12/site-packages/libraft/lib64/cmake/NvidiaCutlass/NvidiaCutlassTargets.cmake:42 (message):
#     Some (but not all) targets in this export set were already defined.
#
#     Targets Defined: nvidia::cutlass::cutlass
#     Targets not yet defined: nvidia::cutlass::tools::util
#
# include(cmake/thirdparty/get_cccl.cmake)
include(${CUML_CPP_SRC}/cmake/thirdparty/get_rmm.cmake)
include(${CUML_CPP_SRC}/cmake/thirdparty/get_raft.cmake)

# --- treelite --- #
# We need to call get_treelite explicitly because we need the correct
# ${TREELITE_LIBS} definition for RF
#
# and it needs to come before find_package(cuml), because it's a PUBLIC
# dependency of cuml::cuml
set(CUML_EXPORT_TREELITE_LINKAGE ON)
set(CUML_PYTHON_TREELITE_TARGET treelite::treelite_static)
# TODO(jameslamb): is it safe for libcuml and cuml to both use their own static treelite?
set(CUML_USE_TREELITE_STATIC ON)
# TODO(jameslamb): should treelite just be included here?
set(CUML_EXCLUDE_TREELITE_FROM_ALL ON)
include(${CUML_CPP_SRC}/cmake/thirdparty/get_treelite.cmake)

# get nlohmann_json and rapidsjson
# TODO(jameslamb): maybe this isn't necessary?
# function(copy_interface_excludes)
#   set(_options "")
#   set(_one_value TARGET INCLUDED_TARGET)
#   set(_multi_value "")
#   cmake_parse_arguments(_CUML_INCLUDES "${_options}" "${_one_value}"
#                         "${_multi_value}" ${ARGN})
#   get_target_property(_includes ${_CUML_INCLUDES_INCLUDED_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
#   target_include_directories(${_CUML_INCLUDES_TARGET} PUBLIC ${_includes})
# endfunction()
# copy_interface_excludes(INCLUDED_TARGET treelite::treelite_static TARGET ${CUML_CPP_TARGET})

# --- cuvs --- #
if(USE_CUVS_WHEEL)
  set(CUML_USE_CUVS_STATIC OFF)
else()
  set(CUML_USE_CUVS_STATIC ON)
endif()

# either way, don't unclude any cuvs stuff in wheel
# (expect anything building against libcuml to provide cuvs headers externally)
set(CUML_EXCLUDE_CUVS_FROM_ALL ON)

# download this again just to get the headers
include(${CUML_CPP_SRC}/cmake/thirdparty/get_cuvs.cmake)

# --- cumlprims_mg --- #
set(CUML_USE_CUMLPRIMS_MG_STATIC ON)
# TODO(jameslamb): figure out how cumlprims_mg should work (re-download in cuml? only install/export the headers in libcuml?)
set(CUML_EXCLUDE_CUMLPRIMS_MG_FROM_ALL ON)
include(${CUML_CPP_SRC}/cmake/thirdparty/get_cumlprims_mg.cmake)

find_package(cuml "${RAPIDS_VERSION}" REQUIRED)

include(rapids-cython-core)

# cuml-cpu does not need libcuml++.so
# TODO(jameslamb): maybe all the stuff above is unnecessary in the CPU-only case?
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
