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

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

set(language_list "CXX")

include(rapids-cuda)
rapids_cuda_init_architectures(cuml-python)
list(APPEND language_list "CUDA")

project(
  cuml-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES ${language_list}
)

################################################################################
# - User Options  --------------------------------------------------------------
option(CUML_UNIVERSAL "Build all cuML Python components." ON)
option(SINGLEGPU "Disable all mnmg components and comms libraries" OFF)
option(USE_LIBCUML_WHEEL "Use libcuml wheel to provide some dependencies" OFF)

# todo: use CMAKE_MESSAGE_CONTEXT for prefix for logging.
# https://github.com/rapidsai/cuml/issues/4843
message(VERBOSE "CUML_PY: Disabling all mnmg components and comms libraries: ${SINGLEGPU}")

set(CUML_ALGORITHMS "ALL" CACHE STRING "Choose which algorithms are built cuML. Can specify individual algorithms or groups in a semicolon-separated list.")

set(CUML_CPP_TARGET "cuml++")
set(CUML_CPP_SRC "../../cpp")

################################################################################
# - Process User Options  ------------------------------------------------------

include(rapids-cpm)
include(rapids-export)
rapids_cpm_init()

# --- treelite --- #
# Need to call get_treelite explicitly because we need the correct
# ${TREELITE_LIBS} definition for RF.
#
# And because cuml Cython code needs the headers to satisfy calls like
# 'cdef extern from "treelite/c_api.h"'

# wheel builds use a static treelite, because the 'libtreelite.so' in 'treelite' wheels
# isn't intended for dynamic linking by third-party projects (e.g. hides its symbols)
if(USE_LIBCUML_WHEEL)
set(CUML_PYTHON_TREELITE_TARGET treelite::treelite_static)
set(CUML_USE_TREELITE_STATIC ON)
else()
set(CUML_PYTHON_TREELITE_TARGET treelite::treelite)
set(CUML_USE_TREELITE_STATIC OFF)
endif()

set(CUML_EXCLUDE_TREELITE_FROM_ALL ON)

include(${CUML_CPP_SRC}/cmake/thirdparty/get_treelite.cmake)

# --- libcuml --- #
find_package(cuml "${RAPIDS_VERSION}" REQUIRED)

set(cuml_sg_libraries cuml::${CUML_CPP_TARGET})
set(cuml_mg_libraries cuml::${CUML_CPP_TARGET})

if(NOT SINGLEGPU)
list(APPEND cuml_mg_libraries cumlprims_mg::cumlprims_mg)
endif()

 ################################################################################
 # - Build Cython artifacts -----------------------------------------------------

include("${CUML_CPP_SRC}/cmake/modules/ConfigureAlgorithms.cmake")
include(cmake/ConfigureCythonAlgorithms.cmake)

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
add_subdirectory(cuml/fil)
add_subdirectory(cuml/linear_model)
add_subdirectory(cuml/manifold)
add_subdirectory(cuml/metrics)
add_subdirectory(cuml/metrics/cluster)
add_subdirectory(cuml/neighbors)
add_subdirectory(cuml/solvers)
add_subdirectory(cuml/svm)
add_subdirectory(cuml/tsa)

add_subdirectory(cuml/experimental/linear_model)

# Include the pth file in the built wheel
install(FILES "_cuml_accel.pth" DESTINATION ".")
