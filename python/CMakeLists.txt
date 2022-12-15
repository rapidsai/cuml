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

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

include(../fetch_rapids.cmake)

set(CUML_VERSION 23.02.00)

# We always need CUDA for cuML because the raft dependency brings in a
# header-only cuco dependency that enables CUDA unconditionally.
include(rapids-cuda)
rapids_cuda_init_architectures(cuml-python)

project(
  cuml-python
  VERSION ${CUML_VERSION}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX CUDA
)

################################################################################
# - User Options  --------------------------------------------------------------
option(CUML_UNIVERSAL "Build all cuML Python components." ON)
option(CUML_CPU "Build only cuML CPU Python components." OFF)
option(CUML_GPU "Build only cuML GPU Python components. Not supported yet, defaulting to CUML_UNIVERSAL" OFF)
option(FIND_CUML_CPP "Search for existing CUML C++ installations before defaulting to local files" OFF)
option(CUML_BUILD_WHEELS "Whether this build is generating a Python wheel." OFF)
option(SINGLEGPU "Disable all mnmg components and comms libraries" OFF)
set(CUML_RAFT_CLONE_ON_PIN OFF)


# todo: use CMAKE_MESSAGE_CONTEXT for prefix for logging.
# https://github.com/rapidsai/cuml/issues/4843
message(VERBOSE "CUML_PY: Searching for existing CUML C++ installations before defaulting to local files: ${FIND_CUML_CPP}")
message(VERBOSE "CUML_PY: Disabling all mnmg components and comms libraries: ${SINGLEGPU}")

set(CUML_ALGORITHMS "ALL" CACHE STRING "Experimental: Choose which algorithms are built into libcuml++.so. Can specify individual algorithms or groups in a semicolon-separated list.")
message(VERBOSE "CUML_PY: Building cuML with algoriths: '${CUML_ALGORITHMS}'.")

set(CUML_CPP_TARGET "cuml++")
set(CUML_CPP_SRC "../cpp")

################################################################################
# - Process User Options  ------------------------------------------------------

# If the user requested it, we attempt to find cuml.
if(FIND_CUML_CPP)
  # We need to call get_treelite explicitly because we need the correct
  # ${TREELITE_LIBS} definition for RF
  include(rapids-cpm)
  include(rapids-export)
  rapids_cpm_init()
  include(../cpp/cmake/thirdparty/get_treelite.cmake)
  find_package(cuml ${CUML_VERSION} REQUIRED)
else()
  set(cuml_FOUND OFF)
endif()

include(rapids-cython)

if(CUML_BUILD_WHEELS)
  set(CUML_PYTHON_TREELITE_TARGET treelite::treelite_static)
else()
  set(CUML_PYTHON_TREELITE_TARGET treelite::treelite)
endif()

if(NOT cuml_FOUND)
  set(BUILD_CUML_TESTS OFF)
  set(BUILD_PRIMS_TESTS OFF)
  set(BUILD_CUML_C_LIBRARY OFF)
  set(BUILD_CUML_EXAMPLES OFF)
  set(BUILD_CUML_BENCH OFF)
  set(BUILD_CUML_PRIMS_BENCH OFF)
  set(CUML_EXPORT_TREELITE_LINKAGE ON)

  set(_exclude_from_all "")
  if(CUML_BUILD_WHEELS)
    # Statically link dependencies if building wheels
    set(CUDA_STATIC_RUNTIME ON)
    set(CUML_USE_RAFT_STATIC ON)
    set(CUML_USE_FAISS_STATIC ON)
    set(CUML_USE_TREELITE_STATIC ON)
    set(CUML_USE_CUMLPRIMS_MG_STATIC ON)
    # Don't install the static libs into wheels
    set(CUML_EXCLUDE_RAFT_FROM_ALL ON)
    set(RAFT_EXCLUDE_FAISS_FROM_ALL ON)
    set(CUML_EXCLUDE_TREELITE_FROM_ALL ON)
    set(CUML_EXCLUDE_CUMLPRIMS_MG_FROM_ALL ON)

    # Don't install the cuML C++ targets into wheels
    set(_exclude_from_all EXCLUDE_FROM_ALL)
  endif()

  add_subdirectory(../cpp cuml-cpp ${_exclude_from_all})

  set(cython_lib_dir cuml)
  install(TARGETS ${CUML_CPP_TARGET} DESTINATION ${cython_lib_dir})
endif()

if(CUML_CPU)
  set(CUML_UNIVERSAL OFF)
  set(SINGLEGPU ON)

  set(CUML_ALGORITHMS "linearregression")
endif()

set(cuml_sg_libraries cuml::${CUML_CPP_TARGET})
set(cuml_mg_libraries cuml::${CUML_CPP_TARGET})

if(NOT SINGLEGPU)
  include("${CUML_CPP_SRC}/cmake/thirdparty/get_cumlprims_mg.cmake")
  set(cuml_mg_libraries
    cuml::${CUML_CPP_TARGET}
    cumlprims_mg::cumlprims_mg
  )
endif()

rapids_cython_init()

include(cmake/ConfigureAlgorithmsHelpers.cmake)

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

if(DEFINED cython_lib_dir)
  rapids_cython_add_rpath_entries(TARGET cuml PATHS "${cython_lib_dir}")
endif()
