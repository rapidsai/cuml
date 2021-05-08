#=============================================================================
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)
include(FetchContent)
FetchContent_Declare(
  rapids-cmake
  GIT_REPOSITORY git@github.com:rapidsai/rapids-cmake.git
  GIT_TAG        origin/branch-0.20
  )
FetchContent_MakeAvailable(rapids-cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(CUML)

project(CUML VERSION 0.20.0 LANGUAGES CXX CUDA)

##############################################################################
# - build type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# this is needed for clang-tidy runs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##############################################################################
# - User Options  ------------------------------------------------------------

option(BUILD_CUML_C_LIBRARY "Build libcuml_c shared library. Contains the cuML C API" ON)
option(BUILD_CUML_CPP_LIBRARY "Build libcuml shared library" ON)
option(BUILD_CUML_TESTS "Build cuML algorithm tests" ON)
option(BUILD_CUML_MG_TESTS "Build cuML multigpu algorithm tests" OFF)
option(BUILD_PRIMS_TESTS "Build ml-prim tests" ON)
option(BUILD_CUML_EXAMPLES "Build C++ API usage examples" ON)
option(BUILD_CUML_BENCH "Build cuML C++ benchmark tests" ON)
option(BUILD_CUML_PRIMS_BENCH "Build ml-prims C++ benchmark tests" ON)
option(BUILD_CUML_STD_COMMS "Build the standard NCCL+UCX Communicator" ON)
option(BUILD_CUML_MPI_COMMS "Build the MPI+NCCL Communicator (used for testing)" OFF)
option(BUILD_STATIC_FAISS "Build the FAISS library for nearest neighbors search on GPU" OFF)
option(BUILD_GTEST "Build the GTEST library for running libcuml++ and prims test executables" OFF)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
option(DISABLE_OPENMP "Disable OpenMP" OFF)
option(ENABLE_CUMLPRIMS_MG "Enable algorithms that use libcumlprims_mg" ON)
option(KERNEL_INFO "Enable kernel resource usage info" OFF)
option(LINE_INFO "Enable lineinfo in nvcc" OFF)
option(NVTX "Enable nvtx markers" OFF)
option(SINGLEGPU "Disable all mnmg components and comms libraries" OFF)
option(USE_CCACHE "Cache build artifacts with ccache" OFF)

message(VERBOSE "CUML: Building libcuml_c shared library. Contains the cuML C API: ${BUILD_CUML_C_LIBRARY}")
message(VERBOSE "CUML: Building libcuml shared library: ${BUILD_CUML_CPP_LIBRARY}")
message(VERBOSE "CUML: Building cuML algorithm tests: ${BUILD_CUML_TESTS}")
message(VERBOSE "CUML: Building cuML multigpu algorithm tests: ${BUILD_CUML_MG_TESTS}")
message(VERBOSE "CUML: Building ml-prims tests: ${BUILD_PRIMS_TESTS}")
message(VERBOSE "CUML: Building C++ API usage examples: ${BUILD_CUML_EXAMPLES}")
message(VERBOSE "CUML: Building cuML C++ benchmark tests: ${BUILD_CUML_BENCH}")
message(VERBOSE "CUML: Building ml-prims C++ benchmark tests: ${BUILD_CUML_PRIMS_BENCH}")
message(VERBOSE "CUML: Building the standard NCCL+UCX Communicator: ${BUILD_CUML_STD_COMMS}")
message(VERBOSE "CUML: Building the MPI+NCCL Communicator (used for testing): ${BUILD_CUML_MPI_COMMS}")
message(VERBOSE "CUML: Building the FAISS library for nearest neighbors search on GPU: ${BUILD_STATIC_FAISS}")
message(VERBOSE "CUML: Building the GTEST library for running libcuml++ and prims test executables: ${BUILD_GTEST}")
message(VERBOSE "CUML: Enabling detection of conda environment for dependencies: ${DETECT_CONDA_ENV}")
message(VERBOSE "CUML: Disabling OpenMP: ${DISABLE_OPENMP}")
message(VERBOSE "CUML: Enabling algorithms that use libcumlprims_mg: ${ENABLE_CUMLPRIMS_MG}")
message(VERBOSE "CUML: Enabling kernel resource usage info: ${KERNEL_INFO}")
message(VERBOSE "CUML: Enabling lineinfo in nvcc: ${LINE_INFO}")
message(VERBOSE "CUML: Enabling nvtx markers: ${NVTX}")
message(VERBOSE "CUML: Disabling all mnmg components and comms libraries: ${SINGLEGPU}")
message(VERBOSE "CUML: Cache build artifacts with ccache: ${USE_CCACHE}")

# Set RMM logging level
set(RMM_LOGGING_LEVEL "INFO" CACHE STRING "Choose the logging level.")
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
message(VERBOSE "CUML: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

##############################################################################
# - Conda environment detection ----------------------------------------------

if(DETECT_CONDA_ENV)
  rapids_cmake_support_conda_env( conda_env MODIFY_PREFIX_PATH )
  if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
      message(STATUS "CUML: No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
      set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
  endif()
endif()

##############################################################################
# - compiler options ---------------------------------------------------------

# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
rapids_find_package(CUDAToolkit REQUIRED
    BUILD_EXPORT_SET raft-exports
    INSTALL_EXPORT_SET raft-exports
    )
include(cmake/modules/ConfigureCUDA.cmake)

##############################################################################
# - Set options based on user defined one  -----------------------------------

# Enabling libcuml enables building libcuml++
if(BUILD_CUML_C_LIBRARY)
  set(BUILD_CUML_CPP_LIBRARY ON)
endif(BUILD_CUML_C_LIBRARY)

# Disabling libcuml++ disables building algorithm tests and examples
if(NOT BUILD_CUML_CPP_LIBRARY)
  set(BUILD_CUML_C_LIBRARY OFF)
  set(BUILD_CUML_TESTS OFF)
  set(BUILD_CUML_MG_TESTS OFF)
  set(BUILD_CUML_EXAMPLES OFF)
endif(NOT BUILD_CUML_CPP_LIBRARY)

# SingleGPU build disables cumlprims_mg and comms components
if(SINGLEGPU)
  message(STATUS "Detected SINGLEGPU build option")
  message(STATUS "Disabling Multi-GPU components and comms libraries")
  set(BUILD_CUML_MG_TESTS OFF)
  set(BUILD_CUML_STD_COMMS OFF)
  set(BUILD_CUML_MPI_COMMS OFF)
  set(ENABLE_CUMLPRIMS_MG OFF)
  set(WITH_UCX OFF)
endif(SINGLEGPU)

if(BUILD_CUML_MG_TESTS AND NOT SINGLEGPU)
  message(STATUS "Detected BUILD_CUML_MG_TESTS set to ON. Enabling BUILD_CUML_MPI_COMMS")
  set(BUILD_CUML_MPI_COMMS ON)
endif(BUILD_CUML_MG_TESTS AND NOT SINGLEGPU)

if(USE_CCACHE)
  set(CMAKE_C_COMPILER_LAUNCHER ccache)
  set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
  set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
endif(USE_CCACHE)

##############################################################################
# - Requirements -------------------------------------------------------------

if (NOT DISABLE_OPENMP OR NOT ${DISABLE_OPENMP})
  find_package(OpenMP)
  if(OpenMP_FOUND)
    message(STATUS "CUML: OpenMP found in ${OPENMP_INCLUDE_DIRS}")
  endif(OpenMP_FOUND)
endif(NOT DISABLE_OPENMP OR NOT ${DISABLE_OPENMP})

# add third party dependencies using CPM
rapids_cpm_init()

include(cmake/thirdparty/get_rmm.cmake)
include(cmake/thirdparty/get_faiss.cmake)
include(cmake/thirdparty/get_treelite.cmake)
include(cmake/thirdparty/get_raft.cmake)

if(NOT SINGLEGPU)
  include(cmake/thirdparty/get_nccl.cmake)
  include(cmake/thirdparty/get_ucx.cmake)
endif()

if(ENABLE_CUMLPRIMS_MG)
  include(cmake/thirdparty/get_cumlprims_mg.cmake)
endif()

if(BUILD_CUML_TESTS OR BUILD_PRIMS_TESTS)
  include(cmake/thirdparty/get_gtest.cmake)
endif()

##############################################################################
# - External Dependencies-----------------------------------------------------

# set(CUML_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE STRING
#   "Path to cuml include directories")

##############################################################################
# - include paths ------------------------------------------------------------

# set(CUML_INCLUDE_DIRECTORIES
#   ${CUML_INCLUDE_DIR}
#   ${SPDLOG_DIR}/src/spdlog/include
#   ${FAISS_INCLUDE_DIRS}
#   ${RAFT_DIR}/cpp/include
#   ${RMM_INCLUDE_DIRS})

# # These directories will be used by dependent targets
# set(CUML_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
# set(CUML_SRC_PRIMS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src_prims)
# set(CUML_TEST_PRIMS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test/prims)

# if(NOT CUB_IS_PART_OF_CTK)
#   list(APPEND CUML_INCLUDE_DIRECTORIES ${CUB_DIR}/src/cub)
# endif(NOT CUB_IS_PART_OF_CTK)

# set(CUML_PRIVATE_LINK_LIBRARIES
#   FAISS::FAISS
#   treelite::treelite
#   treelite::treelite_runtime
#   )

##############################################################################
# - build libcuml++ shared library -------------------------------------------

if(BUILD_CUML_CPP_LIBRARY)

  # single GPU components
  add_library(cuml SHARED
    src/arima/batched_arima.cu
    src/arima/batched_kalman.cu
    src/common/logger.cpp
    src/common/nvtx.cu
    src/datasets/make_arima.cu
    src/datasets/make_blobs.cu
    src/datasets/make_regression.cu
    src/dbscan/dbscan.cu
    src/decisiontree/decisiontree.cu
    src/explainer/kernel_shap.cu
    src/explainer/permutation_shap.cu
    src/fil/fil.cu
    src/fil/infer.cu
    src/glm/glm.cu
    src/genetic/genetic.cu
    src/genetic/node.cu
    src/holtwinters/holtwinters.cu
    src/kmeans/kmeans.cu
    src/knn/knn.cu
    src/knn/knn_sparse.cu
    src/hierarchy/linkage.cu
    src/metrics/accuracy_score.cu
    src/metrics/adjusted_rand_index.cu
    src/metrics/completeness_score.cu
    src/metrics/entropy.cu
    src/metrics/homogeneity_score.cu
    src/metrics/kl_divergence.cu
    src/metrics/mutual_info_score.cu
    src/metrics/pairwise_distance.cu
    src/metrics/r2_score.cu
    src/metrics/rand_index.cu
    src/metrics/silhouette_score.cu
    src/metrics/trustworthiness.cu
    src/metrics/v_measure.cu
    src/pca/pca.cu
    src/randomforest/randomforest.cu
    src/random_projection/rproj.cu
    src/solver/lars.cu
    src/solver/solver.cu
    src/spectral/spectral.cu
    src/svm/svc.cu
    src/svm/svr.cu
    src/svm/ws_util.cu
    src/tsa/auto_arima.cu
    src/tsa/stationarity.cu
    src/tsne/tsne.cu
    src/tsvd/tsvd.cu
    src/umap/umap.cu)

  # mnmg components
  if(NOT SINGLEGPU)
    target_sources(cuml
      PRIVATE
        src/glm/ols_mg.cu
        src/glm/preprocess_mg.cu
        src/glm/ridge_mg.cu
        src/kmeans/kmeans_mg.cu
        src/knn/knn_mg.cu
        src/knn/knn_classify_mg.cu
        src/knn/knn_regress_mg.cu
        src/pca/pca_mg.cu
        src/pca/sign_flip_mg.cu
        src/solver/cd_mg.cu
        src/tsvd/tsvd_mg.cu
    )
  endif(NOT SINGLEGPU)

  add_library(cuml::cuml ALIAS cuml)

  set_target_properties(cuml
    PROPERTIES BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               # set target compile options
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       17
               CUDA_STANDARD_REQUIRED              ON
  )

  target_compile_definitions(cuml
    PUBLIC
      $<$<BOOL:${NVTX}>:NVTX_ENABLED>
      DISABLE_CUSPARSE_DEPRECATED
    PRIVATE
      CUML_CPP_API
  )

  target_compile_options(cuml
    PUBLIC
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda
                                 --expt-relaxed-constexpr
                                 -Xcudafe
                                 --diag_suppress=unrecognized_gcc_pragma
                                 -ccbin ${CMAKE_CXX_COMPILER}>

    PRIVATE
      $<$<AND:$<BOOL:${LINE_INFO}>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>
      $<$<AND:$<BOOL:${KERNEL_INFO}>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas=-v>
  )

  target_include_directories(cuml
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src_prims>
      $<BUILD_INTERFACE:${RAFT_DIR}/cpp/include>
      $<BUILD_INTERFACE:${CUTLASS_DIR}/src/cutlass>
      $<BUILD_INTERFACE:$<$<BOOL:${ENABLE_CUMLPRIMS_MG}>:${cumlprims_mg_INCLUDE_DIRS}>>
    PRIVATE
      $<$<OR:$<BOOL:BUILD_CUML_STD_COMMS>,$<BOOL:BUILD_CUML_MPI_COMMS>>:${NCCL_INCLUDE_DIRS}>
      $<$<BOOL:BUILD_CUML_MPI_COMMS>:${MPI_CXX_INCLUDE_PATH}>
    INTERFACE
      $<INSTALL_INTERFACE:include>
  )

  target_link_libraries(cuml
    PUBLIC
      rmm::rmm
      raft::raft
    PRIVATE
      CUDA::cublas
      CUDA::curand
      CUDA::cusolver
      CUDA::cudart
      CUDA::cusparse
      $<$<BOOL:NVTX>:CUDA::nvToolsExt>
      FAISS::FAISS
      treelite::treelite
      treelite::treelite_runtime
      $<$<BOOL:OpenMP_FOUND>:OpenMP::OpenMP_CXX>
      $<$<OR:$<BOOL:${BUILD_CUML_STD_COMMS}>,$<BOOL:${BUILD_CUML_MPI_COMMS}>>:NCCL::NCCL>
      $<$<BOOL:${BUILD_CUML_MPI_COMMS}>:${MPI_CXX_LIBRARIES}>
      $<$<BOOL:${ENABLE_CUMLPRIMS_MG}>:cumlprims_mg::cumlprims_mg>
  )

  # If we export the libdmlc symbols, they can lead to weird crashes with other
  # libraries that use libdmlc. This just hides the symbols internally.
  target_link_options(cuml PRIVATE "-Wl,--exclude-libs,libdmlc.a")
  # same as above, but for protobuf library
  target_link_options(cuml PRIVATE "-Wl,--exclude-libs,libprotobuf.a")

endif(BUILD_CUML_CPP_LIBRARY)

##############################################################################
# - build libcuml C shared library -------------------------------------------

# if(BUILD_CUML_C_LIBRARY)
#   add_library(cuml_c SHARED
#     src/common/cumlHandle.cpp
#     src/common/cuml_api.cpp
#     src/dbscan/dbscan_api.cpp
#     src/glm/glm_api.cpp
#     src/holtwinters/holtwinters_api.cpp
#     src/knn/knn_api.cpp
#     src/svm/svm_api.cpp
#   )

#   add_library(cuml::cuml_c ALIAS cuml_c)

#   # target_compile_definitions(cuml_c
#   #   PRIVATE
#   #     CUML_C_API)

#   # target_include_directories(cuml_c
#   #   PRIVATE
#   #     ${CUML_SRC_DIR}
#   # )

#   target_link_libraries(cuml_c
#     PUBLIC
#       cuml
#   )

# endif(BUILD_CUML_C_LIBRARY)

##############################################################################
# - build test executables ---------------------------------------------------

# if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
#   include(CTest)
# endif()

# if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR CUML_BUILD_TESTING) AND BUILD_TESTING)
#   add_subdirectory(test ${PROJECT_BINARY_DIR}/test)
# endif()

##############################################################################
# - build examples ------------------------------------------------------------------------------

# if(BUILD_CUML_EXAMPLES)
#   add_subdirectory(examples)
# endif(BUILD_CUML_EXAMPLES)

##############################################################################
# - install targets ----------------------------------------------------------

# include(GNUInstallDirs)
# set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/cuml)

# install(TARGETS
#           ${CUML_CPP_TARGET}
#           ${CUML_C_TARGET}
#         EXPORT
#           cuml-targets
#         DESTINATION
#           ${CMAKE_INSTALL_LIBDIR}
# )

# # Copy the include and raft directories
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/cuml DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# install(DIRECTORY ${RAFT_INCLUDE_DIR} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cuml)

# include(CMakePackageConfigHelpers)
# configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/cuml-config.cmake.in
#                               ${CMAKE_CURRENT_BINARY_DIR}/cuml-config.cmake
#                               INSTALL_DESTINATION ${INSTALL_CONFIGDIR})

# write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/cuml-config-version.cmake
#                                  VERSION ${PROJECT_VERSION}
#                                  COMPATIBILITY SameMinorVersion)

# install(
#   EXPORT cuml-targets
#   FILE cuml-targets.cmake
#   NAMESPACE CUML::
#   DESTINATION ${INSTALL_CONFIGDIR})

# install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cuml-config.cmake
#               ${CMAKE_CURRENT_BINARY_DIR}/cuml-config-version.cmake
#               ${CMAKE_CURRENT_LIST_DIR}/cmake/FindRMM.cmake
#         DESTINATION ${INSTALL_CONFIGDIR})

##############################################################################
# - export targets -----------------------------------------------------------


# Export any Cmake modules
# configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/FindRapidJSON.cmake
#     ${CMAKE_CURRENT_BINARY_DIR}/FindRapidJSON.cmake
#     COPYONLY)

# Export targets
# export(EXPORT cuml-targets
#     FILE ${CMAKE_CURRENT_BINARY_DIR}/cuml-targets.cmake
#     NAMESPACE CUML::)

# #Register package in user's package registry
# export(PACKAGE CUML)

##############################################################################
# - build benchmark executable -----------------------------------------------

# if(BUILD_CUML_BENCH)
#   add_subdirectory(bench ${PROJECT_BINARY_DIR}/bench)
# endif(BUILD_CUML_BENCH)

###################################################################################################
# - make documentation ----------------------------------------------------------------------------
# requires doxygen and graphviz to be installed
# from build directory, run make docs_cugraph

# doc targets for cugraph
find_package(Doxygen 1.8.11)
if(Doxygen_FOUND)
    add_custom_command(OUTPUT CUML_DOXYGEN
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
                       COMMAND doxygen Doxyfile
                       VERBATIM)

    add_custom_target(docs_cuml DEPENDS CUGRAPH_DOXYGEN)
endif()
