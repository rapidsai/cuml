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

set (CMAKE_FIND_NO_INSTALL_PREFIX TRUE FORCE)

cmake_minimum_required(VERSION 3.17...3.19 FATAL_ERROR)

project(CUML VERSION 0.18.0 LANGUAGES C CXX)

##############################################################################
# - build type ---------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release")
endif()

# this is needed for clang-tidy runs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##############################################################################
# - User Options  ------------------------------------------------------------

set(BLAS_LIBRARIES "" CACHE STRING
    "Location of BLAS library for FAISS build.")

option(BUILD_CUML_C_LIBRARY "Build libcuml shared library. Contains the cuML C API" ON)

option(BUILD_CUML_CPP_LIBRARY "Build libcuml++ shared library" ON)

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

set(BUILD_RPATH_USE_ORIGIN "Whether to use relative paths for the build RPATH." TRUE)

set(PARALLEL_LEVEL "" CACHE STRING
    "Sub-projects parallel level for compilation. Currently only affects FAISS" )

set(GPU_ARCHS "" CACHE STRING
    "List of GPU architectures (semicolon-separated) to be compiled for.
    Pass 'ALL' if you want to compile for all supported GPU architectures.
    Empty string means to auto-detect the GPUs on the current system")


##############################################################################
# - Conda environment detection ----------------------------------------------

# conda-build conda package building detection
if("$ENV{CONDA_BUILD}" STREQUAL "1")
    message(STATUS "Detected conda-build call, CMAKE_PREFIX_PATH unchanged.")

# otherwise detecting conda environment
else("$ENV{CONDA_BUILD}" STREQUAL "1")

  if(DETECT_CONDA_ENV)

    if(DEFINED ENV{CONDA_PREFIX})
      set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_PREFIX_PATH}")

      message(STATUS "Detected Conda environment, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")

      if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        message(STATUS "No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
        set (CMAKE_INSTALL_PREFIX $ENV{CONDA_PREFIX})

      endif (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

    else(DEFINED ENV{CONDA_PREFIX})
      message(STATUS "No Conda environment detected, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")

    endif(DEFINED ENV{CONDA_PREFIX})

  else(DETECT_CONDA_ENV)
    message(STATUS "No Conda detection being used")

  endif(DETECT_CONDA_ENV)

endif("$ENV{CONDA_BUILD}" STREQUAL "1")


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

# Create FindPackage.cmake files to use find(package) functionality for
# dependencies

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates)

if(ENABLE_CUMLPRIMS_MG)
  GENERATE_FIND_MODULE(
    NAME         cumlprims_mg
    HEADER_NAME  cumlprims.hpp
    LIBRARY_NAME cumlprims
    LOCATION     cumlprims)
endif(ENABLE_CUMLPRIMS_MG)

if(BUILD_CUML_STD_COMMS OR BUILD_CUML_MPI_COMMS)
GENERATE_FIND_MODULE(
  NAME         NCCL
  HEADER_NAME  nccl.h
  LIBRARY_NAME nccl)
endif(BUILD_CUML_STD_COMMS OR BUILD_CUML_MPI_COMMS)

if(BUILD_CUML_STD_COMMS)
  GENERATE_FIND_MODULE(
    NAME         UCX
    HEADER_NAME  ucp/api/ucp.h
    LIBRARY_NAME ucp)
endif(BUILD_CUML_STD_COMMS)

if(NOT BUILD_STATIC_FAISS)
  GENERATE_FIND_MODULE(
    NAME         FAISS
    HEADER_NAME  faiss/IndexFlat.h
    LIBRARY_NAME faiss)
endif(NOT BUILD_STATIC_FAISS)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR}/cmake)

# append additional directories to CMAKE_MODULE_PATH if we need other modules here

if (NOT DISABLE_OPENMP OR NOT ${DISABLE_OPENMP})
  find_package(OpenMP)
  if(OpenMP_FOUND)
    message(STATUS "OpenMP found in ${OPENMP_INCLUDE_DIRS}")
  endif(OpenMP_FOUND)
endif(NOT DISABLE_OPENMP OR NOT ${DISABLE_OPENMP})

if(NOT DEFINED BLAS_LIBRARIES)
  find_package( BLAS REQUIRED )
else()
  message(STATUS "Manually setting BLAS to ${BLAS_LIBRARIES}")
endif()


##############################################################################
# - External Dependencies-----------------------------------------------------

set(CUML_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE STRING
  "Path to cuml include directories")

##############################################################################
# - dependencies -------------------------------------------------------------

include(cmake/Dependencies.cmake)

##############################################################################
# - include paths ------------------------------------------------------------

set(CUML_INCLUDE_DIRECTORIES
  ${CUML_INCLUDE_DIR}
  ${SPDLOG_DIR}/src/spdlog/include
  ${FAISS_INCLUDE_DIRS}
  ${RAFT_DIR}/cpp/include
  ${RMM_INCLUDE_DIRS})

# These directories will be used by dependent targets
set(CUML_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(CUML_SRC_PRIMS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src_prims)
set(CUML_TEST_PRIMS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test/prims)

if(NOT CUB_IS_PART_OF_CTK)
  list(APPEND CUML_INCLUDE_DIRECTORIES ${CUB_DIR}/src/cub)
endif(NOT CUB_IS_PART_OF_CTK)

# set(CUML_PRIVATE_LINK_LIBRARIES
#   FAISS::FAISS
#   treelite::treelite
#   treelite::treelite_runtime
#   )

##############################################################################
# - build libcuml++ shared library -------------------------------------------

if(BUILD_CUML_CPP_LIBRARY)
  set(CUML_CPP_TARGET "cuml++")

  # single GPU components
  add_library(${CUML_CPP_TARGET} SHARED
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
    target_sources(${CUML_CPP_TARGET}
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

  # Add the alias
  add_library(CUML::${CUML_CPP_TARGET} ALIAS ${CUML_CPP_TARGET})

  # if(OPENMP_FOUND)
  #   set(CUML_PUBLIC_LINK_LIBRARIES ${CUML_PUBLIC_LINK_LIBRARIES} ${OpenMP_CXX_LIB_NAMES} Threads::Threads)
  # endif(OPENMP_FOUND)

  # if(NVTX)
  #   set(CUML_PUBLIC_LINK_LIBRARIES ${CUML_PUBLIC_LINK_LIBRARIES} nvToolsExt)
  #   link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  # endif(NVTX)

  target_compile_definitions(${CUML_CPP_TARGET}
    PUBLIC
      $<$<BOOL:${NVTX}>:NVTX_ENABLED>
      DISABLE_CUSPARSE_DEPRECATED
    PRIVATE
      CUML_CPP_API
  )

  target_compile_options(${CUML_CPP_TARGET}
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

  target_include_directories(${CUML_CPP_TARGET}
    PUBLIC
      $<INSTALL_INTERFACE:include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src_prims>
      $<BUILD_INTERFACE:${RAFT_DIR}/cpp/include>
      $<BUILD_INTERFACE:${CUTLASS_DIR}/src/cutlass>
      $<BUILD_INTERFACE:$<$<BOOL:${ENABLE_CUMLPRIMS_MG}>:${cumlprims_mg_INCLUDE_DIRS}>>
    PRIVATE
      ${CUML_SRC_DIR}
      ${CUML_SRC_PRIMS_DIR}
      ${RMM_INCLUDE_DIRS}
      $<$<NOT:$<BOOL:${CUB_IS_PART_OF_CTK}>>:${CUB_DIR}/src/cub>
      $<$<OR:$<BOOL:BUILD_CUML_STD_COMMS>,$<BOOL:BUILD_CUML_MPI_COMMS>>:${NCCL_INCLUDE_DIRS}>
      $<$<BOOL:BUILD_CUML_MPI_COMMS>:${MPI_CXX_INCLUDE_PATH}>
  )

  target_link_libraries(${CUML_CPP_TARGET}
    PUBLIC
      CUDA::cublas
      CUDA::curand
      CUDA::cusolver
      CUDA::cudart
      CUDA::cusparse
      RMM::RMM
      $<$<BOOL:NVTX>:CUDA::nvToolsExt>
    PRIVATE
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
  target_link_options(${CUML_CPP_TARGET} PRIVATE "-Wl,--exclude-libs,libdmlc.a")
  # same as above, but for protobuf library
  target_link_options(${CUML_CPP_TARGET} PRIVATE "-Wl,--exclude-libs,libprotobuf.a")

endif(BUILD_CUML_CPP_LIBRARY)

##############################################################################
# - build libcuml C shared library -------------------------------------------

if(BUILD_CUML_C_LIBRARY)
  set(CUML_C_TARGET "cuml")

  add_library(${CUML_C_TARGET} SHARED
    src/common/cumlHandle.cpp
    src/common/cuml_api.cpp
    src/dbscan/dbscan_api.cpp
    src/glm/glm_api.cpp
    src/holtwinters/holtwinters_api.cpp
    src/knn/knn_api.cpp
    src/svm/svm_api.cpp
  )
  add_library(CUML::${CUML_C_TARGET} ALIAS ${CUML_C_TARGET})

  # target_compile_definitions(${CUML_C_TARGET}
  #   PRIVATE
  #     CUML_C_API)

  # target_include_directories(${CUML_C_TARGET}
  #   PRIVATE
  #     ${CUML_SRC_DIR}
  # )

  target_link_libraries(${CUML_C_TARGET}
    PUBLIC
      ${CUML_CPP_TARGET}
  )

endif(BUILD_CUML_C_LIBRARY)

##############################################################################
# - build test executables ---------------------------------------------------

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(CTest)
endif()

if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR CUML_BUILD_TESTING) AND BUILD_TESTING)
  add_subdirectory(test ${PROJECT_BINARY_DIR}/test)
endif()

##############################################################################
# - build examples ------------------------------------------------------------------------------

if(BUILD_CUML_EXAMPLES)
  add_subdirectory(examples)
endif(BUILD_CUML_EXAMPLES)

##############################################################################
# - install targets ----------------------------------------------------------

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/cuml)

install(TARGETS
          ${CUML_CPP_TARGET}
          ${CUML_C_TARGET}
        EXPORT
          cuml-targets
        DESTINATION
          ${CMAKE_INSTALL_LIBDIR}
)

# Copy the include and raft directories
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/cuml DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${RAFT_INCLUDE_DIR} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cuml)

include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/cuml-config.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/cuml-config.cmake
                              INSTALL_DESTINATION ${INSTALL_CONFIGDIR})

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/cuml-config-version.cmake
                                 VERSION ${PROJECT_VERSION}
                                 COMPATIBILITY SameMinorVersion)

install(
  EXPORT cuml-targets
  FILE cuml-targets.cmake
  NAMESPACE CUML::
  DESTINATION ${INSTALL_CONFIGDIR})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cuml-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/cuml-config-version.cmake
              ${CMAKE_CURRENT_LIST_DIR}/cmake/FindRMM.cmake
        DESTINATION ${INSTALL_CONFIGDIR})

##############################################################################
# - export targets -----------------------------------------------------------


# Export any Cmake modules
# configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/FindRapidJSON.cmake
#     ${CMAKE_CURRENT_BINARY_DIR}/FindRapidJSON.cmake
#     COPYONLY)

# Export targets
export(EXPORT cuml-targets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/cuml-targets.cmake
    NAMESPACE CUML::)

#Register package in user's package registry
export(PACKAGE CUML)

##############################################################################
# - build benchmark executable -----------------------------------------------

if(BUILD_CUML_BENCH)
  add_subdirectory(bench ${PROJECT_BINARY_DIR}/bench)
endif(BUILD_CUML_BENCH)

##############################################################################
# - doxygen targets ----------------------------------------------------------

include(cmake/doxygen.cmake)
add_doxygen_target(IN_DOXYFILE Doxyfile.in
  OUT_DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  CWD ${CMAKE_CURRENT_BINARY_DIR})
