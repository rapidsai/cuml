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

cmake_minimum_required(VERSION 3.14...3.17 FATAL_ERROR)

project(cuML VERSION 0.18.0 LANGUAGES C CXX CUDA)

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

# todo: change for FindCUDAToolkit when CMake 3.17 is the minimum required
# https://github.com/rapidsai/cuml/issues/2194
find_package(CUDA 10.0 REQUIRED)

# Set the -isystem flag to not have an '=' sign to allow parsing of compile_commands.json
set(CMAKE_INCLUDE_SYSTEM_FLAG_CUDA "-isystem ")

if (NOT DISABLE_OPENMP OR NOT ${DISABLE_OPENMP})
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message(STATUS "OpenMP found in ${OPENMP_INCLUDE_DIRS}")
  endif(OPENMP_FOUND)
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
# - Compiler Options  --------------------------------------------------------

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CUDA_HOST_COMPILER)
  # If CMAKE_CUDA_HOST_COMPILER is set to a nonempty string cmake was called with the environment variable CUDAHOSTCXX set or -DCMAKE_CUDA_HOST_COMPILER
  if(NOT CMAKE_CUDA_HOST_COMPILER STREQUAL CMAKE_CXX_COMPILER)
    message(WARNING "CMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER} and CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} are inconsistent!")
  endif(NOT CMAKE_CUDA_HOST_COMPILER STREQUAL CMAKE_CXX_COMPILER)
else()
  # No attempt to set CMAKE_CUDA_HOST_COMPILER has been made. Make sure CMAKE_CXX_COMPILER is used as CUDA host compiler.
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CMAKE_CXX_COMPILER}")
endif(CMAKE_CUDA_HOST_COMPILER)

if(OPENMP_FOUND)
  message(STATUS "Building with OpenMP support")
  find_package(Threads REQUIRED)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

if(${CMAKE_VERSION} VERSION_LESS "3.17.0")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++14")
endif(${CMAKE_VERSION} VERSION_LESS "3.17.0")

if(LINE_INFO)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif(LINE_INFO)

if(KERNEL_INFO)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-v")
endif(KERNEL_INFO)

if(NVTX)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNVTX_ENABLED")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNVTX_ENABLED")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
endif(NVTX)

if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

if("${GPU_ARCHS}" STREQUAL "")
  include(cmake/EvalGpuArchs.cmake)
  evaluate_gpu_archs(GPU_ARCHS)
endif()


# CUDA 11 onwards cub ships with CTK
if((CUDA_VERSION_MAJOR EQUAL 11) OR (CUDA_VERSION_MAJOR GREATER 11))
  set(CUB_IS_PART_OF_CTK ON)
else()
  set(CUB_IS_PART_OF_CTK OFF)
endif()

if("${GPU_ARCHS}" STREQUAL "ALL")
  set(GPU_ARCHS "60")
  if((CUDA_VERSION_MAJOR EQUAL 9) OR (CUDA_VERSION_MAJOR GREATER 9))
    set(GPU_ARCHS "${GPU_ARCHS};70")
  endif()
  if((CUDA_VERSION_MAJOR EQUAL 10) OR (CUDA_VERSION_MAJOR GREATER 10))
    set(GPU_ARCHS "${GPU_ARCHS};75")
  endif()
  if((CUDA_VERSION_MAJOR EQUAL 11) OR (CUDA_VERSION_MAJOR GREATER 11))
    set(GPU_ARCHS "${GPU_ARCHS};80")
  endif()
endif()

message("-- Building for GPU_ARCHS = ${GPU_ARCHS}")

foreach(arch ${GPU_ARCHS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")
  set(FAISS_GPU_ARCHS "${FAISS_GPU_ARCHS} ${arch}")
endforeach()

list(GET GPU_ARCHS -1 ptx)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${ptx},code=compute_${ptx}")
set(FAISS_GPU_ARCHS "${FAISS_GPU_ARCHS} ${ptx}")

set(CMAKE_CUDA_FLAGS
  "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=unrecognized_gcc_pragma")

# make sure we produce smallest binary size
string(APPEND CMAKE_CUDA_FLAGS " -Xfatbin=-compress-all")

##############################################################################
# - dependencies -------------------------------------------------------------

include(cmake/Dependencies.cmake)

##############################################################################
# - include paths ------------------------------------------------------------

set(CUML_INCLUDE_DIRECTORIES
  ${CUML_INCLUDE_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${CUTLASS_DIR}/src/cutlass
  ${SPDLOG_DIR}/src/spdlog/include
  ${FAISS_INCLUDE_DIRS}
  ${CUDA_CXX_DIR}/include
  ${CUCO_DIR}/include
  ${RAFT_DIR}/cpp/include
  ${RMM_INCLUDE_DIRS})

# These directories will be used by dependent targets
set(CUML_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(CUML_SRC_PRIMS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src_prims)
set(CUML_TEST_PRIMS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test/prims)

if(NOT CUB_IS_PART_OF_CTK)
  list(APPEND CUML_INCLUDE_DIRECTORIES ${CUB_DIR}/src/cub)
endif(NOT CUB_IS_PART_OF_CTK)

set(CUML_PUBLIC_LINK_LIBRARIES
  ${CUDA_cublas_LIBRARY}
  ${CUDA_curand_LIBRARY}
  ${CUDA_cusolver_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_cusparse_LIBRARY})


set(CUML_PRIVATE_LINK_LIBRARIES
  FAISS::FAISS
  treelite::treelite
  treelite::treelite_runtime
  )

if(BUILD_STATIC_FAISS)
  set(CUML_PRIVATE_LINK_LIBRARIES
   ${CUML_PRIVATE_LINK_LIBRARIES}
   ${BLAS_LIBRARIES}
  )
endif(BUILD_STATIC_FAISS)

if(BUILD_CUML_STD_COMMS OR BUILD_CUML_MPI_COMMS)
	list(APPEND CUML_INCLUDE_DIRECTORIES
		${NCCL_INCLUDE_DIRS})

	list(APPEND CUML_PRIVATE_LINK_LIBRARIES
		NCCL::NCCL)
endif(BUILD_CUML_STD_COMMS OR BUILD_CUML_MPI_COMMS)

if(BUILD_CUML_MPI_COMMS)
	list(APPEND CUML_INCLUDE_DIRECTORIES
        ${MPI_CXX_INCLUDE_PATH})

    list(APPEND CUML_PRIVATE_LINK_LIBRARIES
         ${MPI_CXX_LIBRARIES})
endif(BUILD_CUML_MPI_COMMS)

if(ENABLE_CUMLPRIMS_MG)
  list(APPEND CUML_INCLUDE_DIRECTORIES
       ${cumlprims_mg_INCLUDE_DIRS})

  list(APPEND CUML_PRIVATE_LINK_LIBRARIES
       cumlprims_mg::cumlprims_mg)
endif(ENABLE_CUMLPRIMS_MG)

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

  if(OPENMP_FOUND)
    set(CUML_PUBLIC_LINK_LIBRARIES ${CUML_PUBLIC_LINK_LIBRARIES} ${OpenMP_CXX_LIB_NAMES} Threads::Threads)
  endif(OPENMP_FOUND)

  if(NVTX)
    set(CUML_PUBLIC_LINK_LIBRARIES ${CUML_PUBLIC_LINK_LIBRARIES} nvToolsExt)
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  endif(NVTX)

  target_compile_definitions(${CUML_CPP_TARGET}
    PRIVATE
      CUML_CPP_API)

  target_include_directories(${CUML_CPP_TARGET}
    PUBLIC
      ${CUML_INCLUDE_DIRECTORIES}
    PRIVATE
      ${CUML_SRC_DIR}
      ${CUML_SRC_PRIMS_DIR})

  target_link_libraries(${CUML_CPP_TARGET}
    PUBLIC
      ${CUML_PUBLIC_LINK_LIBRARIES}
    PRIVATE
      ${CUML_PRIVATE_LINK_LIBRARIES}
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

  target_compile_definitions(${CUML_C_TARGET}
    PRIVATE
      CUML_C_API)

  target_include_directories(${CUML_C_TARGET}
    PRIVATE 
      ${CUML_SRC_DIR}
  )

  target_link_libraries(${CUML_C_TARGET} PUBLIC ${CUML_CPP_TARGET})
endif(BUILD_CUML_C_LIBRARY)

##############################################################################
# - build test executables ---------------------------------------------------

if(BUILD_CUML_TESTS OR BUILD_CUML_MG_TESTS OR BUILD_PRIMS_TESTS)
  add_subdirectory(test ${PROJECT_BINARY_DIR}/test)
endif(BUILD_CUML_TESTS OR BUILD_CUML_MG_TESTS OR BUILD_PRIMS_TESTS)

##############################################################################
# - build examples ------------------------------------------------------------------------------

if(BUILD_CUML_EXAMPLES)
  add_subdirectory(examples)
endif(BUILD_CUML_EXAMPLES)

##############################################################################
# - install targets ----------------------------------------------------------

install(TARGETS ${CUML_CPP_TARGET}
                ${CUML_C_TARGET}
        DESTINATION lib)

install(DIRECTORY ${CUML_INCLUDE_DIR}/cuml DESTINATION include)
install(DIRECTORY ${RAFT_DIR}/cpp/include/ DESTINATION include/cuml)

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
