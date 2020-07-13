#=============================================================================
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(CUML VERSION 0.15.0 LANGUAGES C CXX CUDA)

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

option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" ON)

option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)

option(DISABLE_OPENMP "Disable OpenMP" OFF)

option(ENABLE_CUMLPRIMS_MG "Enable algorithms that use libcumlprims_mg" ON)

option(EMPTY_MARKER_KERNEL "Enable empty marker kernel after nvtxRangePop" OFF)

option(KERNEL_INFO "Enable kernel resource usage info" OFF)

option(LINE_INFO "Enable lineinfo in nvcc" OFF)

option(NVTX "Enable nvtx markers" OFF)

option(SINGLEGPU "Disable all mnmg components and comms libraries" OFF)

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

##############################################################################
# - Requirements -------------------------------------------------------------

# todo: change for FindCUDAToolkit when CMake 3.17 is the minimum required
# https://github.com/rapidsai/cuml/issues/2194
find_package(CUDA 10.0 REQUIRED)

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

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

##############################################################################
# - External Dependencies-----------------------------------------------------

set(CUDA_nvgraph_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvgraph.so CACHE STRING
  "Path to nvGraph lib")

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

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

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
  if(EMPTY_MARKER_KERNEL)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DENABLE_EMPTY_MARKER_KERNEL")
  endif(EMPTY_MARKER_KERNEL)
endif(NVTX)

if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

if("${GPU_ARCHS}" STREQUAL "")
  include(cmake/EvalGpuArchs.cmake)
  evaluate_gpu_archs(GPU_ARCHS)
endif()

if("${GPU_ARCHS}" STREQUAL "ALL")
  set(GPU_ARCHS "60")
  if((CUDA_VERSION_MAJOR EQUAL 9) OR (CUDA_VERSION_MAJOR GREATER 9))
    set(GPU_ARCHS "${GPU_ARCHS};70")
  endif()
  if((CUDA_VERSION_MAJOR EQUAL 10) OR (CUDA_VERSION_MAJOR GREATER 10))
    set(GPU_ARCHS "${GPU_ARCHS};75")
  endif()
endif()

message("-- Building for GPU_ARCHS = ${GPU_ARCHS}")

foreach(arch ${GPU_ARCHS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")
  set(FAISS_GPU_ARCHS "${FAISS_GPU_ARCHS} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

list(GET GPU_ARCHS -1 ptx)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${ptx},code=compute_${ptx}")
set(FAISS_GPU_ARCHS "${FAISS_GPU_ARCHS} -gencode arch=compute_${ptx},code=compute_${ptx}")

if(CMAKE_COMPILER_IS_GNUCXX)
  if(NOT CMAKE_CXX11_ABI)
    message(STATUS "Disabling the GLIBCXX11 ABI")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0")
  elseif(CMAKE_CXX11_ABI)
    message(STATUS "Enabling the GLIBCXX11 ABI")
  endif(NOT CMAKE_CXX11_ABI)
endif(CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_CUDA_FLAGS
  "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=unrecognized_gcc_pragma")

##############################################################################
# - dependencies -------------------------------------------------------------

include(cmake/Dependencies.cmake)

##############################################################################
# - include paths ------------------------------------------------------------

set(CUML_INCLUDE_DIRECTORIES
  ${CUML_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src_prims
  ${CMAKE_CURRENT_SOURCE_DIR}/test/prims
  ${FAISS_DIR}/src/
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${CUTLASS_DIR}/src/cutlass
  ${CUB_DIR}/src/cub
  ${SPDLOG_DIR}/src/spdlog/include
  ${RAFT_DIR}/cpp/include
  )

set(CUML_PUBLIC_LINK_LIBRARIES
  ${CUDA_cublas_LIBRARY}
  ${CUDA_curand_LIBRARY}
  ${CUDA_cusolver_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_cusparse_LIBRARY}
  ${CUDA_nvgraph_LIBRARY}
  )

set(CUML_PRIVATE_LINK_LIBRARIES
  faisslib
  treelite::treelite
  treelite::treelite_runtime
  )

if(ENABLE_CUMLPRIMS_MG)
  list(APPEND CUML_INCLUDE_DIRECTORIES
       ${CUMLPRIMS_MG_INCLUDE_DIRS})

  list(APPEND CUML_PRIVATE_LINK_LIBRARIES
       CUMLPRIMS_MG::CUMLPRIMS_MG)

endif(ENABLE_CUMLPRIMS_MG)

##############################################################################
# - build libcuml++ shared library -------------------------------------------

if(BUILD_CUML_CPP_LIBRARY)
  set(CUML_CPP_TARGET "cuml++")

  # single GPU components
  add_library(${CUML_CPP_TARGET} SHARED
    src/arima/batched_kalman.cu
    src/arima/batched_arima.cu
    src/common/cumlHandle.cpp
    src/common/cuml_api.cpp
    src/common/cuML_comms_impl.cpp
    src/common/logger.cpp
    src/common/nvtx.cu
    src/comms/cuML_comms_test.cpp
    src/datasets/make_arima.cu
    src/datasets/make_blobs.cu
    src/datasets/make_regression.cu
    src/dbscan/dbscan.cu
    src/decisiontree/decisiontree.cu
    src/fil/fil.cu
    src/fil/infer.cu
    src/glm/glm.cu
    src/holtwinters/holtwinters.cu
    src/kmeans/kmeans.cu
    src/knn/knn.cu
    src/metrics/metrics.cu
    src/metrics/trustworthiness.cu
    src/pca/pca.cu
    src/randomforest/randomforest.cu
    src/random_projection/rproj.cu
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
        src/kmeans/kmeans_mg.cu
        src/knn/knn_mg.cu
        src/pca/pca_mg.cu
        src/pca/sign_flip_mg.cu
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

  target_include_directories(${CUML_CPP_TARGET}
    PRIVATE ${CUML_INCLUDE_DIRECTORIES})

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
    src/common/cuml_api.cpp
    src/dbscan/dbscan_api.cpp
    src/glm/glm_api.cpp
    src/holtwinters/holtwinters_api.cpp
    src/svm/svm_api.cpp)

  target_include_directories(${CUML_C_TARGET}
    PRIVATE ${CUML_INCLUDE_DIRECTORIES})

  target_link_libraries(${CUML_C_TARGET} PUBLIC ${CUML_CPP_TARGET})
endif(BUILD_CUML_C_LIBRARY)

##############################################################################
# - build test executables ---------------------------------------------------

if(BUILD_CUML_TESTS OR BUILD_CUML_MG_TESTS OR BUILD_PRIMS_TESTS)
  add_subdirectory(test ${PROJECT_BINARY_DIR}/test)
endif(BUILD_CUML_TESTS OR BUILD_CUML_MG_TESTS OR BUILD_PRIMS_TESTS)

##############################################################################
# - build comms ------------------------------------------------------------------------------

if(BUILD_CUML_STD_COMMS)
  add_subdirectory(comms/std)
endif(BUILD_CUML_STD_COMMS)

if(BUILD_CUML_MPI_COMMS)
  add_subdirectory(comms/mpi)
endif(BUILD_CUML_MPI_COMMS)

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
