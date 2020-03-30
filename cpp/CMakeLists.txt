#=============================================================================
# Copyright (c) 2018-2019, NVIDIA CORPORATION.
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
cmake_policy(SET CMP0079 NEW)

project(CUML VERSION 0.12.0 LANGUAGES CXX CUDA)

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

##############################################################################
# - User Options  ------------------------------------------------------------

set(BLAS_LIBRARIES "" CACHE STRING
    "Location of BLAS library for FAISS build.")

option(BUILD_CUML_C_LIBRARY "Build libcuml shared library. Contains the cuML C API" ON)

option(BUILD_CUML_CPP_LIBRARY "Build libcuml++ shared library" ON)

option(BUILD_CUML_TESTS "Build cuML algorithm tests" ON)

option(BUILD_CUML_MG_TESTS "Build cuML multigpu algorithm tests" ON)

option(BUILD_PRIMS_TESTS "Build ml-prim tests" ON)

option(BUILD_CUML_EXAMPLES "Build C++ API usage examples" ON)

option(BUILD_CUML_BENCH "Build cuML C++ benchmark tests" ON)

option(BUILD_CUML_PRIMS_BENCH "Build ml-prims C++ benchmark tests" ON)

option(BUILD_CUML_STD_COMMS "Build the standard NCCL+UCX Communicator" ON)

option(BUILD_CUML_MPI_COMMS "Build the MPI+NCCL Communicator (used for testing)" OFF)

option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" ON)

option(DISABLE_OPENMP "Disable OpenMP" OFF)

option(EMPTY_MARKER_KERNEL "Enable empty marker kernel after nvtxRangePop" ON)

option(KERNEL_INFO "Enable kernel resource usage info" OFF)

option(LINE_INFO "Enable lineinfo in nvcc" OFF)

option(NVTX "Enable nvtx markers" OFF)

set(PARALLEL_LEVEL "" CACHE STRING
    "Sub-projects parallel level for compilation. Currently only affects FAISS" )

set(GPU_ARCHS "" CACHE STRING
    "List of GPU architectures (semicolon-separated) to be compiled for.
    Pass 'ALL' if you want to compile for all supported GPU architectures.
    Empty string means to auto-detect the GPUs on the current system")

set(CMAKE_IGNORE_PATH "${CMAKE_INSTALL_DIR}/lib" CACHE STRING
    "Ignore any libs added implicitly from the CMAKE_INSTALL_DIR")

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

##############################################################################
# - Requirements -------------------------------------------------------------

find_package(CUDA 9.0 REQUIRED)

if (NOT DISABLE_OPENMP OR NOT ${DISABLE_OPENMP})
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message(STATUS "OpenMP found in ${OPENMP_INCLUDE_DIRS}")
  endif(OPENMP_FOUND)
endif(NOT DISABLE_OPENMP OR NOT ${DISABLE_OPENMP})

find_package(ZLIB REQUIRED)
if(ZLIB_FOUND)
  message(STATUS "ZLib found in ${ZLIB_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "ZLib not found, please check your settings.")
endif(ZLIB_FOUND)

if(NOT DEFINED BLAS_LIBRARIES)
  find_package( BLAS REQUIRED )
else()
  message(STATUS "Manually setting BLAS to ${BLAS_LIBRARIES}")
endif()

set(Protobuf_USE_STATIC_LIBS ON)
find_package(Protobuf REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

##############################################################################
# - External Dependencies-----------------------------------------------------

set(CUB_DIR ${CMAKE_CURRENT_BINARY_DIR}/cub CACHE STRING
  "Path to cub repo")

set(CUTLASS_DIR ${CMAKE_CURRENT_BINARY_DIR}/cutlass CACHE STRING
  "Path to the cutlass repo")

set(CUDA_nvgraph_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvgraph.so CACHE STRING
  "Path to nvGraph lib")

set(FAISS_DIR ${CMAKE_CURRENT_BINARY_DIR}/faiss CACHE STRING
  "Path to FAISS source directory")

set(TREELITE_DIR ${CMAKE_CURRENT_BINARY_DIR}/treelite CACHE STRING
  "Path to treelite install directory")

set(CUML_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include CACHE STRING
  "Path to cuml include directories")

##############################################################################
# - Compiler Options  --------------------------------------------------------

# TODO: Update to c++14, github issue
set(CMAKE_CXX_STANDARD 11)
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
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++11")
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
# - Cloning header only libraries---------------------------------------------

include(ExternalProject)

ExternalProject_Add(cub
  GIT_REPOSITORY    https://github.com/NVlabs/cub.git
  GIT_TAG           v1.8.0
  PREFIX            ${CUB_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
)

ExternalProject_Add(cutlass
  GIT_REPOSITORY    https://github.com/NVIDIA/cutlass.git
  GIT_TAG           v1.0.1
  PREFIX            ${CUTLASS_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
)

# dependencies will be added in sequence, so if a new project `project_b` is added
# after `project_a`, please add the dependency add_dependencies(project_b project_a)
# This allows the cloning to happen sequentially, enhancing the printing at
# compile time, helping significantly to troubleshoot build issues.
add_dependencies(cutlass cub)

##############################################################################
# - FAISS Build  -------------------------------------------------------------

ExternalProject_Add(faiss
  GIT_REPOSITORY    https://github.com/facebookresearch/faiss.git
  GIT_TAG           v1.6.1
  CONFIGURE_COMMAND LIBS=-pthread
                    CPPFLAGS=-w
                    LDFLAGS=-L${CMAKE_INSTALL_PREFIX}/lib
                            ${CMAKE_CURRENT_BINARY_DIR}/faiss/src/faiss/configure
                            --prefix=${CMAKE_CURRENT_BINARY_DIR}/faiss
                            --with-blas=${BLAS_LIBRARIES}
                            --with-cuda=${CUDA_TOOLKIT_ROOT_DIR}
                            --with-cuda-arch=${FAISS_GPU_ARCHS}
                            -v
  PREFIX            ${FAISS_DIR}
  BUILD_COMMAND     ${CMAKE_MAKE_PROGRAM} -j${PARALLEL_LEVEL} VERBOSE=1
  INSTALL_COMMAND   ${CMAKE_MAKE_PROGRAM} -s install > /dev/null
  UPDATE_COMMAND    ""
  BUILD_IN_SOURCE   1
)

add_dependencies(faiss cutlass)

ExternalProject_Get_Property(faiss install_dir)

add_library(faisslib STATIC IMPORTED)

add_dependencies(faisslib faiss)

set_property(TARGET faisslib PROPERTY IMPORTED_LOCATION ${FAISS_DIR}/lib/libfaiss.a)

##############################################################################
# - treelite build -----------------------------------------------------------

ExternalProject_Add(treelite
    GIT_REPOSITORY    https://github.com/dmlc/treelite.git
    GIT_TAG           6fd01e4f1890950bbcf9b124da24e886751bffe6
    PREFIX            ${TREELITE_DIR}
    CMAKE_ARGS        -DBUILD_SHARED_LIBS=OFF
                      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                      -DENABLE_PROTOBUF=ON
                      -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
    UPDATE_COMMAND    ""
    PATCH_COMMAND     patch -p1 -N < ${CMAKE_CURRENT_SOURCE_DIR}/cmake/treelite_protobuf.patch || true
)

add_dependencies(treelite faiss)

add_library(dmlclib STATIC IMPORTED)
add_library(treelitelib STATIC IMPORTED)
add_library(treelite_runtimelib SHARED IMPORTED)

add_dependencies(dmlclib treelite)
add_dependencies(treelitelib treelite)
add_dependencies(treelite_runtimelib treelite)

set_property(TARGET dmlclib PROPERTY IMPORTED_LOCATION ${TREELITE_DIR}/lib/libdmlc.a)
set_property(TARGET treelitelib PROPERTY IMPORTED_LOCATION ${TREELITE_DIR}/lib/libtreelite.a)
set_property(TARGET treelite_runtimelib PROPERTY IMPORTED_LOCATION ${TREELITE_DIR}/lib/libtreelite_runtime.so)

##############################################################################
# - include paths ------------------------------------------------------------

include_directories(
  ${CUML_INCLUDE_DIR}
  src
  src_prims
  test/prims
  ${FAISS_DIR}/src/
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${CUTLASS_DIR}/src/cutlass
  ${CUB_DIR}/src/cub
  ${TREELITE_DIR}/include
  ${TREELITE_DIR}/include/fmt
  ${ZLIB_INCLUDE_DIRS$})

set(PRIMS_TEST_UTILS
    src_prims/cuda_utils.h
    src_prims/utils.h)

##############################################################################
# - build libcuml++ shared library -------------------------------------------

if(BUILD_CUML_CPP_LIBRARY)

  set(CUML_CPP_TARGET "cuml++")

  add_library(${CUML_CPP_TARGET} SHARED
    src/common/cumlHandle.cpp
    src/common/cuml_api.cpp
    src/common/cuML_comms_impl.cpp
    src/comms/cuML_comms_test.cpp
    src/common/nvtx.cu
    src/datasets/make_blobs.cu
    src/datasets/make_regression.cu
    src/dbscan/dbscan.cu
    src/decisiontree/decisiontree.cu
    src/fil/fil.cu
    src/fil/infer.cu
    src/glm/glm.cu
    src/holtwinters/holtwinters.cu
    src/kalman_filter/lkf_py.cu
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
    src/tsa/stationarity.cu
    src/tsne/tsne.cu
    src/tsvd/tsvd.cu
    src/umap/umap.cu
    src/arima/batched_kalman.cu
    src/arima/batched_arima.cu
    )

  set(CUML_LINK_LIBRARIES
    ${CUDA_cublas_LIBRARY}
    ${CUDA_curand_LIBRARY}
    ${CUDA_cusolver_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_nvgraph_LIBRARY}
    ${ZLIB_LIBRARIES}
    ${Protobuf_LIBRARIES}
    treelitelib
    dmlclib
    faisslib
    )

  if(OPENMP_FOUND)
    set(CUML_LINK_LIBRARIES ${CUML_LINK_LIBRARIES} OpenMP::OpenMP_CXX Threads::Threads)
  endif(OPENMP_FOUND)

  if(NVTX)
    set(CUML_LINK_LIBRARIES ${CUML_LINK_LIBRARIES} nvToolsExt)
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  endif(NVTX)

  target_link_libraries(${CUML_CPP_TARGET} ${CUML_LINK_LIBRARIES})
  # If we export the libdmlc symbols, they can lead to weird crashes with other
  # libraries that use libdmlc. This just hides the symbols internally.
  target_link_options(${CUML_CPP_TARGET} PRIVATE "-Wl,--exclude-libs,libdmlc.a")
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

  target_link_libraries(${CUML_C_TARGET} ${CUML_CPP_TARGET})
endif(BUILD_CUML_C_LIBRARY)

##############################################################################
# - build test executables ---------------------------------------------------

if(BUILD_CUML_TESTS OR BUILD_CUML_MG_TESTS OR BUILD_PRIMS_TESTS)
  add_subdirectory(test ${PROJECT_BINARY_DIR}/test)
endif(BUILD_CUML_TESTS OR BUILD_CUML_MG_TESTS OR BUILD_PRIMS_TESTS)

###################################################################################################
# - build comms ------------------------------------------------------------------------

if(BUILD_CUML_STD_COMMS)
  add_subdirectory(comms/std)
endif(BUILD_CUML_STD_COMMS)

if(BUILD_CUML_MPI_COMMS)
	add_subdirectory(comms/mpi)
endif(BUILD_CUML_MPI_COMMS)

###################################################################################################
# - build examples -------------------------------------------------------------------------

if (BUILD_CUML_EXAMPLES)
  add_subdirectory(examples)
endif(BUILD_CUML_EXAMPLES)

##############################################################################
# - install targets ----------------------------------------------------------

install(TARGETS ${CUML_CPP_TARGET}
                ${CUML_C_TARGET}
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/cuml
        DESTINATION include)

##############################################################################
# - build benchmark executable -----------------------------------------------

add_subdirectory(bench ${PROJECT_BINARY_DIR}/bench)

##############################################################################
# - doxygen targets ----------------------------------------------------------

# include(cmake/doxygen.cmake)
# add_doxygen_target(IN_DOXYFILE src_prims/Doxyfile.in
#   OUT_DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
#   CWD ${CMAKE_CURRENT_BINARY_DIR})
