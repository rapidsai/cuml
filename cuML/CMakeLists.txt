#
 # Copyright (c) 2018, NVIDIA CORPORATION.
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
 #


cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ml LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA 8.0 REQUIRED)
find_package(OpenMP REQUIRED)

# Submodules
set(GTEST_DIR ${PROJECT_SOURCE_DIR}/external/ml-prims/external/googletest CACHE STRING
  "Path to the googletest repo")
set(GTEST_LIBNAME "gtest_main" CACHE STRING
  "Name of the googletest library")
set(MLPRIMS_DIR ${PROJECT_SOURCE_DIR}/external/ml-prims CACHE STRING
  "Path to the ml-prims repo")
set(CUB_DIR ${PROJECT_SOURCE_DIR}/external/ml-prims/external/cub CACHE STRING
  "Path to cub repo")
set(CUTLASS_DIR ${PROJECT_SOURCE_DIR}/external/ml-prims/external/cutlass CACHE STRING
  "Path to the cutlass repo")
set(CUTLASS_DIR_2 "${PROJECT_SOURCE_DIR}/external/cutlass" CACHE STRING
  "Path to the root of cutlass project directory that Thejaswi updated")


# options exposed to users
set(GPU_ARCHS "" CACHE STRING
  "List of GPU architectures (semicolon-separated) to be compiled for")
option(LINEINFO "Enable lineinfo in nvcc" OFF)
option(KERNELINFO "Enable kernel resource usage info" OFF)
option(DEBUG "Get a debug build" OFF)

## start nvcc options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
if(CMAKE_CXX_STANDARD STREQUAL "11")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++11")
endif()
if(LINEINFO)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()
if(KERNELINFO)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-v")
endif()
if(DEBUG)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()
# Generate optimized binary for every known arch
if(NOT "${GPU_ARCHS}")
  set(GPU_ARCHS "35;50;52;60;61")
  # NOTE: NOTE: Add more 'if's for every new arch release!
  if((CUDA_VERSION_MAJOR EQUAL 9) OR (CUDA_VERSION_MAJOR GREATER 9))
    set(GPU_ARCHS "${GPU_ARCHS};70")
  endif()
endif()
foreach(arch ${GPU_ARCHS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()
# Generate PTX (to be JIT'd at runtime) for the latest architecture
# It is assumed that the last arch in the 'archs' is the latest!
list(GET GPU_ARCHS -1 ptx)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${ptx},code=compute_${ptx}")
## end nvcc options

include_directories(src
  ${GTEST_DIR}/googletest/include
  ${MLPRIMS_DIR}/src
  ${MLPRIMS_DIR}/src/dbscan
  ${MLPRIMS_DIR}/test
  ${CUTLASS_DIR}
  ${CUTLASS_DIR_2}
  ${CUB_DIR})

add_subdirectory(${GTEST_DIR}/googletest ${PROJECT_BINARY_DIR}/googletest)

add_executable(ml_test
  test/pca_test.cu
  test/tsvd_test.cu
  test/dbscan_test.cu
  test/kmeans_test.cu
  )

target_link_libraries(ml_test
  ${GTEST_LIBNAME}
  cublas
  curand
  cusolver
  pthread
  z)
