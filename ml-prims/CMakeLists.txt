cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(mlcommon LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA 8.0 REQUIRED)
find_package(OpenMP REQUIRED)

# Submodules
set(GTEST_DIR ${PROJECT_SOURCE_DIR}/external/googletest CACHE STRING
  "Path to the googletest repo")
set(GTEST_LIBNAME "gtest_main" CACHE STRING
  "Name of the googletest library")
set(CUTLASS_DIR ${PROJECT_SOURCE_DIR}/external/cutlass CACHE STRING
  "Path to the cutlass repo")
set(CUB_DIR ${PROJECT_SOURCE_DIR}/external/cub CACHE STRING
  "Path to cub repo")

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
  ${CUTLASS_DIR}
  ${CUB_DIR})

add_subdirectory(${GTEST_DIR}/googletest ${PROJECT_BINARY_DIR}/googletest)

add_executable(mlcommon_test
  test/add.cu
  test/binary_op.cu
  test/cuda_utils.cu
  test/cov.cu
  test/distance.cu
  test/eig.cu
  test/eltwise.cu
  test/eltwise2d.cu
  test/gemm.cu
  test/kselection.cu
  test/math.cu
  test/matrix.cu
  test/matrix_vector_op.cu
  test/mean.cu
  test/mean_center.cu
  test/mnist.cu
  test/mvg.cu
  test/norm.cu
  test/rng.cu
  test/rsvd.cu
  test/stddev.cu
  test/subtract.cu
  test/svd.cu
  test/transpose.cu
  test/unary_op.cu
  test/vector_broadcast.cu)

target_link_libraries(mlcommon_test
  ${GTEST_LIBNAME}
  cublas
  curand
  cusolver
  pthread
  z)
