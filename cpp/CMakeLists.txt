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

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUML VERSION 0.8.0 LANGUAGES CXX CUDA)

###################################################################################################
# - build type ------------------------------------------------------------------------------------

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

###################################################################################################
# - User Options  ---------------------------------------------------------------------------------

option(BUILD_CUML_CPP_LIBRARY "Build libcuml++ shared library" ON)

option(BUILD_CUML_TESTS "Build cuML algorithm tests" ON)

option(BUILD_CUML_MG_TESTS "Build cuML multigpu algorithm tests" ON)

option(BUILD_PRIM_TESTS "Build ml-prim tests" ON)

option(BUILD_CUML_EXAMPLES "Build C++ API usage examples" ON)

option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" ON)

option(DISABLE_OPENMP "Disable OpenMP" OFF)

option(KERNEL_INFO "Enable kernel resource usage info" OFF)

option(LINE_INFO "Enable lineinfo in nvcc" OFF)

set(BLAS_LIBRARIES "" CACHE STRING
    "Location of BLAS library")

set(GPU_ARCHS "" CACHE STRING
  "List of GPU architectures (semicolon-separated) to be compiled for")

set(CMAKE_IGNORE_PATH "${CMAKE_INSTALL_DIR}/lib" CACHE STRING
  "Ignore any libs added implicitly from the CMAKE_INSTALL_DIR")

# Bulding cuml_test or cuml_mg_test executables forces building libcuml++
if(BUILD_CUML_TESTS OR BUILD_CUML_MG_TESTS)
  set(BUILD_CUML_CPP_LIBRARY ON)
endif(BUILD_CUML_TESTS OR BUILD_CUML_MG_TESTS)


###################################################################################################
# - Requirements ----------------------------------------------------------------------------------

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

# set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
# find_package (Threads REQUIRED)
# if(NOT CMAKE_USE_PTHREADS_INIT)
#   message(FATAL_ERROR "pthreads not found, please check your settings")

if(NOT DEFINED BLAS_LIBRARIES)
  find_package( BLAS REQUIRED )
else()
  message(STATUS "Manually setting BLAS to ${BLAS_LIBRARIES}")
endif()

# if(NOT DEFINED LAPACK_LIBRARIES)
#   find_package( LAPACK REQUIRED )
# else()
#   message(STATUS "Manually setting LAPACK to ${LAPACK_LIBRARIES}")
# endif()

###################################################################################################
# - External Dependencies--------------------------------------------------------------------------

set(GTEST_DIR ${PROJECT_SOURCE_DIR}/external/ml-prims/external/googletest/googletest CACHE STRING
  "Path to the googletest repo")

set(GTEST_LIBNAME "gtest_main" CACHE STRING
  "Name of the googletest library")

set(FAISS_DIR ${PROJECT_SOURCE_DIR}/external/faiss CACHE STRING
  "Path to FAISS source directory")

set(MLPRIMS_DIR ${PROJECT_SOURCE_DIR}/src_prims/ CACHE STRING
  "Path to the ml-prims repo")

set(CUB_DIR ${PROJECT_SOURCE_DIR}/external/ml-prims/external/cub CACHE STRING
  "Path to cub repo")

set(CUTLASS_DIR ${PROJECT_SOURCE_DIR}/external/ml-prims/external/cutlass CACHE STRING
  "Path to the cutlass repo")

set(CUDA_nvgraph_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvgraph.so CACHE STRING
  "Path to nvGraph lib")


###################################################################################################
# - Compiler Options  -----------------------------------------------------------------------------

# TODO: Update to c++14, github issue
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(OPENMP_FOUND)
  message(STATUS "Building with OpenMP support")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++11")

if(LINE_INFO)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif(LINE_INFO)

if(KERNEL_INFO)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-v")
endif(KERNEL_INFO)

if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

if(NOT "${GPU_ARCHS}")
  set(GPU_ARCHS "60")
  if((CUDA_VERSION_MAJOR EQUAL 9) OR (CUDA_VERSION_MAJOR GREATER 9))
    set(GPU_ARCHS "${GPU_ARCHS};70")
  endif()
  if((CUDA_VERSION_MAJOR EQUAL 10) OR (CUDA_VERSION_MAJOR GREATER 10))
    set(GPU_ARCHS "${GPU_ARCHS};75")
  endif()
endif()

foreach(arch ${GPU_ARCHS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

list(GET GPU_ARCHS -1 ptx)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${ptx},code=compute_${ptx}")

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

###################################################################################################
# - FAISS Build  ----------------------------------------------------------------------------------

if(BUILD_CUML_CPP_LIBRARY)

  # Configuration of faiss for the correct architectures

  # TODO: Update faiss submodule and use new flags
  file(READ ${FAISS_DIR}/makefile.inc.in CONFIG_FILE)
      string(REPLACE "-Xcudafe --diag_suppress=unrecognized_attribute"
             "--disable-warnings"
             CONFIG_FILE ${CONFIG_FILE})
      string(REPLACE "compute_35,code=\"compute_35\"" "compute_60,code=\"sm_60\""
             CONFIG_FILE ${CONFIG_FILE})
      string(REPLACE "compute_52,code=\"compute_52\"" "compute_61,code=\"sm_61\""
             CONFIG_FILE ${CONFIG_FILE})
      string(REPLACE "compute_60,code=\"compute_60\"" "compute_70,code=\"sm_70\""
             CONFIG_FILE ${CONFIG_FILE})

  if((CUDA_VERSION_MAJOR EQUAL 9))
      # Do not generate Turing on 9.2
      string(REPLACE "-gencode arch=compute_61,code=\"compute_61\" " ""
             CONFIG_FILE ${CONFIG_FILE})

  endif()
  if((CUDA_VERSION_MAJOR EQUAL 10) OR (CUDA_VERSION_MAJOR GREATER 10))
      string(REPLACE "compute_61,code=\"compute_61\"" "compute_70,code=\"sm_75\""
             CONFIG_FILE ${CONFIG_FILE})
      file(WRITE ${FAISS_DIR}/makefile.inc.in "${CONFIG_FILE}")
  endif()

  file(WRITE ${FAISS_DIR}/makefile.inc.in "${CONFIG_FILE}")


  include (ExternalProject)
  ExternalProject_Add(faiss
    SOURCE_DIR ${FAISS_DIR}
    CONFIGURE_COMMAND LIBS=-pthread CPPFLAGS=-w LDFLAGS=-L${CMAKE_INSTALL_PREFIX}/lib ${FAISS_DIR}/configure --prefix=${CMAKE_CURRENT_BINARY_DIR}/faiss --with-blas=${BLAS_LIBRARIES} --with-cuda=${CUDA_TOOLKIT_ROOT_DIR} --quiet
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/faiss/
    BUILD_COMMAND $(MAKE)
    INSTALL_COMMAND $(MAKE) -s install > /dev/null || $(MAKE) && cd gpu && $(MAKE) -s install > /dev/null || $(MAKE)
    BUILD_IN_SOURCE 1
    PREFIX=${CMAKE_CURRENT_BINARY_DIR}/faiss
  )

  ExternalProject_Get_Property(faiss install_dir)

  add_library(faisslib STATIC IMPORTED)
  add_library(gpufaisslib STATIC IMPORTED)

  add_dependencies(faisslib faiss)
  add_dependencies(gpufaisslib faiss)

  set_property(TARGET faisslib PROPERTY IMPORTED_LOCATION ${FAISS_DIR}/libfaiss.a)
  set_property(TARGET gpufaisslib PROPERTY IMPORTED_LOCATION ${FAISS_DIR}/gpu/libgpufaiss.a)

endif(BUILD_CUML_CPP_LIBRARY)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories(
  src
  src_prims
  test/prims
  ${CMAKE_CURRENT_BINARY_DIR}/faiss/include
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${GTEST_DIR}/include
  ${CUTLASS_DIR}
  ${CUB_DIR}
  ${ZLIB_INCLUDE_DIRS$})

add_subdirectory(${GTEST_DIR} ${PROJECT_BINARY_DIR}/googletest)

file(GLOB_RECURSE ml_prims_header "src_prims/*.h" "src_prims/*.hpp")

###################################################################################################
# - build libcuml++ shared library ------------------------------------------------------------------

if(BUILD_CUML_CPP_LIBRARY)

  set(CUML_CPP_TARGET "cuml++")
  add_library(${CUML_CPP_TARGET} SHARED
    src/common/cumlHandle.cpp
    src/common/cuml_api.cpp
    src/dbscan/dbscan.cu
    src/decisiontree/decisiontree.cu
    src/glm/glm.cu
    src/kalman_filter/lkf_py.cu
    src/kmeans/kmeans.cu
    src/knn/knn.cu
    src/metrics/metrics.cu
    src/pca/pca.cu
    src/randomforest/randomforest.cu
    src/solver/solver.cu
    src/tsvd/tsvd.cu
    src/umap/umap.cu)

  set(CUML_LINK_LIBRARIES
    ${CUDA_cublas_LIBRARY}
    ${CUDA_curand_LIBRARY}
    ${CUDA_cusolver_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_nvgraph_LIBRARY}
    ${ZLIB_LIBRARIES}
    gpufaisslib
    faisslib)

  if(OPENMP_FOUND)
    set(CUML_LINK_LIBRARIES ${CUML_LINK_LIBRARIES} OpenMP::OpenMP_CXX pthread)
  endif(OPENMP_FOUND)

  target_link_libraries(${CUML_CPP_TARGET} ${CUML_LINK_LIBRARIES})

endif(BUILD_CUML_CPP_LIBRARY)

###################################################################################################
# - build ml_test executable ----------------------------------------------------------------------

if(BUILD_CUML_TESTS)

  file(GLOB_RECURSE cuml_test_cuda_sources "test/sg/*.cu")

  add_executable(ml_test
    ${cuml_test_cuda_sources}
    ${ml_prims_header})

  target_link_libraries(ml_test
    ${GTEST_LIBNAME}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_curand_LIBRARY}
    ${CUDA_cusolver_LIBRARY}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    gpufaisslib
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_nvgraph_LIBRARY}
    faisslib
    ${CUML_CPP_TARGET}
    pthread
    ${ZLIB_LIBRARIES})

endif(BUILD_CUML_TESTS)

###################################################################################################
# - build ml_mg_test executable -------------------------------------------------------------------

if(BUILD_CUML_MG_TESTS)

file(GLOB_RECURSE cuml_mg_test_cuda_sources "test/mg/*.cu")

add_executable(ml_mg_test
    ${cuml_mg_test_cuda_sources}
    ${ml_prims_header})

target_link_libraries(ml_mg_test
  ${GTEST_LIBNAME}
  ${CUDA_cublas_LIBRARY}
  ${CUDA_curand_LIBRARY}
  ${CUDA_cusolver_LIBRARY}
  ${CUDA_cusparse_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_cusparse_LIBRARY}
  ${CUDA_nvgraph_LIBRARY}
  gpufaisslib
  faisslib
  ${CUML_CPP_TARGET}
  pthread
  ${ZLIB_LIBRARIES})

endif(BUILD_CUML_MG_TESTS)

###################################################################################################
# - build prims_test executable ----------------------------------------------------------------

file(GLOB_RECURSE mlprims_test_cuda_sources "test/prims/*.cu")

set(MLPRIMS_LINK_LIBRARIES
  ${CUDA_cublas_LIBRARY}
  ${CUDA_curand_LIBRARY}
  ${CUDA_cusolver_LIBRARY}
  ${CUDA_cusparse_LIBRARY}
  pthread
  ${ZLIB_LIBRARIES})

add_executable(prims_test
  ${mlprims_test_cuda_sources}
  ${ml_prims_header})

target_link_libraries(prims_test
  ${GTEST_LIBNAME}
  ${MLPRIMS_LINK_LIBRARIES})

###################################################################################################
# - build examples -------------------------------------------------------------------------

if (DISABLE_EXAMPLES OR ${BUILD_CUML_EXAMPLES})
  add_subdirectory(examples)
endif(DISABLE_EXAMPLES OR ${BUILD_CUML_EXAMPLES})

###################################################################################################
# - install targets -------------------------------------------------------------------------------

install(TARGETS ${CUML_CPP_TARGET} DESTINATION lib)

###################################################################################################
# - doxygen targets -------------------------------------------------------------------------------

# include(cmake/doxygen.cmake)
# add_doxygen_target(IN_DOXYFILE ${MLPRIMS_DIR}/Doxyfile.in
#   OUT_DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
#   CWD ${CMAKE_CURRENT_BINARY_DIR})

