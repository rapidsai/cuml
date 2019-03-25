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


set (CMAKE_FIND_NO_INSTALL_PREFIX TRUE FORCE)
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(cuML VERSION 0.5.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

###################################################################################################
# - Requirements ----------------------------------------------------------------------------------

if(NOT DEFINED BLAS_LIBRARIES)
    find_package( BLAS REQUIRED )
else()
	message(STATUS "Manually setting BLAS to ${BLAS_LIBRARIES}")
endif()

find_package(CUDA 9.0 REQUIRED)
if (NOT DISABLE_OPENMP OR NOT ${DISABLE_OPENMP})
    find_package(OpenMP)
endif(NOT DISABLE_OPENMP OR NOT ${DISABLE_OPENMP})

find_package(ZLIB REQUIRED)
if(ZLIB_FOUND)
    message(STATUS "ZLib found in ${ZLIB_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "ZLib not found, please check your settings.")
endif(ZLIB_FOUND)


###################################################################################################
# - Submodules ------------------------------------------------------------------------------------
set(GTEST_DIR ${PROJECT_SOURCE_DIR}/external/ml-prims/external/googletest/googletest CACHE STRING
  "Path to the googletest repo")
set(GTEST_LIBNAME "gtest_main" CACHE STRING
  "Name of the googletest library")
set(FAISS_DIR ${PROJECT_SOURCE_DIR}/external/faiss CACHE STRING
  "Path to FAISS source directory")
set(MLPRIMS_DIR ${PROJECT_SOURCE_DIR}/external/ml-prims CACHE STRING
  "Path to the ml-prims repo")
set(CUB_DIR ${PROJECT_SOURCE_DIR}/external/ml-prims/external/cub CACHE STRING
  "Path to cub repo")
set(CUTLASS_DIR ${PROJECT_SOURCE_DIR}/external/ml-prims/external/cutlass CACHE STRING
  "Path to the cutlass repo")
set(CUDA_nvgraph_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvgraph.so CACHE STRING
  "Path to nvGraph lib")

###################################################################################################
# - User Options  ---------------------------------------------------------------------------------

set(CMAKE_IGNORE_PATH "${CMAKE_INSTALL_DIR}/lib" CACHE STRING
  "Ignore any libs added implicitly from the CMAKE_INSTALL_DIR")
set(GPU_ARCHS "" CACHE STRING
  "List of GPU architectures (semicolon-separated) to be compiled for")
option(LINEINFO "Enable lineinfo in nvcc" OFF)
option(KERNELINFO "Enable kernel resource usage info" OFF)
option(DEBUG "Get a debug build" OFF)

###################################################################################################
# - Compiler Options  -----------------------------------------------------------------------------

## nvcc options

if(OPENMP_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)
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
  set(GPU_ARCHS "60;61")
  # NOTE: NOTE: Add more 'if's for every new arch release!
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
# Generate PTX (to be JIT'd at runtime) for the latest architecture
# It is assumed that the last arch in the 'archs' is the latest!
list(GET GPU_ARCHS -1 ptx)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${ptx},code=compute_${ptx}")

# Configuration of faiss for the correct architectures

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

## end nvcc options

## other compiler options

option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" ON)

if(NOT CMAKE_CXX11_ABI)
        message(STATUS "Disabling the GLIBCXX11 ABI")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0")
elseif(CMAKE_CXX11_ABI)
    message(STATUS "Enabling the GLIBCXX11 ABI")
endif(NOT CMAKE_CXX11_ABI)


## end of other compiler options

###################################################################################################
# - include paths ---------------------------------------------------------------------------------


include (ExternalProject)
ExternalProject_Add(faiss
  SOURCE_DIR ${FAISS_DIR}
  CONFIGURE_COMMAND LIBS=-pthread CPPFLAGS=-w LDFLAGS=-L${CMAKE_INSTALL_PREFIX}/lib ${FAISS_DIR}/configure --prefix=${CMAKE_CURRENT_BINARY_DIR}/faiss --with-blas=${CMAKE_INSTALL_PREFIX}/lib/libopenblas.a --with-cuda=${CUDA_TOOLKIT_ROOT_DIR} --quiet
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/faiss/
  BUILD_COMMAND $(MAKE)
  INSTALL_COMMAND $(MAKE) -s install > /dev/null || $(MAKE) && cd gpu && $(MAKE) -s install > /dev/null || $(MAKE)
  BUILD_IN_SOURCE 1
  BUILD_ALWAYS 0
  PREFIX=${CMAKE_CURRENT_BINARY_DIR}/faiss
)

ExternalProject_Get_Property(faiss install_dir)

add_library(faisslib STATIC IMPORTED)
add_library(gpufaisslib STATIC IMPORTED)

add_dependencies(faisslib faiss)
add_dependencies(gpufaisslib faiss)

set_property(TARGET faisslib PROPERTY IMPORTED_LOCATION ${FAISS_DIR}/libfaiss.a)
set_property(TARGET gpufaisslib PROPERTY IMPORTED_LOCATION ${FAISS_DIR}/gpu/libgpufaiss.a)

include_directories(src
	    ${CMAKE_CURRENT_BINARY_DIR}/faiss/include
	    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${GTEST_DIR}/include
        ${MLPRIMS_DIR}/src
        ${MLPRIMS_DIR}/test
        ${CUTLASS_DIR}
        ${CUB_DIR}
        ${ZLIB_INCLUDE_DIRS$})

add_subdirectory(${GTEST_DIR} ${PROJECT_BINARY_DIR}/googletest)

# Append source file in recursive manner, append header files to target for work with them in IDE
file(GLOB_RECURSE ml_prims_header "${MLPRIMS_DIR}/src/*.h" "${MLPRIMS_DIR}/src/*.hpp")
file(GLOB_RECURSE cuml_test_cuda_sources "test/*.cu")
file(GLOB_RECURSE cuml_mg_test_cuda_sources "test_mg/*.cu")

###################################################################################################
# - build libcuml shared library ------------------------------------------------------------------
add_library(cuml SHARED
            src/pca/pca.cu
            src/tsvd/tsvd.cu
            src/dbscan/dbscan.cu
            src/kmeans/kmeans.cu
            src/glm/glm.cu
            src/glm/qn_c.cu
            src/knn/knn.cu
            src/kalman_filter/lkf_py.cu
            src/common/cumlHandle.cpp
            src/common/cuml_api.cpp
            src/umap/umap.cu
	    src/solver/solver.cu
            )

set(CUML_LINK_LIBRARIES
	${CUDA_cublas_LIBRARY}
	${CUDA_curand_LIBRARY}
	${CUDA_cusolver_LIBRARY}
	${CUDA_CUDART_LIBRARY}
	${CUDA_cusparse_LIBRARY}
	${CUDA_nvgraph_LIBRARY}
	${ZLIB_LIBRARIES}
	gpufaisslib
	faisslib
	${BLAS_LIBRARIES})

if(OPENMP_FOUND)
    set(CUML_LINK_LIBRARIES ${CUML_LINK_LIBRARIES} OpenMP::OpenMP_CXX pthread)
endif(OPENMP_FOUND)

target_link_libraries(cuml ${CUML_LINK_LIBRARIES})

###################################################################################################
# - build test executable -------------------------------------------------------------------------

add_executable(ml_test ${cuml_test_cuda_sources} ${ml_prims_header})

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
		                  gpufaisslib
                      faisslib
                      ${BLAS_LIBRARIES}
                      cuml
                      pthread
                      z)


###################################################################################################
# - build test executable -------------------------------------------------------------------------

add_executable(ml_mg_test ${cuml_mg_test_cuda_sources} ${ml_prims_header})

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
                      ${BLAS_LIBRARIES}
                      cuml
                      pthread
                      ${ZLIB_LIBRARIES})

###################################################################################################
# - build examples -------------------------------------------------------------------------

if (NOT DISABLE_EXAMPLES OR NOT ${DISABLE_EXAMPLES})
    add_subdirectory(examples)
endif(NOT DISABLE_EXAMPLES OR NOT ${DISABLE_EXAMPLES})

###################################################################################################
# - install targets -------------------------------------------------------------------------------

install(TARGETS cuml
        DESTINATION lib)





