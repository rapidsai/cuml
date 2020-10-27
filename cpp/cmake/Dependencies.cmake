#=============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
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

include(ExternalProject)

##############################################################################
# - raft - (header only) -----------------------------------------------------

# Only cloned if RAFT_PATH env variable is not defined

if(DEFINED ENV{RAFT_PATH})
  message(STATUS "RAFT_PATH environment variable detected.")
  message(STATUS "RAFT_DIR set to $ENV{RAFT_PATH}")
  set(RAFT_DIR "$ENV{RAFT_PATH}")

  ExternalProject_Add(raft
    DOWNLOAD_COMMAND  ""
    SOURCE_DIR        ${RAFT_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   "")

else(DEFINED ENV{RAFT_PATH})
  message(STATUS "RAFT_PATH environment variable NOT detected, cloning RAFT")
  set(RAFT_DIR ${CMAKE_CURRENT_BINARY_DIR}/raft CACHE STRING "Path to RAFT repo")

  ExternalProject_Add(raft
    GIT_REPOSITORY    https://github.com/divyegala/raft.git
    GIT_TAG           bug-017-missing_includes
    PREFIX            ${RAFT_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   "")

  # Redefining RAFT_DIR so it coincides with the one inferred by env variable.
  set(RAFT_DIR ${RAFT_DIR}/src/raft/)
endif(DEFINED ENV{RAFT_PATH})


##############################################################################
# - cumlprims (binary dependency) --------------------------------------------

if(ENABLE_CUMLPRIMS_MG)

    if(DEFINED ENV{CUMLPRIMS_MG_PATH})
      set(CUMLPRIMS_MG_PATH ENV{CUMLPRIMS_MG_PATH}})
    endif(DEFINED ENV{CUMLPRIMS_MG_PATH})

    if(NOT CUMLPRIMS_MG_PATH)
      find_package(cumlprims_mg
                   REQUIRED)

    else()
      message("-- Manually setting CUMLPRIMS_MG_PATH to ${CUMLPRIMS_MG_PATH}")
      if(EXISTS "${CUMLPRIMS_MG_PATH}/lib/libcumlprims.so")
        set(CUMLPRIMS_MG_FOUND TRUE)
        set(CUMLPRIMS_MG_INCLUDE_DIRS ${CUMLPRIMS_MG_PATH}/include)
        set(CUMLPRIMS_MG_LIBRARIES ${CUMLPRIMS_MG_PATH}/lib/libcumlprims.so)
      else()
        message(FATAL_ERROR "libcumlprims library NOT found in ${CUMLPRIMS_MG_PATH}")
      endif(EXISTS "${CUMLPRIMS_MG_PATH}/lib/libcumlprims.so")
    endif(NOT CUMLPRIMS_MG_PATH)

endif(ENABLE_CUMLPRIMS_MG)


##############################################################################
# - RMM ----------------------------------------------------------------------

find_path(RMM_INCLUDE_DIRS "rmm"
    HINTS
    "$ENV{RMM_ROOT}/include"
    "$ENV{CONDA_PREFIX}/include/rmm"
    "$ENV{CONDA_PREFIX}/include")

message(STATUS "RMM: RMM_INCLUDE_DIRS set to ${RMM_INCLUDE_DIRS}")

##############################################################################
# - NCCL ---------------------------------------------------------------------

if(BUILD_CUML_MPI_COMMS OR BUILD_CUML_STD_COMMS)
  find_package(NCCL REQUIRED)
endif(BUILD_CUML_MPI_COMMS OR BUILD_CUML_STD_COMMS)

##############################################################################
# - MPI ---------------------------------------------------------------------

if(BUILD_CUML_MPI_COMMS)
  find_package(MPI REQUIRED)
endif(BUILD_CUML_MPI_COMMS)

##############################################################################
# - cub - (header only) ------------------------------------------------------

if(NOT CUB_IS_PART_OF_CTK)
  set(CUB_DIR ${CMAKE_CURRENT_BINARY_DIR}/cub CACHE STRING "Path to cub repo")
  ExternalProject_Add(cub
    GIT_REPOSITORY    https://github.com/thrust/cub.git
    GIT_TAG           1.8.0
    PREFIX            ${CUB_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   "")
endif(NOT CUB_IS_PART_OF_CTK)

##############################################################################
# - cutlass - (header only) --------------------------------------------------

set(CUTLASS_DIR ${CMAKE_CURRENT_BINARY_DIR}/cutlass CACHE STRING
  "Path to the cutlass repo")
ExternalProject_Add(cutlass
  GIT_REPOSITORY    https://github.com/NVIDIA/cutlass.git
  GIT_TAG           v1.0.1
  PREFIX            ${CUTLASS_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   "")

##############################################################################
# - spdlog -------------------------------------------------------------------

set(SPDLOG_DIR ${CMAKE_CURRENT_BINARY_DIR}/spdlog CACHE STRING
  "Path to spdlog install directory")
ExternalProject_Add(spdlog
  GIT_REPOSITORY    https://github.com/gabime/spdlog.git
  GIT_TAG           v1.7.0
  PREFIX            ${SPDLOG_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   "")

##############################################################################
# - faiss --------------------------------------------------------------------

if(BUILD_STATIC_FAISS)
  set(FAISS_DIR ${CMAKE_CURRENT_BINARY_DIR}/faiss CACHE STRING
    "Path to FAISS source directory")
  ExternalProject_Add(faiss
    GIT_REPOSITORY    https://github.com/facebookresearch/faiss.git
    GIT_TAG           a5b850dec6f1cd6c88ab467bfd5e87b0cac2e41d
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
    BUILD_COMMAND     make -j${PARALLEL_LEVEL} VERBOSE=1
    BUILD_BYPRODUCTS  ${FAISS_DIR}/lib/libfaiss.a
    BUILD_ALWAYS      1
    INSTALL_COMMAND   make -s install > /dev/null
    UPDATE_COMMAND    ""
    BUILD_IN_SOURCE   1
    PATCH_COMMAND     patch -p1 -N < ${CMAKE_CURRENT_SOURCE_DIR}/cmake/faiss_cuda11.patch || true)

  ExternalProject_Get_Property(faiss install_dir)
  add_library(FAISS::FAISS STATIC IMPORTED)
  set_property(TARGET FAISS::FAISS PROPERTY
    IMPORTED_LOCATION ${FAISS_DIR}/lib/libfaiss.a)
  # to account for the FAISS file reorg that happened recently after the current
  # pinned commit, just change the following line to
  # set(FAISS_INCLUDE_DIRS "${FAISS_DIR}/src/faiss")
  set(FAISS_INCLUDE_DIRS "${FAISS_DIR}/src")
else()
  set(FAISS_INSTALL_DIR ENV{FAISS_ROOT})
  find_package(FAISS REQUIRED)
endif(BUILD_STATIC_FAISS)

##############################################################################
# - treelite build -----------------------------------------------------------

find_package(Treelite 0.93 REQUIRED)

##############################################################################
# - googletest build -----------------------------------------------------------

if(BUILD_GTEST)
	set(GTEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest CACHE STRING
	  "Path to googletest repo")
	set(GTEST_BINARY_DIR ${PROJECT_BINARY_DIR}/googletest)
	set(GTEST_INSTALL_DIR ${GTEST_BINARY_DIR}/install)
	set(GTEST_LIB ${GTEST_INSTALL_DIR}/lib/libgtest_main.a)
	include(ExternalProject)
	ExternalProject_Add(googletest
	  GIT_REPOSITORY    https://github.com/google/googletest.git
	  GIT_TAG           6ce9b98f541b8bcd84c5c5b3483f29a933c4aefb
	  PREFIX            ${GTEST_DIR}
	  CMAKE_ARGS        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
	                    -DBUILD_SHARED_LIBS=OFF
	                    -DCMAKE_INSTALL_LIBDIR=lib
	  BUILD_BYPRODUCTS  ${GTEST_DIR}/lib/libgtest.a
	                    ${GTEST_DIR}/lib/libgtest_main.a
	  UPDATE_COMMAND    "")

	add_library(GTest::GTest STATIC IMPORTED)
	add_library(GTest::Main STATIC IMPORTED)

	set_property(TARGET GTest::GTest PROPERTY
	  IMPORTED_LOCATION ${GTEST_DIR}/lib/libgtest.a)
	set_property(TARGET GTest::Main PROPERTY
	  IMPORTED_LOCATION ${GTEST_DIR}/lib/libgtest_main.a)

	set(GTEST_INCLUDE_DIRS "${GTEST_DIR}")

	add_dependencies(GTest::GTest googletest)
	add_dependencies(GTest::Main googletest)

else()
	find_package(GTest REQUIRED)
endif(BUILD_GTEST)

##############################################################################
# - googlebench ---------------------------------------------------------------

set(GBENCH_DIR ${CMAKE_CURRENT_BINARY_DIR}/benchmark CACHE STRING
  "Path to google benchmark repo")
set(GBENCH_BINARY_DIR ${PROJECT_BINARY_DIR}/benchmark)
set(GBENCH_INSTALL_DIR ${GBENCH_BINARY_DIR}/install)
set(GBENCH_LIB ${GBENCH_INSTALL_DIR}/lib/libbenchmark.a)
include(ExternalProject)
ExternalProject_Add(benchmark
  GIT_REPOSITORY    https://github.com/google/benchmark.git
  GIT_TAG           bf4f2ea0bd1180b34718ac26eb79b170a4f6290e
  PREFIX            ${GBENCH_DIR}
  CMAKE_ARGS        -DBENCHMARK_ENABLE_GTEST_TESTS=OFF
                    -DBENCHMARK_ENABLE_TESTING=OFF
                    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                    -DCMAKE_BUILD_TYPE=Release
                    -DCMAKE_INSTALL_LIBDIR=lib
  BUILD_BYPRODUCTS  ${GBENCH_DIR}/lib/libbenchmark.a
  UPDATE_COMMAND    "")
add_library(benchmarklib STATIC IMPORTED)
add_dependencies(benchmarklib benchmark)
set_property(TARGET benchmarklib PROPERTY
  IMPORTED_LOCATION ${GBENCH_DIR}/lib/libbenchmark.a)

# dependencies will be added in sequence, so if a new project `project_b` is added
# after `project_a`, please add the dependency add_dependencies(project_b project_a)
# This allows the cloning to happen sequentially, enhancing the printing at
# compile time, helping significantly to troubleshoot build issues.

# TODO: Change to using build.sh and make targets instead of this

if(CUB_IS_PART_OF_CTK)
  add_dependencies(cutlass raft)
else()
  add_dependencies(cub raft)
  add_dependencies(cutlass cub)
endif(CUB_IS_PART_OF_CTK)
add_dependencies(spdlog cutlass)
add_dependencies(GTest::GTest spdlog)
add_dependencies(benchmark GTest::GTest)
add_dependencies(FAISS::FAISS benchmark)
add_dependencies(FAISS::FAISS faiss)
