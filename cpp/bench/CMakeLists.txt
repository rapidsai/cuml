# Copyright (c) 2019, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
cmake_policy(SET CMP0079 NEW)

###################################################################################################
# - build cuml bench executable -------------------------------------------------------------------

set(GBENCH_DIR ${PROJECT_SOURCE_DIR}/external/benchmark)
set(GBENCH_BINARY_DIR ${PROJECT_BINARY_DIR}/benchmark)
set(GBENCH_INSTALL_DIR ${GBENCH_BINARY_DIR}/install)

include(ExternalProject)
ExternalProject_Add(benchmark
  SOURCE_DIR   ${GBENCH_DIR}
  BINARY_DIR   ${GBENCH_BINARY_DIR}/build
  INSTALL_DIR  ${GBENCH_INSTALL_DIR}
  CMAKE_ARGS   -DBENCHMARK_ENABLE_GTEST_TESTS=OFF
               -DBENCHMARK_ENABLE_TESTING=OFF
               -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
               -DCMAKE_BUILD_TYPE=Release
)

set(CUML_SG_BENCH_TARGET "sg_benchmark")
set(ML_BENCH_LINK_LIBRARIES
  ${CUML_CPP_TARGET}
  ${GBENCH_INSTALL_DIR}/lib/libbenchmark.a
)

# (please keep the filenames in alphabetical order)
add_executable(${CUML_SG_BENCH_TARGET}
  sg/dbscan.cu
  sg/kmeans.cu
  sg/main.cpp
  sg/rf_classifier.cu
  )
add_dependencies(${CUML_SG_BENCH_TARGET} ${ClangFormat_TARGET})
add_dependencies(${CUML_SG_BENCH_TARGET} benchmark)
include_directories(${GBENCH_INSTALL_DIR}/include)
target_link_libraries(${CUML_SG_BENCH_TARGET} ${ML_BENCH_LINK_LIBRARIES})
