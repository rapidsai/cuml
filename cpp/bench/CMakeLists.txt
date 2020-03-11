# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

# TREELITE_DIR ${CMAKE_CURRENT_BINARY_DIR}/treelite

set(GBENCH_DIR ${CMAKE_CURRENT_BINARY_DIR}/benchmark)
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
  UPDATE_COMMAND    ""
)

add_library(benchmarklib STATIC IMPORTED)
add_dependencies(benchmarklib benchmark)
set_property(TARGET benchmarklib PROPERTY
  IMPORTED_LOCATION ${GBENCH_DIR}/lib/libbenchmark.a)

if(BUILD_CUML_BENCH)
  set(CUML_SG_BENCH_TARGET "sg_benchmark")
  set(ML_BENCH_LINK_LIBRARIES
    ${CUML_CPP_TARGET}
    benchmarklib
    )
  # (please keep the filenames in alphabetical order)
  add_executable(${CUML_SG_BENCH_TARGET}
    sg/dbscan.cu
    sg/kmeans.cu
    sg/main.cpp
    sg/rf_classifier.cu
    sg/rf_regressor.cu
    sg/umap.cu
    )
  target_link_libraries(${CUML_SG_BENCH_TARGET} ${ML_BENCH_LINK_LIBRARIES})
  target_include_directories(${CUML_SG_BENCH_TARGET}
    PRIVATE ${GBENCH_DIR}/include)
endif(BUILD_CUML_BENCH)

if(BUILD_CUML_PRIMS_BENCH)
  set(CUML_PRIMS_BENCH_TARGET "prims_benchmark")
  # (please keep the filenames in alphabetical order)
  add_executable(${CUML_PRIMS_BENCH_TARGET}
    prims/add.cu
    prims/distance_cosine.cu
    prims/distance_exp_l2.cu
    prims/distance_l1.cu
    prims/distance_unexp_l2.cu
    prims/fused_l2_nn.cu
    prims/map_then_reduce.cu
    prims/matrix_vector_op.cu
    prims/permute.cu
    prims/reduce.cu
    prims/rng.cu
    prims/main.cpp
    )
  add_dependencies(${CUML_PRIMS_BENCH_TARGET} cub)
  add_dependencies(${CUML_PRIMS_BENCH_TARGET} cutlass)
  target_link_libraries(${CUML_PRIMS_BENCH_TARGET} benchmarklib)
  target_include_directories(${CUML_PRIMS_BENCH_TARGET}
    PRIVATE ${GBENCH_DIR}/include
    src_prims)
endif(BUILD_CUML_PRIMS_BENCH)
