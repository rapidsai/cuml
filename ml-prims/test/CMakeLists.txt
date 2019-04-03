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
project(mlcommon_test LANGUAGES CXX CUDA)

include_directories(${GTEST_DIR}/googletest/include)

# single gpu unit-tests
# (please keep the filenames in alphabetical order)
add_executable(mlcommon_test
  add.cu
  binary_op.cu
  ternary_op.cu
  coalesced_reduction.cu
  cuda_utils.cu
  coo.cu
  cov.cu
  csr.cu
  decoupled_lookback.cu
  dist_adj.cu
  dist_cos.cu
  dist_euc_exp.cu
  dist_euc_unexp.cu
  dist_l1.cu
  divide.cu
  eig.cu
  eltwise.cu
  eltwise2d.cu
  gemm.cu
  grid_sync.cu
  kselection.cu
  map_then_reduce.cu
  math.cu
  matrix.cu
  matrix_vector_op.cu
  mean.cu
  mean_center.cu
  metrics.cu
  mvg.cu
  multiply.cu
  norm.cu
  permute.cu
  power.cu
  reduce_rows_by_key.cu
  rng.cu
  rng_int.cu
  rsvd.cu
  sqrt.cu
  stddev.cu
  strided_reduction.cu
  subtract.cu
  sum.cu
  svd.cu
  transpose.cu
  unary_op.cu
  hinge.cu
  linearReg.cu
  log.cu
  logisticReg.cu
  penalty.cu
  sigmoid.cu
  weighted_mean.cu
  columnSort.cu
  )

target_link_libraries(mlcommon_test
  ${GTEST_LIBNAME}
  ${MLPRIMS_LIBS})
