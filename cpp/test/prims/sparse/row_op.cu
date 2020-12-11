/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include <sparse/csr.cuh>
#include <sparse/op/row_op.cuh>

#include <raft/cudart_utils.h>
#include <raft/random/rng.cuh>
#include "test_utils.h"

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

template <typename T>
struct SparseRowOpInputs {
  int m, n, nnz;
  unsigned long long int seed;
};


template <typename T>
class SparseRowOpTest : public ::testing::TestWithParam<SparseRowOpInputs<T>> {
 protected:
  void SetUp() override {}

  void TearDown() override {}

 protected:
  SparseRowOpInputs<T> params;
};

const std::vector<SparseRowOpInputs<float>> inputsf = {{5, 10, 5, 1234ULL}};

typedef SparseRowOpTest<float> CSRRowOpTest;
TEST_P(CSRRowOpTest, Result) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int *ex_scan;
  float *result, *verify;

  int ex_scan_h[4] = {0, 4, 8, 9};

  float verify_h[10] = {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0};

  raft::allocate(verify, 10);
  raft::allocate(ex_scan, 4);
  raft::allocate(result, 10, true);

  raft::update_device(ex_scan, *&ex_scan_h, 4, stream);
  raft::update_device(verify, *&verify_h, 10, stream);

  op::csr_row_op<int, 32>(
    ex_scan, 4, 10,
  [result] __device__(int row, int start_idx, int stop_idx) {
    for (int i = start_idx; i < stop_idx; i++) result[i] = row;
  },
  stream);

  ASSERT_TRUE(
    raft::devArrMatch<float>(verify, result, 10, raft::Compare<float>()));

  cudaStreamDestroy(stream);

  CUDA_CHECK(cudaFree(ex_scan));
  CUDA_CHECK(cudaFree(verify));
  CUDA_CHECK(cudaFree(result));
}

INSTANTIATE_TEST_CASE_P(SparseRowOpTest, CSRRowOpTest, ::testing::ValuesIn(inputsf));

}  // namespace sparse
}  // namespace raft
