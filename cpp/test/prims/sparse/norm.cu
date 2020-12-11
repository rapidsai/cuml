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

#include <sparse/linalg/norm.cuh>
#include <sparse/csr.cuh>
#include <raft/cudart_utils.h>
#include <raft/random/rng.cuh>
#include "test_utils.h"

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

template <typename T>
struct SparseNormInputs {
  int m, n, nnz;
  unsigned long long int seed;
};


template <typename T>
class SparseNormTest : public ::testing::TestWithParam<SparseNormInputs<T>> {
 protected:
  void SetUp() override {}

  void TearDown() override {}

 protected:
  SparseNormInputs<T> params;
};

const std::vector<SparseNormInputs<float>> inputsf = {{5, 10, 5, 1234ULL}};

typedef SparseNormTest<float> CSRRowNormalizeMax;
TEST_P(CSRRowNormalizeMax, Result) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int *ex_scan;
  float *in_vals, *result, *verify;

  int ex_scan_h[4] = {0, 4, 8, 9};
  float in_vals_h[10] = {5.0, 1.0, 0.0, 0.0, 10.0, 1.0, 0.0, 0.0, 1.0, 0.0};

  float verify_h[10] = {1.0, 0.2, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 1, 0.0};

  raft::allocate(in_vals, 10);
  raft::allocate(verify, 10);
  raft::allocate(ex_scan, 4);
  raft::allocate(result, 10, true);

  raft::update_device(ex_scan, *&ex_scan_h, 4, stream);
  raft::update_device(in_vals, *&in_vals_h, 10, stream);
  raft::update_device(verify, *&verify_h, 10, stream);

  linalg::csr_row_normalize_max<32, float>(ex_scan, in_vals, 10, 4, result, stream);

  ASSERT_TRUE(
    raft::devArrMatch<float>(verify, result, 10, raft::Compare<float>()));

  cudaStreamDestroy(stream);

  CUDA_CHECK(cudaFree(ex_scan));
  CUDA_CHECK(cudaFree(in_vals));
  CUDA_CHECK(cudaFree(verify));
  CUDA_CHECK(cudaFree(result));
}

typedef SparseNormTest<float> CSRRowNormalizeL1;
TEST_P(CSRRowNormalizeL1, Result) {
  int *ex_scan;
  float *in_vals, *result, *verify;

  int ex_scan_h[4] = {0, 4, 8, 9};
  float in_vals_h[10] = {1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0};

  float verify_h[10] = {0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1, 0.0};

  raft::allocate(in_vals, 10);
  raft::allocate(verify, 10);
  raft::allocate(ex_scan, 4);
  raft::allocate(result, 10, true);

  raft::update_device(ex_scan, *&ex_scan_h, 4, 0);
  raft::update_device(in_vals, *&in_vals_h, 10, 0);
  raft::update_device(verify, *&verify_h, 10, 0);

  linalg::csr_row_normalize_l1<32, float>(ex_scan, in_vals, 10, 4, result, 0);
  cudaDeviceSynchronize();

  ASSERT_TRUE(
    raft::devArrMatch<float>(verify, result, 10, raft::Compare<float>()));

  CUDA_CHECK(cudaFree(ex_scan));
  CUDA_CHECK(cudaFree(in_vals));
  CUDA_CHECK(cudaFree(verify));
  CUDA_CHECK(cudaFree(result));
}

INSTANTIATE_TEST_CASE_P(SparseNormTest, CSRRowNormalizeMax,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(SparseNormTest, CSRRowNormalizeL1,
                        ::testing::ValuesIn(inputsf));

}  // namespace sparse
}  // namespace raft
