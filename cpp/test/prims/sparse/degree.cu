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
#include <raft/cudart_utils.h>
#include <test_utils.h>
#include <raft/random/rng.cuh>

#include <sparse/linalg/degree.cuh>

#include <iostream>

namespace raft {
namespace sparse {

template <typename T>
struct SparseDegreeInputs {
  int m, n, nnz;
  unsigned long long int seed;
};

template <typename T>
class SparseDegreeTests
  : public ::testing::TestWithParam<SparseDegreeInputs<T>> {
 protected:
  void SetUp() override {}

  void TearDown() override {}

 protected:
  SparseDegreeInputs<T> params;
};

const std::vector<SparseDegreeInputs<float>> inputsf = {{5, 10, 5, 1234ULL}};

typedef SparseDegreeTests<float> COODegree;
TEST_P(COODegree, Result) {
  int *in_rows, *verify, *results;

  int in_rows_h[5] = {0, 0, 1, 2, 2};
  int verify_h[5] = {2, 1, 2, 0, 0};

  raft::allocate(in_rows, 5);
  raft::allocate(verify, 5, true);
  raft::allocate(results, 5, true);

  raft::update_device(in_rows, *&in_rows_h, 5, 0);
  raft::update_device(verify, *&verify_h, 5, 0);

  linalg::coo_degree<32>(in_rows, 5, results, 0);
  cudaDeviceSynchronize();

  ASSERT_TRUE(raft::devArrMatch<int>(verify, results, 5, raft::Compare<int>()));

  CUDA_CHECK(cudaFree(in_rows));
  CUDA_CHECK(cudaFree(verify));
}

typedef SparseDegreeTests<float> COODegreeNonzero;
TEST_P(COODegreeNonzero, Result) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int *in_rows, *verify, *results;
  float *in_vals;

  int in_rows_h[5] = {0, 0, 1, 2, 2};
  float in_vals_h[5] = {0.0, 5.0, 0.0, 1.0, 1.0};
  int verify_h[5] = {1, 0, 2, 0, 0};

  raft::allocate(in_rows, 5);
  raft::allocate(verify, 5, true);
  raft::allocate(results, 5, true);
  raft::allocate(in_vals, 5, true);

  raft::update_device(in_rows, *&in_rows_h, 5, 0);
  raft::update_device(verify, *&verify_h, 5, 0);
  raft::update_device(in_vals, *&in_vals_h, 5, 0);

  linalg::coo_degree_nz<32, float>(in_rows, in_vals, 5, results, stream);
  cudaDeviceSynchronize();

  ASSERT_TRUE(raft::devArrMatch<int>(verify, results, 5, raft::Compare<int>()));

  CUDA_CHECK(cudaFree(in_rows));
  CUDA_CHECK(cudaFree(verify));

  CUDA_CHECK(cudaStreamDestroy(stream));
}

INSTANTIATE_TEST_CASE_P(SparseDegreeTests, COODegree,
                        ::testing::ValuesIn(inputsf));
INSTANTIATE_TEST_CASE_P(SparseDegreeTests, COODegreeNonzero,
                        ::testing::ValuesIn(inputsf));

}  // namespace sparse
}  // namespace raft
