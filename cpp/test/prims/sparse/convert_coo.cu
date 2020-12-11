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

#include <sparse/convert/coo.cuh>
#include <sparse/csr.cuh>

#include <raft/cudart_utils.h>
#include <raft/random/rng.cuh>

#include "test_utils.h"

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

template <typename T>
struct SparseConvertCOOInputs {
  int m, n, nnz;
  unsigned long long int seed;
};

template <typename T>
class SparseConvertCOOTest
  : public ::testing::TestWithParam<SparseConvertCOOInputs<T>> {
 protected:
  void SetUp() override {}

  void TearDown() override {}

 protected:
  SparseConvertCOOInputs<T> params;
};

const std::vector<SparseConvertCOOInputs<float>> inputsf = {
  {5, 10, 5, 1234ULL}};

typedef SparseConvertCOOTest<float> CSRToCOO;
TEST_P(CSRToCOO, Result) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int *ex_scan;
  int *result, *verify;

  int *ex_scan_h = new int[4]{0, 4, 8, 9};
  int *verify_h = new int[10]{0, 0, 0, 0, 1, 1, 1, 1, 2, 3};

  raft::allocate(verify, 10);
  raft::allocate(ex_scan, 4);
  raft::allocate(result, 10, true);

  raft::update_device(ex_scan, ex_scan_h, 4, stream);
  raft::update_device(verify, verify_h, 10, stream);

  convert::csr_to_coo<int, 32>(ex_scan, 4, result, 10, stream);

  ASSERT_TRUE(
    raft::devArrMatch<int>(verify, result, 10, raft::Compare<float>(), stream));

  delete[] ex_scan_h;
  delete[] verify_h;

  CUDA_CHECK(cudaFree(ex_scan));
  CUDA_CHECK(cudaFree(verify));
  CUDA_CHECK(cudaFree(result));

  cudaStreamDestroy(stream);
}

INSTANTIATE_TEST_CASE_P(SparseConvertCOOTest, CSRToCOO,
                        ::testing::ValuesIn(inputsf));

}  // namespace sparse
}  // namespace raft
