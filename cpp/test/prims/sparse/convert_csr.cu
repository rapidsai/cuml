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

#include <raft/mr/device/allocator.hpp>
#include <sparse/convert/csr.cuh>
#include <sparse/coo.cuh>

#include <iostream>

namespace raft {
namespace sparse {

template <typename T>
struct SparseConvertCSRInputs {
  int m, n, nnz;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os,
                           const SparseConvertCSRInputs<T> &dims) {
  return os;
}

template <typename T>
class SparseConvertCSRTest
  : public ::testing::TestWithParam<SparseConvertCSRInputs<T>> {
 protected:
  void SetUp() override {}

  void TearDown() override {}

 protected:
  SparseConvertCSRInputs<T> params;
};

const std::vector<SparseConvertCSRInputs<float>> inputsf = {
  {5, 10, 5, 1234ULL}};

typedef SparseConvertCSRTest<float> SortedCOOToCSR;
TEST_P(SortedCOOToCSR, Result) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::shared_ptr<raft::mr::device::allocator> alloc(
    new raft::mr::device::default_allocator);

  int nnz = 8;

  int *in, *out, *exp;

  int *in_h = new int[nnz]{0, 0, 1, 1, 2, 2, 3, 3};
  int *exp_h = new int[4]{0, 2, 4, 6};

  raft::allocate(in, nnz, true);
  raft::allocate(exp, 4, true);
  raft::allocate(out, 4, true);

  raft::update_device(in, in_h, nnz, stream);
  raft::update_device(exp, exp_h, 4, stream);

  convert::sorted_coo_to_csr<int>(in, nnz, out, 4, alloc, stream);

  ASSERT_TRUE(raft::devArrMatch<int>(out, exp, 4, raft::Compare<int>()));

  cudaStreamDestroy(stream);

  delete[] in_h;
  delete[] exp_h;

  CUDA_CHECK(cudaFree(in));
  CUDA_CHECK(cudaFree(exp));
  CUDA_CHECK(cudaFree(out));
}

typedef SparseConvertCSRTest<float> AdjGraphTest;
TEST_P(AdjGraphTest, Result) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int *row_ind, *result, *verify;
  bool *adj;

  int row_ind_h[3] = {0, 3, 6};
  bool adj_h[18] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  int verify_h[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};

  raft::allocate(row_ind, 3);
  raft::allocate(adj, 18);
  raft::allocate(result, 9, true);
  raft::allocate(verify, 9);

  raft::update_device(row_ind, *&row_ind_h, 3, stream);
  raft::update_device(adj, *&adj_h, 18, stream);
  raft::update_device(verify, *&verify_h, 9, stream);

  convert::csr_adj_graph_batched<int, 32>(row_ind, 6, 9, 3, adj, result,
                                          stream);

  ASSERT_TRUE(raft::devArrMatch<int>(verify, result, 9, raft::Compare<int>()));

  cudaStreamDestroy(stream);

  CUDA_CHECK(cudaFree(row_ind));
  CUDA_CHECK(cudaFree(adj));
  CUDA_CHECK(cudaFree(verify));
  CUDA_CHECK(cudaFree(result));
}

INSTANTIATE_TEST_CASE_P(SparseConvertCSRTest, SortedCOOToCSR,
                        ::testing::ValuesIn(inputsf));
INSTANTIATE_TEST_CASE_P(SparseConvertCSRTest, AdjGraphTest,
                        ::testing::ValuesIn(inputsf));

}  // namespace sparse
}  // namespace raft
