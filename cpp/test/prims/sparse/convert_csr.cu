/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

/**************************** sorted COO to CSR ****************************/

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

INSTANTIATE_TEST_CASE_P(SparseConvertCSRTest, SortedCOOToCSR,
                        ::testing::ValuesIn(inputsf));

/******************************** adj graph ********************************/

template <typename Index_>
struct CSRAdjGraphInputs {
  Index_ n_rows;
  Index_ n_cols;
  std::vector<Index_> row_ind;
  std::vector<uint8_t> adj;  // To avoid vector<bool> optimization
  std::vector<Index_> verify;
};

template <typename Index_>
class CSRAdjGraphTest
  : public ::testing::TestWithParam<CSRAdjGraphInputs<Index_>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<CSRAdjGraphInputs<Index_>>::GetParam();
    cudaStreamCreate(&stream);
    nnz = params.verify.size();

    raft::allocate(row_ind, params.n_rows);
    raft::allocate(adj, params.n_rows * params.n_cols);
    raft::allocate(result, nnz, true);
    raft::allocate(verify, nnz);
  }

  void Run() {
    raft::update_device(row_ind, params.row_ind.data(), params.n_rows, stream);
    raft::update_device(adj, reinterpret_cast<bool *>(params.adj.data()),
                        params.n_rows * params.n_cols, stream);
    raft::update_device(verify, params.verify.data(), nnz, stream);

    convert::csr_adj_graph_batched<Index_, 32>(
      row_ind, params.n_cols, nnz, params.n_rows, adj, result, stream);

    ASSERT_TRUE(
      raft::devArrMatch<Index_>(verify, result, nnz, raft::Compare<Index_>()));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(row_ind));
    CUDA_CHECK(cudaFree(adj));
    CUDA_CHECK(cudaFree(verify));
    CUDA_CHECK(cudaFree(result));
    cudaStreamDestroy(stream);
  }

 protected:
  CSRAdjGraphInputs<Index_> params;
  cudaStream_t stream;
  Index_ nnz;
  Index_ *row_ind, *result, *verify;
  bool *adj;
};

using CSRAdjGraphTestI = CSRAdjGraphTest<int>;
TEST_P(CSRAdjGraphTestI, Result) { Run(); }

using CSRAdjGraphTestL = CSRAdjGraphTest<int64_t>;
TEST_P(CSRAdjGraphTestL, Result) { Run(); }

const std::vector<CSRAdjGraphInputs<int>> csradjgraph_inputs_i = {
  {3,
   6,
   {0, 3, 6},
   {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
   {0, 1, 2, 0, 1, 2, 0, 1, 2}},
};
const std::vector<CSRAdjGraphInputs<int64_t>> csradjgraph_inputs_l = {
  {3,
   6,
   {0, 3, 6},
   {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
   {0, 1, 2, 0, 1, 2, 0, 1, 2}},
};

INSTANTIATE_TEST_CASE_P(SparseConvertCSRTest, CSRAdjGraphTestI,
                        ::testing::ValuesIn(csradjgraph_inputs_i));
INSTANTIATE_TEST_CASE_P(SparseConvertCSRTest, CSRAdjGraphTestL,
                        ::testing::ValuesIn(csradjgraph_inputs_l));

}  // namespace sparse
}  // namespace raft
