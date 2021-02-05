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
#include <sparse/csr.cuh>
#include <sparse/linalg/norm.cuh>

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

enum NormalizeMethod { MAX, L1 };

template <typename Type_f, typename Index_>
struct CSRRowNormalizeInputs {
  NormalizeMethod method;
  std::vector<Index_> ex_scan;
  std::vector<Type_f> in_vals;
  std::vector<Type_f> verify;
};

template <typename Type_f, typename Index_>
class CSRRowNormalizeTest
  : public ::testing::TestWithParam<CSRRowNormalizeInputs<Type_f, Index_>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<
      CSRRowNormalizeInputs<Type_f, Index_>>::GetParam();
    cudaStreamCreate(&stream);

    raft::allocate(in_vals, params.in_vals.size());
    raft::allocate(verify, params.verify.size());
    raft::allocate(ex_scan, params.ex_scan.size());
    raft::allocate(result, params.verify.size(), true);
  }

  void Run() {
    Index_ n_rows = params.ex_scan.size();
    Index_ nnz = params.in_vals.size();

    raft::update_device(ex_scan, params.ex_scan.data(), n_rows, stream);
    raft::update_device(in_vals, params.in_vals.data(), nnz, stream);
    raft::update_device(verify, params.verify.data(), nnz, stream);

    switch (params.method) {
      case MAX:
        linalg::csr_row_normalize_max<32, Type_f>(ex_scan, in_vals, nnz, n_rows,
                                                  result, stream);
        break;
      case L1:
        linalg::csr_row_normalize_l1<32, Type_f>(ex_scan, in_vals, nnz, n_rows,
                                                 result, stream);
        break;
    }

    ASSERT_TRUE(
      raft::devArrMatch<Type_f>(verify, result, nnz, raft::Compare<Type_f>()));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(ex_scan));
    CUDA_CHECK(cudaFree(in_vals));
    CUDA_CHECK(cudaFree(verify));
    CUDA_CHECK(cudaFree(result));
    cudaStreamDestroy(stream);
  }

 protected:
  CSRRowNormalizeInputs<Type_f, Index_> params;
  cudaStream_t stream;
  Index_ *ex_scan;
  Type_f *in_vals, *result, *verify;
};

using CSRRowNormalizeTestF = CSRRowNormalizeTest<float, int>;
TEST_P(CSRRowNormalizeTestF, Result) { Run(); }

using CSRRowNormalizeTestD = CSRRowNormalizeTest<double, int>;
TEST_P(CSRRowNormalizeTestD, Result) { Run(); }

const std::vector<CSRRowNormalizeInputs<float, int>> csrnormalize_inputs_f = {
  {MAX,
   {0, 4, 8, 9},
   {5.0, 1.0, 0.0, 0.0, 10.0, 1.0, 0.0, 0.0, 1.0, 0.0},
   {1.0, 0.2, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 1, 0.0}},
  {L1,
   {0, 4, 8, 9},
   {1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0},
   {0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1, 0.0}},
};
const std::vector<CSRRowNormalizeInputs<double, int>> csrnormalize_inputs_d = {
  {MAX,
   {0, 4, 8, 9},
   {5.0, 1.0, 0.0, 0.0, 10.0, 1.0, 0.0, 0.0, 1.0, 0.0},
   {1.0, 0.2, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 1, 0.0}},
  {L1,
   {0, 4, 8, 9},
   {1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0},
   {0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1, 0.0}},
};

INSTANTIATE_TEST_CASE_P(SparseNormTest, CSRRowNormalizeTestF,
                        ::testing::ValuesIn(csrnormalize_inputs_f));
INSTANTIATE_TEST_CASE_P(SparseNormTest, CSRRowNormalizeTestD,
                        ::testing::ValuesIn(csrnormalize_inputs_d));

}  // namespace sparse
}  // namespace raft
