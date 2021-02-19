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

#include <sparse/csr.cuh>
#include <sparse/op/row_op.cuh>

#include <specializations/raft/cudart_utils.h>
#include <test_utils.h>
#include <raft/random/rng.cuh>

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

template <typename Type_f, typename Index_>
struct CSRRowOpInputs {
  std::vector<Index_> ex_scan;
  std::vector<Type_f> verify;
};

/** Wrapper to call csr_row_op because the enclosing function of a __device__
 *  lambda cannot have private ot protected access within the class. */
template <typename Type_f, typename Index_>
void csr_row_op_wrapper(const Index_ *row_ind, Index_ n_rows, Index_ nnz,
                        Type_f *result, cudaStream_t stream) {
  op::csr_row_op<Index_, 32>(
    row_ind, n_rows, nnz,
    [result] __device__(Index_ row, Index_ start_idx, Index_ stop_idx) {
      for (Index_ i = start_idx; i < stop_idx; i++) result[i] = row;
    },
    stream);
}

template <typename Type_f, typename Index_>
class CSRRowOpTest
  : public ::testing::TestWithParam<CSRRowOpInputs<Type_f, Index_>> {
 protected:
  void SetUp() override {
    params =
      ::testing::TestWithParam<CSRRowOpInputs<Type_f, Index_>>::GetParam();
    cudaStreamCreate(&stream);
    n_rows = params.ex_scan.size();
    nnz = params.verify.size();

    raft::allocate(verify, nnz);
    raft::allocate(ex_scan, n_rows);
    raft::allocate(result, nnz, true);
  }

  void Run() {
    raft::update_device(ex_scan, params.ex_scan.data(), n_rows, stream);
    raft::update_device(verify, params.verify.data(), nnz, stream);

    csr_row_op_wrapper<Type_f, Index_>(ex_scan, n_rows, nnz, result, stream);

    ASSERT_TRUE(
      raft::devArrMatch<Type_f>(verify, result, nnz, raft::Compare<Type_f>()));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(ex_scan));
    CUDA_CHECK(cudaFree(verify));
    CUDA_CHECK(cudaFree(result));
    cudaStreamDestroy(stream);
  }

 protected:
  CSRRowOpInputs<Type_f, Index_> params;
  cudaStream_t stream;
  Index_ n_rows, nnz;
  Index_ *ex_scan;
  Type_f *result, *verify;
};

using CSRRowOpTestF = CSRRowOpTest<float, int>;
TEST_P(CSRRowOpTestF, Result) { Run(); }

using CSRRowOpTestD = CSRRowOpTest<double, int>;
TEST_P(CSRRowOpTestD, Result) { Run(); }

const std::vector<CSRRowOpInputs<float, int>> csrrowop_inputs_f = {
  {{0, 4, 8, 9}, {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0}},
};
const std::vector<CSRRowOpInputs<double, int>> csrrowop_inputs_d = {
  {{0, 4, 8, 9}, {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0}},
};

INSTANTIATE_TEST_CASE_P(SparseRowOpTest, CSRRowOpTestF,
                        ::testing::ValuesIn(csrrowop_inputs_f));
INSTANTIATE_TEST_CASE_P(SparseRowOpTest, CSRRowOpTestD,
                        ::testing::ValuesIn(csrrowop_inputs_d));

}  // namespace sparse
}  // namespace raft
