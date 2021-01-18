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
#include <sparse/linalg/add.cuh>

#include <raft/cudart_utils.h>
#include <test_utils.h>
#include <raft/random/rng.cuh>

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

template <typename Type_f, typename Index_>
struct CSRMatrixVal {
  std::vector<Index_> row_ind;
  std::vector<Index_> row_ind_ptr;
  std::vector<Type_f> values;
};

template <typename Type_f, typename Index_>
struct CSRAddInputs {
  CSRMatrixVal<Type_f, Index_> matrix_a;
  CSRMatrixVal<Type_f, Index_> matrix_b;
  CSRMatrixVal<Type_f, Index_> matrix_verify;
};

template <typename Type_f, typename Index_>
class CSRAddTest
  : public ::testing::TestWithParam<CSRAddInputs<Type_f, Index_>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<CSRAddInputs<Type_f, Index_>>::GetParam();
    n_rows = params.matrix_a.row_ind.size();
    nnz_a = params.matrix_a.row_ind_ptr.size();
    nnz_b = params.matrix_b.row_ind_ptr.size();
    nnz_result = params.matrix_verify.row_ind_ptr.size();

    cudaStreamCreate(&stream);

    raft::allocate(ind_a, n_rows);
    raft::allocate(ind_ptr_a, nnz_a);
    raft::allocate(values_a, nnz_a);

    raft::allocate(ind_b, n_rows);
    raft::allocate(ind_ptr_b, nnz_b);
    raft::allocate(values_b, nnz_b);

    raft::allocate(ind_verify, n_rows);
    raft::allocate(ind_ptr_verify, nnz_result);
    raft::allocate(values_verify, nnz_result);

    raft::allocate(ind_result, n_rows);
    raft::allocate(ind_ptr_result, nnz_result);
    raft::allocate(values_result, nnz_result);
  }

  void Run() {
    std::shared_ptr<MLCommon::deviceAllocator> alloc(
      new raft::mr::device::default_allocator);

    raft::update_device(ind_a, params.matrix_a.row_ind.data(), n_rows, stream);
    raft::update_device(ind_ptr_a, params.matrix_a.row_ind_ptr.data(), nnz_a,
                        stream);
    raft::update_device(values_a, params.matrix_a.values.data(), nnz_a, stream);

    raft::update_device(ind_b, params.matrix_b.row_ind.data(), n_rows, stream);
    raft::update_device(ind_ptr_b, params.matrix_b.row_ind_ptr.data(), nnz_b,
                        stream);
    raft::update_device(values_b, params.matrix_b.values.data(), nnz_b, stream);

    raft::update_device(ind_verify, params.matrix_verify.row_ind.data(), n_rows,
                        stream);
    raft::update_device(ind_ptr_verify, params.matrix_verify.row_ind_ptr.data(),
                        nnz_result, stream);
    raft::update_device(values_verify, params.matrix_verify.values.data(),
                        nnz_result, stream);

    Index_ nnz = linalg::csr_add_calc_inds<Type_f, 32>(
      ind_a, ind_ptr_a, values_a, nnz_a, ind_b, ind_ptr_b, values_b, nnz_b,
      n_rows, ind_result, alloc, stream);

    ASSERT_TRUE(nnz == nnz_result);
    ASSERT_TRUE(raft::devArrMatch<Index_>(ind_verify, ind_result, n_rows,
                                          raft::Compare<Index_>()));

    linalg::csr_add_finalize<Type_f, 32>(
      ind_a, ind_ptr_a, values_a, nnz_a, ind_b, ind_ptr_b, values_b, nnz_b,
      n_rows, ind_result, ind_ptr_result, values_result, stream);

    ASSERT_TRUE(raft::devArrMatch<Index_>(ind_ptr_verify, ind_ptr_result, nnz,
                                          raft::Compare<Index_>()));
    ASSERT_TRUE(raft::devArrMatch<Type_f>(values_verify, values_result, nnz,
                                          raft::Compare<Type_f>()));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(ind_a));
    CUDA_CHECK(cudaFree(ind_b));
    CUDA_CHECK(cudaFree(ind_result));
    CUDA_CHECK(cudaFree(ind_ptr_a));
    CUDA_CHECK(cudaFree(ind_ptr_b));
    CUDA_CHECK(cudaFree(ind_ptr_verify));
    CUDA_CHECK(cudaFree(ind_ptr_result));
    CUDA_CHECK(cudaFree(values_a));
    CUDA_CHECK(cudaFree(values_b));
    CUDA_CHECK(cudaFree(values_verify));
    CUDA_CHECK(cudaFree(values_result));
    cudaStreamDestroy(stream);
  }

 protected:
  CSRAddInputs<Type_f, Index_> params;
  cudaStream_t stream;
  Index_ n_rows, nnz_a, nnz_b, nnz_result;
  Index_ *ind_a, *ind_b, *ind_verify, *ind_result, *ind_ptr_a, *ind_ptr_b,
    *ind_ptr_verify, *ind_ptr_result;
  Type_f *values_a, *values_b, *values_verify, *values_result;
};

using CSRAddTestF = CSRAddTest<float, int>;
TEST_P(CSRAddTestF, Result) { Run(); }

using CSRAddTestD = CSRAddTest<double, int>;
TEST_P(CSRAddTestD, Result) { Run(); }

const std::vector<CSRAddInputs<float, int>> csradd_inputs_f = {
  {{{0, 4, 8, 9},
    {1, 2, 3, 4, 1, 2, 3, 5, 0, 1},
    {1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0}},
   {{0, 4, 8, 9},
    {1, 2, 5, 4, 0, 2, 3, 5, 1, 0},
    {1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0}},
   {{0, 5, 10, 12},
    {1, 2, 3, 4, 5, 1, 2, 3, 5, 0, 0, 1, 1, 0},
    {2.0, 2.0, 0.5, 1.0, 0.5, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}}},
};
const std::vector<CSRAddInputs<double, int>> csradd_inputs_d = {
  {{{0, 4, 8, 9},
    {1, 2, 3, 4, 1, 2, 3, 5, 0, 1},
    {1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0}},
   {{0, 4, 8, 9},
    {1, 2, 5, 4, 0, 2, 3, 5, 1, 0},
    {1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0}},
   {{0, 5, 10, 12},
    {1, 2, 3, 4, 5, 1, 2, 3, 5, 0, 0, 1, 1, 0},
    {2.0, 2.0, 0.5, 1.0, 0.5, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}}},
};

INSTANTIATE_TEST_CASE_P(SparseAddTest, CSRAddTestF,
                        ::testing::ValuesIn(csradd_inputs_f));
INSTANTIATE_TEST_CASE_P(SparseAddTest, CSRAddTestD,
                        ::testing::ValuesIn(csradd_inputs_d));

}  // namespace sparse
}  // namespace raft
