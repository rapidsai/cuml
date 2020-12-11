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

template <typename T>
struct SparseAddInputs {
  int m, n, nnz;
  unsigned long long int seed;
};

template <typename T>
class SparseAddTest : public ::testing::TestWithParam<SparseAddInputs<T>> {
 protected:
  void SetUp() override {}

  void TearDown() override {}

 protected:
  SparseAddInputs<T> params;
};

const std::vector<SparseAddInputs<float>> inputsf = {{5, 10, 5, 1234ULL}};

typedef SparseAddTest<float> CSRSum;
TEST_P(CSRSum, Result) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::shared_ptr<MLCommon::deviceAllocator> alloc(
    new raft::mr::device::default_allocator);

  int *ex_scan, *ind_ptr_a, *ind_ptr_b, *verify_indptr;
  float *in_vals_a, *in_vals_b, *verify;

  int ex_scan_h[4] = {0, 4, 8, 9};

  int indptr_a_h[10] = {1, 2, 3, 4, 1, 2, 3, 5, 0, 1};
  int indptr_b_h[10] = {1, 2, 5, 4, 0, 2, 3, 5, 1, 0};

  float in_vals_h[10] = {1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0};

  float verify_h[14] = {2.0, 2.0, 0.5, 1.0, 0.5, 1.0, 2.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  int verify_indptr_h[14] = {1, 2, 3, 4, 5, 1, 2, 3, 5, 0, 0, 1, 1, 0};

  raft::allocate(in_vals_a, 10);
  raft::allocate(in_vals_b, 10);
  raft::allocate(verify, 14);
  raft::allocate(ex_scan, 4);
  raft::allocate(verify_indptr, 14);

  raft::allocate(ind_ptr_a, 10);
  raft::allocate(ind_ptr_b, 10);

  raft::update_device(ex_scan, *&ex_scan_h, 4, stream);
  raft::update_device(in_vals_a, *&in_vals_h, 10, stream);
  raft::update_device(in_vals_b, *&in_vals_h, 10, stream);
  raft::update_device(verify, *&verify_h, 14, stream);
  raft::update_device(verify_indptr, *&verify_indptr_h, 14, stream);
  raft::update_device(ind_ptr_a, *&indptr_a_h, 10, stream);
  raft::update_device(ind_ptr_b, *&indptr_b_h, 10, stream);

  int *result_ind;
  raft::allocate(result_ind, 4);

  int nnz = linalg::csr_add_calc_inds<float, 32>(
    ex_scan, ind_ptr_a, in_vals_a, 10, ex_scan, ind_ptr_b, in_vals_b, 10, 4,
    result_ind, alloc, stream);

  int *result_indptr;
  float *result_val;
  raft::allocate(result_indptr, nnz);
  raft::allocate(result_val, nnz);

  linalg::csr_add_finalize<float, 32>(
    ex_scan, ind_ptr_a, in_vals_a, 10, ex_scan, ind_ptr_b, in_vals_b, 10, 4,
    result_ind, result_indptr, result_val, stream);

  ASSERT_TRUE(nnz == 14);

  ASSERT_TRUE(
    raft::devArrMatch<float>(verify, result_val, nnz, raft::Compare<float>()));
  ASSERT_TRUE(raft::devArrMatch<int>(verify_indptr, result_indptr, nnz,
                                     raft::Compare<int>()));

  cudaStreamDestroy(stream);

  CUDA_CHECK(cudaFree(ex_scan));
  CUDA_CHECK(cudaFree(in_vals_a));
  CUDA_CHECK(cudaFree(in_vals_b));
  CUDA_CHECK(cudaFree(ind_ptr_a));
  CUDA_CHECK(cudaFree(ind_ptr_b));
  CUDA_CHECK(cudaFree(verify));
  CUDA_CHECK(cudaFree(result_indptr));
  CUDA_CHECK(cudaFree(result_val));
}

INSTANTIATE_TEST_CASE_P(SparseAddTest, CSRSum, ::testing::ValuesIn(inputsf));
}  // namespace sparse
}  // namespace raft
