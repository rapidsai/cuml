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
#include "csr.h"

#include <raft/cudart_utils.h>
#include <raft/random/rng.cuh>
#include "test_utils.h"

#include <iostream>
#include <limits>

namespace MLCommon {
namespace Sparse {

template <typename T>
class CSRTest : public ::testing::TestWithParam<CSRInputs<T>> {
 protected:
  void SetUp() override {}

  void TearDown() override {}

 protected:
  CSRInputs<T> params;
};

const std::vector<CSRInputs<float>> inputsf = {{5, 10, 5, 1234ULL}};

typedef CSRTest<float> CSRToCOO;
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

  csr_to_coo<32>(ex_scan, 4, result, 10, stream);

  ASSERT_TRUE(
    raft::devArrMatch<int>(verify, result, 10, raft::Compare<float>(), stream));

  delete[] ex_scan_h;
  delete[] verify_h;

  CUDA_CHECK(cudaFree(ex_scan));
  CUDA_CHECK(cudaFree(verify));
  CUDA_CHECK(cudaFree(result));

  cudaStreamDestroy(stream);
}

typedef CSRTest<float> CSRRowNormalizeMax;
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

  csr_row_normalize_max<32, float>(ex_scan, in_vals, 10, 4, result, stream);

  ASSERT_TRUE(
    raft::devArrMatch<float>(verify, result, 10, raft::Compare<float>()));

  cudaStreamDestroy(stream);

  CUDA_CHECK(cudaFree(ex_scan));
  CUDA_CHECK(cudaFree(in_vals));
  CUDA_CHECK(cudaFree(verify));
  CUDA_CHECK(cudaFree(result));
}

typedef CSRTest<float> CSRRowNormalizeL1;
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

  csr_row_normalize_l1<32, float>(ex_scan, in_vals, 10, 4, result, 0);
  cudaDeviceSynchronize();

  ASSERT_TRUE(
    raft::devArrMatch<float>(verify, result, 10, raft::Compare<float>()));

  CUDA_CHECK(cudaFree(ex_scan));
  CUDA_CHECK(cudaFree(in_vals));
  CUDA_CHECK(cudaFree(verify));
  CUDA_CHECK(cudaFree(result));
}

typedef CSRTest<float> CSRSum;
TEST_P(CSRSum, Result) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::shared_ptr<deviceAllocator> alloc(
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

  int nnz = csr_add_calc_inds<float, 32>(ex_scan, ind_ptr_a, in_vals_a, 10,
                                         ex_scan, ind_ptr_b, in_vals_b, 10, 4,
                                         result_ind, alloc, stream);

  int *result_indptr;
  float *result_val;
  raft::allocate(result_indptr, nnz);
  raft::allocate(result_val, nnz);

  csr_add_finalize<float, 32>(ex_scan, ind_ptr_a, in_vals_a, 10, ex_scan,
                              ind_ptr_b, in_vals_b, 10, 4, result_ind,
                              result_indptr, result_val, stream);

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

typedef CSRTest<float> CSRRowOpTest;
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

  csr_row_op<int, 32>(
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

typedef CSRTest<float> AdjGraphTest;
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

  csr_adj_graph_batched<int, 32>(row_ind, 6, 9, 3, adj, result, stream);

  ASSERT_TRUE(raft::devArrMatch<int>(verify, result, 9, raft::Compare<int>()));

  cudaStreamDestroy(stream);

  CUDA_CHECK(cudaFree(row_ind));
  CUDA_CHECK(cudaFree(adj));
  CUDA_CHECK(cudaFree(verify));
  CUDA_CHECK(cudaFree(result));
}

typedef CSRTest<float> WeakCCTest;
TEST_P(WeakCCTest, Result) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::shared_ptr<deviceAllocator> alloc(
    new raft::mr::device::default_allocator);
  int *row_ind, *row_ind_ptr, *result, *verify;

  int row_ind_h1[3] = {0, 3, 6};
  int row_ind_ptr_h1[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  int verify_h1[6] = {1, 1, 1, 2147483647, 2147483647, 2147483647};

  int row_ind_h2[3] = {0, 2, 4};
  int row_ind_ptr_h2[5] = {3, 4, 3, 4, 5};
  int verify_h2[6] = {1, 1, 1, 5, 5, 5};

  raft::allocate(row_ind, 3);
  raft::allocate(row_ind_ptr, 9);
  raft::allocate(result, 9, true);
  raft::allocate(verify, 9);

  device_buffer<bool> xa(alloc, stream, 6);
  device_buffer<bool> fa(alloc, stream, 6);
  device_buffer<bool> m(alloc, stream, 1);
  WeakCCState state(xa.data(), fa.data(), m.data());

  /**
     * Run batch #1
     */
  raft::update_device(row_ind, *&row_ind_h1, 3, stream);
  raft::update_device(row_ind_ptr, *&row_ind_ptr_h1, 9, stream);
  raft::update_device(verify, *&verify_h1, 6, stream);

  weak_cc_batched<int, 32>(result, row_ind, row_ind_ptr, 9, 6, 0, 3, &state,
                           stream);

  cudaStreamSynchronize(stream);
  ASSERT_TRUE(raft::devArrMatch<int>(verify, result, 6, raft::Compare<int>()));

  /**
     * Run batch #2
     */
  raft::update_device(row_ind, *&row_ind_h2, 3, stream);
  raft::update_device(row_ind_ptr, *&row_ind_ptr_h2, 5, stream);
  raft::update_device(verify, *&verify_h2, 6, stream);

  weak_cc_batched<int, 32>(result, row_ind, row_ind_ptr, 5, 6, 4, 3, &state,
                           stream);

  ASSERT_TRUE(raft::devArrMatch<int>(verify, result, 6, raft::Compare<int>()));

  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);

  CUDA_CHECK(cudaFree(row_ind));
  CUDA_CHECK(cudaFree(row_ind_ptr));
  CUDA_CHECK(cudaFree(verify));
  CUDA_CHECK(cudaFree(result));
}

INSTANTIATE_TEST_CASE_P(CSRTests, WeakCCTest, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(CSRTests, AdjGraphTest, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(CSRTests, CSRRowOpTest, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(CSRTests, CSRToCOO, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(CSRTests, CSRRowNormalizeMax,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(CSRTests, CSRRowNormalizeL1,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(CSRTests, CSRSum, ::testing::ValuesIn(inputsf));
}  // namespace Sparse
}  // namespace MLCommon
