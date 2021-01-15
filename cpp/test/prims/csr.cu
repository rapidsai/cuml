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

#include <raft/cudart_utils.h>
#include <raft/random/rng.cuh>
#include "test_utils.h"

#include <iostream>
#include <limits>
#include <vector>

constexpr int MAX32 = std::numeric_limits<int>::max();
constexpr int64_t MAX64 = std::numeric_limits<int64_t>::max();

namespace MLCommon {
namespace Sparse {

template <typename Index_>
struct CSRMatrix {
  std::vector<Index_> row_ind;
  std::vector<Index_> row_ind_ptr;
};

template <typename Type_f, typename Index_>
struct CSRMatrixVal {
  std::vector<Index_> row_ind;
  std::vector<Index_> row_ind_ptr;
  std::vector<Type_f> values;
};

/**************************** CSR to COO indices ****************************/

template <typename Index_>
struct CSRtoCOOInputs {
  std::vector<Index_> ex_scan;
  std::vector<Index_> verify;
};

template <typename Index_>
class CSRtoCOOTest : public ::testing::TestWithParam<CSRtoCOOInputs<Index_>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<CSRtoCOOInputs<Index_>>::GetParam();

    cudaStreamCreate(&stream);
    raft::allocate(ex_scan, params.ex_scan.size());
    raft::allocate(verify, params.verify.size());
    raft::allocate(result, params.verify.size(), true);
  }

  void Run() {
    Index_ n_rows = params.ex_scan.size();
    Index_ nnz = params.verify.size();

    raft::update_device(ex_scan, params.ex_scan.data(), n_rows, stream);
    raft::update_device(verify, params.verify.data(), nnz, stream);

    csr_to_coo<Index_, 32>(ex_scan, n_rows, result, nnz, stream);

    ASSERT_TRUE(raft::devArrMatch<Index_>(verify, result, nnz,
                                          raft::Compare<float>(), stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(ex_scan));
    CUDA_CHECK(cudaFree(verify));
    CUDA_CHECK(cudaFree(result));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  CSRtoCOOInputs<Index_> params;
  cudaStream_t stream;
  Index_ *ex_scan, *verify, *result;
};

using CSRtoCOOTestI = CSRtoCOOTest<int>;
TEST_P(CSRtoCOOTestI, Result) { Run(); }

using CSRtoCOOTestL = CSRtoCOOTest<int64_t>;
TEST_P(CSRtoCOOTestL, Result) { Run(); }

const std::vector<CSRtoCOOInputs<int>> csrtocoo_inputs_32 = {
  {{0, 0, 2, 2}, {1, 1, 3}},
  {{0, 4, 8, 9}, {0, 0, 0, 0, 1, 1, 1, 1, 2, 3}},
};
const std::vector<CSRtoCOOInputs<int64_t>> csrtocoo_inputs_64 = {
  {{0, 0, 2, 2}, {1, 1, 3}},
  {{0, 4, 8, 9}, {0, 0, 0, 0, 1, 1, 1, 1, 2, 3}},
};

/*********************** CSR row normalize (max, L1) ***********************/

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
        csr_row_normalize_max<32, Type_f>(ex_scan, in_vals, nnz, n_rows, result,
                                          stream);
        break;
      case L1:
        csr_row_normalize_l1<32, Type_f>(ex_scan, in_vals, nnz, n_rows, result,
                                         stream);
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

/********************************* CSR sum *********************************/

template <typename Type_f, typename Index_>
struct CSRSumInputs {
  CSRMatrixVal<Type_f, Index_> matrix_a;
  CSRMatrixVal<Type_f, Index_> matrix_b;
  CSRMatrixVal<Type_f, Index_> matrix_verify;
};

template <typename Type_f, typename Index_>
class CSRSumTest
  : public ::testing::TestWithParam<CSRSumInputs<Type_f, Index_>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<CSRSumInputs<Type_f, Index_>>::GetParam();
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
    std::shared_ptr<deviceAllocator> alloc(
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

    Index_ nnz = csr_add_calc_inds<Type_f, 32>(
      ind_a, ind_ptr_a, values_a, nnz_a, ind_b, ind_ptr_b, values_b, nnz_b,
      n_rows, ind_result, alloc, stream);

    ASSERT_TRUE(nnz == nnz_result);
    ASSERT_TRUE(raft::devArrMatch<Index_>(ind_verify, ind_result, n_rows,
                                          raft::Compare<Index_>()));

    csr_add_finalize<Type_f, 32>(ind_a, ind_ptr_a, values_a, nnz_a, ind_b,
                                 ind_ptr_b, values_b, nnz_b, n_rows, ind_result,
                                 ind_ptr_result, values_result, stream);

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
  CSRSumInputs<Type_f, Index_> params;
  cudaStream_t stream;
  Index_ n_rows, nnz_a, nnz_b, nnz_result;
  Index_ *ind_a, *ind_b, *ind_verify, *ind_result, *ind_ptr_a, *ind_ptr_b,
    *ind_ptr_verify, *ind_ptr_result;
  Type_f *values_a, *values_b, *values_verify, *values_result;
};

using CSRSumTestF = CSRSumTest<float, int>;
TEST_P(CSRSumTestF, Result) { Run(); }

using CSRSumTestD = CSRSumTest<double, int>;
TEST_P(CSRSumTestD, Result) { Run(); }

const std::vector<CSRSumInputs<float, int>> csrsum_inputs_f = {
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
const std::vector<CSRSumInputs<double, int>> csrsum_inputs_d = {
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

/******************************** CSR row op ********************************/

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
  csr_row_op<Index_, 32>(
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

    csr_adj_graph_batched<Index_, 32>(row_ind, params.n_cols, nnz,
                                      params.n_rows, adj, result, stream);

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

/*********************** Weakly connected components ***********************/

template <typename Index_>
struct WeakCCInputs {
  Index_ N;
  std::vector<int8_t> mask;
  std::vector<CSRMatrix<Index_>> batches;
  std::vector<std::vector<Index_>> verify;
};

/** Wrapper to call weakcc because the enclosing function of a __device__
 *  lambda cannot have private ot protected access within the class. */
template <typename Index_>
void weak_cc_wrapper(Index_ *labels, const Index_ *row_ind,
                     const Index_ *row_ind_ptr, Index_ nnz, Index_ N,
                     Index_ startVertexId, Index_ batchSize, WeakCCState *state,
                     cudaStream_t stream, bool *mask) {
  weak_cc_batched<Index_>(
    labels, row_ind, row_ind_ptr, nnz, N, startVertexId, batchSize, state,
    stream, [mask, N] __device__(Index_ global_id) {
      return global_id < N ? __ldg((char *)mask + global_id) : 0;
    });
}

template <typename Index_>
class WeakCCTest : public ::testing::TestWithParam<WeakCCInputs<Index_>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<WeakCCInputs<Index_>>::GetParam();

    CUDA_CHECK(cudaStreamCreate(&stream));
    std::shared_ptr<deviceAllocator> alloc(
      new raft::mr::device::default_allocator);

    Index_ row_ind_size = params.batches[0].row_ind.size();
    Index_ row_ind_ptr_size = params.batches[0].row_ind_ptr.size();
    for (int i = 1; i < params.batches.size(); i++) {
      row_ind_size =
        max(row_ind_size, (Index_)params.batches[i].row_ind.size());
      row_ind_ptr_size =
        max(row_ind_ptr_size, (Index_)params.batches[i].row_ind_ptr.size());
    }

    raft::allocate(row_ind, row_ind_size);
    raft::allocate(row_ind_ptr, row_ind_ptr_size);
    raft::allocate(result, params.N, true);
    raft::allocate(verify, params.N);
    raft::allocate(mask, params.N);
    raft::allocate(m, 1);
  }

  void Run() {
    params = ::testing::TestWithParam<WeakCCInputs<Index_>>::GetParam();
    Index_ N = params.N;

    WeakCCState state(m);

    raft::update_device(mask, reinterpret_cast<bool *>(params.mask.data()), N,
                        stream);

    Index_ start_id = 0;
    for (int i = 0; i < params.batches.size(); i++) {
      Index_ batch_size = params.batches[i].row_ind.size() - 1;
      Index_ row_ind_size = params.batches[i].row_ind.size();
      Index_ row_ind_ptr_size = params.batches[i].row_ind_ptr.size();

      raft::update_device(row_ind, params.batches[i].row_ind.data(),
                          row_ind_size, stream);
      raft::update_device(row_ind_ptr, params.batches[i].row_ind_ptr.data(),
                          row_ind_ptr_size, stream);
      raft::update_device(verify, params.verify[i].data(), N, stream);

      weak_cc_wrapper<Index_>(result, row_ind, row_ind_ptr, row_ind_ptr_size, N,
                              start_id, batch_size, &state, stream, mask);

      cudaStreamSynchronize(stream);
      ASSERT_TRUE(
        raft::devArrMatch<Index_>(verify, result, N, raft::Compare<Index_>()));

      start_id += batch_size;
    }
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(row_ind));
    CUDA_CHECK(cudaFree(row_ind_ptr));
    CUDA_CHECK(cudaFree(verify));
    CUDA_CHECK(cudaFree(result));
    CUDA_CHECK(cudaFree(mask));
    CUDA_CHECK(cudaFree(m));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  WeakCCInputs<Index_> params;
  cudaStream_t stream;
  Index_ *row_ind, *row_ind_ptr, *result, *verify;
  bool *mask, *m;
};

using WeakCCTestI = WeakCCTest<int>;
TEST_P(WeakCCTestI, Result) { Run(); }

using WeakCCTestL = WeakCCTest<int64_t>;
TEST_P(WeakCCTestL, Result) { Run(); }

// Hand-designed corner cases for weakcc
const std::vector<WeakCCInputs<int>> weakcc_inputs_32 = {
  {6,
   {1, 0, 1, 1, 1, 0},
   {{{0, 2, 5, 7}, {0, 1, 0, 1, 4, 2, 5}},
    {{0, 2, 5, 7}, {3, 4, 1, 3, 4, 2, 5}}},
   {{1, 1, 3, 4, 5, 3}, {1, 4, 3, 4, 4, 3}}},
  {6,
   {1, 0, 1, 0, 1, 0},
   {{{0, 5, 8}, {0, 1, 2, 3, 4, 0, 1, 4}},
    {{0, 5, 8}, {0, 2, 3, 4, 5, 0, 2, 3}},
    {{0, 5, 8}, {0, 1, 2, 4, 5, 2, 4, 5}}},
   {{1, 1, 1, 1, 1, MAX32}, {1, MAX32, 1, 1, 1, 1}, {1, 1, 1, MAX32, 1, 1}}},
  {6,
   {1, 1, 1, 0, 1, 1},
   {{{0, 3, 6}, {0, 1, 2, 0, 1, 3}},
    {{0, 3, 6}, {0, 2, 4, 1, 3, 5}},
    {{0, 3, 6}, {2, 4, 5, 3, 4, 5}}},
   {{1, 1, 1, 1, 5, 6}, {1, 2, 1, 2, 1, 6}, {1, 2, 3, 3, 3, 3}}},
  {8,
   {1, 1, 1, 1, 0, 0, 1, 1},
   {{{0, 2, 5}, {0, 1, 0, 1, 2}},
    {{0, 3, 6}, {1, 2, 3, 2, 3, 4}},
    {{0, 2, 4}, {3, 4, 5, 6}},
    {{0, 2, 5}, {5, 6, 7, 6, 7}}},
   {{1, 1, 1, 4, MAX32, MAX32, 7, 8},
    {1, 2, 2, 2, 2, MAX32, 7, 8},
    {1, 2, 3, 4, 4, 7, 7, 8},
    {1, 2, 3, 4, MAX32, 7, 7, 7}}}};
const std::vector<WeakCCInputs<int64_t>> weakcc_inputs_64 = {
  {6,
   {1, 0, 1, 1, 1, 0},
   {{{0, 2, 5, 7}, {0, 1, 0, 1, 4, 2, 5}},
    {{0, 2, 5, 7}, {3, 4, 1, 3, 4, 2, 5}}},
   {{1, 1, 3, 4, 5, 3}, {1, 4, 3, 4, 4, 3}}},
  {6,
   {1, 0, 1, 0, 1, 0},
   {{{0, 5, 8}, {0, 1, 2, 3, 4, 0, 1, 4}},
    {{0, 5, 8}, {0, 2, 3, 4, 5, 0, 2, 3}},
    {{0, 5, 8}, {0, 1, 2, 4, 5, 2, 4, 5}}},
   {{1, 1, 1, 1, 1, MAX64}, {1, MAX64, 1, 1, 1, 1}, {1, 1, 1, MAX64, 1, 1}}},
  {6,
   {1, 1, 1, 0, 1, 1},
   {{{0, 3, 6}, {0, 1, 2, 0, 1, 3}},
    {{0, 3, 6}, {0, 2, 4, 1, 3, 5}},
    {{0, 3, 6}, {2, 4, 5, 3, 4, 5}}},
   {{1, 1, 1, 1, 5, 6}, {1, 2, 1, 2, 1, 6}, {1, 2, 3, 3, 3, 3}}},
  {8,
   {1, 1, 1, 1, 0, 0, 1, 1},
   {{{0, 2, 5}, {0, 1, 0, 1, 2}},
    {{0, 3, 6}, {1, 2, 3, 2, 3, 4}},
    {{0, 2, 4}, {3, 4, 5, 6}},
    {{0, 2, 5}, {5, 6, 7, 6, 7}}},
   {{1, 1, 1, 4, MAX64, MAX64, 7, 8},
    {1, 2, 2, 2, 2, MAX64, 7, 8},
    {1, 2, 3, 4, 4, 7, 7, 8},
    {1, 2, 3, 4, MAX64, 7, 7, 7}}}};

/**************************** Test instantiation ****************************/

INSTANTIATE_TEST_CASE_P(CSRTests, CSRtoCOOTestI,
                        ::testing::ValuesIn(csrtocoo_inputs_32));
INSTANTIATE_TEST_CASE_P(CSRTests, CSRtoCOOTestL,
                        ::testing::ValuesIn(csrtocoo_inputs_64));

INSTANTIATE_TEST_CASE_P(CSRTests, CSRRowNormalizeTestF,
                        ::testing::ValuesIn(csrnormalize_inputs_f));
INSTANTIATE_TEST_CASE_P(CSRTests, CSRRowNormalizeTestD,
                        ::testing::ValuesIn(csrnormalize_inputs_d));

INSTANTIATE_TEST_CASE_P(CSRTests, CSRSumTestF,
                        ::testing::ValuesIn(csrsum_inputs_f));
INSTANTIATE_TEST_CASE_P(CSRTests, CSRSumTestD,
                        ::testing::ValuesIn(csrsum_inputs_d));

INSTANTIATE_TEST_CASE_P(CSRTests, CSRRowOpTestF,
                        ::testing::ValuesIn(csrrowop_inputs_f));
INSTANTIATE_TEST_CASE_P(CSRTests, CSRRowOpTestD,
                        ::testing::ValuesIn(csrrowop_inputs_d));

INSTANTIATE_TEST_CASE_P(CSRTests, CSRAdjGraphTestI,
                        ::testing::ValuesIn(csradjgraph_inputs_i));
INSTANTIATE_TEST_CASE_P(CSRTests, CSRAdjGraphTestL,
                        ::testing::ValuesIn(csradjgraph_inputs_l));

INSTANTIATE_TEST_CASE_P(CSRTests, WeakCCTestI,
                        ::testing::ValuesIn(weakcc_inputs_32));
INSTANTIATE_TEST_CASE_P(CSRTests, WeakCCTestL,
                        ::testing::ValuesIn(weakcc_inputs_64));

}  // namespace Sparse
}  // namespace MLCommon
