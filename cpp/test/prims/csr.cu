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

#include <raft/cudart_utils.h>
#include <raft/random/rng.cuh>
#include "test_utils.h"

#include <iostream>
#include <limits>
#include <vector>

constexpr int MAX32 = std::numeric_limits<int>::max();
constexpr int64_t MAX64 = std::numeric_limits<int64_t>::max();

namespace raft {
namespace sparse {

template <typename Index_>
struct CSRMatrix {
  std::vector<Index_> row_ind;
  std::vector<Index_> row_ind_ptr;
};

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
    std::shared_ptr<MLCommon::deviceAllocator> alloc(
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

INSTANTIATE_TEST_CASE_P(CSRTests, WeakCCTestI,
                        ::testing::ValuesIn(weakcc_inputs_32));
INSTANTIATE_TEST_CASE_P(CSRTests, WeakCCTestL,
                        ::testing::ValuesIn(weakcc_inputs_64));

}  // namespace sparse
}  // namespace raft
