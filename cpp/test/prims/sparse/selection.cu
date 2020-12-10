/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
#include <cuml/common/logger.hpp>
#include <sparse/selection.cuh>

namespace MLCommon {
namespace Sparse {
namespace Selection {

using namespace raft;

template <typename value_idx, typename value_t>
struct SparseSelectionInputs {
  value_idx n_rows;
  value_idx n_cols;

  std::vector<value_t> dists_h;

  std::vector<value_t> out_dists_ref_h;
  std::vector<value_idx> out_indices_ref_h;

  int k;

  bool select_min;
};

template <typename value_idx, typename value_t>
::std::ostream &operator<<(
  ::std::ostream &os, const SparseSelectionInputs<value_idx, value_t> &dims) {
  return os;
}

template <typename value_idx, typename value_t>
class SparseSelectionTest
  : public ::testing::TestWithParam<SparseSelectionInputs<value_idx, value_t>> {
 protected:
  void make_data() {
    std::vector<value_t> dists_h = params.dists_h;

    allocate(dists, n_rows * n_cols);
    update_device(dists, dists_h.data(), dists_h.size(), stream);

    allocate(inds, n_rows * n_cols);
    iota_fill(inds, n_rows, n_cols, stream);

    std::vector<value_t> out_dists_ref_h = params.out_dists_ref_h;
    std::vector<value_idx> out_indices_ref_h = params.out_indices_ref_h;

    allocate(out_indices_ref, out_indices_ref_h.size());
    allocate(out_dists_ref, out_dists_ref_h.size());

    update_device(out_indices_ref, out_indices_ref_h.data(),
                  out_indices_ref_h.size(), stream);
    update_device(out_dists_ref, out_dists_ref_h.data(), out_dists_ref_h.size(),
                  stream);

    allocate(out_dists, n_rows * k);
    allocate(out_indices, n_rows * k);
  }

  void SetUp() override {
    params = ::testing::TestWithParam<
      SparseSelectionInputs<value_idx, value_t>>::GetParam();
    std::shared_ptr<deviceAllocator> alloc(
      new raft::mr::device::default_allocator);
    CUDA_CHECK(cudaStreamCreate(&stream));

    ML::Logger::get().setLevel(CUML_LEVEL_INFO);

    n_rows = params.n_rows;
    n_cols = params.n_cols;
    k = params.k;

    make_data();

    select_k(dists, inds, n_rows, n_cols, out_dists, out_indices,
             params.select_min, k, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(dists));
    CUDA_CHECK(cudaFree(inds));
    CUDA_CHECK(cudaFree(out_indices));
    CUDA_CHECK(cudaFree(out_dists));
    CUDA_CHECK(cudaFree(out_indices_ref));
    CUDA_CHECK(cudaFree(out_dists_ref));

    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void compare() {
    ASSERT_TRUE(
      devArrMatch(out_dists_ref, out_dists, n_rows * k, Compare<value_t>()));
    ASSERT_TRUE(devArrMatch(out_indices_ref, out_indices, n_rows * k,
                            Compare<value_idx>()));
  }

 protected:
  cudaStream_t stream;

  int n_rows, n_cols, k;

  // input data
  value_t *dists;
  value_idx *inds;

  // output data
  value_idx *out_indices;
  value_t *out_dists;

  value_idx *out_indices_ref;
  value_t *out_dists_ref;

  SparseSelectionInputs<value_idx, value_t> params;
};

const std::vector<SparseSelectionInputs<int, float>> inputs_i32_f = {
  {5,
   5,
   {5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 5.0,
    1.0, 4.0, 5.0, 3.0, 2.0, 4.0, 1.0, 1.0, 3.0, 2.0, 5.0, 4.0},
   {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0,
    4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0},
   {4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 3, 0, 1, 4, 2, 4, 2, 1, 3, 0, 0, 2, 1, 4, 3},
   5,
   true}};
typedef SparseSelectionTest<int, float> SparseSelectionTestF;
TEST_P(SparseSelectionTestF, Result) { compare(); }
INSTANTIATE_TEST_CASE_P(SparseSelectionTest, SparseSelectionTestF,
                        ::testing::ValuesIn(inputs_i32_f));

};  // end namespace Selection
};  // end namespace Sparse
};  // end namespace MLCommon
