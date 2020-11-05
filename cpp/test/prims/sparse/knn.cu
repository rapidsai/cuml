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

#include <cusparse_v2.h>
#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <common/device_buffer.hpp>

#include <raft/sparse/cusparse_wrappers.h>
#include <test_utils.h>
#include <cuml/common/logger.hpp>
#include <sparse/knn.cuh>

namespace MLCommon {
namespace Sparse {
namespace Selection {

using namespace raft;

template <typename value_idx, typename value_t>
struct SparseKNNInputs {
  value_idx n_cols;

  std::vector<value_idx> indptr_h;
  std::vector<value_idx> indices_h;
  std::vector<value_t> data_h;

  std::vector<value_t> out_dists_ref_h;
  std::vector<value_idx> out_indices_ref_h;

  int k;

  int batch_size_index = 2;
  int batch_size_query = 2;

  ML::MetricType metric = ML::MetricType::METRIC_L2;
};

template <typename value_idx, typename value_t>
::std::ostream &operator<<(::std::ostream &os,
                           const SparseKNNInputs<value_idx, value_t> &dims) {
  return os;
}

template <typename value_idx, typename value_t>
class SparseKNNTest
  : public ::testing::TestWithParam<SparseKNNInputs<value_idx, value_t>> {
 protected:
  void make_data() {
    std::vector<value_idx> indptr_h = params.indptr_h;
    std::vector<value_idx> indices_h = params.indices_h;
    std::vector<value_t> data_h = params.data_h;

    allocate(indptr, indptr_h.size());
    allocate(indices, indices_h.size());
    allocate(data, data_h.size());

    update_device(indptr, indptr_h.data(), indptr_h.size(), stream);
    update_device(indices, indices_h.data(), indices_h.size(), stream);
    update_device(data, data_h.data(), data_h.size(), stream);

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
    params =
      ::testing::TestWithParam<SparseKNNInputs<value_idx, value_t>>::GetParam();
    std::shared_ptr<deviceAllocator> alloc(
      new raft::mr::device::default_allocator);
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUSPARSE_CHECK(cusparseCreate(&cusparseHandle));

    ML::Logger::get().setLevel(CUML_LEVEL_INFO);

    n_rows = params.indptr_h.size() - 1;
    nnz = params.indices_h.size();
    k = params.k;

    make_data();

    brute_force_knn<value_idx, value_t>(
      indptr, indices, data, nnz, n_rows, params.n_cols, indptr, indices, data,
      nnz, n_rows, params.n_cols, out_indices, out_dists, k, cusparseHandle,
      alloc, stream, params.batch_size_index, params.batch_size_query,
      params.metric);

    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(indptr));
    CUDA_CHECK(cudaFree(indices));
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(out_indices));
    CUDA_CHECK(cudaFree(out_dists));
    CUDA_CHECK(cudaFree(out_indices_ref));
    CUDA_CHECK(cudaFree(out_dists_ref));

    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void compare() {
    ASSERT_TRUE(devArrMatch(out_dists_ref, out_dists, n_rows * k,
                            CompareApprox<value_t>(1e-4)));
    ASSERT_TRUE(devArrMatch(out_indices_ref, out_indices, n_rows * k,
                            Compare<value_idx>()));
  }

 protected:
  cudaStream_t stream;
  cusparseHandle_t cusparseHandle;

  int n_rows, nnz, k;

  // input data
  value_idx *indptr, *indices;
  value_t *data;

  // output data
  value_idx *out_indices;
  value_t *out_dists;

  value_idx *out_indices_ref;
  value_t *out_dists_ref;

  SparseKNNInputs<value_idx, value_t> params;
};

const std::vector<SparseKNNInputs<int, float>> inputs_i32_f = {
  {9,                                                 // ncols
   {0, 2, 4, 6, 8},                                   // indptr
   {0, 4, 0, 3, 0, 2, 0, 8},                          // indices
   {0.0f, 1.0f, 5.0f, 6.0f, 5.0f, 6.0f, 0.0f, 1.0f},  // data
   {0, 1.41421, 0, 7.87401, 0, 7.87401, 0, 1.41421},  // dists
   {0, 3, 1, 0, 2, 0, 3, 0},                          // inds
   2,
   2,
   ML::MetricType::METRIC_L2}};
typedef SparseKNNTest<int, float> KNNTestF;
TEST_P(KNNTestF, Result) { compare(); }
INSTANTIATE_TEST_CASE_P(SparseKNNTest, KNNTestF,
                        ::testing::ValuesIn(inputs_i32_f));

};  // end namespace Selection
};  // end namespace Sparse
};  // end namespace MLCommon
