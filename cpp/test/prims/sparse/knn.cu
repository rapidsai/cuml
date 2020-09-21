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

#include <common/cudart_utils.h>
#include <cusparse_v2.h>
#include <gtest/gtest.h>
#include <common/device_buffer.hpp>

#include <raft/sparse/cusparse_wrappers.h>
#include <test_utils.h>
#include <cuml/common/logger.hpp>
#include <sparse/knn.cuh>

namespace MLCommon {
namespace Sparse {
namespace Selection {

template <typename value_idx, typename value_t>
struct KNNInputs {};

template <typename value_idx, typename value_t>
::std::ostream &operator<<(::std::ostream &os,
                           const KNNInputs<value_idx, value_t> &dims) {
  return os;
}

template <typename value_idx, typename value_t>
class SparseKNNTest
  : public ::testing::TestWithParam<KNNInputs<value_idx, value_t>> {
 protected:
  void make_data() {
    std::vector<value_idx> indptr_h = {0, 2, 4, 6, 8};
    std::vector<value_idx> indices_h = {0, 4, 0, 3, 0, 2, 0, 8};
    std::vector<value_t> data_h = {0.0f, 1.0f, 5.0f, 6.0f,
                                   5.0f, 6.0f, 0.0f, 1.0f};

    allocate(indptr, 5);
    allocate(indices, 8);
    allocate(data, 8);

    updateDevice(indptr, indptr_h.data(), indptr_h.size(), stream);
    updateDevice(indices, indices_h.data(), indices_h.size(), stream);
    updateDevice(data, data_h.data(), data_h.size(), stream);

    std::vector<value_t> out_dists_ref_h = {1,  0,  0, 61, 25, 0,
                                            61, 25, 0, 1,  0,  0};
    std::vector<value_idx> out_indices_ref_h = {0, 3, 1, 1, 2, 3,
                                                2, 1, 3, 3, 2, 0};

    allocate(out_indices_ref, out_indices_ref_h.size());
    allocate(out_dists_ref, out_dists_ref_h.size());

    updateDevice(out_indices_ref, out_indices_ref_h.data(),
                 out_indices_ref_h.size(), stream);
    updateDevice(out_dists_ref, out_dists_ref_h.data(), out_dists_ref_h.size(),
                 stream);

    allocate(out_dists, 4 * k);
    allocate(out_indices, 4 * k);
  }

  void SetUp() override {
    params =
      ::testing::TestWithParam<KNNInputs<value_idx, value_t>>::GetParam();
    std::shared_ptr<deviceAllocator> alloc(
      new raft::mr::device::default_allocator);
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUSPARSE_CHECK(cusparseCreate(&cusparseHandle));

    k = 2;

    ML::Logger::get().setLevel(CUML_LEVEL_DEBUG);

    make_data();

    brute_force_knn<value_idx, value_t>(
      indptr, indices, data, 8, 4, 9, indptr, indices, data, 8, 4, 9,
      out_indices, out_dists, k, cusparseHandle, alloc, stream, 2, 2,
      ML::MetricType::METRIC_L2);


    std::cout << arr2Str(out_dists, 4 * k, "out_dists", stream) << std::endl;
    std::cout << arr2Str(out_indices, 4 * k, "out_indices", stream) << std::endl;

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
    ASSERT_TRUE(
      devArrMatch(out_dists_ref, out_dists, 4 * k, Compare<value_t>()));
    ASSERT_TRUE(
      devArrMatch(out_indices_ref, out_indices, 4 * k, Compare<value_idx>()));
  }

 protected:
  cudaStream_t stream;
  cusparseHandle_t cusparseHandle;

  // input data
  value_idx *indptr, *indices;
  value_t *data;

  // output data
  value_idx *out_indices;
  value_t *out_dists;

  int k;

  value_idx *out_indices_ref;
  value_t *out_dists_ref;

  KNNInputs<value_idx, value_t> params;
};

const std::vector<KNNInputs<int, float>> inputs_i32_f = {{}};
typedef SparseKNNTest<int, float> KNNTestF;
TEST_P(KNNTestF, Result) { compare(); }
INSTANTIATE_TEST_CASE_P(SparseKNNTest, KNNTestF,
                        ::testing::ValuesIn(inputs_i32_f));

};  // end namespace Selection
};  // end namespace Sparse
};  // end namespace MLCommon
