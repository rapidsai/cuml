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
#include <raft/cudart_utils.h>
#include <common/device_buffer.hpp>

#include <gtest/gtest.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <test_utils.h>
#include <sparse/distance.cuh>
#include <sparse/semiring.cuh>

namespace MLCommon {
namespace Sparse {
namespace Distance {

using namespace raft;

template <typename value_idx, typename value_t>
struct SparseDistanceSemiringInputs {
  value_idx n_cols;

  std::vector<value_idx> indptr_h;
  std::vector<value_idx> indices_h;
  std::vector<value_t> data_h;

  std::vector<value_t> out_dists_ref_h;
};

template <typename value_idx, typename value_t>
::std::ostream &operator<<(
  ::std::ostream &os,
  const SparseDistanceSemiringInputs<value_idx, value_t> &dims) {
  return os;
}

template <typename value_idx, typename value_t>
class SparseDistanceSemiringTest
  : public ::testing::TestWithParam<
      SparseDistanceSemiringInputs<value_idx, value_t>> {
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

    allocate(out_dists_ref, (indptr_h.size() - 1) * (indptr_h.size() - 1));

    update_device(out_dists_ref, out_dists_ref_h.data(), out_dists_ref_h.size(),
                  stream);
  }

  void SetUp() override {
    params = ::testing::TestWithParam<
      SparseDistanceSemiringInputs<value_idx, value_t>>::GetParam();
    std::shared_ptr<deviceAllocator> alloc(
      new raft::mr::device::default_allocator);
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUSPARSE_CHECK(cusparseCreate(&cusparseHandle));


    std::cout << "Making data" << std::endl;
    make_data();

    Distance::distances_config_t<value_idx, value_t> dist_config;
    dist_config.b_nrows = params.indptr_h.size() - 1;
    dist_config.b_ncols = params.n_cols;
    dist_config.b_nnz = params.indices_h.size();
    dist_config.b_indptr = indptr;
    dist_config.b_indices = indices;
    dist_config.b_data = data;
    dist_config.a_nrows = params.indptr_h.size() - 1;
    dist_config.a_ncols = params.n_cols;
    dist_config.a_nnz = params.indices_h.size();
    dist_config.a_indptr = indptr;
    dist_config.a_indices = indices;
    dist_config.a_data = data;
    dist_config.handle = cusparseHandle;
    dist_config.allocator = alloc;
    dist_config.stream = stream;

    int out_size = dist_config.a_nrows * dist_config.b_nrows;

    allocate(out_dists, out_size);

    ML::Logger::get().setLevel(CUML_LEVEL_DEBUG);

    std::cout << "Running dists" << std::endl;


    l1_distances_t<value_idx, value_t> l1_dists(dist_config);

    l1_dists.compute(out_dists);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "Done." << std::endl;
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(indptr));
    CUDA_CHECK(cudaFree(indices));
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(out_dists));
    CUDA_CHECK(cudaFree(out_dists_ref));
  }

  void compare() {
    ASSERT_TRUE(
      devArrMatch(out_dists_ref, out_dists, 16, CompareApprox<value_t>(1e-4)));
  }

 protected:
  cudaStream_t stream;
  cusparseHandle_t cusparseHandle;

  // input data
  value_idx *indptr, *indices;
  value_t *data;

  // output data
  value_t *out_dists, *out_dists_ref;

  SparseDistanceSemiringInputs<value_idx, value_t> params;
};

const std::vector<SparseDistanceSemiringInputs<int, float>> inputs_i32_f = {
  {4,
   {0, 1, 1, 2, 4},
   {3, 2, 0, 1},  // indices
   {0.99296, 0.42180, 0.11687, 0.305869},
   {
     // dense output
     0.0,
     0.99296,
     1.41476,
     1.415707,
     0.99296,
     0.0,
     0.42180,
     0.42274,
     1.41476,
     0.42180,
     0.0,
     0.84454,
     1.41570,
     0.42274,
     0.84454,
     0.0,
   }}};
typedef SparseDistanceSemiringTest<int, float> SparseDistanceSemiringTestF;
TEST_P(SparseDistanceSemiringTestF, Result) { compare(); }
INSTANTIATE_TEST_CASE_P(SparseDistanceSemiringTests,
                        SparseDistanceSemiringTestF,
                        ::testing::ValuesIn(inputs_i32_f));

};  // namespace Distance
};  // end namespace Sparse
};  // end namespace MLCommon
