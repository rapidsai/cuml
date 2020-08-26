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
#include <gtest/gtest.h>
#include <sparse/distances.cuh>
#include "test_utils.h"
#include <cusparse_v2.h>
#include <raft/sparse/cusparse_wrappers.h>

namespace MLCommon {
namespace Sparse {
namespace Selection {


template <typename value_idx, typename value_t>
struct DistancesInputs {};

template <typename value_idx, typename value_t>
::std::ostream &operator<<(::std::ostream &os,
                           const DistancesInputs<value_idx, value_t> &dims) {
  return os;
}


template <typename value_idx, typename value_t>
class DistancesTest : public ::testing::TestWithParam<DistancesInputs<value_idx, value_t>> {
 protected:

	void make_data() {
		std::vector<value_idx> indptr_h = { 0, 2, 4, 6, 8 };
		std::vector<value_idx> indices_h = { 0, 1, 0, 1, 0, 1, 0, 1 };
		std::vector<value_t> data_h = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

		allocate(indptr, 5);
		allocate(indices, 8);
		allocate(data, 8);

		updateDevice(indptr, indptr_h.data(), 5, stream);
		updateDevice(indices, indices_h.data(), 8, stream);
		updateDevice(data, data_h.data(), 8, stream);

		allocate(csc_indptr, 3);
		allocate(csc_indices, 8);

		std::vector<value_idx> csc_indptr_h = { 0, 4, 8 };
		std::vector<value_idx> csc_indices_h = { 0, 1, 2, 3, 0, 1, 2, 3 };

		updateDevice(csc_indptr, csc_indptr_h.data(), 3, stream);
		updateDevice(csc_indices, csc_indices_h.data(), 8, stream);

		out_indptr_h = { 0, 2, 4, 6, 8 };
		out_indices_h = { 0, 1, 0, 1, 0, 1, 0, 1 };
		out_data_h = { 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f };
	}



  void SetUp() override {
    params = ::testing::TestWithParam<DistancesInputs<value_idx, value_t>>::GetParam();
    std::shared_ptr<deviceAllocator> alloc(new defaultDeviceAllocator);
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUSPARSE_CHECK(cusparseCreate(&cusparseHandle));


    make_data();

    std::cout << "Done making data. Running dists" << std::endl;

    Distance::distances_config_t<value_idx, value_t>
      dist_config;
    dist_config.index_nrows = 4;
    dist_config.index_ncols = 2;
    dist_config.index_nnz = 8;
    dist_config.csc_index_indptr = csc_indptr;
    dist_config.csc_index_indices = csc_indices;
    dist_config.csc_index_data = data;
    dist_config.search_nrows = 4;
    dist_config.search_ncols = 2;
    dist_config.search_nnz = 8;
    dist_config.csr_search_indptr = indptr;
    dist_config.csr_search_indices = indices;
    dist_config.csr_search_data = data;
    dist_config.handle = cusparseHandle;
    dist_config.allocator = alloc;
    dist_config.stream = stream;

    allocate(out_indptr, 5);

    Distance::ip_distances_t<value_idx, value_t> compute_dists(dist_config);
    value_idx out_nnz = compute_dists.get_nnz(out_indptr);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "New NNZ: " << out_nnz << std::endl;

    allocate(out_indices, out_nnz);
    allocate(out_data, out_nnz);

    std::cout << "Running compute()" << std::endl;

    compute_dists.compute(out_indptr, out_indices, out_data);

    std::cout << "Synchronizing" << std::endl;

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "Done running dists" << std::endl;

  }

  void TearDown() override {

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(indptr));
    CUDA_CHECK(cudaFree(indices));
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(csc_indptr));
    CUDA_CHECK(cudaFree(csc_indices));
    CUDA_CHECK(cudaFree(out_indptr));
    CUDA_CHECK(cudaFree(out_indices));
    CUDA_CHECK(cudaFree(out_data));
  }

  void compare() {

	  std::cout << arr2Str(out_indptr, 5, "out_indptr", stream) << std::endl;
	  std::cout << arr2Str(out_indices, 16, "out_indices", stream) << std::endl;
	  std::cout << arr2Str(out_data, 16, "out_data", stream) << std::endl;
  }

 protected:

  cudaStream_t stream;
  cusparseHandle_t cusparseHandle;

  // input data
  value_idx *indptr, *indices;
  value_t *data;

  // transposed input
  value_idx *csc_indptr, *csc_indices;

  // output data
  value_idx *out_indptr, *out_indices;
  value_t *out_data;

  std::vector<value_idx> out_indptr_h;
  std::vector<value_idx> out_indices_h;
  std::vector<value_t> out_data_h;

  DistancesInputs<value_idx, value_t> params;
};

const std::vector<DistancesInputs<int, float>> inputs_i32_f = {{}};
typedef DistancesTest<int, float> DistancesTestF;
TEST_P(DistancesTestF, Result) { compare(); }
INSTANTIATE_TEST_CASE_P(DistancesTests, DistancesTestF, ::testing::ValuesIn(inputs_i32_f));

};  // end namespace Selection
};  // end namespace Sparse
};  // end namespace MLCommon
