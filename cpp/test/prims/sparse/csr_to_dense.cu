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
#include <sparse/csr.cuh>

namespace MLCommon {
namespace Sparse {

using namespace raft;

template <typename value_idx, typename value_t>
struct CSRToDenseInputs {
  value_idx nrows;
  value_idx ncols;

  std::vector<value_idx> indptr_h;
  std::vector<value_idx> indices_h;
  std::vector<value_t> data_h;

  std::vector<value_t> out_ref_h;
};

template <typename value_idx, typename value_t>
::std::ostream &operator<<(::std::ostream &os,
                           const CSRToDenseInputs<value_idx, value_t> &dims) {
  return os;
}

template <typename value_idx, typename value_t>
class CSRToDenseTest
  : public ::testing::TestWithParam<CSRToDenseInputs<value_idx, value_t>> {
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

    std::vector<value_t> out_ref_h = params.out_ref_h;

    allocate(out_ref, out_ref_h.size());

    update_device(out_ref, out_ref_h.data(), out_ref_h.size(), stream);

    allocate(out, out_ref_h.size());
  }

  void SetUp() override {
    params = ::testing::TestWithParam<
      CSRToDenseInputs<value_idx, value_t>>::GetParam();
    std::shared_ptr<deviceAllocator> alloc(
      new raft::mr::device::default_allocator);
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUSPARSE_CHECK(cusparseCreate(&handle));

    make_data();

    ML::Logger::get().setLevel(CUML_LEVEL_INFO);

    csr_to_dense(handle, params.nrows, params.ncols, indptr, indices, data,
                 params.nrows, out, stream, true);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUSPARSE_CHECK(cusparseDestroy(handle));
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(indptr));
    CUDA_CHECK(cudaFree(indices));
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(out_ref));
  }

  void compare() {
    ASSERT_TRUE(
      devArrMatch(out, out_ref, params.out_ref_h.size(), Compare<value_t>()));
  }

 protected:
  cudaStream_t stream;
  cusparseHandle_t handle;

  // input data
  value_idx *indptr, *indices;
  value_t *data;

  // output data
  value_t *out;

  // expected output data
  value_t *out_ref;

  CSRToDenseInputs<value_idx, value_t> params;
};

const std::vector<CSRToDenseInputs<int, float>> inputs_i32_f = {
  {4,
   4,
   {0, 2, 4, 6, 8},
   {0, 1, 2, 3, 0, 1, 2, 3},  // indices
   {1.0f, 3.0f, 1.0f, 5.0f, 50.0f, 28.0f, 16.0f, 2.0f},
   {1.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 5.0f, 50.0f, 28.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 16.0f, 2.0f}},
};
typedef CSRToDenseTest<int, float> CSRToDenseTestF;
TEST_P(CSRToDenseTestF, Result) { compare(); }
INSTANTIATE_TEST_CASE_P(CSRToDenseTest, CSRToDenseTestF,
                        ::testing::ValuesIn(inputs_i32_f));

};  // end namespace Sparse
};  // end namespace MLCommon
