/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <common/device_buffer.hpp>
#include <distance/epsilon_neighborhood.cuh>
#include <random/make_blobs.h>
#include "test_utils.h"

namespace MLCommon {
namespace Distance {

template <typename T, typename IdxT>
struct EpsInputs {
  IdxT n_row, n_col, n_centers, n_batches;
  T eps;
};

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const EpsInputs<T, IdxT>& p) {
  return os;
}

template <typename T, typename IdxT>
class EpsNeighTest : public ::testing::TestWithParam<EpsInputs<T, IdxT>> {
 protected:
  void SetUp() override {
    param = ::testing::TestWithParam<EpsInputs<T, IdxT>>::GetParam();
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocate(data, param.n_row * param.n_col);
    allocate(labels, param.n_row);
    allocate(adj, param.n_row * param.n_row / param.n_batches);
    allocate(vd, param.n_row + 1, true);
    allocator.reset(new defaultDeviceAllocator);
    Random::make_blobs<T, IdxT>(
      data, labels, param.n_row, param.n_col, param.n_centers, allocator,
      stream, nullptr, nullptr, T(0.01), false);
    epsUnexpL2SqNeighborhood<T, IdxT>(
      adj, vd, data, data, param.n_row, param.n_row, param.n_col,
      param.eps * param.eps, stream);
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(labels));
    CUDA_CHECK(cudaFree(adj));
    CUDA_CHECK(cudaFree(vd));
  }

  EpsInputs<T, IdxT> param;
  cudaStream_t stream;
  T *data;
  bool *adj;
  IdxT *labels, *vd;
  std::shared_ptr<deviceAllocator> allocator;
};  // class EpsNeighTest

const std::vector<EpsInputs<float, int>> inputsfi = {
  {15000, 16, 5, 1, 2.f},
  {14000, 16, 5, 1, 2.f},
};
typedef EpsNeighTest<float, int> EpsNeighTestFI;
TEST_P(EpsNeighTestFI, Result) {
  ASSERT_TRUE(devArrMatch(param.n_row / param.n_centers, vd, param.n_row,
                          Compare<int>(), stream));
}
INSTANTIATE_TEST_CASE_P(EpsNeighTests, EpsNeighTestFI,
                        ::testing::ValuesIn(inputsfi));

};  // namespace Distance
};  // namespace MLCommon
