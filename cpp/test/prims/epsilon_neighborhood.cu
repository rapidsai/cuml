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
#include <random/make_blobs.h>
#include <common/device_buffer.hpp>
#include <distance/epsilon_neighborhood.cuh>
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
    batchSize = param.n_row / param.n_batches;
    allocate(adj, param.n_row * batchSize);
    allocate(vd, batchSize + 1, true);
    allocator.reset(new defaultDeviceAllocator);
    Random::make_blobs<T, IdxT>(data, labels, param.n_row, param.n_col,
                                param.n_centers, allocator, stream, nullptr,
                                nullptr, T(0.01), false);
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
  T* data;
  bool* adj;
  IdxT *labels, *vd;
  IdxT batchSize;
  std::shared_ptr<deviceAllocator> allocator;
};  // class EpsNeighTest

const std::vector<EpsInputs<float, int>> inputsfi = {
  {15000, 16, 5, 1, 2.f},     {14000, 16, 5, 1, 2.f},
  {15000, 17, 5, 1, 2.f},     {14000, 17, 5, 1, 2.f},
  {15000, 18, 5, 1, 2.f},     {14000, 18, 5, 1, 2.f},
  {15000, 32, 5, 1, 2.f},     {14000, 32, 5, 1, 2.f},
  {20000, 10000, 10, 1, 2.f}, {20000, 10000, 10, 2, 2.f},
};
typedef EpsNeighTest<float, int> EpsNeighTestFI;
TEST_P(EpsNeighTestFI, Result) {
  for (int i = 0; i < param.n_batches; ++i) {
    CUDA_CHECK(
      cudaMemsetAsync(adj, 0, sizeof(bool) * param.n_row * batchSize, stream));
    CUDA_CHECK(cudaMemsetAsync(vd, 0, sizeof(int) * (batchSize + 1), stream));
    epsUnexpL2SqNeighborhood<float, int>(
      adj, vd, data, data + (i * batchSize * param.n_col), param.n_row,
      batchSize, param.n_col, param.eps * param.eps, stream);
    ASSERT_TRUE(devArrMatch(param.n_row / param.n_centers, vd, batchSize,
                            Compare<int>(), stream));
  }
}
INSTANTIATE_TEST_CASE_P(EpsNeighTests, EpsNeighTestFI,
                        ::testing::ValuesIn(inputsfi));

};  // namespace Distance
};  // namespace MLCommon
