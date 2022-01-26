/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include "test_utils.h"
#include <distance/epsilon_neighborhood.cuh>
#include <gtest/gtest.h>
#include <memory>
#include <raft/cudart_utils.h>
#include <random/make_blobs.cuh>
#include <rmm/device_uvector.hpp>

namespace MLCommon {
namespace Distance {

template <typename T, typename IdxT>
struct EpsInputs {
  IdxT n_row, n_col, n_centers, n_batches;
  T eps;
};

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const EpsInputs<T, IdxT>& p)
{
  return os;
}

template <typename T, typename IdxT>
class EpsNeighTest : public ::testing::TestWithParam<EpsInputs<T, IdxT>> {
 protected:
  EpsNeighTest() : data(0, stream), adj(0, stream), labels(0, stream), vd(0, stream) {}

  void SetUp() override
  {
    param = ::testing::TestWithParam<EpsInputs<T, IdxT>>::GetParam();
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));
    data.resize(param.n_row * param.n_col, stream);
    labels.resize(param.n_row, stream);
    batchSize = param.n_row / param.n_batches;
    adj.resize(param.n_row * batchSize, stream);
    vd.resize(batchSize + 1, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(vd.data(), 0, vd.size() * sizeof(IdxT), stream));
    Random::make_blobs<T, IdxT>(data.data(),
                                labels.data(),
                                param.n_row,
                                param.n_col,
                                param.n_centers,
                                stream,
                                true,
                                nullptr,
                                nullptr,
                                T(0.01),
                                false);
  }

  void TearDown() override { RAFT_CUDA_TRY(cudaStreamDestroy(stream)); }

  EpsInputs<T, IdxT> param;
  cudaStream_t stream = 0;
  rmm::device_uvector<T> data;
  rmm::device_uvector<bool> adj;
  rmm::device_uvector<IdxT> labels, vd;
  IdxT batchSize;
};  // class EpsNeighTest

const std::vector<EpsInputs<float, int>> inputsfi = {
  {15000, 16, 5, 1, 2.f},
  {14000, 16, 5, 1, 2.f},
  {15000, 17, 5, 1, 2.f},
  {14000, 17, 5, 1, 2.f},
  {15000, 18, 5, 1, 2.f},
  {14000, 18, 5, 1, 2.f},
  {15000, 32, 5, 1, 2.f},
  {14000, 32, 5, 1, 2.f},
  {20000, 10000, 10, 1, 2.f},
  {20000, 10000, 10, 2, 2.f},
};
typedef EpsNeighTest<float, int> EpsNeighTestFI;
TEST_P(EpsNeighTestFI, Result)
{
  for (int i = 0; i < param.n_batches; ++i) {
    RAFT_CUDA_TRY(cudaMemsetAsync(adj.data(), 0, sizeof(bool) * param.n_row * batchSize, stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(vd.data(), 0, sizeof(int) * (batchSize + 1), stream));
    epsUnexpL2SqNeighborhood<float, int>(adj.data(),
                                         vd.data(),
                                         data.data(),
                                         data.data() + (i * batchSize * param.n_col),
                                         param.n_row,
                                         batchSize,
                                         param.n_col,
                                         param.eps * param.eps,
                                         stream);
    ASSERT_TRUE(raft::devArrMatch(
      param.n_row / param.n_centers, vd.data(), batchSize, raft::Compare<int>(), stream));
  }
}
INSTANTIATE_TEST_CASE_P(EpsNeighTests, EpsNeighTestFI, ::testing::ValuesIn(inputsfi));

};  // namespace Distance
};  // namespace MLCommon
