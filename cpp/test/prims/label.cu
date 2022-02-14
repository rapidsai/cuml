/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <label/classlabels.cuh>

#include "test_utils.h"
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/interruptible.hpp>
#include <rmm/device_uvector.hpp>

#include <iostream>
#include <vector>

namespace MLCommon {
namespace Label {

class LabelTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

typedef LabelTest MakeMonotonicTest;
TEST_F(MakeMonotonicTest, Result)
{
  cudaStream_t stream = 0;
  RAFT_CUDA_TRY(cudaStreamCreate(&stream));

  int m = 12;

  rmm::device_uvector<float> data(m, stream);
  rmm::device_uvector<float> actual(m, stream);
  rmm::device_uvector<float> expected(m, stream);
  RAFT_CUDA_TRY(cudaMemset(data.data(), 0, data.size() * sizeof(float)));
  RAFT_CUDA_TRY(cudaMemset(actual.data(), 0, actual.size() * sizeof(float)));
  RAFT_CUDA_TRY(cudaMemset(expected.data(), 0, expected.size() * sizeof(float)));

  float* data_h = new float[m]{1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 8.0, 7.0, 8.0, 8.0, 25.0, 80.0};

  float* expected_h = new float[m]{1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 5.0, 4.0, 5.0, 5.0, 6.0, 7.0};

  raft::update_device(data.data(), data_h, m, stream);
  raft::update_device(expected.data(), expected_h, m, stream);

  make_monotonic(actual.data(), data.data(), m, stream);

  raft::interruptible::synchronize(stream);

  ASSERT_TRUE(devArrMatch(actual.data(), expected.data(), m, raft::Compare<bool>(), stream));

  RAFT_CUDA_TRY(cudaStreamDestroy(stream));

  delete data_h;
  delete expected_h;
}

TEST(LabelTest, ClassLabels)
{
  cudaStream_t stream = 0;
  RAFT_CUDA_TRY(cudaStreamCreate(&stream));

  int n_rows = 6;
  rmm::device_uvector<float> y_d(n_rows, stream);

  float y_h[] = {2, -1, 1, 2, 1, 1};
  raft::update_device(y_d.data(), y_h, n_rows, stream);

  rmm::device_uvector<float> y_unique_d(n_rows, stream);
  auto n_classes = getUniqueLabels(y_d.data(), n_rows, y_unique_d.data(), stream);
  y_unique_d.resize(n_classes, stream);

  ASSERT_EQ(n_classes, 3);

  float y_unique_exp[] = {-1, 1, 2};
  EXPECT_TRUE(
    devArrMatchHost(y_unique_exp, y_unique_d.data(), n_classes, raft::Compare<float>(), stream));

  rmm::device_uvector<float> y_relabeled_d(n_rows, stream);

  getOvrLabels(y_d.data(), n_rows, y_unique_d.data(), n_classes, y_relabeled_d.data(), 2, stream);

  float y_relabeled_exp[] = {1, -1, -1, 1, -1, -1};
  EXPECT_TRUE(
    devArrMatchHost(y_relabeled_exp, y_relabeled_d.data(), n_rows, raft::Compare<float>(), stream));

  RAFT_CUDA_TRY(cudaStreamDestroy(stream));
}
};  // namespace Label
};  // namespace MLCommon
