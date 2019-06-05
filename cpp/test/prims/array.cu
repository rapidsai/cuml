/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "array/array.h"
#include "array/classlabels.h"

#include <cuda_utils.h>
#include "common/cuml_allocator.hpp"
#include "test_utils.h"

#include <iostream>
#include <vector>

namespace MLCommon {
namespace Array {

class ArrayTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

typedef ArrayTest MakeMonotonicTest;
TEST_F(MakeMonotonicTest, Result) {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  int m = 12;

  float *data, *actual, *expected;

  allocate(data, m, true);
  allocate(actual, m, true);
  allocate(expected, m, true);

  float *data_h =
    new float[m]{1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 8.0, 7.0, 8.0, 8.0, 25.0, 80.0};

  float *expected_h =
    new float[m]{1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 5.0, 4.0, 5.0, 5.0, 6.0, 7.0};

  updateDevice(data, data_h, m, stream);
  updateDevice(expected, expected_h, m, stream);

  make_monotonic(actual, data, m, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  ASSERT_TRUE(devArrMatch(actual, expected, m, Compare<bool>(), stream));

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(data));
  CUDA_CHECK(cudaFree(actual));

  delete data_h;
  delete expected_h;
}

TEST(Arraytest, ClassLabels) {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  std::shared_ptr<deviceAllocator> allocator(new defaultDeviceAllocator);

  int n_rows = 6;
  float *y_d;
  allocate(y_d, n_rows);

  float y_h[] = {2, -1, 1, 2, 1, 1};
  updateDevice(y_d, y_h, n_rows, stream);

  int n_classes;
  float *y_unique_d;
  getUniqueLabels(y_d, n_rows, &y_unique_d, &n_classes, stream, allocator);

  ASSERT_EQ(n_classes, 3);

  float y_unique_exp[] = {-1, 1, 2};
  EXPECT_TRUE(devArrMatchHost(y_unique_exp, y_unique_d, n_classes,
                              Compare<float>(), stream));

  float *y_relabeled_d;
  allocate(y_relabeled_d, n_rows);

  getOvrLabels(y_d, n_rows, y_unique_d, n_classes, y_relabeled_d, 2, stream);

  float y_relabeled_exp[] = {1, -1, -1, 1, -1, -1};
  EXPECT_TRUE(devArrMatchHost(y_relabeled_exp, y_relabeled_d, n_rows,
                              Compare<float>(), stream));

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(y_d));
  CUDA_CHECK(cudaFree(y_unique_d));
  CUDA_CHECK(cudaFree(y_relabeled_d));
}
};  // namespace Array
};  // namespace MLCommon
