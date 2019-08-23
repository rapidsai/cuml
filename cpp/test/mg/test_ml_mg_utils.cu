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

#include <cuda_utils.h>
#include <gtest/gtest.h>
#include <test_utils.h>
#include <iostream>
#include <vector>
#include "ml_mg_utils.h"

namespace ML {

using namespace MLCommon;

/**
 *
 * NOTE: Not exhaustively testing the kNN implementation since
 * we are using FAISS for this. Just testing API to verify the
 * knn.cu class is accepting inputs and providing outputs as
 * expected.
 */
template <typename T>
class ML_MG_UtilsTest : public ::testing::Test {
 protected:
  void basicTest() {
    // make test data on host
    std::vector<T> ptr_h = {1.0, 50.0, 51.0, 1.0, 50.0, 51.0, 1.0, 50.0, 51.0};
    ptr_h.resize(9);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    params = new float *[2];
    sizes = new int[2];

    expected_params = new float *[2];
    expected_sizes = new int[2];

    MLCommon::allocate(expected_params[0], 5);
    MLCommon::updateDevice(expected_params[0], ptr_h.data(), 5, stream);

    expected_sizes[0] = 5;
    expected_sizes[1] = 4;

    MLCommon::allocate(expected_params[1], 4);
    MLCommon::updateDevice(expected_params[1], ptr_h.data() + 5, 4, stream);

    int *devices = new int[2]{0, 1};

    chunk_to_device<float>(ptr_h.data(), 9, 1, devices, params, sizes, 2,
                           stream);

    cudaStreamDestroy(stream);
    delete devices;
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {
    CUDA_CHECK(cudaFree(params[0]));
    CUDA_CHECK(cudaFree(params[1]));
  }

 protected:
  float **params;
  int *sizes;

  float **expected_params;
  int *expected_sizes;
};

typedef ML_MG_UtilsTest<float> ChunkToDeviceTest;
TEST_F(ChunkToDeviceTest, Fit) {
  ASSERT_TRUE(sizes[0] == 5);
  ASSERT_TRUE(sizes[1] == 4);
  ASSERT_TRUE(devArrMatch(expected_params[0], params[0], 5, Compare<float>()));
  ASSERT_TRUE(devArrMatch(expected_params[1], params[1], 4, Compare<float>()));
}

}  // end namespace ML
