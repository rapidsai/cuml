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

#include "score/score.h"
#include <gtest/gtest.h>
#include "random/rng.h"
#include "test_utils.h"

#include "device_allocator.h"

#include <iostream>

namespace MLCommon {
namespace Score {

class ScoreTest : public ::testing::Test {
protected:
  void SetUp() override {}

  void TearDown() override {}

};

typedef ScoreTest ScoreTestHighScore;
TEST(ScoreTestHighScore, Result) {

    float y[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
    float y_hat[5] = {0.12, 0.22, 0.32, 0.42, 0.52};

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    float *d_y;
    MLCommon::allocate(d_y, 5);

    float *d_y_hat;
    MLCommon::allocate(d_y_hat, 5);

    MLCommon::updateDevice(d_y_hat, y_hat, 5, stream);
    MLCommon::updateDevice(d_y, y, 5, stream);

    float result = MLCommon::Score::r2_score(d_y, d_y_hat, 5, stream);
    ASSERT_TRUE(result == 0.98f);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

typedef ScoreTest ScoreTestLowScore;
TEST(ScoreTestLowScore, Result) {

    float y[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
    float y_hat[5] = {0.012, 0.022, 0.032, 0.042, 0.052};

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    float *d_y;
    MLCommon::allocate(d_y, 5);

    float *d_y_hat;
    MLCommon::allocate(d_y_hat, 5);

    MLCommon::updateDevice(d_y_hat, y_hat, 5, stream);
    MLCommon::updateDevice(d_y, y, 5, stream);

    float result = MLCommon::Score::r2_score(d_y, d_y_hat, 5, stream);

    std::cout << "Result: " << result - -3.4012f << std::endl;
    ASSERT_TRUE(result - -3.4012f < 0.00001);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

}}

