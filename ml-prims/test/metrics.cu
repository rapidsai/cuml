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

#include "metrics/metrics.h"
#include <gtest/gtest.h>
#include "random/rng.h"
#include "test_utils.h"

#include "device_allocator.h"

#include <iostream>

namespace MLCommon {
namespace Metrics {

class MetricsTest : public ::testing::Test {
protected:
  void SetUp() override {}

  void TearDown() override {}

};

TEST(MetricsTest, Result) {

    float y[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
    float y_hat[5] = {0.12, 0.22, 0.32, 0.42, 0.52};

    float *d_y;
    MLCommon::allocate(d_y, 5);

    float *d_y_hat;
    MLCommon::allocate(d_y_hat, 5);

    MLCommon::updateDevice(d_y_hat, y_hat, 5);
    MLCommon::updateDevice(d_y, y, 5);

    float result = MLCommon::Metrics::r_squared(d_y, d_y_hat, 5);
    ASSERT_TRUE(result == 0.98f);
}
}}

