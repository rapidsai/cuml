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
#include <score/scores.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include "datasets/digits.h"
#include "tsvd/sparse_svd.cu"

#include "cuda_utils.h"

#include <cuml/common/cuml_allocator.hpp>
#include "common/device_buffer.hpp"

using namespace MLCommon;
using namespace MLCommon::Distance;
using namespace MLCommon::Datasets::Digits;
using namespace ML;

class SparseSVDTest : public ::testing::Test {
 protected:
  void basicTest() {
    cumlHandle handle;

    // Allocate memory
    device_buffer<float> X_d(handle.getDeviceAllocator(), handle.getStream(),
                             n * p);
    MLCommon::updateDevice(X_d.data(), digits.data(), n * p,
                           handle.getStream());
    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  int n = 1797;
  int p = 64;
};

typedef SparseSVDTest SparseSVDTestF;
TEST_F(SparseSVDTestF, Result) {
  printf("Hi!\n");
}
