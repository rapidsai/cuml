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
#include <data_generators.h>
#include <gtest/gtest.h>
#include <test_utils.h>
#include "ml_utils.h"

namespace ML {

using namespace MLCommon;

void demo_class() {
  std::vector<float> data;
  std::vector<int> labels;
  int N = 1000;

  makeClassificationDataHost(data, labels, N, 5, 3, 2, 2);
  int num_ones = 0;
  for (int i = 0; i < N; i++) {
    num_ones += labels[i] == 1;
  }
  // Ensure we have a reasonable class split
  ASSERT_GE(num_ones, 4 * N / 10);
  ASSERT_LE(num_ones, 6 * N / 10);
}

void demo_reg() {
  std::vector<float> X, y, coeff;

  makeRegressionDataHost(X, y, coeff, 100, 10, 5, 0.0f);
  // myPrintHostVector("y", y.data(), y.size());
  // myPrintHostMatrix("X", X.data(), 100, 10, true, std::cout);

  ASSERT_EQ(coeff[5], 0.0);
}

TEST(generators, demo_reg) { demo_reg(); }

TEST(generators, demo_class) {
  demo_class();
  ASSERT_EQ(1, 1);
}

}  // namespace ML