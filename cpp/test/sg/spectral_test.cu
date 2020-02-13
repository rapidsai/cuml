/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cuml/cluster/spectral.hpp>
#include <cuml/cuml.hpp>

#include "random/rng.h"

#include <cuda_utils.h>
#include <gtest/gtest.h>
#include <test_utils.h>
#include <iostream>
#include <vector>

namespace ML {

using namespace MLCommon;

template <typename T>
class SpectralTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}

 protected:
};

typedef SpectralTest<float> TestSpectralClustering;
TEST_F(TestSpectralClustering, Fit) {
  int n = 500;
  int d = 30;
  int k = 3;

  float *X;
  cumlHandle handle;
  MLCommon::allocate(X, n * d);

  Random::Rng r(150, MLCommon::Random::GenTaps);
  r.uniform(X, n * d, -1.0f, 1.0f, handle.getStream());

  int *out;
  MLCommon::allocate(out, n, true);

  ML::Spectral::fit_clusters(handle, X, n, d, k, 10, 1e-3f, out);
  CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(X));
}

typedef SpectralTest<float> TestSpectralEmbedding;
TEST_F(TestSpectralEmbedding, Fit) {
  int n = 500;
  int d = 30;
  int k = 3;

  float *X;
  cumlHandle handle;
  MLCommon::allocate(X, n * d);

  Random::Rng r(150, MLCommon::Random::GenTaps);
  r.uniform(X, n * d, -1.0f, 1.0f, handle.getStream());

  float *out;
  MLCommon::allocate(out, n * 2, true);

  ML::Spectral::fit_embedding(handle, X, n, d, k, 2, out);
  CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(X));
}

}  // end namespace ML
