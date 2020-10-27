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

#include <raft/cudart_utils.h>
#include <cuml/manifold/tsne.h>
#include <datasets/digits.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <common/device_buffer.hpp>
#include <raft/cuda_utils.cuh>
#include <cuml/common/cuml_allocator.hpp>
#include <cuml/common/logger.hpp>
#include <iostream>
#include <score/scores.cuh>
#include <vector>

using namespace MLCommon;
using namespace MLCommon::Score;
using namespace MLCommon::Distance;
using namespace MLCommon::Datasets::Digits;
using namespace ML;

class TSNETest : public ::testing::Test {
 protected:
  void basicTest() {
    raft::handle_t handle;

    // Allocate memory
    device_buffer<float> X_d(handle.get_device_allocator(), handle.get_stream(),
                             n * p);
    raft::update_device(X_d.data(), digits.data(), n * p, handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    device_buffer<float> Y_d(handle.get_device_allocator(), handle.get_stream(),
                             n * 2);

    // Test Barnes Hut
    TSNE_fit(handle, X_d.data(), Y_d.data(), n, p, 2, 90, 0.5, 0.0025, 50, 100,
             1e-5, 12, 250, 0.01, 200, 500, 1000, 1e-7, 0.5, 0.8, -1);

    // Move embeddings to host.
    // This can be used for printing if needed.
    float *embeddings_h = (float *)malloc(sizeof(float) * n * 2);
    assert(embeddings_h != NULL);

    raft::update_host(&embeddings_h[0], Y_d.data(), n * 2, handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    // Transpose the data
    int k = 0;
    float C_contiguous_embedding[n * 2];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < 2; j++)
        C_contiguous_embedding[k++] = embeddings_h[j * n + i];
    }

    // Move transposed embeddings back to device, as trustworthiness requires C contiguous format
    raft::update_device(Y_d.data(), C_contiguous_embedding, n * 2,
                        handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    // Test trustworthiness
    score_bh =
      trustworthiness_score<float,
                            ML::Distance::DistanceType::EucUnexpandedL2Sqrt>(
        X_d.data(), Y_d.data(), n, p, 2, 5, handle.get_device_allocator(),
        handle.get_stream());

    // Test Exact TSNE
    TSNE_fit(handle, X_d.data(), Y_d.data(), n, p, 2, 90, 0.5, 0.0025, 50, 100,
             1e-5, 12, 250, 0.01, 200, 500, 1000, 1e-7, 0.5, 0.8, -1,
             CUML_LEVEL_INFO, false, false);

    raft::update_host(&embeddings_h[0], Y_d.data(), n * 2, handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    // Move embeddings to host.
    // This can be used for printing if needed.
    k = 0;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < 2; j++)
        C_contiguous_embedding[k++] = embeddings_h[j * n + i];
    }

    // Move transposed embeddings back to device, as trustworthiness requires C contiguous format
    raft::update_device(Y_d.data(), C_contiguous_embedding, n * 2,
                        handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    // Test trustworthiness
    score_exact =
      trustworthiness_score<float,
                            ML::Distance::DistanceType::EucUnexpandedL2Sqrt>(
        X_d.data(), Y_d.data(), n, p, 2, 5, handle.get_device_allocator(),
        handle.get_stream());

    // Free space
    free(embeddings_h);
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  int n = 1797;
  int p = 64;
  double score_bh;
  double score_exact;
};

typedef TSNETest TSNETestF;
TEST_F(TSNETestF, Result) {
  if (score_bh < 0.98) CUML_LOG_DEBUG("BH score = %f", score_bh);
  if (score_exact < 0.98) CUML_LOG_DEBUG("Exact score = %f", score_exact);
  ASSERT_TRUE(0.98 < score_bh && 0.98 < score_exact);
}
