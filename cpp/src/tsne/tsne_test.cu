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

#ifndef IF_DEBUG
#define IF_DEBUG 1
#endif

#include <gtest/gtest.h>
#include <score/scores.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include "tsne/digits.h"
#include "tsne/tsne.cu"

#include "cuda_utils.h"

using namespace MLCommon;
using namespace MLCommon::Score;
using namespace MLCommon::Distance;

using namespace ML;

class TSNETest : public ::testing::Test {
 protected:
  void basicTest() {
    cumlHandle handle;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float *X_d, *Y_d;
    MLCommon::allocate(X_d, n * p);
    MLCommon::allocate(Y_d, n * 2);
    MLCommon::updateDevice(X_d, digits.data(), n * p, stream);

    std::cout << "[>>>>]    Starting TSNE....\n";
    TSNE_fit(handle, X_d, Y_d, n, p, 2, 90);
    std::cout << "[>>>>]    Got embeddings!....\n";

    std::cout << "Updating host" << std::endl;
    float *embeddings_h = (float *)malloc(sizeof(float) * n * 2);
    cudaMemcpy(embeddings_h, Y_d, sizeof(float) * n * 2,
               cudaMemcpyDeviceToHost);

    int k = 0;
    float C_contiguous_embedding[n * 2];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < 2; j++)
        C_contiguous_embedding[k++] = embeddings_h[j * n + i];
    }

    float *YY;
    MLCommon::allocate(YY, n * 2);
    MLCommon::updateDevice(YY, C_contiguous_embedding, n * 2, stream);

    std::cout << "DONE!" << std::endl;

    CUDA_CHECK(cudaPeekAtLastError());

    // Test trustworthiness
    // euclidean test
    score = trustworthiness_score<float, EucUnexpandedL2>(
      X_d, YY, n, p, 2, 5, handle.getDeviceAllocator(), stream);

    std::cout << "SCORE: " << score << std::endl;

    free(embeddings_h);
    CUDA_CHECK(cudaFree(Y_d));
    CUDA_CHECK(cudaFree(YY));
    CUDA_CHECK(cudaFree(X_d));

    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  int n = 1797;
  int p = 64;
  double score;
};

typedef TSNETest TSNETestF;
TEST_F(TSNETestF, Result) { ASSERT_TRUE(0.98 < score); }