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
#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "datasets/digits.h"
#include "tsne/tsne.h"

#include "metrics/trustworthiness.h"
#include "cuML.hpp"
#include "distance/distance.h"
#include "knn/knn.hpp"
#include "common/cuml_allocator.hpp"
#include "common/device_buffer.hpp"


using namespace ML;
using namespace MLCommon;
using namespace ML::Metrics;
using namespace std;

using namespace MLCommon;
using namespace MLCommon::Distance;
using namespace MLCommon::Datasets::Digits;



class TSNETest : public ::testing::Test
{
 protected:
  void basicTest()
  {
    cumlHandle handle;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    assert(stream != NULL);


    float *X_d, *Y_d;
    MLCommon::allocate(X_d, n * p);
    assert(X_d != NULL);

    MLCommon::allocate(Y_d, n * 2);
    assert(Y_d != NULL);
    MLCommon::updateDevice(X_d, digits.data(), n * p, stream);


    // Test Barnes Hut
    printf("[Test] Start Barnes Hut\n");
    TSNE_fit(handle, X_d, Y_d, n, p, 2, 90);
    printf("[Test] Completed Barnes Hut\n");


    float *embeddings_h = (float *)malloc(sizeof(float) * n * 2);
    cudaMemcpy(embeddings_h, Y_d, sizeof(float) * n * 2,
               cudaMemcpyDeviceToHost);
    printf("[Test] Completed memcpy\n");


    int k = 0;
    float C_contiguous_embedding[n * 2];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < 2; j++)
        C_contiguous_embedding[k++] = embeddings_h[j * n + i];
    }
    printf("[Test] Completed making C-Contiguous\n");


    float *YY;
    MLCommon::allocate(YY, n * 2);
    assert(YY != NULL);

    MLCommon::updateDevice(YY, C_contiguous_embedding, n * 2, stream);
    CUDA_CHECK(cudaPeekAtLastError());

    printf("[Test] Completed memcpying into device memory\n");

    // Test trustworthiness
    // euclidean test


    printf("[Test] Starting Trustworthiness\n");
    assert(X_d != NULL);
    assert(YY != NULL);
    assert(n != 0);
    assert(p != 0);
    assert(handle.getDeviceAllocator() != NULL);
    assert(stream != NULL);

    score_bh = trustworthiness_score<float,EucUnexpandedL2Sqrt>(handle, X_d, YY, n, p, 2, 5);


    // Test Exact TSNE
    printf("[Test] Start Exact TSNE\n");
    TSNE_fit(handle, X_d, Y_d, n, p, 2, 90, 0.5, 0.0025, 50, 100, 1e-5, 12, 250,
             0.01, 200, 500, 1000, 1e-7, 0.5, 0.8, -1, true, false, false);
    printf("[Test] Completed Exact TSNE\n");

    cudaMemcpy(embeddings_h, Y_d, sizeof(float) * n * 2,
               cudaMemcpyDeviceToHost);
    printf("[Test] Completed memcpy\n");


    k = 0;
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < 2; j++)
        C_contiguous_embedding[k++] = embeddings_h[j * n + i];
    }
    printf("[Test] Completed making C-Contiguous\n");


    MLCommon::updateDevice(YY, C_contiguous_embedding, n * 2, stream);
    CUDA_CHECK(cudaPeekAtLastError());
    printf("[Test] Completed memcpying into device memory\n");


    // Test trustworthiness
    // euclidean test
    printf("[Test] Testing trustworthiness\n");
    assert(X_d != NULL);
    assert(YY != NULL);
    assert(n != 0);
    assert(p != 0);
    assert(handle.getDeviceAllocator() != NULL);
    assert(stream != NULL);

    score_exact = trustworthiness_score<float,EucUnexpandedL2Sqrt>(handle, X_d, YY, n, p, 2, 5);

    // Free space
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
  double score_bh;
  double score_exact;
};

typedef TSNETest TSNETestF;
TEST_F(TSNETestF, Result)
{
  if (score_bh < 0.98)
    printf("BH score = %f\n", score_bh);
  if (score_exact < 0.98)
    printf("Exact score = %f\n", score_exact);

  ASSERT_TRUE(0.98 < score_bh && 0.98 < score_exact);
}
