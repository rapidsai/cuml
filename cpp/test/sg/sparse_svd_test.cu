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
    auto d_alloc = handle.getDeviceAllocator();
    cudaStream_t stream = handle.getStream();

    // Allocate memory
    device_buffer<float> X_d(d_alloc, stream, n*p);

    // SparseSVD only accepts F-Contiguous data, but digits is C-Contiguous
    // Technically, no matter if the data is F or C contiguous, SparseSVD
    // should work regardless. The only difference is U, VT are swapped.
    float *__restrict X_T = (float*) malloc(sizeof(float) * n * p);
    ASSERT(X_T != NULL, "No more memory!");

    #define X_T(i,j)  X_T[(i) + (j)*n]
    #define X(i,j)    X[(i)*p + (j)]

    for (int i = 0; i < n; i++) {
      #pragma omp simd
      for (int j = 0; j < p; j++)
        X_T(i, j) = X(i, j);
    }
    MLCommon::updateDevice(X_d.data(), X_T, n*p, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Allocate U, S, VT
    device_buffer<float> U(d_alloc, stream, n*k);
    device_buffer<float> S(d_alloc, stream, k);
    device_buffer<float> VT(d_alloc, stream, p*k);

    SparseSVD(handle, X_d.data(), n, p, U.data(), S.data(), VT.data(), k);

    #undef X_T
    #undef X

    free(X_T);
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  const int n = 1797;
  const int p = 64;
  const int k = 5;
};


typedef SparseSVDTest SparseSVDTestF;
TEST_F(SparseSVDTestF, Result) {
  printf("Hi!\n");
}
