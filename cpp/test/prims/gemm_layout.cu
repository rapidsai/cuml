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
#include "cuda_utils.h"
#include "linalg/gemm.h"
#include "random/rng.h"
#include "test_utils.h"

namespace MLCommon {
namespace LinAlg {

template <typename T>
struct GemmLayoutInputs {
  int M;
  int N;
  int K;
  bool zLayout;
  bool xLayout;
  bool yLayout;
  unsigned long long int seed;
};

// Reference GEMM implementation.
template <typename T>
__global__ void naiveGemm(T *Z, T *X, T *Y, int M, int N, int K,
                          bool isZColMajor, bool isXColMajor,
                          bool isYColMajor) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;

  for (int m = tidy; m < M; m += (blockDim.y * gridDim.y)) {
    for (int n = tidx; n < N; n += (blockDim.x * gridDim.x)) {
      T temp = T(0.0);
      for (int k = 0; k < K; k++) {
        int xIndex = isXColMajor ? m + k * M : m * K + k;
        int yIndex = isYColMajor ? k + n * K : k * N + n;
        temp += X[xIndex] * Y[yIndex];
      }
      int zIndex = isZColMajor ? m + n * M : m * N + n;
      Z[zIndex] = temp;
    }
  }
}

template <typename T>
class GemmLayoutTest : public ::testing::TestWithParam<GemmLayoutInputs<T>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<GemmLayoutInputs<T>>::GetParam();
    cudaStream_t stream;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUDA_CHECK(cudaStreamCreate(&stream));
    Random::Rng r(params.seed);

    // We compute Z = X * Y and compare against reference result
    // Dimensions of X : M x K
    // Dimensions of Y : K x N
    // Dimensions of Z : M x N

    T *X = NULL;  // Argument X
    T *Y = NULL;  // Argument Y

    size_t xElems = params.M * params.K;
    size_t yElems = params.K * params.N;
    size_t zElems = params.M * params.N;

    CUDA_CHECK(cudaMalloc(&X, xElems * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&Y, yElems * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&refZ, zElems * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&Z, zElems * sizeof(T)));

    r.uniform(X, xElems, T(-10.0), T(10.0), stream);
    r.uniform(Y, yElems, T(-10.0), T(10.0), stream);

    dim3 blocks(ceildiv<int>(params.M, 128), ceildiv<int>(params.N, 4), 1);
    dim3 threads(128, 4, 1);

    naiveGemm<<<blocks, threads>>>(refZ, X, Y, params.M, params.N, params.K,
                                   params.zLayout, params.xLayout,
                                   params.yLayout);

    gemm(handle, Z, X, Y, params.M, params.N, params.K, params.zLayout,
         params.xLayout, params.yLayout, stream);

    CUDA_CHECK(cudaFree(X));
    CUDA_CHECK(cudaFree(Y));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(handle));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(refZ));
    CUDA_CHECK(cudaFree(Z));
  }

 protected:
  GemmLayoutInputs<T> params;
  T *refZ = NULL;  // Reference result for comparison
  T *Z = NULL;     // Computed result
};

const std::vector<GemmLayoutInputs<float>> inputsf = {
  {80, 70, 80, true, true, true, 76433ULL},
  {80, 100, 40, true, true, false, 426646ULL},
  {20, 100, 20, true, false, true, 237703ULL},
  {100, 60, 30, true, false, false, 538004ULL},
  {50, 10, 60, false, true, true, 73012ULL},
  {90, 90, 30, false, true, false, 538147ULL},
  {30, 100, 10, false, false, true, 412352ULL},
  {40, 80, 100, false, false, false, 297941ULL}};

const std::vector<GemmLayoutInputs<double>> inputsd = {
  {10, 70, 40, true, true, true, 535648ULL},
  {30, 30, 30, true, true, false, 956681ULL},
  {70, 80, 50, true, false, true, 875083ULL},
  {80, 90, 70, true, false, false, 50744ULL},
  {90, 90, 30, false, true, true, 506321ULL},
  {40, 100, 70, false, true, false, 638418ULL},
  {80, 50, 30, false, false, true, 701529ULL},
  {50, 80, 60, false, false, false, 893038ULL}};

typedef GemmLayoutTest<float> GemmLayoutTestF;
TEST_P(GemmLayoutTestF, Result) {
  ASSERT_TRUE(
    devArrMatch(refZ, Z, params.M * params.N, CompareApprox<float>(1e-4)));
}

typedef GemmLayoutTest<double> GemmLayoutTestD;
TEST_P(GemmLayoutTestD, Result) {
  ASSERT_TRUE(
    devArrMatch(refZ, Z, params.M * params.N, CompareApprox<float>(1e-6)));
}

INSTANTIATE_TEST_CASE_P(GemmLayoutTests, GemmLayoutTestF,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(GemmLayoutTests, GemmLayoutTestD,
                        ::testing::ValuesIn(inputsd));

}  // end namespace LinAlg
}  // end namespace MLCommon
