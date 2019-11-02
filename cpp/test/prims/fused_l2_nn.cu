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
#include <distance/fused_l2_nn.h>
#include <gtest/gtest.h>
#include <linalg/norm.h>
#include <random/rng.h>
#include "test_utils.h"

namespace MLCommon {
namespace Distance {

template <typename DataT>
__global__ void naiveKernel(int *min, DataT *minDist, DataT *x, DataT *y, int m,
                            int n, int k, int *workspace) {
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) return;
  DataT acc = DataT(0);
  for (int i = 0; i < k; ++i) {
    int xidx = i + midx * k;
    int yidx = i + nidx * k;
    auto diff = x[xidx] - y[yidx];
    acc += diff * diff;
  }
  while (atomicCAS(workspace, 0, 1) == 1)
    ;
  if (acc < minDist[midx]) {
    minDist[midx] = acc;
    min[midx] = nidx;
  }
  __threadfence();
  atomicCAS(workspace, 1, 0);
}

template <typename DataT>
void naive(int *min, DataT *minDist, DataT *x, DataT *y, int m, int n, int k,
           int *workspace, cudaStream_t stream) {
  static const dim3 TPB(32, 16, 1);
  dim3 nblks(ceildiv(m, (int)TPB.x), ceildiv(n, (int)TPB.y), 1);
  CUDA_CHECK(cudaMemsetAsync(workspace, 0, sizeof(int), stream));
  auto blks = ceildiv(m, 256);
  initKernel<DataT, int, int><<<blks, 256, 0, stream>>>(
    min, minDist, m, std::numeric_limits<DataT>::max());
  CUDA_CHECK(cudaGetLastError());
  naiveKernel<DataT>
    <<<nblks, TPB, 0, stream>>>(min, minDist, x, y, m, n, k, workspace);
  CUDA_CHECK(cudaGetLastError());
}

template <typename DataT>
struct Inputs {
  DataT tolerance;
  int m, n, k;
  unsigned long long int seed;
};

template <typename DataT>
class FusedL2NNTest : public ::testing::TestWithParam<Inputs<DataT>> {
 public:
  void SetUp() override {
    params = ::testing::TestWithParam<Inputs<DataT>>::GetParam();
    Random::Rng r(params.seed);
    int m = params.m;
    int n = params.n;
    int k = params.k;
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocate(x, m * k);
    allocate(y, n * k);
    allocate(xn, m);
    allocate(yn, n);
    allocate(minDist_ref, m * n);
    allocate(minDist, m * n);
    allocate(workspace, m);
    allocate(min, m);
    allocate(min_ref, m);
    r.uniform(x, m * k, DataT(-1.0), DataT(1.0), stream);
    r.uniform(y, n * k, DataT(-1.0), DataT(1.0), stream);
    naive(min_ref, minDist_ref, x, y, m, n, k, workspace, stream);
    LinAlg::rowNorm(xn, x, k, m, LinAlg::L2Norm, true, stream);
    LinAlg::rowNorm(yn, y, k, n, LinAlg::L2Norm, true, stream);
    fusedL2NN<DataT, int, int>(min, minDist, x, y, xn, yn, m, n, k, workspace,
                               stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(xn));
    CUDA_CHECK(cudaFree(yn));
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(min_ref));
    CUDA_CHECK(cudaFree(minDist_ref));
    CUDA_CHECK(cudaFree(min));
    CUDA_CHECK(cudaFree(minDist));
  }

 protected:
  Inputs<DataT> params;
  DataT *x, *y, *xn, *yn, *minDist_ref, *minDist;
  int *workspace, *min, *min_ref;
  cudaStream_t stream;
};

///@todo: enable testing of arbitrary values of 'k'
const std::vector<Inputs<float>> inputsf = {
  {0.001f, 32, 32, 32, 1234ULL},   {0.001f, 32, 64, 32, 1234ULL},
  {0.001f, 64, 32, 32, 1234ULL},   {0.001f, 64, 64, 32, 1234ULL},
  {0.001f, 128, 32, 32, 1234ULL},  {0.001f, 128, 64, 32, 1234ULL},
  {0.001f, 128, 128, 64, 1234ULL}, {0.001f, 64, 128, 128, 1234ULL},

  {0.001f, 32, 32, 34, 1234ULL},   {0.001f, 32, 64, 34, 1234ULL},
  {0.001f, 64, 32, 34, 1234ULL},   {0.001f, 64, 64, 34, 1234ULL},
  {0.001f, 128, 32, 34, 1234ULL},  {0.001f, 128, 64, 34, 1234ULL},
  {0.001f, 128, 128, 66, 1234ULL}, {0.001f, 64, 128, 130, 1234ULL},

  {0.001f, 32, 32, 33, 1234ULL},   {0.001f, 32, 64, 33, 1234ULL},
  {0.001f, 64, 32, 33, 1234ULL},   {0.001f, 64, 64, 33, 1234ULL},
  {0.001f, 128, 32, 33, 1234ULL},  {0.001f, 128, 64, 33, 1234ULL},
  {0.001f, 128, 128, 65, 1234ULL}, {0.001f, 64, 128, 129, 1234ULL},
};
typedef FusedL2NNTest<float> FusedL2NNTestF;
TEST_P(FusedL2NNTestF, Result) {
  ASSERT_TRUE(devArrMatch(minDist_ref, minDist, params.m,
                          CompareApprox<float>(params.tolerance)));
  ASSERT_TRUE(devArrMatch(min_ref, min, params.m, Compare<int>()));
}
INSTANTIATE_TEST_CASE_P(FusedL2NNTests, FusedL2NNTestF,
                        ::testing::ValuesIn(inputsf));

///@todo: enable double tests

}  // end namespace Distance
}  // end namespace MLCommon
