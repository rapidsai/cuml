/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
#include "distance/distance.h"
#include "random/rng.h"
#include "test_utils.h"


namespace MLCommon {
namespace Distance {

template <typename DataType>
__global__ void naiveDistanceAdjKernel(bool *dist, const DataType *x, const DataType *y,
                                       int m, int n, int k, DataType eps) {
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n)
    return;
  DataType acc = DataType(0);
  for (int i = 0; i < k; ++i) {
    auto diff = x[i + midx * k] - y[i + nidx * k];
    acc += diff * diff;
  }
  dist[midx * n + nidx] = acc <= eps;
}

template <typename DataType>
void naiveDistanceAdj(bool *dist, const DataType *x, const DataType *y, int m, int n,
                      int k, DataType eps) {
  static const dim3 TPB(16, 32, 1);
  dim3 nblks(ceildiv(m, (int)TPB.x), ceildiv(n, (int)TPB.y), 1);
  naiveDistanceAdjKernel<DataType><<<nblks, TPB>>>(dist, x, y, m, n, k, eps);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename DataType>
struct DistanceAdjInputs {
  DataType eps;
  int m, n, k;
  unsigned long long int seed;
};

template <typename DataType>
::std::ostream &operator<<(::std::ostream &os, const DistanceAdjInputs<DataType> &dims) {
  return os;
}

template <typename DataType>
class DistanceAdjTest : public ::testing::TestWithParam<DistanceAdjInputs<DataType>> {
public:
  void SetUp() override {
    params = ::testing::TestWithParam<DistanceAdjInputs<DataType>>::GetParam();
    Random::Rng r(params.seed);
    int m = params.m;
    int n = params.n;
    int k = params.k;
    allocate(x, m * k);
    allocate(y, n * k);
    allocate(dist_ref, m * n);
    allocate(dist, m * n);
    r.uniform(x, m * k, DataType(-1.0), DataType(1.0));
    r.uniform(y, n * k, DataType(-1.0), DataType(1.0));

    DataType threshold = params.eps;

    naiveDistanceAdj(dist_ref, x, y, m, n, k, threshold);
    char *workspace = nullptr;
    size_t worksize = getWorkspaceSize<EucExpandedL2, DataType, DataType, bool>(x, y, m, n, k);
    if (worksize != 0) {
      allocate(workspace, worksize);
    }

    typedef cutlass::Shape<8, 128, 128> OutputTile_t;

    auto fin_op = [threshold] __device__(DataType d_val, int g_d_idx) {
      (d_val <= threshold) ? (d_val = 1.f) : (d_val = 0.f);
      return d_val;
    };

    distance<EucExpandedL2, DataType, DataType, bool, OutputTile_t>(
      x, y, dist, m, n, k, workspace, worksize, fin_op);
    CUDA_CHECK(cudaFree(workspace));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(dist_ref));
    CUDA_CHECK(cudaFree(dist));
  }

protected:
  DistanceAdjInputs<DataType> params;
  DataType *x, *y;
  bool *dist_ref, *dist;
};


const std::vector<DistanceAdjInputs<float>> inputsf = {
  {0.01f, 1024, 1024, 32, 1234ULL},
  {0.1f, 1024, 1024, 32, 1234ULL},
  {1.0f, 1024, 1024, 32, 1234ULL},
  {10.0f, 1024, 1024, 32, 1234ULL}};
typedef DistanceAdjTest<float> DistanceAdjTestF;
TEST_P(DistanceAdjTestF, Result) {
  ASSERT_TRUE(devArrMatch(dist_ref, dist, params.m, params.n, Compare<bool>()));
}
INSTANTIATE_TEST_CASE_P(DistanceAdjTests, DistanceAdjTestF,
                        ::testing::ValuesIn(inputsf));


const std::vector<DistanceAdjInputs<double>> inputsd = {
  {0.01, 1024, 1024, 32, 1234ULL},
  {0.1, 1024, 1024, 32, 1234ULL},
  {1.0, 1024, 1024, 32, 1234ULL},
  {10.0, 1024, 1024, 32, 1234ULL}};
typedef DistanceAdjTest<double> DistanceAdjTestD;
TEST_P(DistanceAdjTestD, Result) {
  ASSERT_TRUE(devArrMatch(dist_ref, dist, params.m, params.n, Compare<bool>()));
}
INSTANTIATE_TEST_CASE_P(DistanceAdjTests, DistanceAdjTestD,
                        ::testing::ValuesIn(inputsd));

} // end namespace DistanceAdj
} // end namespace MLCommon
