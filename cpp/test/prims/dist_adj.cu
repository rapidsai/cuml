/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
#include <raft/cudart_utils.h>
#include <distance/distance.cuh>
#include <raft/cuda_utils.cuh>
#include <raft/random/rng.cuh>
#include "test_utils.h"

namespace MLCommon {
namespace Distance {

template <typename DataType>
__global__ void naiveDistanceAdjKernel(bool *dist, const DataType *x,
                                       const DataType *y, int m, int n, int k,
                                       DataType eps, bool isRowMajor) {
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) return;
  DataType acc = DataType(0);
  for (int i = 0; i < k; ++i) {
    int xidx = isRowMajor ? i + midx * k : i * m + midx;
    int yidx = isRowMajor ? i + nidx * k : i * n + nidx;
    auto diff = x[xidx] - y[yidx];
    acc += diff * diff;
  }
  int outidx = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx] = acc <= eps;
}

template <typename DataType>
void naiveDistanceAdj(bool *dist, const DataType *x, const DataType *y, int m,
                      int n, int k, DataType eps, bool isRowMajor) {
  static const dim3 TPB(16, 32, 1);
  dim3 nblks(raft::ceildiv(m, (int)TPB.x), raft::ceildiv(n, (int)TPB.y), 1);
  naiveDistanceAdjKernel<DataType>
    <<<nblks, TPB>>>(dist, x, y, m, n, k, eps, isRowMajor);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename DataType>
struct DistanceAdjInputs {
  DataType eps;
  int m, n, k;
  bool isRowMajor;
  unsigned long long int seed;
};

template <typename DataType>
::std::ostream &operator<<(::std::ostream &os,
                           const DistanceAdjInputs<DataType> &dims) {
  return os;
}

template <typename DataType>
class DistanceAdjTest
  : public ::testing::TestWithParam<DistanceAdjInputs<DataType>> {
 public:
  void SetUp() override {
    params = ::testing::TestWithParam<DistanceAdjInputs<DataType>>::GetParam();
    raft::random::Rng r(params.seed);
    int m = params.m;
    int n = params.n;
    int k = params.k;
    bool isRowMajor = params.isRowMajor;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    raft::allocate(x, m * k);
    raft::allocate(y, n * k);
    raft::allocate(dist_ref, m * n);
    raft::allocate(dist, m * n);
    r.uniform(x, m * k, DataType(-1.0), DataType(1.0), stream);
    r.uniform(y, n * k, DataType(-1.0), DataType(1.0), stream);

    DataType threshold = params.eps;

    naiveDistanceAdj(dist_ref, x, y, m, n, k, threshold, isRowMajor);
    char *workspace = nullptr;
    size_t worksize = getWorkspaceSize<raft::distance::DistanceType::L2Expanded,
                                       DataType, DataType, bool>(x, y, m, n, k);
    if (worksize != 0) {
      raft::allocate(workspace, worksize);
    }

    typedef cutlass::Shape<8, 128, 128> OutputTile_t;
    auto fin_op = [threshold] __device__(DataType d_val, int g_d_idx) {
      return d_val <= threshold;
    };
    distance<raft::distance::DistanceType::L2Expanded, DataType, DataType, bool,
             OutputTile_t>(x, y, dist, m, n, k, workspace, worksize, fin_op,
                           stream, isRowMajor);
    CUDA_CHECK(cudaStreamDestroy(stream));
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
  {0.01f, 1024, 1024, 32, true, 1234ULL},
  {0.1f, 1024, 1024, 32, true, 1234ULL},
  {1.0f, 1024, 1024, 32, true, 1234ULL},
  {10.0f, 1024, 1024, 32, true, 1234ULL},
  {0.01f, 1024, 1024, 32, false, 1234ULL},
  {0.1f, 1024, 1024, 32, false, 1234ULL},
  {1.0f, 1024, 1024, 32, false, 1234ULL},
  {10.0f, 1024, 1024, 32, false, 1234ULL},
};
typedef DistanceAdjTest<float> DistanceAdjTestF;
TEST_P(DistanceAdjTestF, Result) {
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(dist_ref, dist, m, n, raft::Compare<bool>()));
}
INSTANTIATE_TEST_CASE_P(DistanceAdjTests, DistanceAdjTestF,
                        ::testing::ValuesIn(inputsf));

const std::vector<DistanceAdjInputs<double>> inputsd = {
  {0.01, 1024, 1024, 32, true, 1234ULL},
  {0.1, 1024, 1024, 32, true, 1234ULL},
  {1.0, 1024, 1024, 32, true, 1234ULL},
  {10.0, 1024, 1024, 32, true, 1234ULL},
  {0.01, 1024, 1024, 32, false, 1234ULL},
  {0.1, 1024, 1024, 32, false, 1234ULL},
  {1.0, 1024, 1024, 32, false, 1234ULL},
  {10.0, 1024, 1024, 32, false, 1234ULL},
};
typedef DistanceAdjTest<double> DistanceAdjTestD;
TEST_P(DistanceAdjTestD, Result) {
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(dist_ref, dist, m, n, raft::Compare<bool>()));
}
INSTANTIATE_TEST_CASE_P(DistanceAdjTests, DistanceAdjTestD,
                        ::testing::ValuesIn(inputsd));

}  // namespace Distance
}  // end namespace MLCommon
