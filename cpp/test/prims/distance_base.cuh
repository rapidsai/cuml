/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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
__global__ void naiveDistanceKernel(DataType* dist,
                                    const DataType* x,
                                    const DataType* y,
                                    int m,
                                    int n,
                                    int k,
                                    raft::distance::DistanceType type,
                                    bool isRowMajor)
{
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) return;
  DataType acc = DataType(0);
  for (int i = 0; i < k; ++i) {
    int xidx  = isRowMajor ? i + midx * k : i * m + midx;
    int yidx  = isRowMajor ? i + nidx * k : i * n + nidx;
    auto diff = x[xidx] - y[yidx];
    acc += diff * diff;
  }
  if (type == raft::distance::DistanceType::L2SqrtExpanded ||
      type == raft::distance::DistanceType::L2SqrtUnexpanded)
    acc = raft::mySqrt(acc);
  int outidx   = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx] = acc;
}

template <typename DataType>
__global__ void naiveL1DistanceKernel(
  DataType* dist, const DataType* x, const DataType* y, int m, int n, int k, bool isRowMajor)
{
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) { return; }

  DataType acc = DataType(0);
  for (int i = 0; i < k; ++i) {
    int xidx  = isRowMajor ? i + midx * k : i * m + midx;
    int yidx  = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a    = x[xidx];
    auto b    = y[yidx];
    auto diff = (a > b) ? (a - b) : (b - a);
    acc += diff;
  }

  int outidx   = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx] = acc;
}

template <typename DataType>
__global__ void naiveCosineDistanceKernel(
  DataType* dist, const DataType* x, const DataType* y, int m, int n, int k, bool isRowMajor)
{
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) { return; }

  DataType acc_a  = DataType(0);
  DataType acc_b  = DataType(0);
  DataType acc_ab = DataType(0);

  for (int i = 0; i < k; ++i) {
    int xidx = isRowMajor ? i + midx * k : i * m + midx;
    int yidx = isRowMajor ? i + nidx * k : i * n + nidx;
    auto a   = x[xidx];
    auto b   = y[yidx];
    acc_a += a * a;
    acc_b += b * b;
    acc_ab += a * b;
  }

  int outidx = isRowMajor ? midx * n + nidx : midx + m * nidx;

  // Use 1.0 - (cosine similarity) to calc the distance
  dist[outidx] = (DataType)1.0 - acc_ab / (raft::mySqrt(acc_a) * raft::mySqrt(acc_b));
}

template <typename DataType>
void naiveDistance(DataType* dist,
                   const DataType* x,
                   const DataType* y,
                   int m,
                   int n,
                   int k,
                   raft::distance::DistanceType type,
                   bool isRowMajor)
{
  static const dim3 TPB(16, 32, 1);
  dim3 nblks(raft::ceildiv(m, (int)TPB.x), raft::ceildiv(n, (int)TPB.y), 1);

  switch (type) {
    case raft::distance::DistanceType::L1:
      naiveL1DistanceKernel<DataType><<<nblks, TPB>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    case raft::distance::DistanceType::L2SqrtUnexpanded:
    case raft::distance::DistanceType::L2Unexpanded:
    case raft::distance::DistanceType::L2SqrtExpanded:
    case raft::distance::DistanceType::L2Expanded:
      naiveDistanceKernel<DataType><<<nblks, TPB>>>(dist, x, y, m, n, k, type, isRowMajor);
      break;
    case raft::distance::DistanceType::CosineExpanded:
      naiveCosineDistanceKernel<DataType><<<nblks, TPB>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    default: FAIL() << "should be here\n";
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename DataType>
struct DistanceInputs {
  DataType tolerance;
  int m, n, k;
  bool isRowMajor;
  unsigned long long int seed;
};

template <typename DataType>
::std::ostream& operator<<(::std::ostream& os, const DistanceInputs<DataType>& dims)
{
  return os;
}

template <raft::distance::DistanceType distanceType, typename DataType>
void distanceLauncher(DataType* x,
                      DataType* y,
                      DataType* dist,
                      DataType* dist2,
                      int m,
                      int n,
                      int k,
                      DistanceInputs<DataType>& params,
                      DataType threshold,
                      char* workspace,
                      size_t worksize,
                      cudaStream_t stream,
                      bool isRowMajor)
{
  auto fin_op = [dist2, threshold] __device__(DataType d_val, int g_d_idx) {
    dist2[g_d_idx] = (d_val < threshold) ? 0.f : d_val;
    return d_val;
  };
  distance<distanceType, DataType, DataType, DataType>(
    x, y, dist, m, n, k, workspace, worksize, fin_op, stream, isRowMajor);
}

template <raft::distance::DistanceType distanceType, typename DataType>
class DistanceTest : public ::testing::TestWithParam<DistanceInputs<DataType>> {
 public:
  void SetUp() override
  {
    params = ::testing::TestWithParam<DistanceInputs<DataType>>::GetParam();
    raft::random::Rng r(params.seed);
    int m           = params.m;
    int n           = params.n;
    int k           = params.k;
    bool isRowMajor = params.isRowMajor;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    raft::allocate(x, m * k);
    raft::allocate(y, n * k);
    raft::allocate(dist_ref, m * n);
    raft::allocate(dist, m * n);
    raft::allocate(dist2, m * n);
    r.uniform(x, m * k, DataType(-1.0), DataType(1.0), stream);
    r.uniform(y, n * k, DataType(-1.0), DataType(1.0), stream);
    naiveDistance(dist_ref, x, y, m, n, k, distanceType, isRowMajor);
    char* workspace = nullptr;
    size_t worksize = getWorkspaceSize<distanceType, DataType, DataType, DataType>(x, y, m, n, k);
    if (worksize != 0) { raft::allocate(workspace, worksize); }

    DataType threshold = -10000.f;
    distanceLauncher<distanceType, DataType>(
      x, y, dist, dist2, m, n, k, params, threshold, workspace, worksize, stream, isRowMajor);
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(workspace));
  }

  void TearDown() override
  {
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(dist_ref));
    CUDA_CHECK(cudaFree(dist));
    CUDA_CHECK(cudaFree(dist2));
  }

 protected:
  DistanceInputs<DataType> params;
  DataType *x, *y, *dist_ref, *dist, *dist2;
};

}  // end namespace Distance
}  // end namespace MLCommon
