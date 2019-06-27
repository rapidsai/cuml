/**
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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
__global__ void naiveDistanceKernel(DataType *dist, const DataType *x,
                                    const DataType *y, int m, int n, int k,
                                    DistanceType type, bool isRowMajor) {
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) return;
  DataType acc = DataType(0);
  for (int i = 0; i < k; ++i) {
    int xidx = isRowMajor ? i + midx * k : i * m + midx;
    int yidx = isRowMajor ? i + nidx * k : i * n + midx;
    auto diff = x[xidx] - y[idx];
    acc += diff * diff;
  }
  if (type == EucExpandedL2Sqrt || type == EucUnexpandedL2Sqrt)
    acc = mySqrt(acc);
  int outidx = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx] = acc;
}

template <typename DataType>
__global__ void naiveL1DistanceKernel(DataType *dist, const DataType *x,
                                      const DataType *y, int m, int n, int k,
                                      bool isRowMajor) {
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) {
    return;
  }

  DataType acc = DataType(0);
  for (int i = 0; i < k; ++i) {
    int xidx = isRowMajor ? i + midx * k : i * m + midx;
    int yidx = isRowMajor ? i + nidx * k : i * n + midx;
    auto a = x[xidx];
    auto b = y[yidx];
    auto diff = (a > b) ? (a - b) : (b - a);
    acc += diff;
  }

  int outidx = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx] = acc;
}

template <typename DataType>
__global__ void naiveCosineDistanceKernel(DataType *dist, const DataType *x,
                                          const DataType *y, int m, int n,
                                          int k, bool isRowMajor) {
  int midx = threadIdx.x + blockIdx.x * blockDim.x;
  int nidx = threadIdx.y + blockIdx.y * blockDim.y;
  if (midx >= m || nidx >= n) {
    return;
  }

  DataType acc_a = DataType(0);
  DataType acc_b = DataType(0);
  DataType acc_ab = DataType(0);

  for (int i = 0; i < k; ++i) {
    int xidx = isRowMajor ? i + midx * k : i * m + midx;
    int yidx = isRowMajor ? i + nidx * k : i * n + midx;
    auto a = x[xidx];
    auto b = y[yidx];
    acc_a += a * a;
    acc_b += b * b;
    acc_ab += a * b;
  }

  int outidx = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx] = acc_ab / (mySqrt(acc_a) * mySqrt(acc_b));
}

template <typename DataType>
void naiveDistance(DataType *dist, const DataType *x, const DataType *y, int m,
                   int n, int k, DistanceType type, bool isRowMajor) {
  static const dim3 TPB(16, 32, 1);
  dim3 nblks(ceildiv(m, (int)TPB.x), ceildiv(n, (int)TPB.y), 1);

  switch (type) {
    case EucUnexpandedL1:
      naiveL1DistanceKernel<DataType>
        <<<nblks, TPB>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    case EucUnexpandedL2Sqrt:
    case EucUnexpandedL2:
    case EucExpandedL2Sqrt:
    case EucExpandedL2:
      naiveDistanceKernel<DataType>
        <<<nblks, TPB>>>(dist, x, y, m, n, k, type, isRowMajor);
      break;
    case EucExpandedCosine:
      naiveCosineDistanceKernel<DataType>
        <<<nblks, TPB>>>(dist, x, y, m, n, k, isRowMajor);
      break;
    default:
      FAIL() << "should be here\n";
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
::std::ostream &operator<<(::std::ostream &os,
                           const DistanceInputs<DataType> &dims) {
  return os;
}

template <DistanceType distanceType, typename DataType, typename OutputTile_t>
void distanceLauncher(DataType *x, DataType *y, DataType *dist, DataType *dist2,
                      int m, int n, int k, DistanceInputs<DataType> &params,
                      DataType threshold, char *workspace, size_t worksize,
                      cudaStream_t stream, bool isRowMajor) {
  auto fin_op = [dist2, threshold] __device__(DataType d_val, int g_d_idx) {
    dist2[g_d_idx] = (d_val < threshold) ? 0.f : d_val;
    return d_val;
  };
  distance<distanceType, DataType, DataType, DataType, OutputTile_t>(
    x, y, dist, m, n, k, workspace, worksize, fin_op, stream, isRowMajor);
}

template <DistanceType distanceType, typename DataType>
class DistanceTest : public ::testing::TestWithParam<DistanceInputs<DataType>> {
 public:
  void SetUp() override {
    params = ::testing::TestWithParam<DistanceInputs<DataType>>::GetParam();
    Random::Rng r(params.seed);
    int m = params.m;
    int n = params.n;
    int k = params.k;
    bool isRowMajor = params.isRowMajor;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocate(x, m * k);
    allocate(y, n * k);
    allocate(dist_ref, m * n);
    allocate(dist, m * n);
    allocate(dist2, m * n);
    r.uniform(x, m * k, DataType(-1.0), DataType(1.0), stream);
    r.uniform(y, n * k, DataType(-1.0), DataType(1.0), stream);
    naiveDistance(dist_ref, x, y, m, n, k, distanceType, isRowMajor);
    char *workspace = nullptr;
    size_t worksize =
      getWorkspaceSize<distanceType, DataType, DataType, DataType>(x, y, m, n,
                                                                   k);
    if (worksize != 0) {
      allocate(workspace, worksize);
    }

    typedef cutlass::Shape<8, 128, 128> OutputTile_t;
    DataType threshold = -10000.f;
    distanceLauncher<distanceType, DataType, OutputTile_t>(
      x, y, dist, dist2, m, n, k, params, threshold, workspace, worksize,
      stream, isRowMajor);
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(workspace));
  }

  void TearDown() override {
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
