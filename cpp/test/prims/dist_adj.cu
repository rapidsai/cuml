/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include "test_utils.h"

#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <distance/distance.cuh>
#include <gtest/gtest.h>

namespace MLCommon {
namespace Distance {

template <typename DataType>
CUML_KERNEL void naiveDistanceAdjKernel(bool* dist,
                                        const DataType* x,
                                        const DataType* y,
                                        int m,
                                        int n,
                                        int k,
                                        DataType eps,
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
  int outidx   = isRowMajor ? midx * n + nidx : midx + m * nidx;
  dist[outidx] = acc <= eps;
}

template <typename DataType>
void naiveDistanceAdj(bool* dist,
                      const DataType* x,
                      const DataType* y,
                      int m,
                      int n,
                      int k,
                      DataType eps,
                      bool isRowMajor)
{
  static const dim3 TPB(16, 32, 1);
  dim3 nblks(raft::ceildiv(m, (int)TPB.x), raft::ceildiv(n, (int)TPB.y), 1);
  naiveDistanceAdjKernel<DataType> < <<nblks, TPB>>(dist, x, y, m, n, k, eps, isRowMajor);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename DataType>
struct DistanceAdjInputs {
  DataType eps;
  int m, n, k;
  bool isRowMajor;
  unsigned long long int seed;
};

template <typename DataType>
::std::ostream& operator<<(::std::ostream& os, const DistanceAdjInputs<DataType>& dims)
{
  return os;
}

template <typename DataType>
class DistanceAdjTest : public ::testing::TestWithParam<DistanceAdjInputs<DataType>> {
 public:
  DistanceAdjTest() : x(0, stream), y(0, stream), dist_ref(0, stream), dist(0, stream) {}

  void SetUp() override
  {
    params = ::testing::TestWithParam < DistanceAdjInputs<DataType>::GetParam();
    raft::random::Rng r(params.seed);
    auto m              = params.m;
    auto n              = params.n;
    auto k              = params.k;
    bool isRowMajor     = params.isRowMajor;
    cudaStream_t stream = 0;
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));
    x        = rmm::device_scalar<DataType>(m * k, stream);
    y        = rmm::device_scalar<DataType>(n * k, stream);
    dist_ref = rmm::device_scalar<bool>(m * n, stream);
    dist     = rmm::device_scalar<bool>(m * n, stream);
    r.uniform(x.data(), m * k, DataType(-1.0), DataType(1.0), stream);
    r.uniform(y.data(), n * k, DataType(-1.0), DataType(1.0), stream);

    DataType threshold = params.eps;

    naiveDistanceAdj(dist_ref.data(), x.data(), y.data(), m, n, k, threshold, isRowMajor);
    size_t worksize =
      getWorkspaceSize<cuvs::distance::DistanceType::L2Expanded, DataType, DataType, bool>(
        x, y, m, n, k);

    rmm::device_uvector<char> workspace(worksize, stream);

    auto fin_op = [threshold] __device__(DataType d_val, int g_d_idx) {
      return d_val <= threshold;
    };
    distance<cuvs::distance::DistanceType::L2Expanded, DataType, DataType, bool>(x.data(),
                                                                                 y.data(),
                                                                                 dist.data(),
                                                                                 m,
                                                                                 n,
                                                                                 k,
                                                                                 workspace.data(),
                                                                                 worksize,
                                                                                 fin_op,
                                                                                 stream,
                                                                                 isRowMajor);
    RAFT_CUDA_TRY(cudaStreamDestroy(stream));
  }

 protected:
  DistanceAdjInputs<DataType> params;
  rmm::device_scalar<DataType> x, y;
  rmm::device_scalar<bool> dist_ref, dist;
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
TEST_P(DistanceAdjTestF, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(dist_ref.data(), dist.data(), m, n, MLCommon::Compare<bool>()));
}
INSTANTIATE_TEST_CASE_P(DistanceAdjTests, DistanceAdjTestF, ::testing::ValuesIn(inputsf));

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
TEST_P(DistanceAdjTestD, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(dist_ref.data(), dist.data(), m, n, MLCommon::Compare<bool>()));
}
INSTANTIATE_TEST_CASE_P(DistanceAdjTests, DistanceAdjTestD, ::testing::ValuesIn(inputsd));

}  // namespace Distance
}  // end namespace MLCommon
