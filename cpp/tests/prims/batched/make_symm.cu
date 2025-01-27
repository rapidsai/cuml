/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "../test_utils.h"

#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>
#include <linalg/batched/make_symm.cuh>
#include <test_utils.h>

namespace MLCommon {
namespace LinAlg {
namespace Batched {

template <typename T>
struct BatchMakeSymmInputs {
  T tolerance;
  int n, batchSize;
  unsigned long long int seed;
};

template <typename T, typename IdxType = int>
::std::ostream& operator<<(::std::ostream& os, const BatchMakeSymmInputs<T>& dims)
{
  return os;
}

template <typename Type>
CUML_KERNEL void naiveBatchMakeSymmKernel(Type* y, const Type* x, int n)
{
  int batch = blockIdx.z;
  int row   = threadIdx.y + blockDim.y * blockIdx.y;
  int col   = threadIdx.x + blockDim.x * blockIdx.x;
  if (row < n && col < n) {
    int idx   = batch * n * n + row * n + col;
    int other = batch * n * n + col * n + row;
    y[idx]    = (x[idx] + x[other]) * Type(0.5);
  }
}

template <typename Type>
void naiveBatchMakeSymm(Type* y, const Type* x, int batchSize, int n, cudaStream_t stream)
{
  dim3 blk(16, 16);
  int nblks = raft::ceildiv<int>(n, blk.x);
  dim3 grid(nblks, nblks, batchSize);
  naiveBatchMakeSymmKernel<Type><<<grid, blk, 0, stream>>>(y, x, n);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename T>
class BatchMakeSymmTest : public ::testing::TestWithParam<BatchMakeSymmInputs<T>> {
 protected:
  BatchMakeSymmTest() : x(0, stream), out_ref(0, stream), out(0, stream) {}

  void SetUp() override
  {
    params = ::testing::TestWithParam<BatchMakeSymmInputs<T>>::GetParam();
    raft::random::Rng r(params.seed);
    int len = params.batchSize * params.n * params.n;
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));

    x.resize(len, stream);
    out_ref.resize(len, stream);
    out.resize(len, stream);

    r.uniform(x.data(), len, T(-1.0), T(1.0), stream);
    naiveBatchMakeSymm(out_ref.data(), x.data(), params.batchSize, params.n, stream);
    make_symm<T, int>(out.data(), x.data(), params.batchSize, params.n, stream);
    RAFT_CUDA_TRY(cudaStreamDestroy(stream));
  }

 protected:
  cudaStream_t stream = 0;
  BatchMakeSymmInputs<T> params;
  rmm::device_uvector<T> x;
  rmm::device_uvector<T> out_ref;
  rmm::device_uvector<T> out;
};

const std::vector<BatchMakeSymmInputs<float>> inputsf = {
  {0.000001f, 128, 32, 1234ULL},
  {0.000001f, 126, 32, 1234ULL},
  {0.000001f, 125, 32, 1234ULL},
};
typedef BatchMakeSymmTest<float> BatchMakeSymmTestF;
TEST_P(BatchMakeSymmTestF, Result)
{
  int len = params.batchSize * params.n * params.n;
  ASSERT_TRUE(
    devArrMatch(out_ref.data(), out.data(), len, MLCommon::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(BatchMakeSymmTests, BatchMakeSymmTestF, ::testing::ValuesIn(inputsf));

typedef BatchMakeSymmTest<double> BatchMakeSymmTestD;
const std::vector<BatchMakeSymmInputs<double>> inputsd = {
  {0.0000001, 128, 32, 1234ULL},
  {0.0000001, 126, 32, 1234ULL},
  {0.0000001, 125, 32, 1234ULL},
};
TEST_P(BatchMakeSymmTestD, Result)
{
  int len = params.batchSize * params.n * params.n;
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), len, MLCommon::CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(BatchMakeSymmTests, BatchMakeSymmTestD, ::testing::ValuesIn(inputsd));

}  // end namespace Batched
}  // end namespace LinAlg
}  // end namespace MLCommon
