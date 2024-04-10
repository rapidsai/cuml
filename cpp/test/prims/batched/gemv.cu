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

#include "../test_utils.h"

#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>
#include <linalg/batched/gemv.cuh>
#include <test_utils.h>

namespace MLCommon {
namespace LinAlg {
namespace Batched {

template <typename T>
struct BatchGemvInputs {
  T tolerance;
  int m, n, batchSize;
  unsigned long long int seed;
};

template <typename T, typename IdxType = int>
::std::ostream& operator<<(::std::ostream& os, const BatchGemvInputs<T>& dims)
{
  return os;
}

template <typename Type>
CUML_KERNEL void naiveBatchGemvKernel(Type* y, const Type* A, const Type* x, int m, int n)
{
  int batch = blockIdx.y;
  int row   = blockIdx.x;
  int col   = threadIdx.x;
  if (row < m && col < n) {
    auto prod = A[batch * m * n + row * n + col] * x[batch * n + col];
    raft::myAtomicAdd(y + batch * m + row, prod);
  }
}

template <typename Type>
void naiveBatchGemv(
  Type* y, const Type* A, const Type* x, int m, int n, int batchSize, cudaStream_t stream)
{
  static int TPB = raft::ceildiv(n, raft::WarpSize) * raft::WarpSize;
  dim3 nblks(m, batchSize);
  naiveBatchGemvKernel<Type><<<nblks, TPB, 0, stream>>>(y, A, x, m, n);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename T>
class BatchGemvTest : public ::testing::TestWithParam<BatchGemvInputs<T>> {
 protected:
  BatchGemvTest() : out_ref(0, stream), out(0, stream) {}

  void SetUp() override
  {
    params = ::testing::TestWithParam<BatchGemvInputs<T>>::GetParam();
    raft::random::Rng r(params.seed);
    int len     = params.batchSize * params.m * params.n;
    int vecleny = params.batchSize * params.m;
    int veclenx = params.batchSize * params.n;
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));

    rmm::device_uvector<T> A(len, stream);
    rmm::device_uvector<T> x(veclenx, stream);
    out_ref.resize(vecleny, stream);
    out.resize(vecleny, stream);

    r.uniform(A.data(), len, T(-1.0), T(1.0), stream);
    r.uniform(x.data(), veclenx, T(-1.0), T(1.0), stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(out_ref.data(), 0, sizeof(T) * vecleny, stream));
    naiveBatchGemv(
      out_ref.data(), A.data(), x.data(), params.m, params.n, params.batchSize, stream);
    gemv<T, int>(out.data(),
                 A.data(),
                 x.data(),
                 nullptr,
                 T(1.0),
                 T(0.0),
                 params.m,
                 params.n,
                 params.batchSize,
                 stream);
  }

  void TearDown() override { RAFT_CUDA_TRY(cudaStreamDestroy(stream)); }

 protected:
  cudaStream_t stream = 0;
  BatchGemvInputs<T> params;
  rmm::device_uvector<T> out_ref;
  rmm::device_uvector<T> out;
};

const std::vector<BatchGemvInputs<float>> inputsf = {
  {0.005f, 128, 128, 32, 1234ULL},
  {0.005f, 128, 126, 32, 1234ULL},
  {0.005f, 128, 125, 32, 1234ULL},
  {0.005f, 126, 128, 32, 1234ULL},
  {0.005f, 126, 126, 32, 1234ULL},
  {0.005f, 126, 125, 32, 1234ULL},
  {0.005f, 125, 128, 32, 1234ULL},
  {0.005f, 125, 126, 32, 1234ULL},
  {0.005f, 125, 125, 32, 1234ULL},
};
typedef BatchGemvTest<float> BatchGemvTestF;
TEST_P(BatchGemvTestF, Result)
{
  int vecleny = params.batchSize * params.m;
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), vecleny, MLCommon::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(BatchGemvTests, BatchGemvTestF, ::testing::ValuesIn(inputsf));

typedef BatchGemvTest<double> BatchGemvTestD;
const std::vector<BatchGemvInputs<double>> inputsd = {
  {0.0000001, 128, 128, 32, 1234ULL},
  {0.0000001, 128, 126, 32, 1234ULL},
  {0.0000001, 128, 125, 32, 1234ULL},
  {0.0000001, 126, 128, 32, 1234ULL},
  {0.0000001, 126, 126, 32, 1234ULL},
  {0.0000001, 126, 125, 32, 1234ULL},
  {0.0000001, 125, 128, 32, 1234ULL},
  {0.0000001, 125, 126, 32, 1234ULL},
  {0.0000001, 125, 125, 32, 1234ULL},
};
TEST_P(BatchGemvTestD, Result)
{
  int vecleny = params.batchSize * params.m;
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), vecleny, MLCommon::CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(BatchGemvTests, BatchGemvTestD, ::testing::ValuesIn(inputsd));

}  // end namespace Batched
}  // end namespace LinAlg
}  // end namespace MLCommon
