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
#include "linalg/batched/make_symm.h"
#include "random/rng.h"
#include "test_utils.h"

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
::std::ostream &operator<<(::std::ostream &os,
                           const BatchMakeSymmInputs<T> &dims) {
  return os;
}

template <typename Type>
__global__ void naiveBatchMakeSymmKernel(Type *y, const Type *x, int n) {
  int batch = blockIdx.z;
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  if (row < n && col < n) {
    int idx = batch * n * n + row * n + col;
    int other = batch * n * n + col * n + row;
    y[idx] = (x[idx] + x[other]) * Type(0.5);
  }
}

template <typename Type>
void naiveBatchMakeSymm(Type *y, const Type *x, int batchSize, int n,
                        cudaStream_t stream) {
  dim3 blk(16, 16);
  int nblks = ceildiv<int>(n, blk.x);
  dim3 grid(nblks, nblks, batchSize);
  naiveBatchMakeSymmKernel<Type><<<grid, blk, 0, stream>>>(y, x, n);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
class BatchMakeSymmTest
  : public ::testing::TestWithParam<BatchMakeSymmInputs<T>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<BatchMakeSymmInputs<T>>::GetParam();
    Random::Rng r(params.seed);
    int len = params.batchSize * params.n * params.n;
    CUDA_CHECK(cudaStreamCreate(&stream));

    allocate(x, len);
    allocate(out_ref, len);
    allocate(out, len);
    r.uniform(x, len, T(-1.0), T(1.0), stream);
    naiveBatchMakeSymm(out_ref, x, params.batchSize, params.n, stream);
    make_symm<T, int>(out, x, params.batchSize, params.n, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(out_ref));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  cudaStream_t stream;
  BatchMakeSymmInputs<T> params;
  T *x, *out_ref, *out;
};

const std::vector<BatchMakeSymmInputs<float>> inputsf = {
  {0.000001f, 128, 32, 1234ULL},
  {0.000001f, 126, 32, 1234ULL},
  {0.000001f, 125, 32, 1234ULL},
};
typedef BatchMakeSymmTest<float> BatchMakeSymmTestF;
TEST_P(BatchMakeSymmTestF, Result) {
  int len = params.batchSize * params.n * params.n;
  ASSERT_TRUE(
    devArrMatch(out_ref, out, len, CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(BatchMakeSymmTests, BatchMakeSymmTestF,
                        ::testing::ValuesIn(inputsf));

typedef BatchMakeSymmTest<double> BatchMakeSymmTestD;
const std::vector<BatchMakeSymmInputs<double>> inputsd = {
  {0.0000001, 128, 32, 1234ULL},
  {0.0000001, 126, 32, 1234ULL},
  {0.0000001, 125, 32, 1234ULL},
};
TEST_P(BatchMakeSymmTestD, Result) {
  int len = params.batchSize * params.n * params.n;
  ASSERT_TRUE(
    devArrMatch(out_ref, out, len, CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(BatchMakeSymmTests, BatchMakeSymmTestD,
                        ::testing::ValuesIn(inputsd));

}  // end namespace Batched
}  // end namespace LinAlg
}  // end namespace MLCommon
