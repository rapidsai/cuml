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
#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.h"
#include "random/rng.h"
#include "stats/minmax.h"
#include "test_utils.h"
#include <limits>

namespace MLCommon {
namespace Stats {

///@todo: need to add tests for verifying the column subsampling feature

template <typename T>
struct MinMaxInputs {
  T tolerance;
  int rows, cols;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const MinMaxInputs<T> &dims) {
  return os;
}

template <typename T>
__global__ void naiveMinMaxInitKernel(int ncols, T* globalmin, T* globalmax,
                                      T init_val) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= ncols)
        return;
    globalmin[tid] = init_val;
    globalmax[tid] = -init_val;
}

template <typename T>
__global__ void naiveMinMaxKernel(const T* data, int nrows, int ncols,
                                  T* globalmin, T* globalmax) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int col = tid / nrows;
    if(col < ncols) {
        myAtomicMin(&globalmin[col], data[tid]);
        myAtomicMax(&globalmax[col], data[tid]);
    }
}

template <typename T>
void naiveMinMax(const T* data, int nrows, int ncols, T* globalmin,
                 T* globalmax, cudaStream_t stream) {
    const int TPB = 128;
    int nblks = ceildiv(ncols, TPB);
    T init_val = std::numeric_limits<T>::max();
    naiveMinMaxInitKernel<<<nblks, TPB, 0, stream>>>(ncols, globalmin,
                                                     globalmax, init_val);
    CUDA_CHECK(cudaGetLastError());
    nblks = ceildiv(nrows*ncols, TPB);
    naiveMinMaxKernel<<<nblks, TPB, 0, stream>>>(data, nrows, ncols, globalmin,
                                                 globalmax);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
class MinMaxTest : public ::testing::TestWithParam<MinMaxInputs<T>> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<MinMaxInputs<T>>::GetParam();
    Random::Rng r(params.seed);
    int len = params.rows * params.cols;
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocate(data, len);
    allocate(minmax_act, 2 * params.cols);
    allocate(minmax_ref, 2 * params.cols);
    r.normal(data, len, (T)0.0, (T)1.0, stream);
    naiveMinMax(data, params.rows, params.cols, minmax_ref,
                minmax_ref+params.cols, stream);
    minmax<T>(data, nullptr, nullptr, params.rows, params.cols, minmax_act,
              minmax_act+params.cols, nullptr, stream);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(minmax_act));
    CUDA_CHECK(cudaFree(minmax_ref));
  }

protected:
  MinMaxInputs<T> params;
  T *data, *minmax_act, *minmax_ref;
  cudaStream_t stream;
};

const std::vector<MinMaxInputs<float>> inputsf = {
  {0.00001f, 1024, 32, 1234ULL},
  {0.00001f, 1024, 64, 1234ULL},
  {0.00001f, 1024, 128, 1234ULL},
  {0.00001f, 1024, 256, 1234ULL},
  {0.00001f, 1024, 512, 1234ULL},
  {0.00001f, 1024, 1024, 1234ULL},
  {0.00001f, 4096, 32, 1234ULL},
  {0.00001f, 4096, 64, 1234ULL},
  {0.00001f, 4096, 128, 1234ULL},
  {0.00001f, 4096, 256, 1234ULL},
  {0.00001f, 4096, 512, 1234ULL},
  {0.00001f, 4096, 1024, 1234ULL},
  {0.00001f, 8192, 32, 1234ULL},
  {0.00001f, 8192, 64, 1234ULL},
  {0.00001f, 8192, 128, 1234ULL},
  {0.00001f, 8192, 256, 1234ULL},
  {0.00001f, 8192, 512, 1234ULL},
  {0.00001f, 8192, 1024, 1234ULL}};

const std::vector<MinMaxInputs<double>> inputsd = {
  {0.0000001, 1024, 32, 1234ULL},
  {0.0000001, 1024, 64, 1234ULL},
  {0.0000001, 1024, 128, 1234ULL},
  {0.0000001, 1024, 256, 1234ULL},
  {0.0000001, 1024, 512, 1234ULL},
  {0.0000001, 1024, 1024, 1234ULL},
  {0.0000001, 4096, 32, 1234ULL},
  {0.0000001, 4096, 64, 1234ULL},
  {0.0000001, 4096, 128, 1234ULL},
  {0.0000001, 4096, 256, 1234ULL},
  {0.0000001, 4096, 512, 1234ULL},
  {0.0000001, 4096, 1024, 1234ULL},
  {0.0000001, 8192, 32, 1234ULL},
  {0.0000001, 8192, 64, 1234ULL},
  {0.0000001, 8192, 128, 1234ULL},
  {0.0000001, 8192, 256, 1234ULL},
  {0.0000001, 8192, 512, 1234ULL},
  {0.0000001, 8192, 1024, 1234ULL}};

typedef MinMaxTest<float> MinMaxTestF;
TEST_P(MinMaxTestF, Result) {
  ASSERT_TRUE(devArrMatch(minmax_ref, minmax_act, 2 * params.cols,
                          CompareApprox<float>(params.tolerance)));
}

typedef MinMaxTest<double> MinMaxTestD;
TEST_P(MinMaxTestD, Result) {
  ASSERT_TRUE(devArrMatch(minmax_ref, minmax_act, 2 * params.cols,
                          CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(MinMaxTests, MinMaxTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(MinMaxTests, MinMaxTestD, ::testing::ValuesIn(inputsd));

} // end namespace Stats
} // end namespace MLCommon
