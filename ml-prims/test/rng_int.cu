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
#include <cub/cub.cuh>
#include "cuda_utils.h"
#include "random/rng.h"
#include "test_utils.h"


namespace MLCommon {
namespace Random {

enum RandomType {
  RNG_Uniform
};

template <typename T, int TPB>
__global__ void meanKernel(float *out, const T *data, int len) {
  typedef cub::BlockReduce<float, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val = tid < len ? data[tid] : T(0);
  float x = BlockReduce(temp_storage).Sum(val);
  __syncthreads();
  float xx = BlockReduce(temp_storage).Sum(val * val);
  __syncthreads();
  if (threadIdx.x == 0) {
    myAtomicAdd(out, x);
    myAtomicAdd(out + 1, xx);
  }
}

template <typename T>
struct RngInputs {
  float tolerance;
  int len;
  // start, end: for uniform
  // mean, sigma: for normal/lognormal
  // mean, beta: for gumbel
  // mean, scale: for logistic and laplace
  // lambda: for exponential
  // sigma: for rayleigh
  T start, end;
  RandomType type;
  GeneratorType gtype;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const RngInputs<T> &dims) {
  return os;
}

template <typename T>
class RngTest : public ::testing::TestWithParam<RngInputs<T>> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<RngInputs<T>>::GetParam();
    Rng r(params.seed, params.gtype);
    allocate(data, params.len);
    allocate(stats, 2, true);
    switch (params.type) {
      case RNG_Uniform:
        r.uniformInt(data, params.len, params.start, params.end);
        break;
    };
    static const int threads = 128;
    meanKernel<T, threads><<<ceildiv(params.len, threads), threads>>>(
      stats, data, params.len);
    updateHost<float>(h_stats, stats, 2);
    h_stats[0] /= params.len;
    h_stats[1] = (h_stats[1] / params.len) - (h_stats[0] * h_stats[0]);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(stats));
  }

  void getExpectedMeanVar(float meanvar[2]) {
    switch (params.type) {
      case RNG_Uniform:
        meanvar[0] = (params.start + params.end) * 0.5f;
        meanvar[1] = params.end - params.start;
        meanvar[1] = meanvar[1] * meanvar[1] / 12.f;
        break;
    };
  }

protected:
  RngInputs<T> params;
  T *data;
  float *stats;
  float h_stats[2]; // mean, var
};

typedef RngTest<uint32_t> RngTestU32;
const std::vector<RngInputs<uint32_t>> inputs_u32 = {
  {0.1f, 32 * 1024, 0, 20, RNG_Uniform, GenPhilox, 1234ULL},
  {0.1f, 8 * 1024, 0, 20, RNG_Uniform, GenPhilox, 1234ULL},

  {0.1f, 32 * 1024, 0, 20, RNG_Uniform, GenTaps, 1234ULL},
  {0.1f, 8 * 1024, 0, 20, RNG_Uniform, GenTaps, 1234ULL},

  {0.1f, 32 * 1024, 0, 20, RNG_Uniform, GenKiss99, 1234ULL},
  {0.1f, 8 * 1024, 0, 20, RNG_Uniform, GenKiss99, 1234ULL}};
TEST_P(RngTestU32, Result) {
  float meanvar[2];
  getExpectedMeanVar(meanvar);
  ASSERT_TRUE(
    match(meanvar[0], h_stats[0], CompareApprox<float>(params.tolerance)));
  ASSERT_TRUE(
    match(meanvar[1], h_stats[1], CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(RngTests, RngTestU32, ::testing::ValuesIn(inputs_u32));

} // end namespace Random
} // end namespace MLCommon
