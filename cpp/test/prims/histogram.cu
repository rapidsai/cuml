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

#include <cuda_utils.h>
#include <gtest/gtest.h>
#include <random/rng.h>
#include <stats/histogram.h>
#include "test_utils.h"

namespace MLCommon {
namespace Stats {

// Note: this kernel also updates the input vector to take care of OOB bins!
__global__ void naiveHistKernel(int* bins, int nbins, int* in, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < n; tid += stride) {
    int id = in[tid];
    if (id < 0)
      id = 0;
    else if (id >= nbins)
      id = nbins - 1;
    in[tid] = id;
    atomicAdd(bins + id, 1);
  }
}

void naiveHist(int* bins, int nbins, int* in, int n, cudaStream_t stream) {
  const int TPB = 128;
  int nblks = ceildiv(n, TPB);
  naiveHistKernel<<<nblks, TPB, 0, stream>>>(bins, nbins, in, n);
  CUDA_CHECK(cudaGetLastError());
}

struct HistInputs {
  int n, nbins;
  bool isNormal;
  HistType type;
  int start, end;
  unsigned long long int seed;
};

class HistTest : public ::testing::TestWithParam<HistInputs> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<HistInputs>::GetParam();
    Random::Rng r(params.seed);
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocate(in, params.n);
    if (params.isNormal) {
      r.normalInt(in, params.n, params.start, params.end, stream);
    } else {
      r.uniformInt(in, params.n, params.start, params.end, stream);
    }
    allocate(bins, params.nbins);
    allocate(ref_bins, params.nbins);
    CUDA_CHECK(
      cudaMemsetAsync(ref_bins, 0, sizeof(int) * params.nbins, stream));
    naiveHist(ref_bins, params.nbins, in, params.n, stream);
    histogram<int>(params.type, bins, params.nbins, in, params.n, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(bins));
    CUDA_CHECK(cudaFree(ref_bins));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  cudaStream_t stream;
  HistInputs params;
  int* in;
  int *bins, *ref_bins;
};

static const int oneK = 1024;
static const int oneM = oneK * oneK;
const std::vector<HistInputs> inputs = {
  {oneM, 2 * oneM, false, HistTypeGmem, 0, 2 * oneM, 1234ULL},
  {oneM, 2 * oneM, true, HistTypeGmem, 1000, 50, 1234ULL},
  {oneM + 1, 2 * oneM, false, HistTypeGmem, 0, 2 * oneM, 1234ULL},
  {oneM + 1, 2 * oneM, true, HistTypeGmem, 1000, 50, 1234ULL},
  {oneM + 2, 2 * oneM, false, HistTypeGmem, 0, 2 * oneM, 1234ULL},
  {oneM + 2, 2 * oneM, true, HistTypeGmem, 1000, 50, 1234ULL},

  {oneM, 2 * oneM, false, HistTypeGmemWarp, 0, 2 * oneM, 1234ULL},
  {oneM, 2 * oneM, true, HistTypeGmemWarp, 1000, 50, 1234ULL},
  {oneM + 1, 2 * oneM, false, HistTypeGmemWarp, 0, 2 * oneM, 1234ULL},
  {oneM + 1, 2 * oneM, true, HistTypeGmemWarp, 1000, 50, 1234ULL},
  {oneM + 2, 2 * oneM, false, HistTypeGmemWarp, 0, 2 * oneM, 1234ULL},
  {oneM + 2, 2 * oneM, true, HistTypeGmemWarp, 1000, 50, 1234ULL},

  {oneM, 2 * oneK, false, HistTypeSmem, 0, 2 * oneK, 1234ULL},
  {oneM, 2 * oneK, true, HistTypeSmem, 1000, 50, 1234ULL},
  {oneM + 1, 2 * oneK, false, HistTypeSmem, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 2 * oneK, true, HistTypeSmem, 1000, 50, 1234ULL},
  {oneM + 2, 2 * oneK, false, HistTypeSmem, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 2 * oneK, true, HistTypeSmem, 1000, 50, 1234ULL},

  {oneM, 2 * oneK, false, HistTypeSmemBits16, 0, 2 * oneK, 1234ULL},
  {oneM, 2 * oneK, true, HistTypeSmemBits16, 1000, 50, 1234ULL},
  {oneM + 1, 2 * oneK, false, HistTypeSmemBits16, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 2 * oneK, true, HistTypeSmemBits16, 1000, 50, 1234ULL},
  {oneM + 2, 2 * oneK, false, HistTypeSmemBits16, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 2 * oneK, true, HistTypeSmemBits16, 1000, 50, 1234ULL},

  {oneM, 2 * oneK, false, HistTypeSmemBits8, 0, 2 * oneK, 1234ULL},
  {oneM, 2 * oneK, true, HistTypeSmemBits8, 1000, 50, 1234ULL},
  {oneM + 1, 2 * oneK, false, HistTypeSmemBits8, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 2 * oneK, true, HistTypeSmemBits8, 1000, 50, 1234ULL},
  {oneM + 2, 2 * oneK, false, HistTypeSmemBits8, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 2 * oneK, true, HistTypeSmemBits8, 1000, 50, 1234ULL},

  ///@todo: enable after this kernel has been fixed
  // {oneM, 2 * oneM, false, HistTypeSmemHash, 0, 2 * oneM, 1234ULL},
  // {oneM, 2 * oneM, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  // {oneM + 1, 2 * oneM, false, HistTypeSmemHash, 0, 2 * oneM, 1234ULL},
  // {oneM + 1, 2 * oneM, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  // {oneM + 2, 2 * oneM, false, HistTypeSmemHash, 0, 2 * oneM, 1234ULL},
  // {oneM + 2, 2 * oneM, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  // {oneM, 2 * oneK, false, HistTypeSmemHash, 0, 2 * oneK, 1234ULL},
  // {oneM, 2 * oneK, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  // {oneM + 1, 2 * oneK, false, HistTypeSmemHash, 0, 2 * oneK, 1234ULL},
  // {oneM + 1, 2 * oneK, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  // {oneM + 2, 2 * oneK, false, HistTypeSmemHash, 0, 2 * oneK, 1234ULL},
  // {oneM + 2, 2 * oneK, true, HistTypeSmemHash, 1000, 50, 1234ULL},

  {oneM, 2 * oneM, false, HistTypeAuto, 0, 2 * oneM, 1234ULL},
  {oneM, 2 * oneM, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM + 1, 2 * oneM, false, HistTypeAuto, 0, 2 * oneM, 1234ULL},
  {oneM + 1, 2 * oneM, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM + 2, 2 * oneM, false, HistTypeAuto, 0, 2 * oneM, 1234ULL},
  {oneM + 2, 2 * oneM, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM, 2 * oneK, false, HistTypeAuto, 0, 2 * oneK, 1234ULL},
  {oneM, 2 * oneK, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM + 1, 2 * oneK, false, HistTypeAuto, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 2 * oneK, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM + 2, 2 * oneK, false, HistTypeAuto, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 2 * oneK, true, HistTypeAuto, 1000, 50, 1234ULL},
};
TEST_P(HistTest, Result) {
  ASSERT_TRUE(devArrMatch(ref_bins, bins, params.nbins, Compare<int>()));
}
INSTANTIATE_TEST_CASE_P(HistTests, HistTest, ::testing::ValuesIn(inputs));

}  // end namespace Stats
}  // end namespace MLCommon
