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
__global__ void naiveHistKernel(int* bins, int nbins, int* in, int nrows) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  auto offset = blockIdx.y * nrows;
  auto binOffset = blockIdx.y * nbins;
  for (; tid < nrows; tid += stride) {
    int id = in[offset + tid];
    if (id < 0)
      id = 0;
    else if (id >= nbins)
      id = nbins - 1;
    in[offset + tid] = id;
    atomicAdd(bins + binOffset + id, 1);
  }
}

void naiveHist(int* bins, int nbins, int* in, int nrows, int ncols,
               cudaStream_t stream) {
  const int TPB = 128;
  int nblksx = ceildiv(nrows, TPB);
  dim3 blks(nblksx, ncols);
  naiveHistKernel<<<blks, TPB, 0, stream>>>(bins, nbins, in, nrows);
  CUDA_CHECK(cudaGetLastError());
}

struct HistInputs {
  int nrows, ncols, nbins;
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
    int len = params.nrows * params.ncols;
    allocate(in, len);
    if (params.isNormal) {
      r.normalInt(in, len, params.start, params.end, stream);
    } else {
      r.uniformInt(in, len, params.start, params.end, stream);
    }
    allocate(bins, params.nbins * params.ncols);
    allocate(ref_bins, params.nbins * params.ncols);
    CUDA_CHECK(cudaMemsetAsync(
      ref_bins, 0, sizeof(int) * params.nbins * params.ncols, stream));
    naiveHist(ref_bins, params.nbins, in, params.nrows, params.ncols, stream);
    histogram<int>(params.type, bins, params.nbins, in, params.nrows,
                   params.ncols, stream);
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
  {oneM, 1, 2 * oneM, false, HistTypeGmem, 0, 2 * oneM, 1234ULL},
  {oneM, 1, 2 * oneM, true, HistTypeGmem, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneM, false, HistTypeGmem, 0, 2 * oneM, 1234ULL},
  {oneM + 1, 1, 2 * oneM, true, HistTypeGmem, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneM, false, HistTypeGmem, 0, 2 * oneM, 1234ULL},
  {oneM + 2, 1, 2 * oneM, true, HistTypeGmem, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneM, false, HistTypeGmem, 0, 2 * oneM, 1234ULL},
  {oneM, 21, 2 * oneM, true, HistTypeGmem, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneM, false, HistTypeGmem, 0, 2 * oneM, 1234ULL},
  {oneM + 1, 21, 2 * oneM, true, HistTypeGmem, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneM, false, HistTypeGmem, 0, 2 * oneM, 1234ULL},
  {oneM + 2, 21, 2 * oneM, true, HistTypeGmem, 1000, 50, 1234ULL},

  {oneM, 1, 2 * oneK, false, HistTypeSmem, 0, 2 * oneK, 1234ULL},
  {oneM, 1, 2 * oneK, true, HistTypeSmem, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneK, false, HistTypeSmem, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 1, 2 * oneK, true, HistTypeSmem, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneK, false, HistTypeSmem, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 1, 2 * oneK, true, HistTypeSmem, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneK, false, HistTypeSmem, 0, 2 * oneK, 1234ULL},
  {oneM, 21, 2 * oneK, true, HistTypeSmem, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneK, false, HistTypeSmem, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 21, 2 * oneK, true, HistTypeSmem, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneK, false, HistTypeSmem, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 21, 2 * oneK, true, HistTypeSmem, 1000, 50, 1234ULL},

  {oneM, 1, 2 * oneK, false, HistTypeSmemMatchAny, 0, 2 * oneK, 1234ULL},
  {oneM, 1, 2 * oneK, true, HistTypeSmemMatchAny, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneK, false, HistTypeSmemMatchAny, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 1, 2 * oneK, true, HistTypeSmemMatchAny, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneK, false, HistTypeSmemMatchAny, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 1, 2 * oneK, true, HistTypeSmemMatchAny, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneK, false, HistTypeSmemMatchAny, 0, 2 * oneK, 1234ULL},
  {oneM, 21, 2 * oneK, true, HistTypeSmemMatchAny, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneK, false, HistTypeSmemMatchAny, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 21, 2 * oneK, true, HistTypeSmemMatchAny, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneK, false, HistTypeSmemMatchAny, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 21, 2 * oneK, true, HistTypeSmemMatchAny, 1000, 50, 1234ULL},

  {oneM, 1, 2 * oneK, false, HistTypeSmemBits16, 0, 2 * oneK, 1234ULL},
  {oneM, 1, 2 * oneK, true, HistTypeSmemBits16, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneK, false, HistTypeSmemBits16, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 1, 2 * oneK, true, HistTypeSmemBits16, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneK, false, HistTypeSmemBits16, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 1, 2 * oneK, true, HistTypeSmemBits16, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneK, false, HistTypeSmemBits16, 0, 2 * oneK, 1234ULL},
  {oneM, 21, 2 * oneK, true, HistTypeSmemBits16, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneK, false, HistTypeSmemBits16, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 21, 2 * oneK, true, HistTypeSmemBits16, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneK, false, HistTypeSmemBits16, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 21, 2 * oneK, true, HistTypeSmemBits16, 1000, 50, 1234ULL},

  {oneM, 1, 2 * oneK, false, HistTypeSmemBits8, 0, 2 * oneK, 1234ULL},
  {oneM, 1, 2 * oneK, true, HistTypeSmemBits8, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneK, false, HistTypeSmemBits8, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 1, 2 * oneK, true, HistTypeSmemBits8, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneK, false, HistTypeSmemBits8, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 1, 2 * oneK, true, HistTypeSmemBits8, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneK, false, HistTypeSmemBits8, 0, 2 * oneK, 1234ULL},
  {oneM, 21, 2 * oneK, true, HistTypeSmemBits8, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneK, false, HistTypeSmemBits8, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 21, 2 * oneK, true, HistTypeSmemBits8, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneK, false, HistTypeSmemBits8, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 21, 2 * oneK, true, HistTypeSmemBits8, 1000, 50, 1234ULL},

  {oneM, 1, 2 * oneK, false, HistTypeSmemBits4, 0, 2 * oneK, 1234ULL},
  {oneM, 1, 2 * oneK, true, HistTypeSmemBits4, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneK, false, HistTypeSmemBits4, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 1, 2 * oneK, true, HistTypeSmemBits4, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneK, false, HistTypeSmemBits4, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 1, 2 * oneK, true, HistTypeSmemBits4, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneK, false, HistTypeSmemBits4, 0, 2 * oneK, 1234ULL},
  {oneM, 21, 2 * oneK, true, HistTypeSmemBits4, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneK, false, HistTypeSmemBits4, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 21, 2 * oneK, true, HistTypeSmemBits4, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneK, false, HistTypeSmemBits4, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 21, 2 * oneK, true, HistTypeSmemBits4, 1000, 50, 1234ULL},

  {oneM, 1, 2 * oneK, false, HistTypeSmemBits2, 0, 2 * oneK, 1234ULL},
  {oneM, 1, 2 * oneK, true, HistTypeSmemBits2, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneK, false, HistTypeSmemBits2, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 1, 2 * oneK, true, HistTypeSmemBits2, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneK, false, HistTypeSmemBits2, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 1, 2 * oneK, true, HistTypeSmemBits2, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneK, false, HistTypeSmemBits2, 0, 2 * oneK, 1234ULL},
  {oneM, 21, 2 * oneK, true, HistTypeSmemBits2, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneK, false, HistTypeSmemBits2, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 21, 2 * oneK, true, HistTypeSmemBits2, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneK, false, HistTypeSmemBits2, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 21, 2 * oneK, true, HistTypeSmemBits2, 1000, 50, 1234ULL},

  {oneM, 1, 2 * oneK, false, HistTypeSmemBits1, 0, 2 * oneK, 1234ULL},
  {oneM, 1, 2 * oneK, true, HistTypeSmemBits1, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneK, false, HistTypeSmemBits1, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 1, 2 * oneK, true, HistTypeSmemBits1, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneK, false, HistTypeSmemBits1, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 1, 2 * oneK, true, HistTypeSmemBits1, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneK, false, HistTypeSmemBits1, 0, 2 * oneK, 1234ULL},
  {oneM, 21, 2 * oneK, true, HistTypeSmemBits1, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneK, false, HistTypeSmemBits1, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 21, 2 * oneK, true, HistTypeSmemBits1, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneK, false, HistTypeSmemBits1, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 21, 2 * oneK, true, HistTypeSmemBits1, 1000, 50, 1234ULL},

  {oneM, 1, 2 * oneM, false, HistTypeSmemHash, 0, 2 * oneM, 1234ULL},
  {oneM, 1, 2 * oneM, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneM, false, HistTypeSmemHash, 0, 2 * oneM, 1234ULL},
  {oneM + 1, 1, 2 * oneM, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneM, false, HistTypeSmemHash, 0, 2 * oneM, 1234ULL},
  {oneM + 2, 1, 2 * oneM, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM, 1, 2 * oneK, false, HistTypeSmemHash, 0, 2 * oneK, 1234ULL},
  {oneM, 1, 2 * oneK, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneK, false, HistTypeSmemHash, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 1, 2 * oneK, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneK, false, HistTypeSmemHash, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 1, 2 * oneK, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneM, false, HistTypeSmemHash, 0, 2 * oneM, 1234ULL},
  {oneM, 21, 2 * oneM, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneM, false, HistTypeSmemHash, 0, 2 * oneM, 1234ULL},
  {oneM + 1, 21, 2 * oneM, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneM, false, HistTypeSmemHash, 0, 2 * oneM, 1234ULL},
  {oneM + 2, 21, 2 * oneM, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneK, false, HistTypeSmemHash, 0, 2 * oneK, 1234ULL},
  {oneM, 21, 2 * oneK, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneK, false, HistTypeSmemHash, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 21, 2 * oneK, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneK, false, HistTypeSmemHash, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 21, 2 * oneK, true, HistTypeSmemHash, 1000, 50, 1234ULL},

  {oneM, 1, 2 * oneM, false, HistTypeAuto, 0, 2 * oneM, 1234ULL},
  {oneM, 1, 2 * oneM, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneM, false, HistTypeAuto, 0, 2 * oneM, 1234ULL},
  {oneM + 1, 1, 2 * oneM, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneM, false, HistTypeAuto, 0, 2 * oneM, 1234ULL},
  {oneM + 2, 1, 2 * oneM, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM, 1, 2 * oneK, false, HistTypeAuto, 0, 2 * oneK, 1234ULL},
  {oneM, 1, 2 * oneK, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneK, false, HistTypeAuto, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 1, 2 * oneK, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneK, false, HistTypeAuto, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 1, 2 * oneK, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneM, false, HistTypeAuto, 0, 2 * oneM, 1234ULL},
  {oneM, 21, 2 * oneM, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneM, false, HistTypeAuto, 0, 2 * oneM, 1234ULL},
  {oneM + 1, 21, 2 * oneM, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneM, false, HistTypeAuto, 0, 2 * oneM, 1234ULL},
  {oneM + 2, 21, 2 * oneM, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneK, false, HistTypeAuto, 0, 2 * oneK, 1234ULL},
  {oneM, 21, 2 * oneK, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneK, false, HistTypeAuto, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 21, 2 * oneK, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneK, false, HistTypeAuto, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 21, 2 * oneK, true, HistTypeAuto, 1000, 50, 1234ULL},
};
TEST_P(HistTest, Result) {
  ASSERT_TRUE(
    devArrMatch(ref_bins, bins, params.nbins * params.ncols, Compare<int>()));
}
INSTANTIATE_TEST_CASE_P(HistTests, HistTest, ::testing::ValuesIn(inputs));

}  // end namespace Stats
}  // end namespace MLCommon
