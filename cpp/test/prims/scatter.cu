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

#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include "common/scatter.h"
#include "cuda_utils.h"
#include "random/rng.h"
#include "test_utils.h"

namespace MLCommon {

template <typename DataT, typename IdxT>
__global__ void naiveScatterKernel(DataT *out, const DataT *in, const IdxT *idx,
                                   IdxT len) {
  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < len) {
    out[tid] = in[idx[tid]];
  }
}

template <typename DataT, typename IdxT>
void naiveScatter(DataT *out, const DataT *in, const IdxT *idx, IdxT len,
                  cudaStream_t stream) {
  int nblks = ceildiv<int>(len, 128);
  naiveScatterKernel<DataT, IdxT><<<nblks, 128, 0, stream>>>(out, in, idx, len);
}

struct ScatterInputs {
  int len;
  unsigned long long int seed;
};

template <typename DataT>
class ScatterTest : public ::testing::TestWithParam<ScatterInputs> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<ScatterInputs>::GetParam();
    Random::Rng r(params.seed);
    CUDA_CHECK(cudaStreamCreate(&stream));
    int len = params.len;
    allocate(in, len);
    allocate(ref_out, len);
    allocate(out, len);
    allocate(idx, len);
    r.uniform(in, len, DataT(-1.0), DataT(1.0), stream);
    {
      std::vector<int> h_idx(len, 0);
      for (int i=0; i<len; ++i) {
        h_idx[i] = i;
      }
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(h_idx.begin(), h_idx.end(), g);
      updateDevice(idx, &(h_idx[0]), len, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    naiveScatter(ref_out, in, idx, len, stream);
    scatter(out, in, idx, len, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(ref_out));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(idx));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  cudaStream_t stream;
  ScatterInputs params;
  DataT *in, *ref_out, *out;
  int *idx;
};

const std::vector<ScatterInputs> inputs = {
  {128, 1234ULL}, {129, 1234ULL}, {130, 1234ULL}};

typedef ScatterTest<float> ScatterTestF;
TEST_P(ScatterTestF, Result) {
  ASSERT_TRUE(devArrMatch(out, ref_out, params.len, Compare<float>()));
}
INSTANTIATE_TEST_CASE_P(ScatterTests, ScatterTestF,
                        ::testing::ValuesIn(inputs));

typedef ScatterTest<double> ScatterTestD;
TEST_P(ScatterTestD, Result) {
  ASSERT_TRUE(devArrMatch(out, ref_out, params.len, Compare<double>()));
}
INSTANTIATE_TEST_CASE_P(ScatterTests, ScatterTestD,
                        ::testing::ValuesIn(inputs));

}  // end namespace MLCommon
