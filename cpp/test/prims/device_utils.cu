/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <common/device_utils.cuh>
#include "test_utils.h"

namespace MLCommon {

/*
 * Testing Methodology:
 * 0. Testing with a kernel of only one block is enough to verify this prim
 * 1. Assume that the threads in the block contain the following values:
 *       0         1   2  ....  NThreads - 1
 *       NThreads  ......................
 *       ................................
 *       ...................... blockDim.x - 1
 * 2. This means, the resulting output of batchedBlockReduce<int, NThreads>
 *    will be NThreads values and each of them is just a column-wise sum of
 *    the above matrix
 * 3. Repeat this for different block dimensions
 * 4. Repeat this for different values of NThreads
 */

template <int NThreads>
__global__ void batchedBlockReduceTestKernel(int* out) {
  extern __shared__ char smem[];
  int val = threadIdx.x;
  val = batchedBlockReduce<int, NThreads>(val, reinterpret_cast<char*>(smem));
  int gid = threadIdx.x / NThreads;
  int lid = threadIdx.x % NThreads;
  if (gid == 0) {
    out[lid] = val;
  }
}

struct BatchedBlockReduceInputs {
  int blkDim;
};

template <int NThreads>
void batchedBlockReduceTest(int* out, const BatchedBlockReduceInputs& param,
                            cudaStream_t stream) {
  size_t smemSize = sizeof(int) * (param.blkDim / WarpSize) * NThreads;
  batchedBlockReduceTestKernel<NThreads>
    <<<1, param.blkDim, smemSize, stream>>>(out);
  CUDA_CHECK(cudaGetLastError());
}

::std::ostream& operator<<(::std::ostream& os,
                           const BatchedBlockReduceInputs& dims) {
  return os;
}

template <int NThreads>
class BatchedBlockReduceTest
  : public ::testing::TestWithParam<BatchedBlockReduceInputs> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<BatchedBlockReduceInputs>::GetParam();
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocate(out, NThreads, true);
    allocate(refOut, NThreads, true);
    computeRef();
    batchedBlockReduceTest<NThreads>(out, params, stream);
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(refOut));
  }

  void computeRef() {
    int* ref = new int[NThreads];
    int nGroups = params.blkDim / NThreads;
    for (int i = 0; i < NThreads; ++i) {
      ref[i] = 0;
      for (int j = 0; j < nGroups; ++j) {
        ref[i] += j * NThreads + i;
      }
    }
    updateDevice(refOut, ref, NThreads, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    delete[] ref;
  }

 protected:
  BatchedBlockReduceInputs params;
  int *out, *refOut;
  cudaStream_t stream;
};

typedef BatchedBlockReduceTest<8> BBTest8;
typedef BatchedBlockReduceTest<16> BBTest16;
typedef BatchedBlockReduceTest<32> BBTest32;

const std::vector<BatchedBlockReduceInputs> inputs = {
  {32}, {64}, {128}, {256}, {512},
};

TEST_P(BBTest8, Result) {
  ASSERT_TRUE(devArrMatch(refOut, out, 8, Compare<int>()));
}
INSTANTIATE_TEST_CASE_P(BatchedBlockReduceTests, BBTest8,
                        ::testing::ValuesIn(inputs));

}  // end namespace MLCommon
