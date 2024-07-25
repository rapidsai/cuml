/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <common/device_utils.cuh>

#include <cuml/common/utils.hpp>

#include <raft/core/interruptible.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

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
CUML_KERNEL void batchedBlockReduceTestKernel(int* out)
{
  extern __shared__ char smem[];
  int val = threadIdx.x;
  val     = batchedBlockReduce<int, NThreads>(val, reinterpret_cast<char*>(smem));
  int gid = threadIdx.x / NThreads;
  int lid = threadIdx.x % NThreads;
  if (gid == 0) { out[lid] = val; }
}

struct BatchedBlockReduceInputs {
  int blkDim;
};

template <int NThreads>
void batchedBlockReduceTest(int* out, const BatchedBlockReduceInputs& param, cudaStream_t stream)
{
  size_t smemSize = sizeof(int) * (param.blkDim / raft::WarpSize) * NThreads;
  batchedBlockReduceTestKernel<NThreads><<<1, param.blkDim, smemSize, stream>>>(out);
  RAFT_CUDA_TRY(cudaGetLastError());
}

::std::ostream& operator<<(::std::ostream& os, const BatchedBlockReduceInputs& dims) { return os; }

template <int NThreads>
class BatchedBlockReduceTest : public ::testing::TestWithParam<BatchedBlockReduceInputs> {
 protected:
  BatchedBlockReduceTest() : out(0, stream), refOut(0, stream) {}

  void SetUp() override
  {
    params = ::testing::TestWithParam<BatchedBlockReduceInputs>::GetParam();
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));
    out.resize(NThreads, stream);
    refOut.resize(NThreads, stream);
    RAFT_CUDA_TRY(cudaMemset(out.data(), 0, out.size() * sizeof(int)));
    RAFT_CUDA_TRY(cudaMemset(refOut.data(), 0, refOut.size() * sizeof(int)));
    computeRef();
    batchedBlockReduceTest<NThreads>(out.data(), params, stream);
  }

  void TearDown() override { RAFT_CUDA_TRY(cudaStreamDestroy(stream)); }

  void computeRef()
  {
    int* ref    = new int[NThreads];
    int nGroups = params.blkDim / NThreads;
    for (int i = 0; i < NThreads; ++i) {
      ref[i] = 0;
      for (int j = 0; j < nGroups; ++j) {
        ref[i] += j * NThreads + i;
      }
    }
    raft::update_device(refOut.data(), ref, NThreads, stream);
    raft::interruptible::synchronize(stream);
    delete[] ref;
  }

 protected:
  BatchedBlockReduceInputs params;
  rmm::device_uvector<int> out, refOut;
  cudaStream_t stream = 0;
};

typedef BatchedBlockReduceTest<8> BBTest8;
typedef BatchedBlockReduceTest<16> BBTest16;
typedef BatchedBlockReduceTest<32> BBTest32;

const std::vector<BatchedBlockReduceInputs> inputs = {
  {32},
  {64},
  {128},
  {256},
  {512},
};

TEST_P(BBTest8, Result)
{
  ASSERT_TRUE(devArrMatch(refOut.data(), out.data(), 8, MLCommon::Compare<int>()));
}
INSTANTIATE_TEST_CASE_P(BatchedBlockReduceTests, BBTest8, ::testing::ValuesIn(inputs));

}  // end namespace MLCommon
