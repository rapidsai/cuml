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

#include <common/fast_int_div.cuh>

#include <cuml/common/utils.hpp>

#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

namespace MLCommon {

TEST(FastIntDiv, CpuTest)
{
  for (int i = 0; i < 100; ++i) {
    // get a positive divisor
    int divisor;
    do {
      divisor = rand();
    } while (divisor <= 0);
    FastIntDiv fid(divisor);
    // run it against a few random numbers and compare the outputs
    for (int i = 0; i < 10000; ++i) {
      auto num      = rand();
      auto correct  = num / divisor;
      auto computed = num / fid;
      ASSERT_EQ(correct, computed) << " divisor=" << divisor << " num=" << num;
      num      = rand();
      correct  = num % divisor;
      computed = num % fid;
      ASSERT_EQ(correct, computed) << " divisor=" << divisor << " num=" << num;
      num      = -num;
      correct  = num / divisor;
      computed = num / fid;
      ASSERT_EQ(correct, computed) << " divisor=" << divisor << " num=" << num;
      num      = rand();
      correct  = num % divisor;
      computed = num % fid;
      ASSERT_EQ(correct, computed) << " divisor=" << divisor << " num=" << num;
    }
  }
}

CUML_KERNEL void fastIntDivTestKernel(
  int* computed, int* correct, const int* in, FastIntDiv fid, int divisor, int len)
{
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < len) {
    computed[tid]       = in[tid] % fid;
    correct[tid]        = in[tid] % divisor;
    computed[len + tid] = -in[tid] % fid;
    correct[len + tid]  = -in[tid] % divisor;
  }
}

TEST(FastIntDiv, GpuTest)
{
  cudaStream_t stream = 0;
  RAFT_CUDA_TRY(cudaStreamCreate(&stream));

  static const int len = 100000;
  static const int TPB = 128;
  rmm::device_uvector<int> computed(len * 2, stream);
  rmm::device_uvector<int> correct(len * 2, stream);
  rmm::device_uvector<int> in(len, stream);
  for (int i = 0; i < 100; ++i) {
    // get a positive divisor
    int divisor;
    do {
      divisor = rand();
    } while (divisor <= 0);
    FastIntDiv fid(divisor);
    // run it against a few random numbers and compare the outputs
    std::vector<int> h_in(len);
    for (int i = 0; i < len; ++i) {
      h_in[i] = rand();
    }
    raft::update_device(in.data(), h_in.data(), len, stream);
    int nblks = raft::ceildiv(len, TPB);
    fastIntDivTestKernel<<<nblks, TPB, 0, 0>>>(
      computed.data(), correct.data(), in.data(), fid, divisor, len);
    RAFT_CUDA_TRY(cudaStreamSynchronize(0));
    ASSERT_TRUE(devArrMatch(correct.data(), computed.data(), len * 2, MLCommon::Compare<int>()))
      << " divisor=" << divisor;
  }
}

FastIntDiv dummyFunc(int num)
{
  FastIntDiv fd(num);
  return fd;
}

TEST(FastIntDiv, IncorrectUsage)
{
  ASSERT_THROW(dummyFunc(-1), raft::exception);
  ASSERT_THROW(dummyFunc(0), raft::exception);
}

}  // namespace MLCommon
