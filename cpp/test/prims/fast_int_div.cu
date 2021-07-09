/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <raft/cudart_utils.h>
#include <common/fast_int_div.cuh>
#include "test_utils.h"

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

__global__ void fastIntDivTestKernel(
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
  static const int len = 100000;
  static const int TPB = 128;
  int *computed, *correct, *in;
  raft::allocate(computed, len * 2);
  raft::allocate(correct, len * 2);
  raft::allocate(in, len);
  for (int i = 0; i < 100; ++i) {
    // get a positive divisor
    int divisor;
    do {
      divisor = rand();
    } while (divisor <= 0);
    FastIntDiv fid(divisor);
    // run it against a few random numbers and compare the outputs
    int* h_in = new int[len];
    for (int i = 0; i < len; ++i) {
      h_in[i] = rand();
    }
    raft::update_device(in, h_in, len, 0);
    int nblks = raft::ceildiv(len, TPB);
    fastIntDivTestKernel<<<nblks, TPB, 0, 0>>>(computed, correct, in, fid, divisor, len);
    CUDA_CHECK(cudaStreamSynchronize(0));
    ASSERT_TRUE(devArrMatch(correct, computed, len * 2, raft::Compare<int>()))
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
