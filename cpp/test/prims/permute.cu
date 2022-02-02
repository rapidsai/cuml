/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
#include <algorithm>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/random/rng.hpp>
#include <random/permute.cuh>
#include <vector>

namespace MLCommon {
namespace Random {

template <typename T>
struct PermInputs {
  int N, D;
  bool needPerms, needShuffle, rowMajor;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const PermInputs<T>& dims)
{
  return os;
}

template <typename T>
class PermTest : public ::testing::TestWithParam<PermInputs<T>> {
 protected:
  PermTest() : in(0, stream), out(0, stream), outPerms(0, stream) {}

  void SetUp() override
  {
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));
    params = ::testing::TestWithParam<PermInputs<T>>::GetParam();
    // forcefully set needPerms, since we need it for unit-testing!
    if (params.needShuffle) { params.needPerms = true; }
    raft::random::Rng r(params.seed);
    int N               = params.N;
    int D               = params.D;
    int len             = N * D;
    cudaStream_t stream = 0;
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));
    if (params.needPerms) {
      outPerms.resize(N, stream);
      outPerms_ptr = outPerms.data();
    }
    if (params.needShuffle) {
      in.resize(len, stream);
      out.resize(len, stream);
      in_ptr  = in.data();
      out_ptr = out.data();
      r.uniform(in_ptr, len, T(-1.0), T(1.0), stream);
    }
    permute(outPerms_ptr, out_ptr, in_ptr, D, N, params.rowMajor, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  void TearDown() override { RAFT_CUDA_TRY(cudaStreamDestroy(stream)); }

 protected:
  PermInputs<T> params;
  rmm::device_uvector<T> in, out;
  T* in_ptr  = nullptr;
  T* out_ptr = nullptr;
  rmm::device_uvector<int> outPerms;
  int* outPerms_ptr   = nullptr;
  cudaStream_t stream = 0;
};

template <typename T, typename L>
::testing::AssertionResult devArrMatchRange(
  const T* actual, size_t size, T start, L eq_compare, bool doSort = true, cudaStream_t stream = 0)
{
  std::vector<T> act_h(size);
  raft::update_host<T>(&(act_h[0]), actual, size, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  if (doSort) std::sort(act_h.begin(), act_h.end());
  for (size_t i(0); i < size; ++i) {
    auto act      = act_h[i];
    auto expected = start + i;
    if (!eq_compare(expected, act)) {
      return ::testing::AssertionFailure()
             << "actual=" << act << " != expected=" << expected << " @" << i;
    }
  }
  return ::testing::AssertionSuccess();
}

template <typename T, typename L>
::testing::AssertionResult devArrMatchShuffle(const int* perms,
                                              const T* out,
                                              const T* in,
                                              int D,
                                              int N,
                                              bool rowMajor,
                                              L eq_compare,
                                              cudaStream_t stream = 0)
{
  std::vector<int> h_perms(N);
  raft::update_host<int>(&(h_perms[0]), perms, N, stream);
  std::vector<T> h_out(N * D), h_in(N * D);
  raft::update_host<T>(&(h_out[0]), out, N * D, stream);
  raft::update_host<T>(&(h_in[0]), in, N * D, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < D; ++j) {
      int outPos    = rowMajor ? i * D + j : j * N + i;
      int inPos     = rowMajor ? h_perms[i] * D + j : j * N + h_perms[i];
      auto act      = h_out[outPos];
      auto expected = h_in[inPos];
      if (!eq_compare(expected, act)) {
        return ::testing::AssertionFailure()
               << "actual=" << act << " != expected=" << expected << " @" << i << ", " << j;
      }
    }
  }
  return ::testing::AssertionSuccess();
}

const std::vector<PermInputs<float>> inputsf = {
  // only generate permutations
  {32, 8, true, false, true, 1234ULL},
  {32, 8, true, false, true, 1234567890ULL},
  {1024, 32, true, false, true, 1234ULL},
  {1024, 32, true, false, true, 1234567890ULL},
  {2 * 1024, 32, true, false, true, 1234ULL},
  {2 * 1024, 32, true, false, true, 1234567890ULL},
  {2 * 1024 + 500, 32, true, false, true, 1234ULL},
  {2 * 1024 + 500, 32, true, false, true, 1234567890ULL},
  {100000, 32, true, false, true, 1234ULL},
  {100000, 32, true, false, true, 1234567890ULL},
  {100001, 33, true, false, true, 1234567890ULL},
  // permute and shuffle the data row major
  {32, 8, true, true, true, 1234ULL},
  {32, 8, true, true, true, 1234567890ULL},
  {1024, 32, true, true, true, 1234ULL},
  {1024, 32, true, true, true, 1234567890ULL},
  {2 * 1024, 32, true, true, true, 1234ULL},
  {2 * 1024, 32, true, true, true, 1234567890ULL},
  {2 * 1024 + 500, 32, true, true, true, 1234ULL},
  {2 * 1024 + 500, 32, true, true, true, 1234567890ULL},
  {100000, 32, true, true, true, 1234ULL},
  {100000, 32, true, true, true, 1234567890ULL},
  {100001, 31, true, true, true, 1234567890ULL},
  // permute and shuffle the data column major
  {32, 8, true, true, false, 1234ULL},
  {32, 8, true, true, false, 1234567890ULL},
  {1024, 32, true, true, false, 1234ULL},
  {1024, 32, true, true, false, 1234567890ULL},
  {2 * 1024, 32, true, true, false, 1234ULL},
  {2 * 1024, 32, true, true, false, 1234567890ULL},
  {2 * 1024 + 500, 32, true, true, false, 1234ULL},
  {2 * 1024 + 500, 32, true, true, false, 1234567890ULL},
  {100000, 32, true, true, false, 1234ULL},
  {100000, 32, true, true, false, 1234567890ULL},
  {100001, 33, true, true, false, 1234567890ULL}};

typedef PermTest<float> PermTestF;
TEST_P(PermTestF, Result)
{
  if (params.needPerms) {
    ASSERT_TRUE(devArrMatchRange(outPerms_ptr, params.N, 0, raft::Compare<int>()));
  }
  if (params.needShuffle) {
    ASSERT_TRUE(devArrMatchShuffle(
      outPerms_ptr, out_ptr, in_ptr, params.D, params.N, params.rowMajor, raft::Compare<float>()));
  }
}
INSTANTIATE_TEST_CASE_P(PermTests, PermTestF, ::testing::ValuesIn(inputsf));

const std::vector<PermInputs<double>> inputsd = {
  // only generate permutations
  {32, 8, true, false, true, 1234ULL},
  {32, 8, true, false, true, 1234567890ULL},
  {1024, 32, true, false, true, 1234ULL},
  {1024, 32, true, false, true, 1234567890ULL},
  {2 * 1024, 32, true, false, true, 1234ULL},
  {2 * 1024, 32, true, false, true, 1234567890ULL},
  {2 * 1024 + 500, 32, true, false, true, 1234ULL},
  {2 * 1024 + 500, 32, true, false, true, 1234567890ULL},
  {100000, 32, true, false, true, 1234ULL},
  {100000, 32, true, false, true, 1234567890ULL},
  {100001, 33, true, false, true, 1234567890ULL},
  // permute and shuffle the data row major
  {32, 8, true, true, true, 1234ULL},
  {32, 8, true, true, true, 1234567890ULL},
  {1024, 32, true, true, true, 1234ULL},
  {1024, 32, true, true, true, 1234567890ULL},
  {2 * 1024, 32, true, true, true, 1234ULL},
  {2 * 1024, 32, true, true, true, 1234567890ULL},
  {2 * 1024 + 500, 32, true, true, true, 1234ULL},
  {2 * 1024 + 500, 32, true, true, true, 1234567890ULL},
  {100000, 32, true, true, true, 1234ULL},
  {100000, 32, true, true, true, 1234567890ULL},
  {100001, 31, true, true, true, 1234567890ULL},
  // permute and shuffle the data column major
  {32, 8, true, true, false, 1234ULL},
  {32, 8, true, true, false, 1234567890ULL},
  {1024, 32, true, true, false, 1234ULL},
  {1024, 32, true, true, false, 1234567890ULL},
  {2 * 1024, 32, true, true, false, 1234ULL},
  {2 * 1024, 32, true, true, false, 1234567890ULL},
  {2 * 1024 + 500, 32, true, true, false, 1234ULL},
  {2 * 1024 + 500, 32, true, true, false, 1234567890ULL},
  {100000, 32, true, true, false, 1234ULL},
  {100000, 32, true, true, false, 1234567890ULL},
  {100001, 33, true, true, false, 1234567890ULL}};
typedef PermTest<double> PermTestD;
TEST_P(PermTestD, Result)
{
  if (params.needPerms) {
    ASSERT_TRUE(devArrMatchRange(outPerms_ptr, params.N, 0, raft::Compare<int>()));
  }
  if (params.needShuffle) {
    ASSERT_TRUE(devArrMatchShuffle(
      outPerms_ptr, out_ptr, in_ptr, params.D, params.N, params.rowMajor, raft::Compare<double>()));
  }
}
INSTANTIATE_TEST_CASE_P(PermTests, PermTestD, ::testing::ValuesIn(inputsd));

}  // end namespace Random
}  // end namespace MLCommon
