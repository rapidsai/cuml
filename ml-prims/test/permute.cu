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

#include "cuda_utils.h"
#include "random/permute.h"
#include "random/rng.h"
#include "test_utils.h"
#include <algorithm>
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
::std::ostream &operator<<(::std::ostream &os, const PermInputs<T> &dims) {
  return os;
}

template <typename T>
class PermTest : public ::testing::TestWithParam<PermInputs<T>> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<PermInputs<T>>::GetParam();
    // forcefully set needPerms, since we need it for unit-testing!
    if(params.needShuffle) {
      params.needPerms = true;
    }
    Random::Rng r(params.seed);
    int N = params.N;
    int D = params.D;
    int len = N * D;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    if(params.needPerms)
      allocate(outPerms, N);
    else
      outPerms = nullptr;
    if(params.needShuffle) {
        allocate(in, len);
        allocate(out, len);
        r.uniform(in, len, T(-1.0), T(1.0), stream);
    } else {
        in = out = nullptr;
    }
    char *workspace = nullptr;
    size_t workspaceSize;
    permute(outPerms, out, in, D, N, params.rowMajor, workspace, workspaceSize, stream);
    allocate(workspace, workspaceSize);
    permute(outPerms, out, in, D, N, params.rowMajor, workspace, workspaceSize, stream);
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override {
    if(params.needPerms)
      CUDA_CHECK(cudaFree(outPerms));
    if(params.needShuffle) {
      CUDA_CHECK(cudaFree(in));
      CUDA_CHECK(cudaFree(out));
    }
  }

protected:
  PermInputs<T> params;
  T *in, *out;
  int *outPerms;
};


template <typename T, typename L>
::testing::AssertionResult devArrMatchRange(const T *actual, size_t size,
                                            T start, L eq_compare,
                                            bool doSort = true) {
  std::vector<T> act_h(size);
  updateHost<T>(&(act_h[0]), actual, size);
  if(doSort)
    std::sort(act_h.begin(), act_h.end());
  for (size_t i(0); i < size; ++i) {
    auto act = act_h[i];
    auto expected = start + i;
    if (!eq_compare(expected, act)) {
      return ::testing::AssertionFailure()
             << "actual=" << act << " != expected=" << expected << " @" << i;
    }
  }
  return ::testing::AssertionSuccess();
}

template <typename T, typename L>
::testing::AssertionResult devArrMatchShuffle(const int *perms, const T *out,
                                              const T *in, int D, int N,
                                              bool rowMajor, L eq_compare) {
  std::vector<int> h_perms(N);
  updateHost<int>(&(h_perms[0]), perms, N);
  std::vector<T> h_out(N * D), h_in(N * D);
  updateHost<T>(&(h_out[0]), out, N * D);
  updateHost<T>(&(h_in[0]), in, N * D);
  for (int i = 0; i < N; ++i) {
      for (int j = 0; j < D; ++j) {
          int outPos = rowMajor? i * D + j : j * N + i;
          int inPos = rowMajor? h_perms[i] * D + j : j * N + h_perms[i];
          auto act = h_out[outPos];
          auto expected = h_in[inPos];
          if (!eq_compare(expected, act)) {
              return ::testing::AssertionFailure()
                  << "actual=" << act << " != expected=" << expected << " @"
                  << i << ", " << j;
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
  {2*1024, 32, true, false, true, 1234ULL},
  {2*1024, 32, true, false, true, 1234567890ULL},
  {2*1024+500, 32, true, false, true, 1234ULL},
  {2*1024+500, 32, true, false, true, 1234567890ULL},
  {100000, 32, true, false, true, 1234ULL},
  {100000, 32, true, false, true, 1234567890ULL},
  // permute and shuffle the data
  {32, 8, true, true, true, 1234ULL},
  {32, 8, true, true, true, 1234567890ULL},
  {1024, 32, true, true, true, 1234ULL},
  {1024, 32, true, true, true, 1234567890ULL},
  {2*1024, 32, true, true, true, 1234ULL},
  {2*1024, 32, true, true, true, 1234567890ULL},
  {2*1024+500, 32, true, true, true, 1234ULL},
  {2*1024+500, 32, true, true, true, 1234567890ULL},
  {100000, 32, true, true, true, 1234ULL},
  {100000, 32, true, true, true, 1234567890ULL}};
typedef PermTest<float> PermTestF;
TEST_P(PermTestF, Result) {
  if(params.needPerms) {
    ASSERT_TRUE(devArrMatchRange(outPerms, params.N, 0, Compare<int>()));
  }
  if(params.needShuffle) {
    ASSERT_TRUE(devArrMatchShuffle(outPerms, out, in, params.D, params.N,
                                   params.rowMajor, Compare<float>()));
  }
}
INSTANTIATE_TEST_CASE_P(PermTests, PermTestF, ::testing::ValuesIn(inputsf));


const std::vector<PermInputs<double>> inputsd = {
  // only generate permutations
  {32, 8, true, false, true, 1234ULL},
  {32, 8, true, false, true, 1234567890ULL},
  {1024, 32, true, false, true, 1234ULL},
  {1024, 32, true, false, true, 1234567890ULL},
  {2*1024, 32, true, false, true, 1234ULL},
  {2*1024, 32, true, false, true, 1234567890ULL},
  {2*1024+500, 32, true, false, true, 1234ULL},
  {2*1024+500, 32, true, false, true, 1234567890ULL},
  {100000, 32, true, false, true, 1234ULL},
  {100000, 32, true, false, true, 1234567890ULL},
  // permute and shuffle the data
  {32, 8, true, true, true, 1234ULL},
  {32, 8, true, true, true, 1234567890ULL},
  {1024, 32, true, true, true, 1234ULL},
  {1024, 32, true, true, true, 1234567890ULL},
  {2*1024, 32, true, true, true, 1234ULL},
  {2*1024, 32, true, true, true, 1234567890ULL},
  {2*1024+500, 32, true, true, true, 1234ULL},
  {2*1024+500, 32, true, true, true, 1234567890ULL},
  {100000, 32, true, true, true, 1234ULL},
  {100000, 32, true, true, true, 1234567890ULL}};
typedef PermTest<double> PermTestD;
TEST_P(PermTestD, Result) {
  if(params.needPerms) {
    ASSERT_TRUE(devArrMatchRange(outPerms, params.N, 0, Compare<int>()));
  }
  if(params.needShuffle) {
    ASSERT_TRUE(devArrMatchShuffle(outPerms, out, in, params.D, params.N,
                                   params.rowMajor, Compare<double>()));
  }
}
INSTANTIATE_TEST_CASE_P(PermTests, PermTestD, ::testing::ValuesIn(inputsd));

} // end namespace Random
} // end namespace MLCommon
