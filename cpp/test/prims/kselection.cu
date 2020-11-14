/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
#include <stdlib.h>
#include <algorithm>
#include <limits>
#include <raft/random/rng.cuh>
#include <selection/kselection.cuh>

namespace MLCommon {
namespace Selection {

template <typename TypeV, typename TypeK, int N, int TPB, bool Greater>
__global__ void sortTestKernel(TypeK *key) {
  KVArray<TypeV, TypeK, N, Greater> arr;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    arr.arr[i].val = (TypeV)raft::laneId();
    arr.arr[i].key = (TypeK)raft::laneId();
  }
  raft::warpFence();
  arr.sort();
  raft::warpFence();
#pragma unroll
  for (int i = 0; i < N; ++i)
    arr.arr[i].store(nullptr, key + threadIdx.x + i * TPB);
}

template <typename TypeV, typename TypeK, int N, int TPB, bool Greater>
void sortTest(TypeK *key) {
  TypeK *dkey;
  CUDA_CHECK(cudaMalloc((void **)&dkey, sizeof(TypeK) * TPB * N));
  sortTestKernel<TypeV, TypeK, N, TPB, Greater><<<1, TPB>>>(dkey);
  CUDA_CHECK(cudaPeekAtLastError());
  raft::update_host<TypeK>(key, dkey, TPB * N, 0);
  CUDA_CHECK(cudaFree(dkey));
}

/************************************************************************/
/********************** Add the function for CPU test *******************/
/************************************************************************/
template <typename TypeV, typename TypeK, bool Greater>
int cmp(KVPair<TypeV, TypeK> a, KVPair<TypeV, TypeK> b) {
  if (Greater == 0) {
    return a.val > b.val;
  } else {
    return a.val < b.val;
  }
}

template <typename TypeV, typename TypeK, bool Greater>
void partSortKVPair(KVPair<TypeV, TypeK> *arr, int N, int k) {
  std::partial_sort(arr, arr + k, arr + N, cmp<TypeV, TypeK, Greater>);
}
template <typename TypeV, typename TypeK, int N, bool Greater>
void sortKVArray(KVArray<TypeV, TypeK, N, Greater> &arr) {
  std::sort(arr.arr, arr.arr + N, cmp<TypeV, TypeK, Greater>);
}
template <typename TypeV, typename TypeK, bool Greater>
::testing::AssertionResult checkResult(TypeV *d_arr, TypeV *d_outv,
                                       TypeK *d_outk, int rows, int N, int k,
                                       TypeV tolerance) {
  for (int rIndex = 0; rIndex < rows; rIndex++) {
    // input data
    TypeV *h_arr = new TypeV[N];
    raft::update_host(h_arr, d_arr + rIndex * N, N, 0);
    KVPair<TypeV, TypeK> *topk = new KVPair<TypeV, TypeK>[N];
    for (int j = 0; j < N; j++) {
      topk[j].val = h_arr[j];
      topk[j].key = j;
    }
    // result reference
    TypeV *h_outv = new TypeV[k];
    raft::update_host(h_outv, d_outv + rIndex * k, k, 0);
    TypeK *h_outk = new TypeK[k];
    raft::update_host(h_outk, d_outk + rIndex * k, k, 0);
    // calculate the result
    partSortKVPair<TypeV, TypeK, Greater>(topk, N, k);

    // check result
    for (int j = 0; j < k; j++) {
      // std::cout<<"Get value at ("<<rIndex<<" "<<j<<") Cpu "
      //  <<topk[j].val<<" "<<topk[j].key<<" Gpu "<<h_outv[j]<<" "
      //<<h_outk[j] <<std::endl<<std::endl;

      if (abs(h_outv[j] - topk[j].val) > tolerance) {
        return ::testing::AssertionFailure()
               << "actual=" << topk[j].val << " != expected=" << h_outv[j];
      }
    }
    // delete resource
    delete[] h_arr;
    delete[] h_outv;
    delete[] h_outk;
    delete[] topk;
  }
  return ::testing::AssertionSuccess();
}

// Structure  WarpTopKInputs
template <typename T>
struct WarpTopKInputs {
  T tolerance;
  int rows;                     // batch size
  int cols;                     // N the length of variables
  int k;                        // the top-k value
  unsigned long long int seed;  // seed to generate data
};
template <typename T>
::std::ostream &operator<<(::std::ostream &os, const WarpTopKInputs<T> &dims) {
  return os;
}

// Define functions WarpTopKTest
template <typename T>
class WarpTopKTest : public ::testing::TestWithParam<WarpTopKInputs<T>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<WarpTopKInputs<T>>::GetParam();
    raft::random::Rng r(params.seed);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    raft::allocate(arr, params.rows * params.cols);
    raft::allocate(outk, params.rows * params.k);
    raft::allocate(outv, params.rows * params.k);
    r.uniform(arr, params.rows * params.cols, T(-1.0), T(1.0), stream);

    static const bool Sort = false;
    static const bool Greater = true;
    warpTopK<T, int, Greater, Sort>(outv, outk, arr, params.k, params.rows,
                                    params.cols, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(outv));
    CUDA_CHECK(cudaFree(outk));
    CUDA_CHECK(cudaFree(arr));
  }

 protected:
  WarpTopKInputs<T> params;
  T *arr, *outv;
  int *outk;
};

// Parameters
// Milestone 1: Verify the result of current implementation
// Milestone 2: Support all the values of k between 1 and 1024; both inclusive
// Milestone 2.1: Using the POC code to Support all the values
const std::vector<WarpTopKInputs<float>> inputs2_0 = {
  {0.00000001, 2, 1024, 256, 1234ULL}};
const std::vector<WarpTopKInputs<float>> inputs2_1 = {
  {0.00000001, 4, 2048, 1024, 1234ULL}};
const std::vector<WarpTopKInputs<float>> inputs2_2 = {
  {0.00000001, 4, 2048, 1, 1234ULL}};

// Milestone 2.2: Using the full thread queue and warp queue  code to support
// all the values
// @TODO: Milestone 3: Support not sorted
// @TODO: Milestone 4: Support multi-gpu

// Define the function TEST_P
typedef WarpTopKTest<float> TestD2_0;
typedef WarpTopKTest<float> TestD2_1;
typedef WarpTopKTest<float> TestD2_2;
TEST_P(TestD2_0, Result) {
  const static bool Greater = true;
  ASSERT_TRUE((checkResult<float, int, Greater>(
    arr, outv, outk, params.rows, params.cols, params.k, params.tolerance)));
}
TEST_P(TestD2_1, Result) {
  const static bool Greater = true;
  ASSERT_TRUE((checkResult<float, int, Greater>(
    arr, outv, outk, params.rows, params.cols, params.k, params.tolerance)));
}
TEST_P(TestD2_2, Result) {
  const static bool Greater = true;
  ASSERT_TRUE((checkResult<float, int, Greater>(
    arr, outv, outk, params.rows, params.cols, params.k, params.tolerance)));
}

// Instantiate
INSTANTIATE_TEST_CASE_P(WarpTopKTests, TestD2_0,
                        ::testing::ValuesIn(inputs2_0));
INSTANTIATE_TEST_CASE_P(WarpTopKTests, TestD2_1,
                        ::testing::ValuesIn(inputs2_1));
INSTANTIATE_TEST_CASE_P(WarpTopKTests, TestD2_2,
                        ::testing::ValuesIn(inputs2_2));

}  // end namespace Selection
}  // end namespace MLCommon
