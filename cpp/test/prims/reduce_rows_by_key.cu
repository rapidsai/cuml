/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <gtest/gtest.h>
#include <iostream>
#include <linalg/reduce_rows_by_key.cuh>
#include <raft/cudart_utils.h>
#include <raft/random/rng.hpp>

namespace MLCommon {
namespace LinAlg {

template <typename Type>
__global__ void naiveReduceRowsByKeyKernel(const Type* d_A,
                                           int lda,
                                           uint32_t* d_keys,
                                           const Type* d_weight,
                                           char* d_char_keys,
                                           int nrows,
                                           int ncols,
                                           int nkeys,
                                           Type* d_sums)
{
  int c = threadIdx.x + blockIdx.x * blockDim.x;
  if (c >= ncols) return;
  int this_key = threadIdx.y + blockIdx.y * blockDim.y;

  Type sum = 0.0;
  for (int r = 0; r < nrows; r++) {
    if (this_key != d_keys[r]) continue;
    Type wt = 1;
    if (d_weight) wt = d_weight[r];
    sum += d_A[lda * r + c] * wt;
  }
  d_sums[this_key * ncols + c] = sum;
}
template <typename Type>
void naiveReduceRowsByKey(const Type* d_A,
                          int lda,
                          uint32_t* d_keys,
                          const Type* d_weight,
                          char* d_char_keys,
                          int nrows,
                          int ncols,
                          int nkeys,
                          Type* d_sums,
                          cudaStream_t stream)
{
  cudaMemset(d_sums, 0, sizeof(Type) * nkeys * ncols);

  naiveReduceRowsByKeyKernel<<<dim3((ncols + 31) / 32, nkeys), dim3(32, 1), 0, stream>>>(
    d_A, lda, d_keys, d_weight, d_char_keys, nrows, ncols, nkeys, d_sums);
}

template <typename T>
struct ReduceRowsInputs {
  T tolerance;
  int nobs;
  uint32_t cols;
  uint32_t nkeys;
  unsigned long long int seed;
  bool weighted;
  T max_weight;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const ReduceRowsInputs<T>& dims)
{
  return os;
}

template <typename T>
class ReduceRowTest : public ::testing::TestWithParam<ReduceRowsInputs<T>> {
 public:
  ReduceRowTest()
    : params(::testing::TestWithParam<ReduceRowsInputs<T>>::GetParam()),
      stream(handle.get_stream()),
      in(params.nobs * params.cols, stream),
      out(params.nkeys * params.cols, stream),
      out_ref(params.nkeys * params.cols, stream),
      keys(params.nobs, stream),
      scratch_buf(params.nobs, stream)
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::Rng r(params.seed);
    raft::random::Rng r_int(params.seed);

    int nobs       = params.nobs;
    uint32_t cols  = params.cols;
    uint32_t nkeys = params.nkeys;
    r.uniform(in.data(), nobs * cols, T(0.0), T(2.0 / nobs), stream);
    r_int.uniformInt(keys.data(), nobs, (uint32_t)0, nkeys, stream);

    rmm::device_uvector<T> weight(0, stream);
    if (params.weighted) {
      weight.resize(nobs, stream);
      raft::random::Rng r(params.seed, raft::random::GeneratorType::GenPhilox);
      r.uniform(weight.data(), nobs, T(1), params.max_weight, stream);
    }

    naiveReduceRowsByKey(in.data(),
                         cols,
                         keys.data(),
                         params.weighted ? weight.data() : nullptr,
                         scratch_buf.data(),
                         nobs,
                         cols,
                         nkeys,
                         out_ref.data(),
                         stream);
    if (params.weighted) {
      reduce_rows_by_key(in.data(),
                         cols,
                         keys.data(),
                         params.weighted ? weight.data() : nullptr,
                         scratch_buf.data(),
                         nobs,
                         cols,
                         nkeys,
                         out.data(),
                         stream);
    } else {
      reduce_rows_by_key(
        in.data(), cols, keys.data(), scratch_buf.data(), nobs, cols, nkeys, out.data(), stream);
    }
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  ReduceRowsInputs<T> params;
  raft::handle_t handle;
  cudaStream_t stream = 0;

  int device_count = 0;
  rmm::device_uvector<T> in, out, out_ref;
  rmm::device_uvector<uint32_t> keys;
  rmm::device_uvector<char> scratch_buf;
};

// ReduceRowTestF
// 128 Obs, 32 cols, 6 clusters
const std::vector<ReduceRowsInputs<float>> inputsf2 = {{0.000001f, 128, 32, 6, 1234ULL, false},
                                                       {0.000001f, 128, 32, 6, 1234ULL, true, 1.0},
                                                       {0.000001f, 128, 32, 6, 1234ULL, true, 2.0}};
typedef ReduceRowTest<float> ReduceRowTestF;
TEST_P(ReduceRowTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(out_ref.data(),
                                out.data(),
                                params.cols * params.nkeys,
                                raft::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReduceRowTests, ReduceRowTestF, ::testing::ValuesIn(inputsf2));

// ReduceRowTestD
// 128 Obs, 32 cols, 6 clusters, double precision
const std::vector<ReduceRowsInputs<double>> inputsd2 = {
  {0.00000001, 128, 32, 6, 1234ULL, false},
  {0.00000001, 128, 32, 6, 1234ULL, true, 2.0},
  {0.00000001, 128, 32, 6, 1234ULL, true, 8.0}};
typedef ReduceRowTest<double> ReduceRowTestD;
TEST_P(ReduceRowTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(out_ref.data(),
                                out.data(),
                                params.cols * params.nkeys,
                                raft::CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReduceRowTests, ReduceRowTestD, ::testing::ValuesIn(inputsd2));

// ReduceRowTestSmallnKey
// 128 Obs, 32 cols, 3 clusters
const std::vector<ReduceRowsInputs<float>> inputsf_small_nkey = {
  {0.000001f, 128, 32, 3, 1234ULL, false},
  {0.000001f, 128, 32, 3, 1234ULL, true, 5.0},
  {0.000001f, 128, 32, 3, 1234ULL, true, 8.0}};
typedef ReduceRowTest<float> ReduceRowTestSmallnKey;
TEST_P(ReduceRowTestSmallnKey, Result)
{
  ASSERT_TRUE(raft::devArrMatch(out_ref.data(),
                                out.data(),
                                params.cols * params.nkeys,
                                raft::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReduceRowTests,
                        ReduceRowTestSmallnKey,
                        ::testing::ValuesIn(inputsf_small_nkey));

// ReduceRowTestBigSpace
// 512 Obs, 1024 cols, 32 clusters, double precision
const std::vector<ReduceRowsInputs<double>> inputsd_big_space = {
  {0.00000001, 512, 1024, 40, 1234ULL, false},
  {0.00000001, 512, 1024, 40, 1234ULL, true, 4.0},
  {0.00000001, 512, 1024, 40, 1234ULL, true, 16.0}};
typedef ReduceRowTest<double> ReduceRowTestBigSpace;
TEST_P(ReduceRowTestBigSpace, Result)
{
  ASSERT_TRUE(raft::devArrMatch(out_ref.data(),
                                out.data(),
                                params.cols * params.nkeys,
                                raft::CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReduceRowTests,
                        ReduceRowTestBigSpace,
                        ::testing::ValuesIn(inputsd_big_space));

// ReduceRowTestManyObs
// 100000 Obs, 37 cols, 32 clusters
const std::vector<ReduceRowsInputs<float>> inputsf_many_obs = {
  {0.00001f, 100000, 37, 32, 1234ULL, false},
  {0.00001f, 100000, 37, 32, 1234ULL, true, 4.0},
  {0.00001f, 100000, 37, 32, 1234ULL, true, 16.0}};
typedef ReduceRowTest<float> ReduceRowTestManyObs;
TEST_P(ReduceRowTestManyObs, Result)
{
  ASSERT_TRUE(raft::devArrMatch(out_ref.data(),
                                out.data(),
                                params.cols * params.nkeys,
                                raft::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReduceRowTests,
                        ReduceRowTestManyObs,
                        ::testing::ValuesIn(inputsf_many_obs));

// ReduceRowTestManyClusters
// 100000 Obs, 37 cols, 2048 clusters
const std::vector<ReduceRowsInputs<float>> inputsf_many_cluster = {
  {0.00001f, 100000, 37, 2048, 1234ULL, false},
  {0.00001f, 100000, 37, 2048, 1234ULL, true, 32.0},
  {0.00001f, 100000, 37, 2048, 1234ULL, true, 16.0}};
typedef ReduceRowTest<float> ReduceRowTestManyClusters;
TEST_P(ReduceRowTestManyClusters, Result)
{
  ASSERT_TRUE(raft::devArrMatch(out_ref.data(),
                                out.data(),
                                params.cols * params.nkeys,
                                raft::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReduceRowTests,
                        ReduceRowTestManyClusters,
                        ::testing::ValuesIn(inputsf_many_cluster));

}  // end namespace LinAlg
}  // end namespace MLCommon
