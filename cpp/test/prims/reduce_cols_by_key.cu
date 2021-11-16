/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <linalg/reduce_cols_by_key.cuh>
#include <raft/random/rng.hpp>
#include "test_utils.h"

namespace MLCommon {
namespace LinAlg {

template <typename T>
void naiveReduceColsByKey(const T* in,
                          const uint32_t* keys,
                          T* out_ref,
                          uint32_t nrows,
                          uint32_t ncols,
                          uint32_t nkeys,
                          cudaStream_t stream)
{
  std::vector<uint32_t> h_keys(ncols, 0u);
  raft::copy(&(h_keys[0]), keys, ncols, stream);
  std::vector<T> h_in(nrows * ncols);
  raft::copy(&(h_in[0]), in, nrows * ncols, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<T> out(nrows * nkeys, T(0));
  for (uint32_t i = 0; i < nrows; ++i) {
    for (uint32_t j = 0; j < ncols; ++j) {
      out[i * nkeys + h_keys[j]] += h_in[i * ncols + j];
    }
  }
  raft::copy(out_ref, &(out[0]), nrows * nkeys, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename T>
struct ReduceColsInputs {
  T tolerance;
  uint32_t rows;
  uint32_t cols;
  uint32_t nkeys;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const ReduceColsInputs<T>& dims)
{
  return os;
}

template <typename T>
class ReduceColsTest : public ::testing::TestWithParam<ReduceColsInputs<T>> {
 protected:
  ReduceColsTest() : in(0, stream), out_ref(0, stream), out(0, stream), keys(0, stream) {}

  void SetUp() override
  {
    params = ::testing::TestWithParam<ReduceColsInputs<T>>::GetParam();
    raft::random::Rng r(params.seed);
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto nrows = params.rows;
    auto ncols = params.cols;
    auto nkeys = params.nkeys;
    in.resize(nrows * ncols, stream);
    keys.resize(ncols, stream);
    out_ref.resize(nrows * nkeys, stream);
    out.resize(nrows * nkeys, stream);
    r.uniform(in.data(), nrows * ncols, T(-1.0), T(1.0), stream);
    r.uniformInt(keys.data(), ncols, 0u, params.nkeys, stream);
    naiveReduceColsByKey(in.data(), keys.data(), out_ref.data(), nrows, ncols, nkeys, stream);
    reduce_cols_by_key(in.data(), keys.data(), out.data(), nrows, ncols, nkeys, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown() override { CUDA_CHECK(cudaStreamDestroy(stream)); }

 protected:
  cudaStream_t stream = 0;
  ReduceColsInputs<T> params;
  rmm::device_uvector<T> in, out_ref, out;
  rmm::device_uvector<uint32_t> keys;
};

const std::vector<ReduceColsInputs<float>> inputsf = {{0.0001f, 128, 32, 6, 1234ULL},
                                                      {0.0005f, 121, 63, 10, 1234ULL}};
typedef ReduceColsTest<float> ReduceColsTestF;
TEST_P(ReduceColsTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(out_ref.data(),
                                out.data(),
                                params.rows * params.nkeys,
                                raft::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReduceColsTests, ReduceColsTestF, ::testing::ValuesIn(inputsf));

const std::vector<ReduceColsInputs<double>> inputsd2 = {{0.0000001, 128, 32, 6, 1234ULL},
                                                        {0.0000001, 121, 63, 10, 1234ULL}};
typedef ReduceColsTest<double> ReduceColsTestD;
TEST_P(ReduceColsTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(out_ref.data(),
                                out.data(),
                                params.rows * params.nkeys,
                                raft::CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReduceColsTests, ReduceColsTestD, ::testing::ValuesIn(inputsd2));

}  // end namespace LinAlg
}  // end namespace MLCommon
