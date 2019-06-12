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

#include <gtest/gtest.h>
#include "linalg/reduce_cols_by_key.h"
#include "random/rng.h"
#include "test_utils.h"

namespace MLCommon {
namespace LinAlg {

template <typename T>
void naiveReduceColsByKey(const T *in, const uint32_t *keys, T *out_ref,
                          uint32_t nrows, uint32_t ncols, uint32_t nkeys,
                          cudaStream_t stream) {
  std::vector<uint32_t> h_keys(0, ncols);
  copy(&(h_keys[0]), keys, ncols, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<T> out(T(0), nrows*nkeys);
  for (uint32_t i=0; i<nrows; ++i) {
    for (uint32_t j=0; j<ncols; ++j) {
      out[i*nkeys+h_keys[j]] += in[i*ncols+j];
    }
  }
  copy(out_ref, &(out[0]), nrows*nkeys, stream);
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
::std::ostream &operator<<(::std::ostream &os,
                           const ReduceColsInputs<T> &dims) {
  return os;
}

template <typename T>
class ReduceColsTest : public ::testing::TestWithParam<ReduceColsInputs<T>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<ReduceColsInputs<T>>::GetParam();
    Random::Rng r(params.seed);
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto nrows = params.rows;
    auto ncols = params.cols;
    auto nkeys = params.nkeys;
    allocate(in, nrows * ncols);
    allocate(keys, ncols);
    allocate(out, nrows * nkeys);
    r.uniform(in, nrows * ncols, T(-1.0), T(1.0), stream);
    r.uniformInt(keys, ncols, 0u, params.nkeys, stream);
    naiveReduceColsByKey(in, keys, out_ref, nrows, ncols, nkeys, stream);
    reduce_cols_by_key(in, keys, out, nrows, ncols, nkeys, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out_ref));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(keys));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  cudaStream_t stream;
  ReduceColsInputs<T> params;
  T *in, *out_ref, *out;
  uint32_t *keys;
};

const std::vector<ReduceColsInputs<float>> inputsf = {
  {0.000001f, 128, 32, 6, 1234ULL},
  {0.000001f, 121, 63, 10, 1234ULL}};
typedef ReduceColsTest<float> ReduceColsTestF;
TEST_P(ReduceColsTestF, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.rows * params.nkeys,
                          CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReduceColsTests, ReduceColsTestF,
                        ::testing::ValuesIn(inputsf));

const std::vector<ReduceColsInputs<double>> inputsd2 = {
  {0.00000001, 128, 32, 6, 1234ULL},
  {0.00000001, 121, 63, 10, 1234ULL}};
typedef ReduceColsTest<double> ReduceColsTestD;
TEST_P(ReduceColsTestD, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.rows * params.nkeys,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReduceColsTests, ReduceColsTestD,
                        ::testing::ValuesIn(inputsd2));

}  // end namespace LinAlg
}  // end namespace MLCommon
