/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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
#include <linalg/ternary_op.cuh>
#include <raft/random/rng.hpp>
#include "test_utils.h"

namespace MLCommon {
namespace LinAlg {

template <typename InType, typename IdxType = int, typename OutType = InType>
struct BinaryOpInputs {
  InType tolerance;
  IdxType len;
  unsigned long long int seed;
};

template <typename InType, typename IdxType = int, typename OutType = InType>
::std::ostream& operator<<(::std::ostream& os, const BinaryOpInputs<InType, IdxType, OutType>& d)
{
  return os;
}

template <typename T>
class ternaryOpTest : public ::testing::TestWithParam<BinaryOpInputs<T>> {
 public:
  void SetUp() override
  {
    params = ::testing::TestWithParam<BinaryOpInputs<T>>::GetParam();
    raft::random::Rng rng(params.seed);

    int len             = params.len;
    cudaStream_t stream = 0;
    CUDA_CHECK(cudaStreamCreate(&stream));
    raft::allocate(in1, len, stream);
    raft::allocate(in2, len, stream);
    raft::allocate(in3, len, stream);
    raft::allocate(out_add_ref, len, stream);
    raft::allocate(out_mul_ref, len, stream);
    raft::allocate(out_add, len, stream);
    raft::allocate(out_mul, len, stream);

    rng.fill(out_add_ref, len, T(6.0), stream);
    rng.fill(out_mul_ref, len, T(6.0), stream);
    rng.fill(in1, len, T(1.0), stream);
    rng.fill(in2, len, T(2.0), stream);
    rng.fill(in3, len, T(3.0), stream);

    auto add = [] __device__(T a, T b, T c) { return a + b + c; };
    auto mul = [] __device__(T a, T b, T c) { return a * b * c; };
    ternaryOp(out_add, in1, in2, in3, len, add, stream);
    ternaryOp(out_mul, in1, in2, in3, len, mul, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override
  {
    CUDA_CHECK(cudaFree(in1));
    CUDA_CHECK(cudaFree(in2));
    CUDA_CHECK(cudaFree(in3));
    CUDA_CHECK(cudaFree(out_mul_ref));
    CUDA_CHECK(cudaFree(out_add_ref));
    CUDA_CHECK(cudaFree(out_add));
    CUDA_CHECK(cudaFree(out_mul));
  }

 protected:
  BinaryOpInputs<T> params;
  T *in1, *in2, *in3, *out_add_ref, *out_mul_ref, *out_add, *out_mul;
};

const std::vector<BinaryOpInputs<float>> inputsf = {{0.000001f, 1024 * 1024, 1234ULL},
                                                    {0.000001f, 1024 * 1024 + 2, 1234ULL},
                                                    {0.000001f, 1024 * 1024 + 1, 1234ULL}};
typedef ternaryOpTest<float> ternaryOpTestF;
TEST_P(ternaryOpTestF, Result)
{
  ASSERT_TRUE(
    devArrMatch(out_add_ref, out_add, params.len, raft::CompareApprox<float>(params.tolerance)));
  ASSERT_TRUE(
    devArrMatch(out_mul_ref, out_mul, params.len, raft::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ternaryOpTests, ternaryOpTestF, ::testing::ValuesIn(inputsf));

const std::vector<BinaryOpInputs<double>> inputsd = {{0.00000001, 1024 * 1024, 1234ULL},
                                                     {0.00000001, 1024 * 1024 + 2, 1234ULL},
                                                     {0.00000001, 1024 * 1024 + 1, 1234ULL}};
typedef ternaryOpTest<double> ternaryOpTestD;
TEST_P(ternaryOpTestD, Result)
{
  ASSERT_TRUE(
    devArrMatch(out_add_ref, out_add, params.len, raft::CompareApprox<double>(params.tolerance)));
  ASSERT_TRUE(
    devArrMatch(out_mul_ref, out_mul, params.len, raft::CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ternaryOpTests, ternaryOpTestD, ::testing::ValuesIn(inputsd));

}  // end namespace LinAlg
}  // end namespace MLCommon
