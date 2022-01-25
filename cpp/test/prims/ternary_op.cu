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
#include <gtest/gtest.h>
#include <linalg/ternary_op.cuh>
#include <raft/cudart_utils.h>
#include <raft/random/rng.hpp>

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
  ternaryOpTest()
    : params(::testing::TestWithParam<BinaryOpInputs<T>>::GetParam()),
      stream(handle.get_stream()),
      out_add_ref(params.len, stream),
      out_add(params.len, stream),
      out_mul_ref(params.len, stream),
      out_mul(params.len, stream)
  {
  }

  void SetUp() override
  {
    raft::random::Rng rng(params.seed);
    int len = params.len;
    rmm::device_uvector<T> in1(len, stream);
    rmm::device_uvector<T> in2(len, stream);
    rmm::device_uvector<T> in3(len, stream);

    rng.fill(out_add_ref.data(), len, T(6.0), stream);
    rng.fill(out_mul_ref.data(), len, T(6.0), stream);
    rng.fill(in1.data(), len, T(1.0), stream);
    rng.fill(in2.data(), len, T(2.0), stream);
    rng.fill(in3.data(), len, T(3.0), stream);

    auto add = [] __device__(T a, T b, T c) { return a + b + c; };
    auto mul = [] __device__(T a, T b, T c) { return a * b * c; };
    ternaryOp(out_add.data(), in1.data(), in2.data(), in3.data(), len, add, stream);
    ternaryOp(out_mul.data(), in1.data(), in2.data(), in3.data(), len, mul, stream);
  }

 protected:
  BinaryOpInputs<T> params;
  raft::handle_t handle;
  cudaStream_t stream = 0;

  rmm::device_uvector<T> out_add_ref, out_add, out_mul_ref, out_mul;
};

const std::vector<BinaryOpInputs<float>> inputsf = {{0.000001f, 1024 * 1024, 1234ULL},
                                                    {0.000001f, 1024 * 1024 + 2, 1234ULL},
                                                    {0.000001f, 1024 * 1024 + 1, 1234ULL}};
typedef ternaryOpTest<float> ternaryOpTestF;
TEST_P(ternaryOpTestF, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_add_ref.data(), out_add.data(), params.len, raft::CompareApprox<float>(params.tolerance)));
  ASSERT_TRUE(devArrMatch(
    out_mul_ref.data(), out_mul.data(), params.len, raft::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ternaryOpTests, ternaryOpTestF, ::testing::ValuesIn(inputsf));

const std::vector<BinaryOpInputs<double>> inputsd = {{0.00000001, 1024 * 1024, 1234ULL},
                                                     {0.00000001, 1024 * 1024 + 2, 1234ULL},
                                                     {0.00000001, 1024 * 1024 + 1, 1234ULL}};
typedef ternaryOpTest<double> ternaryOpTestD;
TEST_P(ternaryOpTestD, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_add_ref.data(), out_add.data(), params.len, raft::CompareApprox<double>(params.tolerance)));
  ASSERT_TRUE(devArrMatch(
    out_mul_ref.data(), out_mul.data(), params.len, raft::CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ternaryOpTests, ternaryOpTestD, ::testing::ValuesIn(inputsd));

}  // end namespace LinAlg
}  // end namespace MLCommon
