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
#include "linalg/unary_op.h"
#include "random/rng.h"
#include "test_utils.h"
#include "unary_op.h"

namespace MLCommon {
namespace LinAlg {

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename InType, typename IdxType = int, typename OutType = InType>
void unaryOpLaunch(OutType *out, const InType *in, InType scalar, IdxType len,
                   cudaStream_t stream) {
  if (in == nullptr) {
    auto op = [scalar] __device__(OutType * ptr, IdxType idx) {
      *ptr = static_cast<OutType>(scalar * idx);
    };
    writeOnlyUnaryOp<OutType, decltype(op), IdxType>(out, len, op, stream);
  } else {
    auto op = [scalar] __device__(InType in) {
      return static_cast<OutType>(in * scalar);
    };
    unaryOp<InType, decltype(op), IdxType, OutType>(out, in, len, op, stream);
  }
}

template <typename InType, typename IdxType, typename OutType = InType>
class UnaryOpTest
  : public ::testing::TestWithParam<UnaryOpInputs<InType, IdxType, OutType>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<
      UnaryOpInputs<InType, IdxType, OutType>>::GetParam();
    Random::Rng r(params.seed);
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto len = params.len;
    allocate(in, len);
    allocate(out_ref, len);
    allocate(out, len);
    r.uniform(in, len, InType(-1.0), InType(1.0), stream);
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out_ref));
    CUDA_CHECK(cudaFree(out));
  }

  virtual void DoTest() {
    auto len = params.len;
    auto scalar = params.scalar;
    naiveScale(out_ref, in, scalar, len, stream);
    unaryOpLaunch(out, in, scalar, len, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                            CompareApprox<OutType>(params.tolerance)));
  }

  UnaryOpInputs<InType, IdxType, OutType> params;
  InType *in;
  OutType *out_ref, *out;
  cudaStream_t stream;
};

template <typename OutType, typename IdxType>
class WriteOnlyUnaryOpTest : public UnaryOpTest<OutType, IdxType, OutType> {
 protected:
  void DoTest() override {
    auto len = this->params.len;
    auto scalar = this->params.scalar;
    naiveScale(this->out_ref, (OutType *)nullptr, scalar, len, this->stream);
    unaryOpLaunch(this->out, (OutType *)nullptr, scalar, len, this->stream);
    CUDA_CHECK(cudaStreamSynchronize(this->stream));
    ASSERT_TRUE(devArrMatch(this->out_ref, this->out, this->params.len,
                            CompareApprox<OutType>(this->params.tolerance)));
  }
};

#define UNARY_OP_TEST(Name, inputs)  \
  TEST_P(Name, Result) { DoTest(); } \
  INSTANTIATE_TEST_CASE_P(UnaryOpTests, Name, ::testing::ValuesIn(inputs))

const std::vector<UnaryOpInputs<float, int>> inputsf_i32 = {
  {0.000001f, 1024 * 1024, 2.f, 1234ULL}};
typedef UnaryOpTest<float, int> UnaryOpTestF_i32;
UNARY_OP_TEST(UnaryOpTestF_i32, inputsf_i32);
typedef WriteOnlyUnaryOpTest<float, int> WriteOnlyUnaryOpTestF_i32;
UNARY_OP_TEST(WriteOnlyUnaryOpTestF_i32, inputsf_i32);

const std::vector<UnaryOpInputs<float, size_t>> inputsf_i64 = {
  {0.000001f, 1024 * 1024, 2.f, 1234ULL}};
typedef UnaryOpTest<float, size_t> UnaryOpTestF_i64;
UNARY_OP_TEST(UnaryOpTestF_i64, inputsf_i64);
typedef WriteOnlyUnaryOpTest<float, size_t> WriteOnlyUnaryOpTestF_i64;
UNARY_OP_TEST(WriteOnlyUnaryOpTestF_i64, inputsf_i64);

const std::vector<UnaryOpInputs<float, int, double>> inputsf_i32_d = {
  {0.000001f, 1024 * 1024, 2.f, 1234ULL}};
typedef UnaryOpTest<float, int, double> UnaryOpTestF_i32_D;
UNARY_OP_TEST(UnaryOpTestF_i32_D, inputsf_i32_d);

const std::vector<UnaryOpInputs<double, int>> inputsd_i32 = {
  {0.00000001, 1024 * 1024, 2.0, 1234ULL}};
typedef UnaryOpTest<double, int> UnaryOpTestD_i32;
UNARY_OP_TEST(UnaryOpTestD_i32, inputsd_i32);
typedef WriteOnlyUnaryOpTest<double, int> WriteOnlyUnaryOpTestD_i32;
UNARY_OP_TEST(WriteOnlyUnaryOpTestD_i32, inputsd_i32);

const std::vector<UnaryOpInputs<double, size_t>> inputsd_i64 = {
  {0.00000001, 1024 * 1024, 2.0, 1234ULL}};
typedef UnaryOpTest<double, size_t> UnaryOpTestD_i64;
UNARY_OP_TEST(UnaryOpTestD_i64, inputsd_i64);
typedef WriteOnlyUnaryOpTest<double, size_t> WriteOnlyUnaryOpTestD_i64;
UNARY_OP_TEST(WriteOnlyUnaryOpTestD_i64, inputsd_i64);

}  // end namespace LinAlg
}  // end namespace MLCommon
