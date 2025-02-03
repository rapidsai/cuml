/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <raft/linalg/add.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

template <typename T, typename IdxType = int>
struct DevScalarInputs {
  T tolerance;
  IdxType len;
  T scalar;
  bool add;
  unsigned long long int seed;
};

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename T, typename IdxType = int>
void unaryOpLaunch(T* out, const T* in, T scalar, IdxType len, bool add, cudaStream_t stream)
{
  raft::linalg::unaryOp(
    out,
    in,
    len,
    [scalar, add] __device__(T in) { return add ? in + scalar : in - scalar; },
    stream);
}

template <typename T, typename IdxType>
class DevScalarTest : public ::testing::TestWithParam<DevScalarInputs<T, IdxType>> {
 protected:
  DevScalarTest() : in(0, stream), out_ref(0, stream), out(0, stream), scalar(stream) {}

  void SetUp() override
  {
    params = ::testing::TestWithParam<DevScalarInputs<T, IdxType>>::GetParam();
    raft::random::Rng r(params.seed);
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));

    auto len = params.len;

    in.resize(len, stream);
    out_ref.resize(len, stream);
    out.resize(len, stream);

    raft::update_device(scalar.data(), &params.scalar, 1, stream);
    r.uniform(in.data(), len, T(-1.0), T(1.0), stream);
    unaryOpLaunch(out_ref.data(), in.data(), params.scalar, len, params.add, stream);
    if (params.add) {
      addDevScalar(out.data(), in.data(), scalar.data(), len, stream);
    } else {
      subtractDevScalar(out.data(), in.data(), scalar.data(), len, stream);
    }
    RAFT_CUDA_TRY(cudaStreamDestroy(stream));
  }

 protected:
  cudaStream_t stream = 0;
  DevScalarInputs<T, IdxType> params;
  rmm::device_uvector<T> in, out_ref, out;
  rmm::device_scalar<T> scalar;
};

const std::vector<DevScalarInputs<float, int>> inputsf_i32 = {
  {0.000001f, 1024 * 1024, 2.f, true, 1234ULL}, {0.000001f, 1024 * 1024, 2.f, false, 1234ULL}};
typedef DevScalarTest<float, int> DevScalarTestF_i32;
TEST_P(DevScalarTestF_i32, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), params.len, MLCommon::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(DevScalarTests, DevScalarTestF_i32, ::testing::ValuesIn(inputsf_i32));

const std::vector<DevScalarInputs<float, size_t>> inputsf_i64 = {
  {0.000001f, 1024 * 1024, 2.f, true, 1234ULL}, {0.000001f, 1024 * 1024, 2.f, false, 1234ULL}};
typedef DevScalarTest<float, size_t> DevScalarTestF_i64;
TEST_P(DevScalarTestF_i64, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), params.len, MLCommon::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(DevScalarTests, DevScalarTestF_i64, ::testing::ValuesIn(inputsf_i64));

const std::vector<DevScalarInputs<double, int>> inputsd_i32 = {
  {0.00000001, 1024 * 1024, 2.0, true, 1234ULL}, {0.00000001, 1024 * 1024, 2.0, false, 1234ULL}};
typedef DevScalarTest<double, int> DevScalarTestD_i32;
TEST_P(DevScalarTestD_i32, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), params.len, MLCommon::CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(DevScalarTests, DevScalarTestD_i32, ::testing::ValuesIn(inputsd_i32));

const std::vector<DevScalarInputs<double, size_t>> inputsd_i64 = {
  {0.00000001, 1024 * 1024, 2.0, true, 1234ULL}, {0.00000001, 1024 * 1024, 2.0, false, 1234ULL}};
typedef DevScalarTest<double, size_t> DevScalarTestD_i64;
TEST_P(DevScalarTestD_i64, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), params.len, MLCommon::CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(DevScalarTests, DevScalarTestD_i64, ::testing::ValuesIn(inputsd_i64));

}  // end namespace linalg
}  // end namespace raft
