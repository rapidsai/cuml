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
#include <metrics/dispersion.cuh>
#include <raft/cuda_utils.cuh>
#include <raft/interruptible.hpp>
#include <raft/random/rng.hpp>
#include <rmm/device_uvector.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

namespace MLCommon {
namespace Metrics {

template <typename T>
struct DispersionInputs {
  T tolerance;
  int dim, clusters;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const DispersionInputs<T>& dims)
{
  return os;
}

template <typename T>
class DispersionTest : public ::testing::TestWithParam<DispersionInputs<T>> {
 protected:
  DispersionTest() : exp_mean(0, stream), act_mean(0, stream) {}

  void SetUp() override
  {
    params = ::testing::TestWithParam<DispersionInputs<T>>::GetParam();
    raft::random::Rng r(params.seed);
    int len = params.clusters * params.dim;
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));
    rmm::device_uvector<T> data(len, stream);
    rmm::device_uvector<int> counts(params.clusters, stream);
    exp_mean.resize(params.dim, stream);
    act_mean.resize(params.dim, stream);
    r.uniform(data.data(), len, (T)-1.0, (T)1.0, stream);
    r.uniformInt(counts.data(), params.clusters, 1, 100, stream);
    std::vector<int> h_counts(params.clusters, 0);
    raft::update_host(&(h_counts[0]), counts.data(), params.clusters, stream);
    npoints = 0;
    for (const auto& val : h_counts) {
      npoints += val;
    }
    actualVal = dispersion(
      data.data(), counts.data(), act_mean.data(), params.clusters, npoints, params.dim, stream);
    expectedVal = T(0);
    std::vector<T> h_data(len, T(0));
    raft::update_host(&(h_data[0]), data.data(), len, stream);
    std::vector<T> mean(params.dim, T(0));
    for (int i = 0; i < params.clusters; ++i) {
      for (int j = 0; j < params.dim; ++j) {
        mean[j] += h_data[i * params.dim + j] * T(h_counts[i]);
      }
    }
    for (int i = 0; i < params.dim; ++i) {
      mean[i] /= T(npoints);
    }
    raft::update_device(exp_mean.data(), &(mean[0]), params.dim, stream);
    for (int i = 0; i < params.clusters; ++i) {
      for (int j = 0; j < params.dim; ++j) {
        auto diff = h_data[i * params.dim + j] - mean[j];
        expectedVal += diff * diff * T(h_counts[i]);
      }
    }
    expectedVal = sqrt(expectedVal);
    raft::interruptible::synchronize(stream);
  }

  void TearDown() override { RAFT_CUDA_TRY(cudaStreamDestroy(stream)); }

 protected:
  DispersionInputs<T> params;
  rmm::device_uvector<T> exp_mean, act_mean;
  cudaStream_t stream = 0;
  int npoints;
  T expectedVal, actualVal;
};

const std::vector<DispersionInputs<float>> inputsf = {
  {0.001f, 10, 1000, 1234ULL}, {0.001f, 100, 100, 1234ULL}, {0.001f, 1000, 1000, 1234ULL}};
typedef DispersionTest<float> DispersionTestF;
TEST_P(DispersionTestF, Result)
{
  auto eq = raft::CompareApprox<float>(params.tolerance);
  ASSERT_TRUE(devArrMatch(exp_mean.data(), act_mean.data(), params.dim, eq));
  ASSERT_TRUE(match(expectedVal, actualVal, eq));
}
INSTANTIATE_TEST_CASE_P(DispersionTests, DispersionTestF, ::testing::ValuesIn(inputsf));

const std::vector<DispersionInputs<double>> inputsd = {
  {0.001, 10, 1000, 1234ULL}, {0.001, 100, 100, 1234ULL}, {0.001, 1000, 1000, 1234ULL}};
typedef DispersionTest<double> DispersionTestD;
TEST_P(DispersionTestD, Result)
{
  auto eq = raft::CompareApprox<double>(params.tolerance);
  ASSERT_TRUE(devArrMatch(exp_mean.data(), act_mean.data(), params.dim, eq));
  ASSERT_TRUE(match(expectedVal, actualVal, eq));
}
INSTANTIATE_TEST_CASE_P(DispersionTests, DispersionTestD, ::testing::ValuesIn(inputsd));

}  // end namespace Metrics
}  // end namespace MLCommon
