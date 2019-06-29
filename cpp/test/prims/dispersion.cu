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
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "cuda_utils.h"
#include "random/rng.h"
#include "metrics/dispersion.h"
#include "test_utils.h"

namespace MLCommon {
namespace Metrics {

template <typename T>
struct DispersionInputs {
  T tolerance;
  int dim, clusters;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const DispersionInputs<T> &dims) {
  return os;
}

template <typename T>
class DispersionTest : public ::testing::TestWithParam<DispersionInputs<T>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<DispersionInputs<T>>::GetParam();
    Random::Rng r(params.seed);
    int len = params.clusters * params.dim;
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocator.reset(new defaultDeviceAllocator);
    allocate(data, len);
    allocate(counts, params.clusters)
    r.uniform(data, len, (T)-1.0, (T)1.0, stream);
    r.uniformInt(counts, params.clusters, 1, 100, stream);
    std::vector<int> h_counts(params.clusters, 0);
    updateHost(&(h_counts[0]), counts, params.clusters, stream);
    npoints = 0;
    for (const auto &val : h_counts) {
      npoints += val;
    }
    computedVal = dispersion(data, counts, nullptr, params.clusters, npoints,
                             params.dim, allocator, stream);
    actualVal = T(0);
    std_vector<T> h_data(len, T(0));
    updateHost(&(h_data[0]), data, len, stream);
    std::vector<T> mean(params.dim, T(0));
    for (int i = 0; i < params.clusters; ++i) {
      for (int j = 0; j < params.dim; ++j) {
        mean[j] += h_data[i * params.dim + j] * T(h_counts[i]);
      }
    }
    for (int i = 0; i < params.dim; ++i) {
      mean[i] /= T(npoints);
    }
    for (int i = 0; i < params.clusters; ++i) {
      for (int j = 0; j < params.dim; ++j) {
        auto diff = h_data[i * params.dim + j] - mean[j];
        actualVal += diff * diff * T(h_counts[i]);
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(counts));
  }

 protected:
  DispersionInputs<T> params;
  T *data;
  int *counts;
  cudaStream_t stream;
  int npoints;
  std::shared_ptr<deviceAllocator> allocator;
  T actualVal, computedVal;
};


const std::vector<DispersionInputs<float>> inputsf = {
  {0.001f, 10, 1000, 1234ULL}, {0.001f, 100, 100, 1234ULL},
  {0.001f, 1000, 1000, 1234ULL}};
typedef DispersionTest<float> DispersionTestF;
TEST_P(DispersionTestF, Result) {
  ASSERT_TRUE(match(actualVal, computedVal,
                    CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(DispersionTests, DispersionTestF,
                        ::testing::ValuesIn(inputsf));

const std::vector<DispersionInputs<double>> inputsd = {
  {0.001, 10, 1000, 1234ULL}, {0.001, 100, 100, 1234ULL},
  {0.001, 1000, 1000, 1234ULL}};
typedef DispersionTest<double> DispersionTestD;
TEST_P(DispersionTestD, Result) {
  ASSERT_TRUE(match(actualVal, computedVal,
                    CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(DispersionTests, DispersionTestD,
                        ::testing::ValuesIn(inputsd));

}  // end namespace Metrics
}  // end namespace MLCommon
