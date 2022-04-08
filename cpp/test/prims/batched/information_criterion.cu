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

#include <test_utils.h>

#include <metrics/batched/information_criterion.cuh>

#include <raft/cudart_utils.h>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

namespace MLCommon {
namespace Metrics {
namespace Batched {

template <typename T>
void naive_ic(
  T* h_ic, const T* h_loglike, IC_Type ic_type, int n_params, int batch_size, int n_samples)
{
  T ic_base{};
  T N = static_cast<T>(n_params);
  T M = static_cast<T>(n_samples);
  switch (ic_type) {
    case AIC: ic_base = (T)2 * N; break;
    case AICc: ic_base = (T)2 * (N + (N * (N + (T)1)) / (M - N - (T)1)); break;
    case BIC: ic_base = std::log(M) * N; break;
  }
#pragma omp parallel for
  for (int bid = 0; bid < batch_size; bid++) {
    h_ic[bid] = ic_base - (T)2.0 * h_loglike[bid];
  }
}

template <typename T>
struct BatchedICInputs {
  int batch_size;
  int n_params;
  int n_samples;
  IC_Type ic_type;
  T tolerance;
};

template <typename T>
class BatchedICTest : public ::testing::TestWithParam<BatchedICInputs<T>> {
 protected:
  BatchedICTest()
    : params(::testing::TestWithParam<BatchedICInputs<T>>::GetParam()),
      res_d(sizeof(T) * params.batch_size, stream)
  {
  }

  void SetUp() override
  {
    using std::vector;

    // Create stream
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));

    // Create arrays
    std::vector<T> loglike_h = std::vector<T>(params.batch_size);
    res_h.resize(params.batch_size);
    rmm::device_uvector<T> loglike_d(sizeof(T) * params.batch_size, stream);

    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> udis(0.001, 1.0);  // 0 has no log
    for (int i = 0; i < params.batch_size; i++)
      loglike_h[i] = std::log(udis(gen));

    // Copy the data to the device
    raft::update_device(loglike_d.data(), loglike_h.data(), params.batch_size, stream);

    // Compute the tested results
    information_criterion(res_d.data(),
                          loglike_d.data(),
                          params.ic_type,
                          params.n_params,
                          params.batch_size,
                          params.n_samples,
                          stream);

    // Compute the expected results
    naive_ic(res_h.data(),
             loglike_h.data(),
             params.ic_type,
             params.n_params,
             params.batch_size,
             params.n_samples);
  }

 protected:
  cudaStream_t stream = 0;
  BatchedICInputs<T> params;
  rmm::device_uvector<T> res_d;
  std::vector<T> res_h;
};

// Test parameters (op, n_batches, m, n, p, q, tolerance)
const std::vector<BatchedICInputs<double>> inputsd = {
  {1, 5, 52, AIC, 1e-3}, {10, 7, 100, AICc, 1e-3}, {67, 2, 350, BIC, 1e-3}};

// Test parameters (op, n_batches, m, n, p, q, tolerance)
const std::vector<BatchedICInputs<float>> inputsf = {
  {1, 5, 52, AIC, 1e-3}, {10, 7, 100, AICc, 1e-3}, {67, 2, 350, BIC, 1e-3}};

using BatchedICTestD = BatchedICTest<double>;
using BatchedICTestF = BatchedICTest<float>;
TEST_P(BatchedICTestD, Result)
{
  ASSERT_TRUE(devArrMatchHost(res_h.data(),
                              res_d.data(),
                              params.batch_size,
                              raft::CompareApprox<double>(params.tolerance),
                              stream));
}
TEST_P(BatchedICTestF, Result)
{
  ASSERT_TRUE(devArrMatchHost(res_h.data(),
                              res_d.data(),
                              params.batch_size,
                              raft::CompareApprox<float>(params.tolerance),
                              stream));
}

INSTANTIATE_TEST_CASE_P(BatchedICTests, BatchedICTestD, ::testing::ValuesIn(inputsd));
INSTANTIATE_TEST_CASE_P(BatchedICTests, BatchedICTestF, ::testing::ValuesIn(inputsf));

}  // namespace Batched
}  // namespace Metrics
}  // namespace MLCommon
