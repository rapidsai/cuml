/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <thrust/count.h>
#include <thrust/device_vector.h>

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>
#include <random/make_arima.cuh>
#include "test_utils.h"

namespace MLCommon {
namespace Random {

/* This test only proves that the generator runs without errors, not
 * correctness! */

struct MakeArimaInputs {
  int batch_size, n_obs;
  int p, d, q, P, D, Q, s, k;
  raft::random::GeneratorType gtype;
  uint64_t seed;
};

template <typename T>
class MakeArimaTest : public ::testing::TestWithParam<MakeArimaInputs> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<MakeArimaInputs>::GetParam();

    // Scales of the different random components
    T scale = 1.0, noise_scale = 0.2;
    T intercept_scale =
      params.d + params.D == 0 ? 1.0 : (params.d + params.D == 1 ? 0.2 : 0.01);

    ML::ARIMAOrder order = {params.p, params.d, params.q, params.P,
                            params.D, params.Q, params.s, params.k};

    allocator.reset(new raft::mr::device::default_allocator);
    CUDA_CHECK(cudaStreamCreate(&stream));

    raft::allocate(data, params.batch_size * params.n_obs);

    // Create the time series dataset
    make_arima(data, params.batch_size, params.n_obs, order, allocator, stream,
               scale, noise_scale, intercept_scale, params.seed, params.gtype);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  MakeArimaInputs params;
  T *data;
  std::shared_ptr<deviceAllocator> allocator;
  cudaStream_t stream;
};

const std::vector<MakeArimaInputs> make_arima_inputs = {
  {100, 200, 1, 1, 2, 0, 0, 0, 0, 1, raft::random::GenPhilox, 1234ULL},
  {1000, 100, 3, 0, 0, 1, 1, 0, 4, 1, raft::random::GenPhilox, 1234ULL},
  {10000, 150, 2, 1, 2, 0, 1, 2, 4, 0, raft::random::GenPhilox, 1234ULL}};

typedef MakeArimaTest<float> MakeArimaTestF;
TEST_P(MakeArimaTestF, Result) { CUDA_CHECK(cudaStreamSynchronize(stream)); }
INSTANTIATE_TEST_CASE_P(MakeArimaTests, MakeArimaTestF,
                        ::testing::ValuesIn(make_arima_inputs));

typedef MakeArimaTest<double> MakeArimaTestD;
TEST_P(MakeArimaTestD, Result) { CUDA_CHECK(cudaStreamSynchronize(stream)); }
INSTANTIATE_TEST_CASE_P(MakeArimaTests, MakeArimaTestD,
                        ::testing::ValuesIn(make_arima_inputs));

}  // end namespace Random
}  // end namespace MLCommon
