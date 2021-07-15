/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <random>
#include <vector>

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>
#include <raft/mr/device/allocator.hpp>

#include "test_utils.h"

#include <cuml/common/device_buffer.hpp>

#include <timeSeries/fillna.cuh>

namespace MLCommon {
namespace TimeSeries {

using namespace std;

struct SeriesDescriptor {
  int leading_nan;
  int random_nan;
  int trailing_nan;
};

template <typename T>
struct FillnaInputs {
  int batch_size;
  int n_obs;
  std::vector<SeriesDescriptor> descriptors;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const FillnaInputs<T>& dims)
{
  return os;
}

template <typename T>
class FillnaTest : public ::testing::TestWithParam<FillnaInputs<T>> {
 protected:
  void basicTest()
  {
    raft::handle_t handle;

    params = ::testing::TestWithParam<FillnaInputs<T>>::GetParam();

    device_buffer<T> y(
      handle.get_device_allocator(), handle.get_stream(), params.n_obs * params.batch_size);

    std::vector<T> h_y(params.n_obs * params.batch_size);

    /* Generate random data */
    std::default_random_engine generator(params.seed);
    std::uniform_real_distribution<T> real_distribution(-2.0, 2.0);
    std::uniform_int_distribution<int> int_distribution(0, params.n_obs - 1);
    for (int i = 0; i < params.n_obs * params.batch_size; i++)
      h_y[i] = real_distribution(generator);
    for (int bid = 0; bid < params.batch_size; bid++) {
      for (int i = 0; i < params.descriptors[bid].leading_nan; i++)
        h_y[bid * params.n_obs + i] = nan("");
      for (int i = 0; i < params.descriptors[bid].trailing_nan; i++)
        h_y[(bid + 1) * params.n_obs - 1 - i] = nan("");
      for (int i = 0; i < params.descriptors[bid].random_nan; i++) {
        h_y[bid * params.n_obs + int_distribution(generator)] = nan("");
      }
    }

    /* Copy to device */
    raft::update_device(
      y.data(), h_y.data(), params.n_obs * params.batch_size, handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    /* Compute using tested prims */
    fillna(y.data(),
           params.batch_size,
           params.n_obs,
           handle.get_device_allocator(),
           handle.get_stream());

    /* Compute reference results */
    for (int bid = 0; bid < params.batch_size; bid++) {
      // Forward pass
      for (int i = 1; i < params.n_obs; i++) {
        if (std::isnan(h_y[bid * params.n_obs + i]))
          h_y[bid * params.n_obs + i] = h_y[bid * params.n_obs + i - 1];
      }

      // Backward pass
      for (int i = params.n_obs - 2; i >= 0; i--) {
        if (std::isnan(h_y[bid * params.n_obs + i]))
          h_y[bid * params.n_obs + i] = h_y[bid * params.n_obs + i + 1];
      }
    }

    /* Check results */
    match = devArrMatchHost(h_y.data(),
                            y.data(),
                            params.n_obs * params.batch_size,
                            raft::Compare<T>(),
                            handle.get_stream());
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  FillnaInputs<T> params;

  testing::AssertionResult match = testing::AssertionFailure();
};

const std::vector<FillnaInputs<float>> gemm_inputsf = {
  {3, 42, {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}}, 12345U},
  {4, 100, {{70, 0, 0}, {0, 20, 0}, {0, 0, 63}, {31, 25, 33}, {20, 15, 42}}, 12345U},
};

const std::vector<FillnaInputs<double>> gemm_inputsd = {
  {3, 42, {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}}, 12345U},
  {4, 100, {{70, 0, 0}, {0, 20, 0}, {0, 0, 63}, {31, 25, 33}, {20, 15, 42}}, 12345U},
};

typedef FillnaTest<float> FillnaTestF;
TEST_P(FillnaTestF, Result) { EXPECT_TRUE(match); }

typedef FillnaTest<double> FillnaTestD;
TEST_P(FillnaTestD, Result) { EXPECT_TRUE(match); }

INSTANTIATE_TEST_CASE_P(FillnaTests, FillnaTestF, ::testing::ValuesIn(gemm_inputsf));

INSTANTIATE_TEST_CASE_P(FillnaTests, FillnaTestD, ::testing::ValuesIn(gemm_inputsd));

}  // namespace TimeSeries
}  // namespace MLCommon