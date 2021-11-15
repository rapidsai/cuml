/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cuml/fil/multi_sum.cuh>

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>
#include <raft/random/rng.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <gtest/gtest.h>

#include <cstddef>

template <typename T>
__device__ void serial_multi_sum(const T* in, T* out, int n_groups, int n_values)
{
  __syncthreads();
  if (threadIdx.x < n_groups) {
    int reduction_id = threadIdx.x;
    T sum            = 0;
    for (int i = 0; i < n_values; ++i)
      sum += in[reduction_id + i * n_groups];
    out[reduction_id] = sum;
  }
  __syncthreads();
}

// the most threads a block can have
const int max_threads = 1024;

struct MultiSumTestParams {
  int radix;     // number of elements summed to 1 at each stage of the sum
  int n_groups;  // number of independent sums
  int n_values;  // number of elements to add in each sum
};

template <int R, typename T>
__device__ void test_single_radix(T thread_value, MultiSumTestParams p, int* block_error_flag)
{
  __shared__ T work[max_threads], correct_result[max_threads];
  work[threadIdx.x] = thread_value;
  serial_multi_sum(work, correct_result, p.n_groups, p.n_values);
  T sum = multi_sum<R>(work, p.n_groups, p.n_values);
  if (threadIdx.x < p.n_groups && 1e-4 < fabsf(sum - correct_result[threadIdx.x])) {
    atomicAdd(block_error_flag, 1);
  }
}

template <typename T>
__global__ void test_multi_sum_k(T* data, MultiSumTestParams* params, int* error_flags)
{
  MultiSumTestParams p = params[blockIdx.x];
  switch (p.radix) {
    case 2: test_single_radix<2>(data[threadIdx.x], p, &error_flags[blockIdx.x]); break;
    case 3: test_single_radix<3>(data[threadIdx.x], p, &error_flags[blockIdx.x]); break;
    case 4: test_single_radix<4>(data[threadIdx.x], p, &error_flags[blockIdx.x]); break;
    case 5: test_single_radix<5>(data[threadIdx.x], p, &error_flags[blockIdx.x]); break;
    case 6: test_single_radix<6>(data[threadIdx.x], p, &error_flags[blockIdx.x]); break;
  }
}

template <typename T>
class MultiSumTest : public testing::TestWithParam<int> {
 protected:
  void SetUp() override
  {
    block_dim_x = GetParam();
    data_d.resize(block_dim_x);
    this->generate_data();

    for (int radix = 2; radix <= 6; ++radix) {
      for (int n_groups = 1; n_groups < 15; ++n_groups) {  // >2x the max radix
        // 1..50 (if block_dim_x permits)
        for (int n_values = 1; n_values <= std::min(block_dim_x, 50) / n_groups; ++n_values)
          params_h.push_back({.radix = radix, .n_groups = n_groups, .n_values = n_values});
        // block_dim_x - 50 .. block_dim_x (if positive)
        // up until 50 would be included in previous loop
        for (int n_values = std::max(block_dim_x - 50, 51) / n_groups;
             n_values <= block_dim_x / n_groups;
             ++n_values)
          params_h.push_back({.radix = radix, .n_groups = n_groups, .n_values = n_values});
      }
    }
    params_d = params_h;
    error_d.resize(params_h.size());
    thrust::fill_n(error_d.begin(), params_h.size(), 0);
  }

  void check()
  {
    T* data_p               = data_d.data().get();
    MultiSumTestParams* p_p = params_d.data().get();
    int* error_p            = error_d.data().get();

    test_multi_sum_k<<<params_h.size(), block_dim_x>>>(data_p, p_p, error_p);
    CUDA_CHECK(cudaPeekAtLastError());
    error = error_d;
    CUDA_CHECK(cudaDeviceSynchronize());
    for (std::size_t i = 0; i < params_h.size(); ++i) {
      ASSERT(error[i] == 0,
             "test # %lu: block_dim_x %d multi_sum<%d>(on %d sets sized"
             " %d) gave wrong result",
             i,
             block_dim_x,
             params_h[i].radix,
             params_h[i].n_values,
             params_h[i].n_groups);
    }
  }

  virtual void generate_data() = 0;

  // parameters
  raft::handle_t handle;
  int block_dim_x;
  thrust::host_vector<MultiSumTestParams> params_h;
  thrust::device_vector<MultiSumTestParams> params_d;
  thrust::host_vector<int> error;
  thrust::device_vector<int> error_d;
  thrust::device_vector<T> data_d;
};

std::vector<int> block_sizes = []() {
  std::vector<int> res;
  for (int i = 2; i < 50; ++i)
    res.push_back(i);
  for (int i = max_threads - 50; i <= max_threads; ++i)
    res.push_back(i);
  return res;
}();

class MultiSumTestFloat : public MultiSumTest<float> {
 public:
  void generate_data()
  {
    raft::random::Rng r(4321);
    r.uniform(data_d.data().get(), data_d.size(), -1.0f, 1.0f, cudaStreamDefault);
  }
};
TEST_P(MultiSumTestFloat, Import) { check(); }
INSTANTIATE_TEST_CASE_P(FilTests, MultiSumTestFloat, testing::ValuesIn(block_sizes));

class MultiSumTestInt : public MultiSumTest<int> {
 public:
  void generate_data()
  {
    raft::random::Rng r(4321);
    r.uniformInt(data_d.data().get(), data_d.size(), -123'456, 123'456, cudaStreamDefault);
  }
};
TEST_P(MultiSumTestInt, Import) { check(); }
INSTANTIATE_TEST_CASE_P(FilTests, MultiSumTestInt, testing::ValuesIn(block_sizes));
