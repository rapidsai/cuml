/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <decisiontree/quantile/quantile.h>
#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <raft/linalg/transpose.h>
#include <test_utils.h>
#include <cuml/datasets/make_blobs.hpp>
#include <cuml/datasets/make_regression.hpp>
#include <cuml/ensemble/randomforest.hpp>
#include <metrics/scores.cuh>
#include <raft/random/rng_impl.cuh>

namespace ML {

using namespace MLCommon;

// N O T E
// If the generated data has duplicate values at the quantile boundary, the
// test will fail. Probability of such a occurrence is low but should that
// happen, change the seed to workaround the issue.

struct inputs {
  int n_rows;
  int n_bins;
  uint64_t seed;
};

// Generate data with some outliers
template <typename T>
__global__ void generateData(T* data, int length, uint64_t seed)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  raft::random::detail::Kiss99Generator gen(seed, 0, 0);

  T num = static_cast<T>(0.0);
  uint32_t toss, multiplier;

  for (int i = tid; i < length; i += blockDim.x * gridDim.x) {
    // Generate a number
    gen.next(multiplier);
    multiplier &= 0xFF;

    gen.next(toss);
    toss &= 0xFF;

    gen.next(num);
    // Generate 5% outliers
    // value of toss is in [0, 255], 5 % of that is 13
    if (toss < 13) {
      // Number between [-multiplier, +multiplier]
      data[i] = multiplier * (1 - 2 * num);
    } else {
      // Number between [-1, 1]
      data[i] = (1 - 2 * num);
    }
  }
  return;
}

template <typename T>
__global__ void computeHistogram(int* histogram, T* data, int length, T* quantiles, int n_bins)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < length; i += blockDim.x * gridDim.x) {
    T num = data[i];
    for (int j = 0; j < n_bins; j++) {
      if (num <= quantiles[j]) {
        atomicAdd(&histogram[j], 1);
        break;
      }
    }
  }
  return;
}

template <typename T>
class RFQuantileTest : public ::testing::TestWithParam<inputs> {
 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<inputs>::GetParam();

    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.reset(new raft::handle_t());
    handle->set_stream(stream);
    auto allocator   = handle->get_device_allocator();
    auto h_allocator = handle->get_host_allocator();

    data        = (T*)allocator->allocate(params.n_rows * sizeof(T), stream);
    quantiles   = (T*)allocator->allocate(params.n_bins * sizeof(T), stream);
    histogram   = (int*)allocator->allocate(params.n_bins * sizeof(int), stream);
    h_histogram = (int*)h_allocator->allocate(params.n_bins * sizeof(int), stream);

    CUDA_CHECK(cudaMemset(histogram, 0, params.n_bins * sizeof(int)));
    const int TPB = 128;
    int numBlocks = raft::ceildiv(params.n_rows, TPB);
    generateData<<<numBlocks, TPB, 0, stream>>>(data, params.n_rows, params.seed);
    DT::computeQuantiles(quantiles, params.n_bins, data, params.n_rows, 1, allocator, stream);

    computeHistogram<<<numBlocks, TPB, 0, stream>>>(
      histogram, data, params.n_rows, quantiles, params.n_bins);

    CUDA_CHECK(cudaMemcpyAsync(
      h_histogram, histogram, params.n_bins * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown() override
  {
    auto allocator   = handle->get_device_allocator();
    auto h_allocator = handle->get_host_allocator();

    allocator->deallocate(data, params.n_rows * sizeof(T), stream);
    allocator->deallocate(quantiles, params.n_bins * sizeof(T), stream);
    allocator->deallocate(histogram, params.n_bins * sizeof(int), stream);
    h_allocator->deallocate(h_histogram, params.n_bins * sizeof(int), stream);

    handle.reset();
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void test_histogram()
  {
    int max_items_per_bin = raft::ceildiv(params.n_rows, params.n_bins);
    int min_items_per_bin = max_items_per_bin - 1;
    int total_items       = 0;
    for (int b = 0; b < params.n_bins; b++) {
      ASSERT_TRUE(h_histogram[b] == max_items_per_bin || h_histogram[b] == min_items_per_bin)
        << "No. samples in bin[" << b << "] = " << h_histogram[b] << " Expected "
        << max_items_per_bin << " or " << min_items_per_bin << std::endl;
      total_items += h_histogram[b];
    }
    ASSERT_EQ(params.n_rows, total_items)
      << "Some samples from dataset are either missed of double counted in "
         "quantile bins"
      << std::endl;
  }

 protected:
  std::shared_ptr<raft::handle_t> handle;
  cudaStream_t stream;
  inputs params;

  T *data, *quantiles;
  bool result;
  int *histogram, *h_histogram;
};

//-------------------------------------------------------------------------------------------------------------------------------------
const std::vector<inputs> inputs = {{1000, 16, 6078587519764079670LLU},
                                    {1130, 32, 4884670006177930266LLU},
                                    {1752, 67, 9175325892580481371LLU},
                                    {2307, 99, 9507819643927052255LLU},
                                    {5000, 128, 9507819643927052255LLU}};

typedef RFQuantileTest<float> RFQuantileTestF;
TEST_P(RFQuantileTestF, test) { test_histogram(); }

INSTANTIATE_TEST_CASE_P(RFQuantileTests, RFQuantileTestF, ::testing::ValuesIn(inputs));

typedef RFQuantileTest<double> RFQuantileTestD;
TEST_P(RFQuantileTestD, test) { test_histogram(); }

INSTANTIATE_TEST_CASE_P(RFQuantileTests, RFQuantileTestD, ::testing::ValuesIn(inputs));

}  // end namespace ML
