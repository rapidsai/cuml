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
#include <cub/cub.cuh>
#include "cuda_utils.h"
#include "random/make_blobs.h"
#include "test_utils.h"

namespace MLCommon {
namespace Random {

template <typename T, int TPB>
__global__ void meanKernel(T* out, const T* data, int len) {
  typedef cub::BlockReduce<T, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  T val = tid < len ? data[tid] : T(0);
  T x = BlockReduce(temp_storage).Sum(val);
  __syncthreads();
  T xx = BlockReduce(temp_storage).Sum(val * val);
  __syncthreads();
  if (threadIdx.x == 0) {
    myAtomicAdd(out, x);
    myAtomicAdd(out + 1, xx);
  }
}

template <typename T>
struct MakeBlobsInputs {
  T tolerance;
  int rows, cols, n_clusters;
  T mean, std;
  GeneratorType gtype;
  uint64_t seed;
};

template <typename T>
class MakeBlobsTest
  : public ::testing::TestWithParam<MakeBlobsInputs<T>> {
 protected:
  void SetUp() override {
    // Tests are configured with their expected test-values sigma. For example,
    // 4 x sigma indicates the test shouldn't fail 99.9% of the time.
    num_sigma = 10;
    allocator.reset(new defaultDeviceAllocator);
    params = ::testing::TestWithParam<MakeBlobsInputs<T>>::GetParam();
    int len = params.rows * params.cols;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    Rng r(params.seed, params.gtype);
    allocate(data, len);
    allocate(labels, params.rows);
    allocate(stats, 2, true);
    allocate(mu_vec, params.cols * params.n_clusters);
    r.fill(mu_vec, params.cols * params.n_clusters, params.mean, stream);
    T* sigma_vec = nullptr;
    make_blobs(data, labels, params.rows, params.cols, params.n_clusters,
               allocator, stream, mu_vec, sigma_vec, params.std, false,
               (T)-10.0, (T)10.0, params.seed, params.gtype);
    static const int threads = 128;
    meanKernel<T, threads>
      <<<ceildiv(len, threads), threads, 0, stream>>>(stats, data, len);
    updateHost<T>(h_stats, stats, 2, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    h_stats[0] /= len;
    h_stats[1] = (h_stats[1] / len) - (h_stats[0] * h_stats[0]);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(labels));
    CUDA_CHECK(cudaFree(stats));
    CUDA_CHECK(cudaFree(mu_vec));
  }

  void getExpectedMeanVar(T meanvar[2]) {
    meanvar[0] = params.mean;
    meanvar[1] = params.std * params.std;
  }

 protected:
  MakeBlobsInputs<T> params;
  int *labels;
  T *data, *stats, *mu_vec;
  T h_stats[2];  // mean, var
  std::shared_ptr<deviceAllocator> allocator;
  int num_sigma;
};

typedef MakeBlobsTest<float> MakeBlobsTestF;
const std::vector<MakeBlobsInputs<float>> inputsf_t = {
  {0.0055, 32, 1024, 3, 1.f, 1.f, GenPhilox, 1234ULL},
  {0.011, 8, 1024, 3, 1.f, 1.f, GenPhilox, 1234ULL},
  {0.0055, 32, 1024, 3, 1.f, 1.f, GenTaps, 1234ULL},
  {0.011, 8, 1024, 3, 1.f, 1.f, GenTaps, 1234ULL},
  {0.0055, 32, 1024, 3, 1.f, 1.f, GenKiss99, 1234ULL},
  {0.011, 8, 1024, 3, 1.f, 1.f, GenKiss99, 1234ULL},

  {0.0055, 32, 1024, 5, 1.f, 1.f, GenPhilox, 1234ULL},
  {0.011, 8, 1024, 5, 1.f, 1.f, GenPhilox, 1234ULL},
  {0.0055, 32, 1024, 5, 1.f, 1.f, GenTaps, 1234ULL},
  {0.011, 8, 1024, 5, 1.f, 1.f, GenTaps, 1234ULL},
  {0.0055, 32, 1024, 5, 1.f, 1.f, GenKiss99, 1234ULL},
  {0.011, 8, 1024, 5, 1.f, 1.f, GenKiss99, 1234ULL}};

TEST_P(MakeBlobsTestF, Result) {
  float meanvar[2];
  getExpectedMeanVar(meanvar);
  ASSERT_TRUE(match(meanvar[0], h_stats[0],
                    CompareApprox<float>(num_sigma * params.tolerance)));
  ASSERT_TRUE(match(meanvar[1], h_stats[1],
                    CompareApprox<float>(num_sigma * params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(MakeBlobsTests, MakeBlobsTestF,
                        ::testing::ValuesIn(inputsf_t));

typedef MakeBlobsTest<double> MakeBlobsTestD;
const std::vector<MakeBlobsInputs<double>> inputsd_t = {
  {0.0055, 32, 1024, 3, 1.0, 1.0, GenPhilox, 1234ULL},
  {0.011, 8, 1024, 3, 1.0, 1.0, GenPhilox, 1234ULL},
  {0.0055, 32, 1024, 3, 1.0, 1.0, GenTaps, 1234ULL},
  {0.011, 8, 1024, 3, 1.0, 1.0, GenTaps, 1234ULL},
  {0.0055, 32, 1024, 3, 1.0, 1.0, GenKiss99, 1234ULL},
  {0.011, 8, 1024, 3, 1.0, 1.0, GenKiss99, 1234ULL},

  {0.0055, 32, 1024, 5, 1.0, 1.0, GenPhilox, 1234ULL},
  {0.011, 8, 1024, 5, 1.0, 1.0, GenPhilox, 1234ULL},
  {0.0055, 32, 1024, 5, 1.0, 1.0, GenTaps, 1234ULL},
  {0.011, 8, 1024, 5, 1.0, 1.0, GenTaps, 1234ULL},
  {0.0055, 32, 1024, 5, 1.0, 1.0, GenKiss99, 1234ULL},
  {0.011, 8, 1024, 5, 1.0, 1.0, GenKiss99, 1234ULL}};
TEST_P(MakeBlobsTestD, Result) {
  double meanvar[2];
  getExpectedMeanVar(meanvar);
  ASSERT_TRUE(match(meanvar[0], h_stats[0],
                    CompareApprox<double>(num_sigma * params.tolerance)));
  ASSERT_TRUE(match(meanvar[1], h_stats[1],
                    CompareApprox<double>(num_sigma * params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(MakeBlobsTests, MakeBlobsTestD,
                        ::testing::ValuesIn(inputsd_t));

}  // end namespace Random
}  // end namespace MLCommon
