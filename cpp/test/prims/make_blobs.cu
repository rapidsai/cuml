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

#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <cub/cub.cuh>
#include <raft/cuda_utils.cuh>
#include <random/make_blobs.cuh>
#include "test_utils.h"

namespace MLCommon {
namespace Random {

template <typename T>
__global__ void meanKernel(T* out,
                           int* lens,
                           const T* data,
                           const int* labels,
                           int nrows,
                           int ncols,
                           int nclusters,
                           bool row_major)
{
  int tid   = threadIdx.x + blockIdx.x * blockDim.x;
  int rowid = row_major ? tid / ncols : tid % nrows;
  int colid = row_major ? tid % ncols : tid / nrows;
  if (rowid < nrows && colid < ncols) {
    T val     = data[tid];
    int label = labels[rowid];
    int idx   = row_major ? label * ncols + colid : colid * nclusters + label;
    raft::myAtomicAdd(out + idx * 2, val);
    raft::myAtomicAdd(out + idx * 2 + 1, val * val);
    if (colid == 0) { raft::myAtomicAdd(lens + label, 1); }
  }
}

template <typename T>
__global__ void compute_mean_var(
  T* out, const T* stats, int* lens, int nrows, int ncols, bool row_major)
{
  int tid    = threadIdx.x + blockIdx.x * blockDim.x;
  int rowid  = row_major ? tid / ncols : tid % nrows;
  int colid  = row_major ? tid % ncols : tid / nrows;
  int stride = nrows * ncols;
  if (rowid < nrows && colid < ncols) {
    int len           = lens[rowid];
    auto mean         = stats[tid * 2] / len;
    out[tid]          = mean;
    out[tid + stride] = (stats[tid * 2 + 1] / len) - (mean * mean);
  }
}

template <typename T>
struct MakeBlobsInputs {
  T tolerance;
  int rows, cols, n_clusters;
  T std;
  bool row_major, shuffle;
  raft::random::GeneratorType gtype;
  uint64_t seed;
};

template <typename T>
class MakeBlobsTest : public ::testing::TestWithParam<MakeBlobsInputs<T>> {
 protected:
  void SetUp() override
  {
    // Tests are configured with their expected test-values sigma. For example,
    // 4 x sigma indicates the test shouldn't fail 99.9% of the time.
    num_sigma = 50;
    params    = ::testing::TestWithParam<MakeBlobsInputs<T>>::GetParam();
    auto len  = params.rows * params.cols;
    CUDA_CHECK(cudaStreamCreate(&stream));
    raft::random::Rng r(params.seed, params.gtype);
    raft::allocate(data, len, stream);
    raft::allocate(labels, params.rows, stream);
    raft::allocate(stats, 2 * params.n_clusters * params.cols, stream, true);
    raft::allocate(mean_var, 2 * params.n_clusters * params.cols, stream, true);
    raft::allocate(mu_vec, params.cols * params.n_clusters, stream);
    raft::allocate(lens, params.n_clusters, stream, true);
    r.uniform(mu_vec, params.cols * params.n_clusters, T(-10.0), T(10.0), stream);
    T* sigma_vec = nullptr;
    make_blobs(data,
               labels,
               params.rows,
               params.cols,
               params.n_clusters,
               stream,
               params.row_major,
               mu_vec,
               sigma_vec,
               params.std,
               params.shuffle,
               T(-10.0),
               T(10.0),
               params.seed,
               params.gtype);
    static const int threads = 128;
    meanKernel<T><<<raft::ceildiv(len, threads), threads, 0, stream>>>(
      stats, lens, data, labels, params.rows, params.cols, params.n_clusters, params.row_major);
    int len1 = params.n_clusters * params.cols;
    compute_mean_var<T><<<raft::ceildiv(len1, threads), threads, 0, stream>>>(
      mean_var, stats, lens, params.n_clusters, params.cols, params.row_major);
  }

  void TearDown() override
  {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(labels));
    CUDA_CHECK(cudaFree(stats));
    CUDA_CHECK(cudaFree(mu_vec));
  }

  void check()
  {
    int len      = params.n_clusters * params.cols;
    auto compare = raft::CompareApprox<T>(num_sigma * params.tolerance);
    ASSERT_TRUE(raft::devArrMatch(mu_vec, mean_var, len, compare));
    ASSERT_TRUE(raft::devArrMatch(params.std, mean_var + len, len, compare));
  }

 protected:
  cudaStream_t stream = 0;
  MakeBlobsInputs<T> params;
  int *labels, *lens;
  T *data, *stats, *mu_vec, *mean_var;
  int num_sigma;
};

typedef MakeBlobsTest<float> MakeBlobsTestF;
const std::vector<MakeBlobsInputs<float>> inputsf_t = {
  {0.0055, 1024, 32, 3, 1.f, true, false, raft::random::GenPhilox, 1234ULL},
  {0.011, 1024, 8, 3, 1.f, true, false, raft::random::GenPhilox, 1234ULL},
  {0.0055, 1024, 32, 3, 1.f, true, false, raft::random::GenTaps, 1234ULL},
  {0.011, 1024, 8, 3, 1.f, true, false, raft::random::GenTaps, 1234ULL},
  {0.0055, 1024, 32, 3, 1.f, true, false, raft::random::GenKiss99, 1234ULL},
  {0.011, 1024, 8, 3, 1.f, true, false, raft::random::GenKiss99, 1234ULL},
  {0.0055, 1024, 32, 3, 1.f, false, false, raft::random::GenPhilox, 1234ULL},
  {0.011, 1024, 8, 3, 1.f, false, false, raft::random::GenPhilox, 1234ULL},
  {0.0055, 1024, 32, 3, 1.f, false, false, raft::random::GenTaps, 1234ULL},
  {0.011, 1024, 8, 3, 1.f, false, false, raft::random::GenTaps, 1234ULL},
  {0.0055, 1024, 32, 3, 1.f, false, false, raft::random::GenKiss99, 1234ULL},
  {0.011, 1024, 8, 3, 1.f, false, false, raft::random::GenKiss99, 1234ULL},
  {0.0055, 1024, 32, 3, 1.f, true, true, raft::random::GenPhilox, 1234ULL},
  {0.011, 1024, 8, 3, 1.f, true, true, raft::random::GenPhilox, 1234ULL},
  {0.0055, 1024, 32, 3, 1.f, true, true, raft::random::GenTaps, 1234ULL},
  {0.011, 1024, 8, 3, 1.f, true, true, raft::random::GenTaps, 1234ULL},
  {0.0055, 1024, 32, 3, 1.f, true, true, raft::random::GenKiss99, 1234ULL},
  {0.011, 1024, 8, 3, 1.f, true, true, raft::random::GenKiss99, 1234ULL},
  {0.0055, 1024, 32, 3, 1.f, false, true, raft::random::GenPhilox, 1234ULL},
  {0.011, 1024, 8, 3, 1.f, false, true, raft::random::GenPhilox, 1234ULL},
  {0.0055, 1024, 32, 3, 1.f, false, true, raft::random::GenTaps, 1234ULL},
  {0.011, 1024, 8, 3, 1.f, false, true, raft::random::GenTaps, 1234ULL},
  {0.0055, 1024, 32, 3, 1.f, false, true, raft::random::GenKiss99, 1234ULL},
  {0.011, 1024, 8, 3, 1.f, false, true, raft::random::GenKiss99, 1234ULL},

  {0.0055, 5003, 32, 5, 1.f, true, false, raft::random::GenPhilox, 1234ULL},
  {0.011, 5003, 8, 5, 1.f, true, false, raft::random::GenPhilox, 1234ULL},
  {0.0055, 5003, 32, 5, 1.f, true, false, raft::random::GenTaps, 1234ULL},
  {0.011, 5003, 8, 5, 1.f, true, false, raft::random::GenTaps, 1234ULL},
  {0.0055, 5003, 32, 5, 1.f, true, false, raft::random::GenKiss99, 1234ULL},
  {0.011, 5003, 8, 5, 1.f, true, false, raft::random::GenKiss99, 1234ULL},
  {0.0055, 5003, 32, 5, 1.f, false, false, raft::random::GenPhilox, 1234ULL},
  {0.011, 5003, 8, 5, 1.f, false, false, raft::random::GenPhilox, 1234ULL},
  {0.0055, 5003, 32, 5, 1.f, false, false, raft::random::GenTaps, 1234ULL},
  {0.011, 5003, 8, 5, 1.f, false, false, raft::random::GenTaps, 1234ULL},
  {0.0055, 5003, 32, 5, 1.f, false, false, raft::random::GenKiss99, 1234ULL},
  {0.011, 5003, 8, 5, 1.f, false, false, raft::random::GenKiss99, 1234ULL},
  {0.0055, 5003, 32, 5, 1.f, true, true, raft::random::GenPhilox, 1234ULL},
  {0.011, 5003, 8, 5, 1.f, true, true, raft::random::GenPhilox, 1234ULL},
  {0.0055, 5003, 32, 5, 1.f, true, true, raft::random::GenTaps, 1234ULL},
  {0.011, 5003, 8, 5, 1.f, true, true, raft::random::GenTaps, 1234ULL},
  {0.0055, 5003, 32, 5, 1.f, true, true, raft::random::GenKiss99, 1234ULL},
  {0.011, 5003, 8, 5, 1.f, true, true, raft::random::GenKiss99, 1234ULL},
  {0.0055, 5003, 32, 5, 1.f, false, true, raft::random::GenPhilox, 1234ULL},
  {0.011, 5003, 8, 5, 1.f, false, true, raft::random::GenPhilox, 1234ULL},
  {0.0055, 5003, 32, 5, 1.f, false, true, raft::random::GenTaps, 1234ULL},
  {0.011, 5003, 8, 5, 1.f, false, true, raft::random::GenTaps, 1234ULL},
  {0.0055, 5003, 32, 5, 1.f, false, true, raft::random::GenKiss99, 1234ULL},
  {0.011, 5003, 8, 5, 1.f, false, true, raft::random::GenKiss99, 1234ULL},
};

TEST_P(MakeBlobsTestF, Result) { check(); }
INSTANTIATE_TEST_CASE_P(MakeBlobsTests, MakeBlobsTestF, ::testing::ValuesIn(inputsf_t));

typedef MakeBlobsTest<double> MakeBlobsTestD;
const std::vector<MakeBlobsInputs<double>> inputsd_t = {
  {0.0055, 1024, 32, 3, 1.0, true, false, raft::random::GenPhilox, 1234ULL},
  {0.011, 1024, 8, 3, 1.0, true, false, raft::random::GenPhilox, 1234ULL},
  {0.0055, 1024, 32, 3, 1.0, true, false, raft::random::GenTaps, 1234ULL},
  {0.011, 1024, 8, 3, 1.0, true, false, raft::random::GenTaps, 1234ULL},
  {0.0055, 1024, 32, 3, 1.0, true, false, raft::random::GenKiss99, 1234ULL},
  {0.011, 1024, 8, 3, 1.0, true, false, raft::random::GenKiss99, 1234ULL},
  {0.0055, 1024, 32, 3, 1.0, false, false, raft::random::GenPhilox, 1234ULL},
  {0.011, 1024, 8, 3, 1.0, false, false, raft::random::GenPhilox, 1234ULL},
  {0.0055, 1024, 32, 3, 1.0, false, false, raft::random::GenTaps, 1234ULL},
  {0.011, 1024, 8, 3, 1.0, false, false, raft::random::GenTaps, 1234ULL},
  {0.0055, 1024, 32, 3, 1.0, false, false, raft::random::GenKiss99, 1234ULL},
  {0.011, 1024, 8, 3, 1.0, false, false, raft::random::GenKiss99, 1234ULL},
  {0.0055, 1024, 32, 3, 1.0, true, true, raft::random::GenPhilox, 1234ULL},
  {0.011, 1024, 8, 3, 1.0, true, true, raft::random::GenPhilox, 1234ULL},
  {0.0055, 1024, 32, 3, 1.0, true, true, raft::random::GenTaps, 1234ULL},
  {0.011, 1024, 8, 3, 1.0, true, true, raft::random::GenTaps, 1234ULL},
  {0.0055, 1024, 32, 3, 1.0, true, true, raft::random::GenKiss99, 1234ULL},
  {0.011, 1024, 8, 3, 1.0, true, true, raft::random::GenKiss99, 1234ULL},
  {0.0055, 1024, 32, 3, 1.0, false, true, raft::random::GenPhilox, 1234ULL},
  {0.011, 1024, 8, 3, 1.0, false, true, raft::random::GenPhilox, 1234ULL},
  {0.0055, 1024, 32, 3, 1.0, false, true, raft::random::GenTaps, 1234ULL},
  {0.011, 1024, 8, 3, 1.0, false, true, raft::random::GenTaps, 1234ULL},
  {0.0055, 1024, 32, 3, 1.0, false, true, raft::random::GenKiss99, 1234ULL},
  {0.011, 1024, 8, 3, 1.0, false, true, raft::random::GenKiss99, 1234ULL},

  {0.0055, 5003, 32, 5, 1.0, true, false, raft::random::GenPhilox, 1234ULL},
  {0.011, 5003, 8, 5, 1.0, true, false, raft::random::GenPhilox, 1234ULL},
  {0.0055, 5003, 32, 5, 1.0, true, false, raft::random::GenTaps, 1234ULL},
  {0.011, 5003, 8, 5, 1.0, true, false, raft::random::GenTaps, 1234ULL},
  {0.0055, 5003, 32, 5, 1.0, true, false, raft::random::GenKiss99, 1234ULL},
  {0.011, 5003, 8, 5, 1.0, true, false, raft::random::GenKiss99, 1234ULL},
  {0.0055, 5003, 32, 5, 1.0, false, false, raft::random::GenPhilox, 1234ULL},
  {0.011, 5003, 8, 5, 1.0, false, false, raft::random::GenPhilox, 1234ULL},
  {0.0055, 5003, 32, 5, 1.0, false, false, raft::random::GenTaps, 1234ULL},
  {0.011, 5003, 8, 5, 1.0, false, false, raft::random::GenTaps, 1234ULL},
  {0.0055, 5003, 32, 5, 1.0, false, false, raft::random::GenKiss99, 1234ULL},
  {0.011, 5003, 8, 5, 1.0, false, false, raft::random::GenKiss99, 1234ULL},
  {0.0055, 5003, 32, 5, 1.0, true, true, raft::random::GenPhilox, 1234ULL},
  {0.011, 5003, 8, 5, 1.0, true, true, raft::random::GenPhilox, 1234ULL},
  {0.0055, 5003, 32, 5, 1.0, true, true, raft::random::GenTaps, 1234ULL},
  {0.011, 5003, 8, 5, 1.0, true, true, raft::random::GenTaps, 1234ULL},
  {0.0055, 5003, 32, 5, 1.0, true, true, raft::random::GenKiss99, 1234ULL},
  {0.011, 5003, 8, 5, 1.0, true, true, raft::random::GenKiss99, 1234ULL},
  {0.0055, 5003, 32, 5, 1.0, false, true, raft::random::GenPhilox, 1234ULL},
  {0.011, 5003, 8, 5, 1.0, false, true, raft::random::GenPhilox, 1234ULL},
  {0.0055, 5003, 32, 5, 1.0, false, true, raft::random::GenTaps, 1234ULL},
  {0.011, 5003, 8, 5, 1.0, false, true, raft::random::GenTaps, 1234ULL},
  {0.0055, 5003, 32, 5, 1.0, false, true, raft::random::GenKiss99, 1234ULL},
  {0.011, 5003, 8, 5, 1.0, false, true, raft::random::GenKiss99, 1234ULL},
};
TEST_P(MakeBlobsTestD, Result) { check(); }
INSTANTIATE_TEST_CASE_P(MakeBlobsTests, MakeBlobsTestD, ::testing::ValuesIn(inputsd_t));

}  // end namespace Random
}  // end namespace MLCommon
