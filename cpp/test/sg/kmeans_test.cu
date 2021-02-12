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
#include <test_utils.h>
#include <raft/cuda_utils.cuh>
#include <vector>

#include <thrust/fill.h>
#include <cuml/cluster/kmeans.hpp>
#include <cuml/common/cuml_allocator.hpp>
#include <cuml/common/device_buffer.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/cuml.hpp>
#include <cuml/datasets/make_blobs.hpp>
#include <cuml/metrics/metrics.hpp>

namespace ML {

using namespace MLCommon;
using namespace Datasets;
using namespace Metrics;

template <typename T>
struct KmeansInputs {
  int n_row;
  int n_col;
  int n_clusters;
  T tol;
  bool weighted;
};

template <typename T>
class KmeansTest : public ::testing::TestWithParam<KmeansInputs<T>> {
 protected:
  void basicTest() {
    raft::handle_t handle;
    testparams = ::testing::TestWithParam<KmeansInputs<T>>::GetParam();

    int n_samples = testparams.n_row;
    int n_features = testparams.n_col;
    params.n_clusters = testparams.n_clusters;
    params.tol = testparams.tol;
    params.n_init = 5;
    params.seed = 1;
    params.oversampling_factor = 0;

    device_buffer<T> X(handle.get_device_allocator(), handle.get_stream(),
                       n_samples * n_features);

    device_buffer<int> labels(handle.get_device_allocator(),
                              handle.get_stream(), n_samples);

    make_blobs(handle, X.data(), labels.data(), n_samples, n_features,
               params.n_clusters, true, nullptr, nullptr, 1.0, false, -10.0f,
               10.0f, 1234ULL);

    raft::allocate(d_labels, n_samples);
    raft::allocate(d_labels_ref, n_samples);
    raft::allocate(d_centroids, params.n_clusters * n_features);

    if (testparams.weighted) {
      raft::allocate(d_sample_weight, n_samples);
      thrust::fill(thrust::cuda::par.on(handle.get_stream()), d_sample_weight,
                   d_sample_weight + n_samples, 1);
    } else {
      d_sample_weight = nullptr;
    }

    raft::copy(d_labels_ref, labels.data(), n_samples, handle.get_stream());

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    T inertia = 0;
    int n_iter = 0;

    kmeans::fit_predict(handle, params, X.data(), n_samples, n_features,
                        d_sample_weight, d_centroids, d_labels, inertia,
                        n_iter);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    score = adjusted_rand_index(handle, d_labels_ref, d_labels, n_samples);

    if (score < 1.0) {
      std::stringstream ss;
      ss << "Expected: "
         << raft::arr2Str(d_labels_ref, 25, "d_labels_ref",
                          handle.get_stream());
      CUML_LOG_DEBUG(ss.str().c_str());
      ss.str(std::string());
      ss << "Actual: "
         << raft::arr2Str(d_labels, 25, "d_labels", handle.get_stream());
      CUML_LOG_DEBUG(ss.str().c_str());
      CUML_LOG_DEBUG("Score = %lf", score);
    }
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_labels_ref));
    CUDA_CHECK(cudaFree(d_sample_weight));
  }

 protected:
  KmeansInputs<T> testparams;
  int *d_labels, *d_labels_ref;
  T *d_centroids, *d_sample_weight;
  double score;
  ML::kmeans::KMeansParams params;
};

const std::vector<KmeansInputs<float>> inputsf2 = {
  {1000, 32, 5, 0.0001, true},      {1000, 32, 5, 0.0001, false},
  {1000, 100, 20, 0.0001, true},    {1000, 100, 20, 0.0001, false},
  {10000, 32, 10, 0.0001, true},    {10000, 32, 10, 0.0001, false},
  {10000, 100, 50, 0.0001, true},   {10000, 100, 50, 0.0001, false},
  {10000, 1000, 200, 0.0001, true}, {10000, 1000, 200, 0.0001, false}};

const std::vector<KmeansInputs<double>> inputsd2 = {
  {1000, 32, 5, 0.0001, true},      {1000, 32, 5, 0.0001, false},
  {1000, 100, 20, 0.0001, true},    {1000, 100, 20, 0.0001, false},
  {10000, 32, 10, 0.0001, true},    {10000, 32, 10, 0.0001, false},
  {10000, 100, 50, 0.0001, true},   {10000, 100, 50, 0.0001, false},
  {10000, 1000, 200, 0.0001, true}, {10000, 1000, 200, 0.0001, false}};

typedef KmeansTest<float> KmeansTestF;
TEST_P(KmeansTestF, Result) { ASSERT_TRUE(score == 1.0); }

typedef KmeansTest<double> KmeansTestD;
TEST_P(KmeansTestD, Result) { ASSERT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestD,
                        ::testing::ValuesIn(inputsd2));

}  // end namespace ML
