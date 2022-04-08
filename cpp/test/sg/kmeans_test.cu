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

#include <gtest/gtest.h>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <test_utils.h>
#include <vector>

#include <cuml/cluster/kmeans.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/datasets/make_blobs.hpp>
#include <cuml/metrics/metrics.hpp>
#include <thrust/fill.h>

namespace ML {

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
  KmeansTest()
    : d_labels(0, stream),
      d_labels_ref(0, stream),
      d_centroids(0, stream),
      d_sample_weight(0, stream)
  {
  }

  void basicTest()
  {
    raft::handle_t handle;
    testparams = ::testing::TestWithParam<KmeansInputs<T>>::GetParam();

    int n_samples              = testparams.n_row;
    int n_features             = testparams.n_col;
    params.n_clusters          = testparams.n_clusters;
    params.tol                 = testparams.tol;
    params.n_init              = 5;
    params.seed                = 1;
    params.oversampling_factor = 0;

    auto stream = handle.get_stream();
    rmm::device_uvector<T> X(n_samples * n_features, stream);
    rmm::device_uvector<int> labels(n_samples, stream);

    make_blobs(handle,
               X.data(),
               labels.data(),
               n_samples,
               n_features,
               params.n_clusters,
               true,
               nullptr,
               nullptr,
               1.0,
               false,
               -10.0f,
               10.0f,
               1234ULL);

    d_labels.resize(n_samples, stream);
    d_labels_ref.resize(n_samples, stream);
    d_centroids.resize(params.n_clusters * n_features, stream);

    T* d_sample_weight_ptr = nullptr;
    if (testparams.weighted) {
      d_sample_weight.resize(n_samples, stream);
      d_sample_weight_ptr = d_sample_weight.data();
      thrust::fill(
        thrust::cuda::par.on(stream), d_sample_weight_ptr, d_sample_weight_ptr + n_samples, 1);
    }

    raft::copy(d_labels_ref.data(), labels.data(), n_samples, stream);

    handle.sync_stream(stream);

    T inertia  = 0;
    int n_iter = 0;

    kmeans::fit_predict(handle,
                        params,
                        X.data(),
                        n_samples,
                        n_features,
                        d_sample_weight_ptr,
                        d_centroids.data(),
                        d_labels.data(),
                        inertia,
                        n_iter);

    handle.sync_stream(stream);

    score = adjusted_rand_index(handle, d_labels_ref.data(), d_labels.data(), n_samples);

    if (score < 1.0) {
      std::stringstream ss;
      ss << "Expected: " << raft::arr2Str(d_labels_ref.data(), 25, "d_labels_ref", stream);
      CUML_LOG_DEBUG(ss.str().c_str());
      ss.str(std::string());
      ss << "Actual: " << raft::arr2Str(d_labels.data(), 25, "d_labels", stream);
      CUML_LOG_DEBUG(ss.str().c_str());
      CUML_LOG_DEBUG("Score = %lf", score);
    }
  }

  void SetUp() override { basicTest(); }

 protected:
  cudaStream_t stream = 0;
  KmeansInputs<T> testparams;
  rmm::device_uvector<int> d_labels;
  rmm::device_uvector<int> d_labels_ref;
  rmm::device_uvector<T> d_centroids;
  rmm::device_uvector<T> d_sample_weight;
  double score;
  ML::kmeans::KMeansParams params;
};

const std::vector<KmeansInputs<float>> inputsf2 = {{1000, 32, 5, 0.0001, true},
                                                   {1000, 32, 5, 0.0001, false},
                                                   {1000, 100, 20, 0.0001, true},
                                                   {1000, 100, 20, 0.0001, false},
                                                   {10000, 32, 10, 0.0001, true},
                                                   {10000, 32, 10, 0.0001, false},
                                                   {10000, 100, 50, 0.0001, true},
                                                   {10000, 100, 50, 0.0001, false},
                                                   {10000, 1000, 200, 0.0001, true},
                                                   {10000, 1000, 200, 0.0001, false}};

const std::vector<KmeansInputs<double>> inputsd2 = {{1000, 32, 5, 0.0001, true},
                                                    {1000, 32, 5, 0.0001, false},
                                                    {1000, 100, 20, 0.0001, true},
                                                    {1000, 100, 20, 0.0001, false},
                                                    {10000, 32, 10, 0.0001, true},
                                                    {10000, 32, 10, 0.0001, false},
                                                    {10000, 100, 50, 0.0001, true},
                                                    {10000, 100, 50, 0.0001, false},
                                                    {10000, 1000, 200, 0.0001, true},
                                                    {10000, 1000, 200, 0.0001, false}};

typedef KmeansTest<float> KmeansTestF;
TEST_P(KmeansTestF, Result) { ASSERT_TRUE(score == 1.0); }

typedef KmeansTest<double> KmeansTestD;
TEST_P(KmeansTestD, Result) { ASSERT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestD, ::testing::ValuesIn(inputsd2));

}  // end namespace ML
