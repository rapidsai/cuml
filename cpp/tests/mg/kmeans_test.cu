/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <cuml/cluster/kmeans.hpp>
#include <cuml/cluster/kmeans_params.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/datasets/make_blobs.hpp>
#include <cuml/metrics/metrics.hpp>

#include <raft/comms/std_comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/fill.h>

#include <gtest/gtest.h>
#include <nccl.h>
#include <stdio.h>
#include <test_utils.h>

#include <vector>

#define NCCLCHECK(cmd)                                                                        \
  do {                                                                                        \
    ncclResult_t res = cmd;                                                                   \
    if (res != ncclSuccess) {                                                                 \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
      exit(EXIT_FAILURE);                                                                     \
    }                                                                                         \
  } while (0)

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
    : stream(handle.get_stream()),
      d_labels(0, stream),
      d_labels_ref(0, stream),
      d_centroids(0, stream),
      d_sample_weight(0, stream)
  {
  }

  void basicTest()
  {
    testparams = ::testing::TestWithParam<KmeansInputs<T>>::GetParam();
    ncclComm_t nccl_comm;
    NCCLCHECK(ncclCommInitAll(&nccl_comm, 1, {0}));
    raft::comms::build_comms_nccl_only(&handle, nccl_comm, 1, 0);

    int n_samples              = testparams.n_row;
    int n_features             = testparams.n_col;
    params.n_clusters          = testparams.n_clusters;
    params.tol                 = testparams.tol;
    params.n_init              = 5;
    params.rng_state.seed      = 1;
    params.oversampling_factor = 1;

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

    kmeans::fit(handle,
                params,
                X.data(),
                n_samples,
                n_features,
                d_sample_weight_ptr,
                d_centroids.data(),
                inertia,
                n_iter);

    kmeans::predict(handle,
                    params,
                    d_centroids.data(),
                    X.data(),
                    n_samples,
                    n_features,
                    d_sample_weight_ptr,
                    true,
                    d_labels.data(),
                    inertia);

    score = adjusted_rand_index(handle, d_labels_ref.data(), d_labels.data(), n_samples);
    handle.sync_stream(stream);

    if (score < 0.99) {
      std::stringstream ss;
      ss << "Expected: " << raft::arr2Str(d_labels_ref.data(), 25, "d_labels_ref", stream);
      CUML_LOG_WARN(ss.str().c_str());
      ss.str(std::string());
      ss << "Actual: " << raft::arr2Str(d_labels.data(), 25, "d_labels", stream);
      CUML_LOG_WARN(ss.str().c_str());
      CUML_LOG_WARN("Score = %lf", score);
    }

    ncclCommDestroy(nccl_comm);
  }

  void SetUp() override { basicTest(); }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;
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
                                                   {10000, 100, 50, 0.0001, false}};

const std::vector<KmeansInputs<double>> inputsd2 = {{1000, 32, 5, 0.0001, true},
                                                    {1000, 32, 5, 0.0001, false},
                                                    {1000, 100, 20, 0.0001, true},
                                                    {1000, 100, 20, 0.0001, false},
                                                    {10000, 32, 10, 0.0001, true},
                                                    {10000, 32, 10, 0.0001, false},
                                                    {10000, 100, 50, 0.0001, true},
                                                    {10000, 100, 50, 0.0001, false}};

typedef KmeansTest<float> KmeansTestF;
TEST_P(KmeansTestF, Result) { ASSERT_TRUE(score >= 0.99); }

typedef KmeansTest<double> KmeansTestD;
TEST_P(KmeansTestD, Result) { ASSERT_TRUE(score >= 0.99); }

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestD, ::testing::ValuesIn(inputsd2));

}  // end namespace ML
