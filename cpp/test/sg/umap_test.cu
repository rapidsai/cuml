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
#include <iostream>
#include <vector>

#include <cuml/manifold/umapparams.h>
#include <datasets/digits.h>
#include <raft/cudart_utils.h>
#include <cuml/common/cuml_allocator.hpp>
#include <cuml/common/device_buffer.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/cuml.hpp>
#include <cuml/neighbors/knn.hpp>
#include <distance/distance.cuh>
#include <metrics/trustworthiness.cuh>
#include <raft/cuda_utils.cuh>
#include <umap/runner.cuh>

using namespace ML;
using namespace ML::Metrics;

using namespace std;

using namespace MLCommon;
using namespace MLCommon::Distance;
using namespace MLCommon::Datasets::Digits;

class UMAPTest : public ::testing::Test {
 protected:
  void xformTest() {
    raft::handle_t handle;

    cudaStream_t stream = handle.get_stream();

    UMAPParams *umap_params = new UMAPParams();
    umap_params->n_neighbors = 10;
    umap_params->init = 1;
    umap_params->verbosity = CUML_LEVEL_INFO;

    UMAPAlgo::find_ab(umap_params, handle.get_device_allocator(), stream);

    device_buffer<float> X_d(handle.get_device_allocator(), handle.get_stream(),
                             n_samples * n_features);

    MLCommon::updateDevice(X_d.data(), digits.data(), n_samples * n_features,
                           handle.get_stream());

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    device_buffer<float> embeddings(handle.get_device_allocator(),
                                    handle.get_stream(),
                                    n_samples * umap_params->n_components);

    UMAPAlgo::_fit<float, 256>(handle, X_d.data(), n_samples, n_features,
                               nullptr, nullptr, umap_params,
                               embeddings.data());

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    device_buffer<float> xformed(handle.get_device_allocator(),
                                 handle.get_stream(),
                                 n_samples * umap_params->n_components);

    UMAPAlgo::_transform<float, 256>(
      handle, X_d.data(), n_samples, n_features, nullptr, nullptr, X_d.data(),
      n_samples, embeddings.data(), n_samples, umap_params, xformed.data());

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    xformed_score = trustworthiness_score<float, L2SqrtUnexpanded>(
      handle, X_d.data(), xformed.data(), n_samples, n_features,
      umap_params->n_components, umap_params->n_neighbors);
  }

  void fitTest() {
    raft::handle_t handle;

    cudaStream_t stream = handle.get_stream();

    UMAPParams *umap_params = new UMAPParams();
    umap_params->n_neighbors = 10;
    umap_params->init = 1;
    umap_params->verbosity = CUML_LEVEL_INFO;

    UMAPAlgo::find_ab(umap_params, handle.get_device_allocator(), stream);

    device_buffer<float> X_d(handle.get_device_allocator(), handle.get_stream(),
                             n_samples * n_features);

    MLCommon::updateDevice(X_d.data(), digits.data(), n_samples * n_features,
                           handle.get_stream());

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    device_buffer<float> embeddings(handle.get_device_allocator(),
                                    handle.get_stream(),
                                    n_samples * umap_params->n_components);

    UMAPAlgo::_fit<float, 256>(handle, X_d.data(), n_samples, n_features,
                               nullptr, nullptr, umap_params,
                               embeddings.data());

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    fit_score = trustworthiness_score<float, L2SqrtUnexpanded>(
      handle, X_d.data(), embeddings.data(), n_samples, n_features,
      umap_params->n_components, umap_params->n_neighbors);
  }

  void supervisedTest() {
    raft::handle_t handle;

    cudaStream_t stream = handle.get_stream();

    UMAPParams *umap_params = new UMAPParams();
    umap_params->n_neighbors = 10;
    umap_params->init = 1;
    umap_params->verbosity = CUML_LEVEL_INFO;

    UMAPAlgo::find_ab(umap_params, handle.get_device_allocator(), stream);

    device_buffer<float> X_d(handle.get_device_allocator(), handle.get_stream(),
                             n_samples * n_features);
    device_buffer<float> Y_d(handle.get_device_allocator(), handle.get_stream(),
                             n_samples * 2);

    MLCommon::updateDevice(X_d.data(), digits.data(), n_samples * n_features,
                           handle.get_stream());

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    device_buffer<float> embeddings(handle.get_device_allocator(),
                                    handle.get_stream(),
                                    n_samples * umap_params->n_components);

    UMAPAlgo::_fit<float, 256>(handle, X_d.data(), Y_d.data(), n_samples,
                               n_features, nullptr, nullptr, umap_params,
                               embeddings.data());

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    supervised_score = trustworthiness_score<float, L2SqrtUnexpanded>(
      handle, X_d.data(), embeddings.data(), n_samples, n_features,
      umap_params->n_components, umap_params->n_neighbors);
  }

  void fitWithKNNTest() {
    raft::handle_t handle;

    UMAPParams *umap_params = new UMAPParams();
    umap_params->n_neighbors = 10;
    umap_params->init = 1;
    umap_params->verbosity = CUML_LEVEL_INFO;

    UMAPAlgo::find_ab(umap_params, handle.get_device_allocator(),
                      handle.get_stream());

    device_buffer<float> X_d(handle.get_device_allocator(), handle.get_stream(),
                             n_samples * n_features);

    MLCommon::updateDevice(X_d.data(), digits.data(), n_samples * n_features,
                           handle.get_stream());

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    device_buffer<float> embeddings(handle.get_device_allocator(),
                                    handle.get_stream(),
                                    n_samples * umap_params->n_components);

    MLCommon::device_buffer<int64_t> knn_indices(
      handle.get_device_allocator(), handle.get_stream(),
      n_samples * umap_params->n_components);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    MLCommon::device_buffer<float> knn_dists(
      handle.get_device_allocator(), handle.get_stream(),
      n_samples * umap_params->n_components);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    std::vector<float *> ptrs(1);
    std::vector<int> sizes(1);
    ptrs[0] = X_d.data();
    sizes[0] = n_samples;

    raft::spatial::knn::brute_force_knn(
      handle, ptrs, sizes, n_features, X_d.data(), n_samples,
      knn_indices.data(), knn_dists.data(), umap_params->n_neighbors);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    UMAPAlgo::_fit<float, 256>(
      handle, X_d.data(), n_samples, n_features,
      //knn_indices.data(), knn_dists.data(), umap_params,
      nullptr, nullptr, umap_params, embeddings.data());

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    fit_with_knn_score = trustworthiness_score<float, L2SqrtUnexpanded>(
      handle, X_d.data(), embeddings.data(), n_samples, n_features,
      umap_params->n_components, umap_params->n_neighbors);
  }

  void SetUp() override {
    fitTest();
    xformTest();
    supervisedTest();
    fitWithKNNTest();

    CUML_LOG_DEBUG("fit_score=%lf", fit_score);
    CUML_LOG_DEBUG("xform_score=%lf", xformed_score);
    CUML_LOG_DEBUG("supervised_score=%f", supervised_score);
    CUML_LOG_DEBUG("fit_with_knn_score=%lf", fit_with_knn_score);
  }

  void TearDown() override {}

 protected:
  double fit_score;
  double xformed_score;
  double supervised_score;
  double fit_with_knn_score;
};

typedef UMAPTest UMAPTestF;
TEST_F(UMAPTestF, Result) {
  ASSERT_TRUE(fit_score > 0.98);
  ASSERT_TRUE(xformed_score > 0.80);
  ASSERT_TRUE(supervised_score > 0.98);
  ASSERT_TRUE(fit_with_knn_score > 0.96);
}
