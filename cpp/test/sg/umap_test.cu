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

#include "distance/distance.h"

#include "datasets/digits.h"

#include <cuml/manifold/umapparams.h>
#include <metrics/trustworthiness.h>
#include <cuml/common/cuml_allocator.hpp>
#include <cuml/cuml.hpp>
#include <cuml/neighbors/knn.hpp>

#include "common/device_buffer.hpp"
#include "umap/runner.h"

#include <cuda_utils.h>

#include <iostream>
#include <vector>

using namespace ML;
using namespace ML::Metrics;

using namespace std;

using namespace MLCommon;
using namespace MLCommon::Distance;
using namespace MLCommon::Datasets::Digits;

/**
  * Kernel downcasting embeddings from doubles to floats
  */
template <int TPB_X>
__global__ void apply_downcasting_kernel(double *before, float *after,
                                         int n_vertices, int n_components) {
  int vertice_idx = (blockIdx.x * TPB_X) + threadIdx.x;
  if (vertice_idx < n_vertices) {
    before += vertice_idx * n_components;
    after += vertice_idx * n_components;
    for (int d = 0; d < n_components; d++) {
      after[d] = before[d];
    }
  }
}

class UMAPTest : public ::testing::Test {
 protected:
  void xformTest() {
    cumlHandle handle;

    cudaStream_t stream = handle.getStream();

    UMAPParams *umap_params = new UMAPParams();
    umap_params->n_neighbors = 10;
    umap_params->init = 1;
    umap_params->verbose = false;

    UMAPAlgo::find_ab(umap_params, handle.getDeviceAllocator(), stream);

    device_buffer<float> X_d(handle.getDeviceAllocator(), handle.getStream(),
                             n_samples * n_features);

    MLCommon::updateDevice(X_d.data(), digits.data(), n_samples * n_features,
                           handle.getStream());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    device_buffer<double> embeddings(handle.getDeviceAllocator(),
                                     handle.getStream(),
                                     n_samples * umap_params->n_components);

    UMAPAlgo::_fit<float, 256>(handle, X_d.data(), n_samples, n_features,
                               umap_params, embeddings.data());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    device_buffer<double> xformed(handle.getDeviceAllocator(),
                                  handle.getStream(),
                                  n_samples * umap_params->n_components);

    UMAPAlgo::_transform<float, 256>(handle, X_d.data(), n_samples, n_features,
                                     X_d.data(), n_samples, embeddings.data(),
                                     n_samples, umap_params, xformed.data());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    // Downcasting embedding for trustworthiness score
    device_buffer<float> xformed_float(handle.getDeviceAllocator(),
                                       handle.getStream(),
                                       n_samples * umap_params->n_components);
    apply_downcasting_kernel<256>
      <<<grid, blk, 0, stream>>>(xformed.data(), xformed_float.data(),
                                 n_samples, umap_params->n_components);

    xformed_score = trustworthiness_score<float, EucUnexpandedL2Sqrt>(
      handle, X_d.data(), xformed_float.data(), n_samples, n_features,
      umap_params->n_components, umap_params->n_neighbors);
  }

  void fitTest() {
    cumlHandle handle;

    cudaStream_t stream = handle.getStream();

    UMAPParams *umap_params = new UMAPParams();
    umap_params->n_neighbors = 10;
    umap_params->init = 1;
    umap_params->verbose = false;

    UMAPAlgo::find_ab(umap_params, handle.getDeviceAllocator(), stream);

    device_buffer<float> X_d(handle.getDeviceAllocator(), handle.getStream(),
                             n_samples * n_features);

    MLCommon::updateDevice(X_d.data(), digits.data(), n_samples * n_features,
                           handle.getStream());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    device_buffer<double> embeddings(handle.getDeviceAllocator(),
                                     handle.getStream(),
                                     n_samples * umap_params->n_components);

    UMAPAlgo::_fit<float, 256>(handle, X_d.data(), n_samples, n_features,
                               umap_params, embeddings.data());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    // Downcasting embedding for trustworthiness score
    device_buffer<float> embeddings_float(
      handle.getDeviceAllocator(), handle.getStream(),
      n_samples * umap_params->n_components);
    apply_downcasting_kernel<256>
      <<<grid, blk, 0, stream>>>(embeddings.data(), embeddings_float.data(),
                                 n_samples, umap_params->n_components);

    fit_score = trustworthiness_score<float, EucUnexpandedL2Sqrt>(
      handle, X_d.data(), embeddings_float.data(), n_samples, n_features,
      umap_params->n_components, umap_params->n_neighbors);
  }

  void supervisedTest() {
    cumlHandle handle;

    cudaStream_t stream = handle.getStream();

    UMAPParams *umap_params = new UMAPParams();
    umap_params->n_neighbors = 10;
    umap_params->init = 1;
    umap_params->verbose = false;

    UMAPAlgo::find_ab(umap_params, handle.getDeviceAllocator(), stream);

    device_buffer<float> X_d(handle.getDeviceAllocator(), handle.getStream(),
                             n_samples * n_features);
    device_buffer<float> Y_d(handle.getDeviceAllocator(), handle.getStream(),
                             n_samples * 2);

    MLCommon::updateDevice(X_d.data(), digits.data(), n_samples * n_features,
                           handle.getStream());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    device_buffer<double> embeddings(handle.getDeviceAllocator(),
                                     handle.getStream(),
                                     n_samples * umap_params->n_components);

    UMAPAlgo::_fit<float, 256>(handle, X_d.data(), Y_d.data(), n_samples,
                               n_features, umap_params, embeddings.data());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    // Downcasting embedding for trustworthiness score
    device_buffer<float> embeddings_float(
      handle.getDeviceAllocator(), handle.getStream(),
      n_samples * umap_params->n_components);
    apply_downcasting_kernel<256>
      <<<grid, blk, 0, stream>>>(embeddings.data(), embeddings_float.data(),
                                 n_samples, umap_params->n_components);

    supervised_score = trustworthiness_score<float, EucUnexpandedL2Sqrt>(
      handle, X_d.data(), embeddings_float.data(), n_samples, n_features,
      umap_params->n_components, umap_params->n_neighbors);
  }

  void SetUp() override {
    grid = dim3(MLCommon::ceildiv(n_samples, 256), 1, 1);
    blk = dim3(256, 1, 1);

    fitTest();
    xformTest();
    supervisedTest();

    std::cout << "fit_score=" << fit_score << std::endl;
    std::cout << "xform_score=" << xformed_score << std::endl;
    std::cout << "supervised_score=" << supervised_score << std::endl;
  }

  void TearDown() override {}

 protected:
  double fit_score;
  double xformed_score;
  double supervised_score;
  dim3 grid;
  dim3 blk;
};

typedef UMAPTest UMAPTestF;
TEST_F(UMAPTestF, Result) {
  ASSERT_TRUE(fit_score > 0.98);
  ASSERT_TRUE(xformed_score > 0.80);
  ASSERT_TRUE(supervised_score > 0.98);
}
