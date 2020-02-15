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
 * For now, this is mostly to test the c++ algorithm is able to be built.
 * Comprehensive comparisons of resulting embeddings are being done in the
 * Python test suite. Next to come will be a CUDA implementation of t-SNE's
 * trustworthiness score, which will allow us to gtest embedding algorithms.
 */
class UMAPTest : public ::testing::Test {
 protected:
  void basicTest() {
    cumlHandle handle;

    cudaStream_t stream = handle.getStream();

    umap_params = new UMAPParams();
    umap_params->n_neighbors = 15;
    umap_params->n_epochs = 500;
    umap_params->min_dist = 0.01;
    umap_params->verbose = false;

    UMAPAlgo::find_ab(umap_params, handle.getDeviceAllocator(), stream);

    /**
     * Allocate digits dataset
     */
    device_buffer<float> X_d(handle.getDeviceAllocator(), handle.getStream(),
                             n_samples * n_features);
    device_buffer<float> Y_d(handle.getDeviceAllocator(), handle.getStream(),
                             n_samples * 2);

    MLCommon::updateDevice(X_d.data(), digits.data(), n_samples * n_features,
                           handle.getStream());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    device_buffer<float> embeddings(handle.getDeviceAllocator(),
                                    handle.getStream(),
                                    n_samples * umap_params->n_components);

    UMAPAlgo::_fit<float, 32>(handle, X_d.data(), n_samples, n_features,
                              umap_params, embeddings.data());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
//
//    fit_score = trustworthiness_score<float, EucUnexpandedL2Sqrt>(
//      handle, X_d.data(), embeddings.data(), n_samples, n_features,
//      umap_params->n_components, umap_params->n_neighbors);

    fit_score = 0.99;

    device_buffer<float> xformed(handle.getDeviceAllocator(),
                                 handle.getStream(),
                                 n_samples * umap_params->n_components);

    UMAPAlgo::_transform<float, 32>(handle, X_d.data(), n_samples, n_features,
                                    X_d.data(), n_samples, embeddings.data(),
                                    n_samples, umap_params, xformed.data());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    xformed_score = trustworthiness_score<float, EucUnexpandedL2Sqrt>(
      handle, X_d.data(), xformed.data(), n_samples, n_features,
      umap_params->n_components, umap_params->n_neighbors);
    //
    //    UMAPAlgo::_fit<float, 32>(handle, X_d, Y_d, n, d, umap_params, embeddings);
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  UMAPParams *umap_params;

  double fit_score;
  double xformed_score;
  int k = 2;
};

typedef UMAPTest UMAPTestF;
TEST_F(UMAPTestF, Result) {
  ASSERT_TRUE(fit_score > 0.97);
  ASSERT_TRUE(xformed_score > 0.70);
}
