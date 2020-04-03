/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "linalg/reduce_rows_by_key.h"
#include "random/make_blobs.h"

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

template <typename T>
__global__ void has_nan_kernel(T* data, size_t len, bool* answer) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= len) return;
  if (isnan(data[tid])) {
    *answer = true;
  }
}

template <typename T>
int has_nan(T* data, size_t len, cudaStream_t stream) {
  dim3 blk(256);
  dim3 grid(MLCommon::ceildiv(len, (size_t)blk.x));
  bool* d_answer;
  cudaMalloc(&d_answer, sizeof(bool));
  CUDA_CHECK(cudaMemsetAsync(d_answer, 0, sizeof(bool), stream));
  bool h_answer;
  has_nan_kernel<<<grid, blk, 0, stream>>>(data, len, d_answer);
  updateHost(&h_answer, d_answer, 1, stream);
  cudaFree(d_answer);
  return h_answer;
}

class UMAPParametrizableTest : public ::testing::Test {
 protected:
  struct TestParams {
    bool supervised;
    int n_samples;
    int n_features;
    int n_clusters;
    double min_trustworthiness;
  };

  void test(TestParams& test_params, UMAPParams& umap_params) {
    std::cout << "\n[" << test_params.supervised << "-" << test_params.n_samples
              << "-" << test_params.n_features << "-" << test_params.n_clusters
              << "-" << test_params.min_trustworthiness << "]" << std::endl;

    cumlHandle handle;
    cudaStream_t stream = handle.getStream();
    auto alloc = handle.getDeviceAllocator();
    int n_samples = test_params.n_samples;
    int n_features = test_params.n_features;

    UMAPAlgo::find_ab(&umap_params, alloc, stream);

    device_buffer<float> X_d(alloc, stream, n_samples * n_features);
    device_buffer<int> y_d(alloc, stream, n_samples);

    Random::make_blobs<float, int>(X_d.data(), y_d.data(), n_samples,
                                   n_features, test_params.n_clusters, alloc,
                                   stream, nullptr, nullptr);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    MLCommon::LinAlg::convert_array((float*)y_d.data(), y_d.data(), n_samples,
                                    stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    device_buffer<float> embeddings(alloc, stream,
                                    n_samples * umap_params.n_components);

    if (test_params.supervised) {
      UMAPAlgo::_fit<float, 256>(handle, X_d.data(), (float*)y_d.data(),
                                 n_samples, n_features, nullptr, nullptr,
                                 &umap_params, embeddings.data());
    } else {
      UMAPAlgo::_fit<float, 256>(handle, X_d.data(), n_samples, n_features,
                                 nullptr, nullptr, &umap_params,
                                 embeddings.data());
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    device_buffer<float> xformed(alloc, stream,
                                 n_samples * umap_params.n_components);

    UMAPAlgo::_transform<float, 256>(handle, X_d.data(), n_samples,
                                     umap_params.n_components, nullptr, nullptr,
                                     X_d.data(), n_samples, embeddings.data(),
                                     n_samples, &umap_params, xformed.data());

    CUDA_CHECK(cudaStreamSynchronize(stream));

    double trustworthiness = trustworthiness_score<float, EucUnexpandedL2Sqrt>(
      handle, X_d.data(), xformed.data(), n_samples, n_features,
      umap_params.n_components, umap_params.n_neighbors);

    ASSERT_TRUE(
      !has_nan(xformed.data(), n_samples * umap_params.n_components, stream));

    std::cout << "min. expected trustworthiness: "
              << test_params.min_trustworthiness << std::endl;
    std::cout << "trustworthiness: " << trustworthiness << std::endl;
    ASSERT_TRUE(trustworthiness > test_params.min_trustworthiness);
  }

  void SetUp() override {
    std::vector<TestParams> test_params_vec = {{false, 10000, 200, 42, 0.45},
                                               {true, 10000, 200, 42, 0.45}};

    std::vector<UMAPParams> umap_params_vec(4);
    umap_params_vec[0].n_components = 2;
    umap_params_vec[0].random_state = 42;
    umap_params_vec[0].multicore_implem = false;

    umap_params_vec[1].n_components = 10;
    umap_params_vec[1].random_state = 42;
    umap_params_vec[1].multicore_implem = false;

    umap_params_vec[2].n_components = 21;
    umap_params_vec[2].random_state = 42;
    umap_params_vec[2].multicore_implem = false;

    umap_params_vec[3].n_components = 25;
    umap_params_vec[3].random_state = 42;
    umap_params_vec[3].multicore_implem = false;

    for (auto& umap_params : umap_params_vec) {
      for (auto& test_params : test_params_vec) {
        test(test_params, umap_params);
      }
    }
  }

  void TearDown() override {}
};

typedef UMAPParametrizableTest UMAPParametrizableTest;
TEST_F(UMAPParametrizableTest, Result) {}
