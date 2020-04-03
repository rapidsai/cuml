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
  bool val = data[tid];
  if (val != val) {
    *answer = true;
  }
}

template <typename T>
bool has_nan(T* data, size_t len, std::shared_ptr<deviceAllocator> alloc,
             cudaStream_t stream) {
  dim3 blk(256);
  dim3 grid(MLCommon::ceildiv(len, (size_t)blk.x));
  bool h_answer = false;
  device_buffer<bool> d_answer(alloc, stream, 1);
  updateDevice(d_answer.data(), &h_answer, 1, stream);
  has_nan_kernel<<<grid, blk, 0, stream>>>(data, len, d_answer.data());
  updateHost(&h_answer, d_answer.data(), 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return h_answer;
}

template <typename T>
__global__ void are_equal_kernel(T* embedding1, T* embedding2, size_t len,
                                 bool* answer) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= len) return;
  if (embedding1[tid] != embedding2[tid]) {
    *answer = false;
  }
}

template <typename T>
bool are_equal(T* embedding1, T* embedding2, size_t len,
               std::shared_ptr<deviceAllocator> alloc, cudaStream_t stream) {
  dim3 blk(256);
  dim3 grid(MLCommon::ceildiv(len, (size_t)blk.x));
  bool h_answer = true;
  device_buffer<bool> d_answer(alloc, stream, 1);
  updateDevice(d_answer.data(), &h_answer, 1, stream);
  are_equal_kernel<<<grid, blk, 0, stream>>>(embedding1, embedding2, len,
                                             d_answer.data());
  updateHost(&h_answer, d_answer.data(), 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return h_answer;
}

class UMAPParametrizableTest : public ::testing::Test {
 protected:
  struct TestParams {
    bool fit_transform;
    bool supervised;
    bool knn_params;
    int n_samples;
    int n_features;
    int n_clusters;
    double min_trustworthiness;
  };

  void get_embedding(cumlHandle& handle, float* X, float* y,
                     float* embedding_ptr, TestParams& test_params,
                     UMAPParams& umap_params) {
    cudaStream_t stream = handle.getStream();
    auto alloc = handle.getDeviceAllocator();
    int& n_samples = test_params.n_samples;
    int& n_features = test_params.n_features;

    device_buffer<int64_t>* knn_indices_b;
    device_buffer<float>* knn_dists_b;
    int64_t* knn_indices = nullptr;
    float* knn_dists = nullptr;
    if (test_params.knn_params) {
      knn_indices_b = new device_buffer<int64_t>(
        alloc, stream, n_samples * umap_params.n_neighbors);
      knn_dists_b = new device_buffer<float>(
        alloc, stream, n_samples * umap_params.n_neighbors);
      knn_indices = knn_indices_b->data();
      knn_dists = knn_dists_b->data();

      std::vector<float*> ptrs(1);
      std::vector<int> sizes(1);
      ptrs[0] = X;
      sizes[0] = n_samples;

      MLCommon::Selection::brute_force_knn(
        ptrs, sizes, n_features, X, n_samples, knn_indices, knn_dists,
        umap_params.n_neighbors, alloc, stream);

      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    float* model_embedding = nullptr;
    device_buffer<float>* model_embedding_b;
    if (test_params.fit_transform) {
      model_embedding = embedding_ptr;
    } else {
      model_embedding_b = new device_buffer<float>(
        alloc, stream, n_samples * umap_params.n_components);
      model_embedding = model_embedding_b->data();
    }

    CUDA_CHECK(cudaMemsetAsync(
      model_embedding, 0, n_samples * umap_params.n_components * sizeof(float),
      stream));

    if (test_params.supervised) {
      UMAPAlgo::_fit<float, 256>(handle, X, y, n_samples, n_features,
                                 knn_indices, knn_dists, &umap_params,
                                 model_embedding);
    } else {
      UMAPAlgo::_fit<float, 256>(handle, X, n_samples, n_features, knn_indices,
                                 knn_dists, &umap_params, model_embedding);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!test_params.fit_transform) {
      CUDA_CHECK(cudaMemsetAsync(
        embedding_ptr, 0, n_samples * umap_params.n_components * sizeof(float),
        stream));

      UMAPAlgo::_transform<float, 256>(
        handle, X, n_samples, umap_params.n_components, knn_indices, knn_dists,
        X, n_samples, model_embedding, n_samples, &umap_params, embedding_ptr);

      CUDA_CHECK(cudaStreamSynchronize(stream));

      delete model_embedding_b;
    }

    if (test_params.knn_params) {
      delete knn_indices_b;
      delete knn_dists_b;
    }
  }

  void assertions(cumlHandle& handle, float* X, float* embedding_ptr,
                  TestParams& test_params, UMAPParams& umap_params) {
    cudaStream_t stream = handle.getStream();
    auto alloc = handle.getDeviceAllocator();
    int& n_samples = test_params.n_samples;
    int& n_features = test_params.n_features;

    ASSERT_TRUE(!has_nan(embedding_ptr, n_samples * umap_params.n_components,
                         alloc, stream));

    double trustworthiness = trustworthiness_score<float, EucUnexpandedL2Sqrt>(
      handle, X, embedding_ptr, n_samples, n_features, umap_params.n_components,
      umap_params.n_neighbors);

    std::cout << "min. expected trustworthiness: "
              << test_params.min_trustworthiness << std::endl;
    std::cout << "trustworthiness: " << trustworthiness << std::endl;
    ASSERT_TRUE(trustworthiness > test_params.min_trustworthiness);
  }

  void test(TestParams& test_params, UMAPParams& umap_params) {
    std::cout << "\numap_params : [" << std::boolalpha
              << umap_params.n_neighbors << "-" << umap_params.n_components
              << "-" << umap_params.n_epochs << "-" << umap_params.random_state
              << "-" << umap_params.multicore_implem << "]" << std::endl;

    std::cout << "test_params : [" << std::boolalpha
              << test_params.fit_transform << "-" << test_params.supervised
              << "-" << test_params.knn_params << "-" << test_params.n_samples
              << "-" << test_params.n_features << "-" << test_params.n_clusters
              << "-" << test_params.min_trustworthiness << "]" << std::endl;

    cumlHandle handle;
    cudaStream_t stream = handle.getStream();
    auto alloc = handle.getDeviceAllocator();
    int& n_samples = test_params.n_samples;
    int& n_features = test_params.n_features;

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

    device_buffer<float> embeddings1(alloc, stream,
                                     n_samples * umap_params.n_components);
    float* e1 = embeddings1.data();

    get_embedding(handle, X_d.data(), (float*)y_d.data(), e1, test_params,
                  umap_params);

    assertions(handle, X_d.data(), e1, test_params, umap_params);

    if (!umap_params.multicore_implem) {
      device_buffer<float> embeddings2(alloc, stream,
                                       n_samples * umap_params.n_components);
      float* e2 = embeddings2.data();
      get_embedding(handle, X_d.data(), (float*)y_d.data(), e2, test_params,
                    umap_params);

      ASSERT_TRUE(
        are_equal(e1, e2, n_samples * umap_params.n_components, alloc, stream));
    }
  }

  void SetUp() override {
    std::vector<TestParams> test_params_vec = {
      {true, true, false, 10000, 200, 42, 0.45},
      {true, false, false, 10000, 200, 42, 0.45},
      {false, true, false, 10000, 200, 42, 0.45},
      {false, false, false, 10000, 200, 42, 0.45},
      {true, false, true, 10000, 200, 42, 0.45}};

    std::vector<UMAPParams> umap_params_vec(8);
    umap_params_vec[0].n_components = 2;
    umap_params_vec[0].n_epochs = 500;
    umap_params_vec[0].random_state = 42;
    umap_params_vec[0].multicore_implem = false;

    umap_params_vec[1].n_components = 10;
    umap_params_vec[1].n_epochs = 500;
    umap_params_vec[1].random_state = 42;
    umap_params_vec[1].multicore_implem = false;

    umap_params_vec[2].n_components = 21;
    umap_params_vec[2].n_epochs = 500;
    umap_params_vec[2].random_state = 42;
    umap_params_vec[2].multicore_implem = false;

    umap_params_vec[3].n_components = 25;
    umap_params_vec[3].n_epochs = 500;
    umap_params_vec[3].random_state = 42;
    umap_params_vec[3].multicore_implem = false;

    umap_params_vec[4].n_components = 2;
    umap_params_vec[4].n_epochs = 500;
    umap_params_vec[4].multicore_implem = true;

    umap_params_vec[5].n_components = 10;
    umap_params_vec[5].n_epochs = 500;
    umap_params_vec[5].multicore_implem = true;

    umap_params_vec[6].n_components = 21;
    umap_params_vec[6].n_epochs = 500;
    umap_params_vec[6].multicore_implem = true;

    umap_params_vec[7].n_components = 25;
    umap_params_vec[7].n_epochs = 500;
    umap_params_vec[7].multicore_implem = true;

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
