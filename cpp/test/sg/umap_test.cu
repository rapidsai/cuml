/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cuml/common/logger.hpp>
#include <cuml/cuml.hpp>
#include <cuml/neighbors/knn.hpp>

#include <random/make_blobs.h>
#include <linalg/reduce_rows_by_key.h>

#include "common/device_buffer.hpp"
#include "umap/runner.cuh"

#include <common/cudart_utils.h>
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
                                 double* diff) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= len) return;
  if (embedding1[tid] != embedding2[tid]) {
    *diff += abs(embedding1[tid] - embedding2[tid]);
  }
}

template <typename T>
bool are_equal(T* embedding1, T* embedding2, size_t len,
               std::shared_ptr<deviceAllocator> alloc, cudaStream_t stream) {
  dim3 blk(32);
  dim3 grid(MLCommon::ceildiv(len, (size_t)blk.x));
  double h_answer = 0.;
  device_buffer<double> d_answer(alloc, stream, 1);
  updateDevice(d_answer.data(), &h_answer, 1, stream);
  are_equal_kernel<<<grid, blk, 0, stream>>>(embedding1, embedding2, len,
                                             d_answer.data());
  updateHost(&h_answer, d_answer.data(), 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  double tolerance = 1.0;
  if (h_answer > tolerance) {
    std::cout << "Not equal, difference : " << h_answer << std::endl;
    return false;
  }
  return true;
}


class UMAPTest : public ::testing::Test {
 protected:
  void xformTest() {
    cumlHandle handle;

    cudaStream_t stream = handle.getStream();

    UMAPParams umap_params;
    umap_params.n_neighbors = 10;
    umap_params.init = 1;
    umap_params.verbosity = CUML_LEVEL_INFO;

    UMAPAlgo::find_ab(&umap_params, handle.getDeviceAllocator(), stream);

    device_buffer<float> X_d(handle.getDeviceAllocator(), handle.getStream(),
                             n_samples * n_features);

    MLCommon::updateDevice(X_d.data(), digits.data(), n_samples * n_features,
                           handle.getStream());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    device_buffer<float> embeddings(handle.getDeviceAllocator(),
                                    handle.getStream(),
                                    n_samples * umap_params.n_components);

    UMAPAlgo::_fit<float, 256>(handle, X_d.data(), n_samples, n_features,
                               nullptr, nullptr, &umap_params,
                               embeddings.data());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    device_buffer<float> xformed(handle.getDeviceAllocator(),
                                 handle.getStream(),
                                 n_samples * umap_params.n_components);

    UMAPAlgo::_transform<float, 256>(
      handle, X_d.data(), n_samples, n_features, nullptr, nullptr, X_d.data(),
      n_samples, embeddings.data(), n_samples, &umap_params, xformed.data());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    xformed_score = trustworthiness_score<float, EucUnexpandedL2Sqrt>(
      handle, X_d.data(), xformed.data(), n_samples, n_features,
      umap_params.n_components, umap_params.n_neighbors);


    std::cout << "ALL DONE xform test!" << std::endl;

  }

  void fitTest() {
    cumlHandle handle;

    cudaStream_t stream = handle.getStream();

    UMAPParams umap_params;
    umap_params.n_neighbors = 10;
    umap_params.init = 1;
    umap_params.verbosity = CUML_LEVEL_INFO;

    UMAPAlgo::find_ab(&umap_params, handle.getDeviceAllocator(), stream);

    device_buffer<float> X_d(handle.getDeviceAllocator(), handle.getStream(),
                             n_samples * n_features);

    MLCommon::updateDevice(X_d.data(), digits.data(), n_samples * n_features,
                           handle.getStream());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    device_buffer<float> embeddings(handle.getDeviceAllocator(),
                                    handle.getStream(),
                                    n_samples * umap_params.n_components);

    UMAPAlgo::_fit<float, 256>(handle, X_d.data(), n_samples, n_features,
                               nullptr, nullptr, &umap_params,
                               embeddings.data());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    fit_score = trustworthiness_score<float, EucUnexpandedL2Sqrt>(
      handle, X_d.data(), embeddings.data(), n_samples, n_features,
      umap_params.n_components, umap_params.n_neighbors);


    std::cout << "ALL DONE fit test!" << std::endl;

  }

  void supervisedTest() {
    cumlHandle handle;

    cudaStream_t stream = handle.getStream();

    UMAPParams umap_params;
    umap_params.n_neighbors = 10;
    umap_params.init = 1;
    umap_params.verbosity = CUML_LEVEL_INFO;

    UMAPAlgo::find_ab(&umap_params, handle.getDeviceAllocator(), stream);

    device_buffer<float> X_d(handle.getDeviceAllocator(), handle.getStream(),
                             n_samples * n_features);
    device_buffer<float> Y_d(handle.getDeviceAllocator(), handle.getStream(),
                             n_samples * 2);

    MLCommon::updateDevice(X_d.data(), digits.data(), n_samples * n_features,
                           handle.getStream());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    device_buffer<float> embeddings(handle.getDeviceAllocator(),
                                    handle.getStream(),
                                    n_samples * umap_params.n_components);

    UMAPAlgo::_fit<float, 256>(handle, X_d.data(), Y_d.data(), n_samples,
                               n_features, nullptr, nullptr, &umap_params,
                               embeddings.data());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    supervised_score = trustworthiness_score<float, EucUnexpandedL2Sqrt>(
      handle, X_d.data(), embeddings.data(), n_samples, n_features,
      umap_params.n_components, umap_params.n_neighbors);


    std::cout << "ALL DONE!" << std::endl;

  }

  void fitWithKNNTest() {
    cumlHandle handle;

    UMAPParams umap_params;
    umap_params.n_neighbors = 10;
    umap_params.init = 1;
    umap_params.verbosity = CUML_LEVEL_INFO;

    UMAPAlgo::find_ab(&umap_params, handle.getDeviceAllocator(),
                      handle.getStream());

    device_buffer<float> X_d(handle.getDeviceAllocator(), handle.getStream(),
                             n_samples * n_features);

    MLCommon::updateDevice(X_d.data(), digits.data(), n_samples * n_features,
                           handle.getStream());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    device_buffer<float> embeddings(handle.getDeviceAllocator(),
                                    handle.getStream(),
                                    n_samples * umap_params.n_components);

    MLCommon::device_buffer<int64_t> knn_indices(
      handle.getDeviceAllocator(), handle.getStream(),
      n_samples * umap_params.n_neighbors);

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    MLCommon::device_buffer<float> knn_dists(
      handle.getDeviceAllocator(), handle.getStream(),
      n_samples * umap_params.n_neighbors);

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    std::vector<float *> ptrs(1);
    std::vector<int> sizes(1);
    ptrs[0] = X_d.data();
    sizes[0] = n_samples;

    MLCommon::Selection::brute_force_knn(
      ptrs, sizes, n_features, X_d.data(), n_samples, knn_indices.data(),
      knn_dists.data(), umap_params.n_neighbors, handle.getDeviceAllocator(),
      handle.getStream());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    UMAPAlgo::_fit<float, 256>(
      handle, X_d.data(), n_samples, n_features,
      //knn_indices.data(), knn_dists.data(), umap_params,
      nullptr, nullptr, &umap_params, embeddings.data());

    CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));

    fit_with_knn_score = trustworthiness_score<float, EucUnexpandedL2Sqrt>(
      handle, X_d.data(), embeddings.data(), n_samples, n_features,
      umap_params.n_components, umap_params.n_neighbors);


    std::cout << "ALL DONE knn test!" << std::endl;

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

    CUDA_CHECK(cudaStreamSynchronize(stream));

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

      CUDA_CHECK(cudaStreamSynchronize(stream));

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
                                   stream, nullptr, nullptr,
                                   0.1, false, -100.0, 100.0, 1234L);

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

    // Disable reproducibility tests after transformation
    if (!test_params.fit_transform) {
      return;
    }

    if (!umap_params.multicore_implem) {
      device_buffer<float> embeddings2(alloc, stream,
                                       n_samples * umap_params.n_components);
      float* e2 = embeddings2.data();
      get_embedding(handle, X_d.data(), (float*)y_d.data(), e2, test_params,
                    umap_params);

      ASSERT_TRUE(
        are_equal(e1, e2, n_samples * umap_params.n_components, alloc, stream));
    }


    std::cout << "ALL DONE!" << std::endl;
  }

  void SetUp() override {
    std::vector<TestParams> test_params_vec = {
      {false, false, false, 2000, 500, 5, 0.45},
      {true, false, false, 2000, 500, 5, 0.98},
      {false, true, false, 2000, 500, 5, 0.50},
      {false, false, true, 2000, 500, 5, 0.98},
      {true, true, false, 2000, 500, 5, 0.98},
      {true, false, true, 2000, 500, 5, 0.98},
      {false, true, true, 2000, 500, 5, 0.98},
      {true, true, true, 2000, 500, 5, 0.98}};

    std::vector<UMAPParams> umap_params_vec(4);
    umap_params_vec[0].n_components = 2;
    umap_params_vec[0].multicore_implem = true;

    umap_params_vec[1].n_components = 10;
    umap_params_vec[1].multicore_implem = true;

    umap_params_vec[2].n_components = 21;
    umap_params_vec[2].random_state = 42;
    umap_params_vec[2].multicore_implem = false;
    umap_params_vec[2].optim_batch_size = 0;  // use default value
    umap_params_vec[2].n_epochs = 500;

    umap_params_vec[3].n_components = 25;
    umap_params_vec[3].random_state = 42;
    umap_params_vec[3].multicore_implem = false;
    umap_params_vec[3].optim_batch_size = 0;  // use default value
    umap_params_vec[3].n_epochs = 500;

    for (auto& umap_params : umap_params_vec) {
      for (auto& test_params : test_params_vec) {
        test(test_params, umap_params);
      }
    }
  }

  void TearDown() override {}
};

typedef UMAPTest UMAPTestF;
TEST_F(UMAPTestF, Result) {
  ASSERT_TRUE(fit_score > 0.98);
  ASSERT_TRUE(xformed_score > 0.80);
  ASSERT_TRUE(supervised_score > 0.98);
  ASSERT_TRUE(fit_with_knn_score > 0.96);
}

typedef UMAPParametrizableTest UMAPParametrizableTest;
TEST_F(UMAPParametrizableTest, Result) {}
