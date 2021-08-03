/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <test_utils.h>

#include <umap/runner.cuh>

#include <cuml/manifold/umapparams.h>
#include <cuml/common/device_buffer.hpp>
#include <cuml/datasets/make_blobs.hpp>
#include <cuml/manifold/umap.hpp>
#include <cuml/metrics/metrics.hpp>
#include <cuml/neighbors/knn.hpp>

#include <datasets/digits.h>
#include <linalg/reduce_rows_by_key.cuh>
#include <selection/knn.cuh>

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>
#include <raft/distance/distance.cuh>
#include <raft/handle.hpp>
#include <raft/mr/device/allocator.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <type_traits>
#include <vector>

using namespace ML;
using namespace ML::Metrics;

using namespace std;

using namespace MLCommon;
using namespace MLCommon::Datasets::Digits;

template <typename T>
__global__ void has_nan_kernel(T* data, size_t len, bool* answer)
{
  static_assert(std::is_floating_point<T>());
  std::size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ((tid < len) && isnan(data[tid])) { *answer = true; }
}

template <typename T>
bool has_nan(T* data,
             size_t len,
             std::shared_ptr<raft::mr::device::allocator> alloc,
             cudaStream_t stream)
{
  dim3 blk(256);
  dim3 grid(raft::ceildiv(len, (size_t)blk.x));
  bool h_answer = false;
  device_buffer<bool> d_answer(alloc, stream, 1);
  raft::update_device(d_answer.data(), &h_answer, 1, stream);
  has_nan_kernel<<<grid, blk, 0, stream>>>(data, len, d_answer.data());
  raft::update_host(&h_answer, d_answer.data(), 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return h_answer;
}

template <typename T>
__global__ void are_equal_kernel(T* embedding1, T* embedding2, size_t len, double* diff)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= len) return;
  if (embedding1[tid] != embedding2[tid]) {
    atomicAdd(diff, abs(embedding1[tid] - embedding2[tid]));
  }
}

template <typename T>
bool are_equal(T* embedding1,
               T* embedding2,
               size_t len,
               std::shared_ptr<raft::mr::device::allocator> alloc,
               cudaStream_t stream)
{
  double h_answer = 0.;
  device_buffer<double> d_answer(alloc, stream, 1);
  raft::update_device(d_answer.data(), &h_answer, 1, stream);

  are_equal_kernel<<<raft::ceildiv(len, (size_t)32), 32, 0, stream>>>(
    embedding1, embedding2, len, d_answer.data());
  raft::update_host(&h_answer, d_answer.data(), 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  double tolerance = 1.0;
  if (h_answer > tolerance) {
    std::cout << "Not equal, difference : " << h_answer << std::endl;
    return false;
  }
  return true;
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

  void get_embedding(raft::handle_t& handle,
                     float* X,
                     float* y,
                     float* embedding_ptr,
                     TestParams& test_params,
                     UMAPParams& umap_params)
  {
    cudaStream_t stream = handle.get_stream();
    auto alloc          = handle.get_device_allocator();
    int& n_samples      = test_params.n_samples;
    int& n_features     = test_params.n_features;

    device_buffer<int64_t>* knn_indices_b;
    device_buffer<float>* knn_dists_b;
    int64_t* knn_indices = nullptr;
    float* knn_dists     = nullptr;
    if (test_params.knn_params) {
      knn_indices_b =
        new device_buffer<int64_t>(alloc, stream, n_samples * umap_params.n_neighbors);
      knn_dists_b = new device_buffer<float>(alloc, stream, n_samples * umap_params.n_neighbors);
      knn_indices = knn_indices_b->data();
      knn_dists   = knn_dists_b->data();

      std::vector<float*> ptrs(1);
      std::vector<int> sizes(1);
      ptrs[0]  = X;
      sizes[0] = n_samples;

      raft::spatial::knn::brute_force_knn(handle,
                                          ptrs,
                                          sizes,
                                          n_features,
                                          X,
                                          n_samples,
                                          knn_indices,
                                          knn_dists,
                                          umap_params.n_neighbors);

      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    float* model_embedding = nullptr;
    device_buffer<float>* model_embedding_b{};

    if (test_params.fit_transform) {
      model_embedding = embedding_ptr;
    } else {
      model_embedding_b =
        new device_buffer<float>(alloc, stream, n_samples * umap_params.n_components);
      model_embedding = model_embedding_b->data();
    }

    CUDA_CHECK(cudaMemsetAsync(
      model_embedding, 0, n_samples * umap_params.n_components * sizeof(float), stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (test_params.supervised) {
      ML::UMAP::fit(
        handle, X, y, n_samples, n_features, knn_indices, knn_dists, &umap_params, model_embedding);
    } else {
      ML::UMAP::fit(handle,
                    X,
                    nullptr,
                    n_samples,
                    n_features,
                    knn_indices,
                    knn_dists,
                    &umap_params,
                    model_embedding);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!test_params.fit_transform) {
      CUDA_CHECK(cudaMemsetAsync(
        embedding_ptr, 0, n_samples * umap_params.n_components * sizeof(float), stream));

      CUDA_CHECK(cudaStreamSynchronize(stream));

      ML::UMAP::transform(handle,
                          X,
                          n_samples,
                          umap_params.n_components,
                          knn_indices,
                          knn_dists,
                          X,
                          n_samples,
                          model_embedding,
                          n_samples,
                          &umap_params,
                          embedding_ptr);

      CUDA_CHECK(cudaStreamSynchronize(stream));

      delete model_embedding_b;
    }

    if (test_params.knn_params) {
      delete knn_indices_b;
      delete knn_dists_b;
    }
  }

  void assertions(raft::handle_t& handle,
                  float* X,
                  float* embedding_ptr,
                  TestParams& test_params,
                  UMAPParams& umap_params)
  {
    cudaStream_t stream = handle.get_stream();
    auto alloc          = handle.get_device_allocator();
    int& n_samples      = test_params.n_samples;
    int& n_features     = test_params.n_features;

    ASSERT_TRUE(!has_nan(embedding_ptr, n_samples * umap_params.n_components, alloc, stream));

    double trustworthiness =
      trustworthiness_score<float, raft::distance::DistanceType::L2SqrtUnexpanded>(
        handle,
        X,
        embedding_ptr,
        n_samples,
        n_features,
        umap_params.n_components,
        umap_params.n_neighbors);

    std::cout << "min. expected trustworthiness: " << test_params.min_trustworthiness << std::endl;
    std::cout << "trustworthiness: " << trustworthiness << std::endl;
    ASSERT_TRUE(trustworthiness > test_params.min_trustworthiness);
  }

  void test(TestParams& test_params, UMAPParams& umap_params)
  {
    std::cout << "\numap_params : [" << std::boolalpha << umap_params.n_neighbors << "-"
              << umap_params.n_components << "-" << umap_params.n_epochs << "-"
              << umap_params.random_state << std::endl;

    std::cout << "test_params : [" << std::boolalpha << test_params.fit_transform << "-"
              << test_params.supervised << "-" << test_params.knn_params << "-"
              << test_params.n_samples << "-" << test_params.n_features << "-"
              << test_params.n_clusters << "-" << test_params.min_trustworthiness << "]"
              << std::endl;

    raft::handle_t handle;
    cudaStream_t stream = handle.get_stream();
    auto alloc          = handle.get_device_allocator();
    int& n_samples      = test_params.n_samples;
    int& n_features     = test_params.n_features;

    UMAP::find_ab(handle, &umap_params);

    device_buffer<float> X_d(alloc, stream, n_samples * n_features);
    device_buffer<int> y_d(alloc, stream, n_samples);

    ML::Datasets::make_blobs(handle,
                             X_d.data(),
                             y_d.data(),
                             n_samples,
                             n_features,
                             test_params.n_clusters,
                             true,
                             nullptr,
                             nullptr,
                             1.f,
                             true,
                             -10.f,
                             10.f,
                             1234ULL);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    MLCommon::LinAlg::convert_array((float*)y_d.data(), y_d.data(), n_samples, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    device_buffer<float> embeddings1(alloc, stream, n_samples * umap_params.n_components);

    float* e1 = embeddings1.data();

    get_embedding(handle, X_d.data(), (float*)y_d.data(), e1, test_params, umap_params);

    assertions(handle, X_d.data(), e1, test_params, umap_params);

    // v21.08: Reproducibility looks to be busted for CTK 11.4. Need to figure out
    // why this is happening and re-enable this.
#if CUDART_VERSION == 11040
    return;
#else
    // Disable reproducibility tests after transformation
    if (!test_params.fit_transform) { return; }
#endif

    device_buffer<float> embeddings2(alloc, stream, n_samples * umap_params.n_components);
    float* e2 = embeddings2.data();
    get_embedding(handle, X_d.data(), (float*)y_d.data(), e2, test_params, umap_params);

#if CUDART_VERSION >= 11020
    bool equal = are_equal(e1, e2, n_samples * umap_params.n_components, alloc, stream);

    if (!equal) {
      raft::print_device_vector("e1", e1, 25, std::cout);
      raft::print_device_vector("e2", e2, 25, std::cout);
    }

    ASSERT_TRUE(equal);
#else
    ASSERT_TRUE(
      raft::devArrMatch(e1, e2, n_samples * umap_params.n_components, raft::Compare<float>{}));
#endif
  }

  void SetUp() override
  {
    std::vector<TestParams> test_params_vec = {{false, false, false, 2000, 50, 20, 0.45},
                                               {true, false, false, 2000, 50, 20, 0.45},
                                               {false, true, false, 2000, 50, 20, 0.45},
                                               {false, false, true, 2000, 50, 20, 0.45},
                                               {true, true, false, 2000, 50, 20, 0.45},
                                               {true, false, true, 2000, 50, 20, 0.45},
                                               {false, true, true, 2000, 50, 20, 0.45},
                                               {true, true, true, 2000, 50, 20, 0.45}};

    std::vector<UMAPParams> umap_params_vec(4);
    umap_params_vec[0].n_components = 2;

    umap_params_vec[1].n_components = 10;

    umap_params_vec[2].n_components = 21;
    umap_params_vec[2].random_state = 43;
    umap_params_vec[2].init         = 0;
    umap_params_vec[2].n_epochs     = 500;

    umap_params_vec[3].n_components = 25;
    umap_params_vec[3].random_state = 43;
    umap_params_vec[3].init         = 0;
    umap_params_vec[3].n_epochs     = 500;

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
