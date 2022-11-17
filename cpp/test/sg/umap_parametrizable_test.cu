/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cuml/datasets/make_blobs.hpp>
#include <cuml/manifold/umap.hpp>
#include <cuml/manifold/umapparams.h>
#include <cuml/metrics/metrics.hpp>
#include <cuml/neighbors/knn.hpp>
#include <datasets/digits.h>

#if defined RAFT_NN_COMPILED
#include <raft/spatial/knn/specializations.hpp>
#endif

#include <test_utils.h>

#include <datasets/digits.h>
#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/spatial/knn/knn.cuh>

#include <raft/core/handle.hpp>
#include <raft/distance/distance.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <umap/runner.cuh>

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
bool has_nan(T* data, size_t len, cudaStream_t stream)
{
  dim3 blk(256);
  dim3 grid(raft::ceildiv(len, (size_t)blk.x));
  bool h_answer = false;
  rmm::device_scalar<bool> d_answer(stream);
  raft::update_device(d_answer.data(), &h_answer, 1, stream);
  has_nan_kernel<<<grid, blk, 0, stream>>>(data, len, d_answer.data());
  h_answer = d_answer.value(stream);
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
bool are_equal(T* embedding1, T* embedding2, size_t len, cudaStream_t stream)
{
  double h_answer = 0.;
  rmm::device_scalar<double> d_answer(stream);
  raft::update_device(d_answer.data(), &h_answer, 1, stream);
  are_equal_kernel<<<raft::ceildiv(len, (size_t)32), 32, 0, stream>>>(
    embedding1, embedding2, len, d_answer.data());
  h_answer = d_answer.value(stream);

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
    bool refine;
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
    int& n_samples      = test_params.n_samples;
    int& n_features     = test_params.n_features;

    rmm::device_uvector<int64_t>* knn_indices_b{};
    rmm::device_uvector<float>* knn_dists_b{};
    int64_t* knn_indices{};
    float* knn_dists{};
    if (test_params.knn_params) {
      knn_indices_b = new rmm::device_uvector<int64_t>(n_samples * umap_params.n_neighbors, stream);
      knn_dists_b   = new rmm::device_uvector<float>(n_samples * umap_params.n_neighbors, stream);
      knn_indices   = knn_indices_b->data();
      knn_dists     = knn_dists_b->data();

      std::vector<float*> ptrs(1);
      std::vector<int> sizes(1);
      ptrs[0]  = X;
      sizes[0] = n_samples;

      raft::spatial::knn::brute_force_knn<long, float, int>(handle,
                                                            ptrs,
                                                            sizes,
                                                            n_features,
                                                            X,
                                                            n_samples,
                                                            knn_indices,
                                                            knn_dists,
                                                            umap_params.n_neighbors);

      handle.sync_stream(stream);
    }

    float* model_embedding = nullptr;
    rmm::device_uvector<float>* model_embedding_b{};
    if (test_params.fit_transform) {
      model_embedding = embedding_ptr;
    } else {
      model_embedding_b =
        new rmm::device_uvector<float>(n_samples * umap_params.n_components, stream);
      model_embedding = model_embedding_b->data();
    }

    RAFT_CUDA_TRY(cudaMemsetAsync(
      model_embedding, 0, n_samples * umap_params.n_components * sizeof(float), stream));

    handle.sync_stream(stream);

    auto graph = raft::sparse::COO<float, int>(stream);

    if (test_params.supervised) {
      ML::UMAP::fit(handle,
                    X,
                    y,
                    n_samples,
                    n_features,
                    knn_indices,
                    knn_dists,
                    &umap_params,
                    model_embedding,
                    &graph);
    } else {
      ML::UMAP::fit(handle,
                    X,
                    nullptr,
                    n_samples,
                    n_features,
                    knn_indices,
                    knn_dists,
                    &umap_params,
                    model_embedding,
                    &graph);
    }

    if (test_params.refine) {
      std::cout << "using refine";
      if (test_params.supervised) {
        auto cgraph_coo =
          ML::UMAP::get_graph(handle, X, y, n_samples, n_features, nullptr, nullptr, &umap_params);
        ML::UMAP::refine(
          handle, X, n_samples, n_features, cgraph_coo.get(), &umap_params, model_embedding);
      } else {
        auto cgraph_coo = ML::UMAP::get_graph(
          handle, X, nullptr, n_samples, n_features, nullptr, nullptr, &umap_params);
        ML::UMAP::refine(
          handle, X, n_samples, n_features, cgraph_coo.get(), &umap_params, model_embedding);
      }
    }
    handle.sync_stream(stream);

    if (!test_params.fit_transform) {
      RAFT_CUDA_TRY(cudaMemsetAsync(
        embedding_ptr, 0, n_samples * umap_params.n_components * sizeof(float), stream));

      handle.sync_stream(stream);

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

      handle.sync_stream(stream);

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
    int& n_samples      = test_params.n_samples;
    int& n_features     = test_params.n_features;

    ASSERT_TRUE(!has_nan(embedding_ptr, n_samples * umap_params.n_components, stream));

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
              << test_params.supervised << "-" << test_params.refine << "-"
              << test_params.knn_params << "-" << test_params.n_samples << "-"
              << test_params.n_features << "-" << test_params.n_clusters << "-"
              << test_params.min_trustworthiness << "]" << std::endl;

    raft::handle_t handle;
    cudaStream_t stream = handle.get_stream();
    int& n_samples      = test_params.n_samples;
    int& n_features     = test_params.n_features;

    UMAP::find_ab(handle, &umap_params);

    rmm::device_uvector<float> X_d(n_samples * n_features, stream);
    rmm::device_uvector<int> y_d(n_samples, stream);

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

    handle.sync_stream(stream);

    raft::linalg::convert_array((float*)y_d.data(), y_d.data(), n_samples, stream);

    handle.sync_stream(stream);

    rmm::device_uvector<float> embeddings1(n_samples * umap_params.n_components, stream);

    float* e1 = embeddings1.data();

#if CUDART_VERSION >= 11020
    // Always use random init w/ CUDA 11.2. For some reason the
    // spectral solver doesn't always converge w/ this CUDA version.
    umap_params.init         = 0;
    umap_params.random_state = 43;
    umap_params.n_epochs     = 500;
#endif
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

    rmm::device_uvector<float> embeddings2(n_samples * umap_params.n_components, stream);
    float* e2 = embeddings2.data();
    get_embedding(handle, X_d.data(), (float*)y_d.data(), e2, test_params, umap_params);

#if CUDART_VERSION >= 11020
    auto equal = are_equal(e1, e2, n_samples * umap_params.n_components, stream);

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
    std::vector<TestParams> test_params_vec = {{false, false, false, true, 2000, 50, 20, 0.45},
                                               {true, false, false, false, 2000, 50, 20, 0.45},
                                               {false, true, false, true, 2000, 50, 20, 0.45},
                                               {false, false, true, false, 2000, 50, 20, 0.45},
                                               {true, true, false, true, 2000, 50, 20, 0.45},
                                               {true, false, true, false, 2000, 50, 20, 0.45},
                                               {false, true, true, true, 2000, 50, 20, 0.45},
                                               {true, true, true, false, 2000, 50, 20, 0.45}};

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
