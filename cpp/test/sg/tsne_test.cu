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

#include <cuml/manifold/tsne.h>
#include <cuml/metrics/metrics.hpp>

#include <datasets/boston.h>
#include <datasets/breast_cancer.h>
#include <datasets/diabetes.h>
#include <datasets/digits.h>
#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuml/common/device_buffer.hpp>
#include <cuml/common/logger.hpp>
#include <iostream>
#include <raft/mr/device/allocator.hpp>
#include <tsne/distances.cuh>
#include <vector>

using namespace MLCommon;
using namespace MLCommon::Datasets;
using namespace ML;
using namespace ML::Metrics;

struct TSNEInput {
  int n, p;
  std::vector<float> dataset;
  double trustworthiness_threshold;
};

class TSNETest : public ::testing::TestWithParam<TSNEInput> {
 protected:
  void assert_score(double score, const char* test, const double threshold)
  {
    printf("%s", test);
    printf("score = %f\n", score);
    ASSERT_TRUE(threshold < score);
  }

  double runTest(TSNE_ALGORITHM algo, bool knn = false)
  {
    raft::handle_t handle;

    // Allocate memory
    device_buffer<float> X_d(handle.get_device_allocator(), handle.get_stream(), n * p);
    raft::update_device(X_d.data(), dataset.data(), n * p, handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    device_buffer<float> Y_d(handle.get_device_allocator(), handle.get_stream(), n * 2);

    MLCommon::device_buffer<int64_t> knn_indices(
      handle.get_device_allocator(), handle.get_stream(), n * 90);

    MLCommon::device_buffer<float> knn_dists(
      handle.get_device_allocator(), handle.get_stream(), n * 90);

    manifold_dense_inputs_t<float> input(X_d.data(), Y_d.data(), n, p);
    knn_graph<int64_t, float> k_graph(n, 90, knn_indices.data(), knn_dists.data());

    if (knn) TSNE::get_distances(handle, input, k_graph, handle.get_stream());

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    model_params.n_neighbors   = 90;
    model_params.min_grad_norm = 1e-12;
    model_params.verbosity     = CUML_LEVEL_DEBUG;
    model_params.algorithm     = algo;

    TSNE_fit(handle,
             X_d.data(),                       // X
             Y_d.data(),                       // embeddings
             n,                                // n_pts
             p,                                // n_ftr
             knn ? knn_indices.data() : NULL,  // knn_indices
             knn ? knn_dists.data() : NULL,    // knn_dists
             model_params);                    // model parameters

    float* embeddings_h = (float*)malloc(sizeof(float) * n * 2);
    assert(embeddings_h != NULL);
    raft::update_host(&embeddings_h[0], Y_d.data(), n * 2, handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    // Move embeddings to host.
    // This can be used for printing if needed.
    int k = 0;
    float C_contiguous_embedding[n * 2];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < 2; j++)
        C_contiguous_embedding[k++] = embeddings_h[j * n + i];
    }

    // Move transposed embeddings back to device, as trustworthiness requires C contiguous format
    raft::update_device(Y_d.data(), C_contiguous_embedding, n * 2, handle.get_stream());

    free(embeddings_h);

    // Test trustworthiness
    return trustworthiness_score<float, raft::distance::DistanceType::L2SqrtUnexpanded>(
      handle, X_d.data(), Y_d.data(), n, p, 2, 5);
  }

  void basicTest()
  {
    printf("BH\n");
    score_bh = runTest(TSNE_ALGORITHM::BARNES_HUT);
    printf("EXACT\n");
    score_exact = runTest(TSNE_ALGORITHM::EXACT);
    printf("FFT\n");
    score_fft = runTest(TSNE_ALGORITHM::FFT);

    printf("KNN BH\n");
    knn_score_bh = runTest(TSNE_ALGORITHM::BARNES_HUT, true);
    printf("KNN EXACT\n");
    knn_score_exact = runTest(TSNE_ALGORITHM::EXACT, true);
    printf("KNN FFT\n");
    knn_score_fft = runTest(TSNE_ALGORITHM::FFT, true);
  }

  void SetUp() override
  {
    params                    = ::testing::TestWithParam<TSNEInput>::GetParam();
    n                         = params.n;
    p                         = params.p;
    dataset                   = params.dataset;
    trustworthiness_threshold = params.trustworthiness_threshold;
    basicTest();
  }

  void TearDown() override {}

 protected:
  TSNEInput params;
  TSNEParams model_params;
  std::vector<float> dataset;
  int n, p;
  double score_bh;
  double score_exact;
  double score_fft;
  double knn_score_bh;
  double knn_score_exact;
  double knn_score_fft;
  double trustworthiness_threshold;
};

const std::vector<TSNEInput> inputs = {
  {Digits::n_samples, Digits::n_features, Digits::digits, 0.98},
  {Boston::n_samples, Boston::n_features, Boston::boston, 0.98},
  {BreastCancer::n_samples, BreastCancer::n_features, BreastCancer::breast_cancer, 0.98},
  {Diabetes::n_samples, Diabetes::n_features, Diabetes::diabetes, 0.90}};

typedef TSNETest TSNETestF;
TEST_P(TSNETestF, Result)
{
  assert_score(score_bh, "bh\n", trustworthiness_threshold);
  assert_score(score_exact, "exact\n", trustworthiness_threshold);
  assert_score(score_fft, "fft\n", trustworthiness_threshold);
  assert_score(knn_score_bh, "knn_bh\n", trustworthiness_threshold);
  assert_score(knn_score_exact, "knn_exact\n", trustworthiness_threshold);
  assert_score(knn_score_fft, "knn_fft\n", trustworthiness_threshold);
}

INSTANTIATE_TEST_CASE_P(TSNETests, TSNETestF, ::testing::ValuesIn(inputs));
