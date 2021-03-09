/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <datasets/digits.h>
#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <stdio.h>
#include <stdlib.h>
#include <common/device_buffer.hpp>
#include <cuml/common/cuml_allocator.hpp>
#include <cuml/common/logger.hpp>
#include <iostream>
#include <metrics/scores.cuh>
#include <tsne/distances.cuh>
#include <vector>

using namespace MLCommon;
using namespace MLCommon::Score;
using namespace MLCommon::Datasets::Digits;
using namespace ML;

class TSNETest : public ::testing::Test {
 protected:
  void assert_score(double score, const char *test) {
    printf("%s", test);
    if (score < 0.98) printf("score = %f\n", score);
    ASSERT_TRUE(0.98 < score);
  }

  double runTest(TSNE_ALGORITHM algo, bool knn = false) {
    raft::handle_t handle;

    // Allocate memory
    device_buffer<float> X_d(handle.get_device_allocator(), handle.get_stream(),
                             n * p);
    raft::update_device(X_d.data(), digits.data(), n * p, handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    device_buffer<float> Y_d(handle.get_device_allocator(), handle.get_stream(),
                             n * 2);

    MLCommon::device_buffer<int64_t> knn_indices(handle.get_device_allocator(),
                                                 handle.get_stream(), n * 90);

    MLCommon::device_buffer<float> knn_dists(handle.get_device_allocator(),
                                             handle.get_stream(), n * 90);

    manifold_dense_inputs_t<float> input(X_d.data(), Y_d.data(), n, p);
    knn_graph<int64_t, float> k_graph(n, 90, knn_indices.data(),
                                      knn_dists.data());

    if (knn) TSNE::get_distances(handle, input, k_graph, handle.get_stream());

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    // Test Exact TSNE
    TSNE_fit(handle,
             X_d.data(),                       // X
             Y_d.data(),                       // embeddings
             n,                                // n_pts
             p,                                // n_ftr
             knn ? knn_indices.data() : NULL,  // knn_indices
             knn ? knn_dists.data() : NULL,    // knn_dists
             2,                                // n_components
             90,                               // k
             0.5,                              // theta
             0.0025,                           // epssq
             50,                               // perplexity
             100,                              // perplex_max_iter
             1e-5,                             // perplex_tol
             12,                               // early_exagg
             1.0,                              // late_exagg
             250,                              // exagg_iter
             0.01,                             // min_gain
             200,                              // pre_learn_rate
             500,                              // post_learn_rate
             1000,                             // max_iter
             1e-7,                             // min_grad_norm
             0.5,                              // pre_momentum
             0.8,                              // post_momentum
             -1,                               // rand_state
             CUML_LEVEL_DEBUG,                 // verbosity
             true,                             // init
             algo);                            // algo

    float *embeddings_h = (float *)malloc(sizeof(float) * n * 2);
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
    raft::update_device(Y_d.data(), C_contiguous_embedding, n * 2,
                        handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    // Free space
    free(embeddings_h);

    // Test trustworthiness
    double score =
      trustworthiness_score<float,
                            raft::distance::DistanceType::L2SqrtUnexpanded>(
        X_d.data(), Y_d.data(), n, p, 2, 5, handle.get_device_allocator(),
        handle.get_stream());

    return score;
  }

  void basicTest() {
    // for (int i = 0; i < 5; i++) {
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
    // }
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  int n = 1797;
  int p = 64;
  double score_bh;
  double score_exact;
  double score_fft;
  double knn_score_bh;
  double knn_score_exact;
  double knn_score_fft;
};

typedef TSNETest TSNETestF;
TEST_F(TSNETestF, Result) {
  assert_score(score_bh, "bh\n");
  assert_score(score_exact, "exact\n");
  assert_score(score_fft, "fft\n");
  assert_score(knn_score_bh, "knn_bh\n");
  assert_score(knn_score_exact, "knn_exact\n");
  assert_score(knn_score_fft, "knn_fft\n");
}
