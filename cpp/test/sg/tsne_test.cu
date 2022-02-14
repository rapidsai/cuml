/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <raft/linalg/map.hpp>

#include <cuml/common/logger.hpp>
#include <datasets/boston.h>
#include <datasets/breast_cancer.h>
#include <datasets/diabetes.h>
#include <datasets/digits.h>
#include <gtest/gtest.h>
#include <iostream>
#include <raft/cudart_utils.h>
#include <stdio.h>
#include <stdlib.h>
#include <tsne/distances.cuh>
#include <tsne/tsne_runner.cuh>
#include <tsne/utils.cuh>
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

float get_kl_div(TSNEParams& params,
                 raft::sparse::COO<float, int64_t>& input_matrix,
                 float* emb_dists,
                 size_t n,
                 cudaStream_t stream)
{
  const size_t total_nn = 2 * n * params.n_neighbors;

  rmm::device_uvector<float> Qs_vec(total_nn, stream);
  float* Ps      = input_matrix.vals();
  float* Qs      = Qs_vec.data();
  float* KL_divs = Qs;

  // Normalize Ps
  float P_sum = thrust::reduce(rmm::exec_policy(stream), Ps, Ps + total_nn);
  raft::linalg::scalarMultiply(Ps, Ps, 1.0f / P_sum, total_nn, stream);

  // Build Qs
  auto get_emb_dist = [=] __device__(const int64_t i, const int64_t j) {
    return emb_dists[i * n + j];
  };
  raft::linalg::map(Qs, total_nn, get_emb_dist, stream, input_matrix.rows(), input_matrix.cols());

  const float dof      = fmaxf(params.dim - 1, 1);  // degree of freedom
  const float exponent = (dof + 1.0) / 2.0;
  raft::linalg::unaryOp(
    Qs,
    Qs,
    total_nn,
    [=] __device__(float dist) { return __powf(dof / (dof + dist), exponent); },
    stream);

  float kl_div = compute_kl_div(Ps, Qs, KL_divs, total_nn, stream);
  return kl_div;
}

class TSNETest : public ::testing::TestWithParam<TSNEInput> {
 protected:
  struct TSNEResults;

  void assert_results(const char* test, TSNEResults& results)
  {
    bool test_tw      = results.trustworthiness > trustworthiness_threshold;
    double kl_div_tol = 0.2;
    bool test_kl_div  = results.kl_div_ref - kl_div_tol < results.kl_div &&
                       results.kl_div < results.kl_div_ref + kl_div_tol;

    if (!test_tw || !test_kl_div) {
      std::cout << "Testing " << test << ":" << std::endl;
      std::cout << "\ttrustworthiness = " << results.trustworthiness << std::endl;
      std::cout << "\tkl_div = " << results.kl_div << std::endl;
      std::cout << "\tkl_div_ref = " << results.kl_div_ref << std::endl;
      std::cout << std::endl;
    }
    ASSERT_TRUE(test_tw);
    ASSERT_TRUE(test_kl_div);
  }

  TSNEResults runTest(TSNE_ALGORITHM algo, bool knn = false)
  {
    raft::handle_t handle;
    auto stream = handle.get_stream();
    TSNEResults results;

    // Setup parameters
    model_params.algorithm     = algo;
    model_params.dim           = 2;
    model_params.n_neighbors   = 90;
    model_params.min_grad_norm = 1e-12;
    model_params.verbosity     = CUML_LEVEL_DEBUG;

    // Allocate memory
    rmm::device_uvector<float> X_d(n * p, stream);
    raft::update_device(X_d.data(), dataset.data(), n * p, stream);
    rmm::device_uvector<float> Y_d(n * model_params.dim, stream);
    rmm::device_uvector<int64_t> input_indices(0, stream);
    rmm::device_uvector<float> input_dists(0, stream);
    rmm::device_uvector<float> pw_emb_dists(n * n, stream);

    // Run TSNE
    manifold_dense_inputs_t<float> input(X_d.data(), Y_d.data(), n, p);
    knn_graph<int64_t, float> k_graph(n, model_params.n_neighbors, nullptr, nullptr);

    if (knn) {
      input_indices.resize(n * model_params.n_neighbors, stream);
      input_dists.resize(n * model_params.n_neighbors, stream);
      k_graph.knn_indices = input_indices.data();
      k_graph.knn_dists   = input_dists.data();
      TSNE::get_distances(handle, input, k_graph, stream);
    }
    handle.sync_stream(stream);
    TSNE_runner<manifold_dense_inputs_t<float>, knn_indices_dense_t, float> runner(
      handle, input, k_graph, model_params);
    results.kl_div = runner.run();

    // Compute embedding's pairwise distances
    pairwise_distance(handle,
                      Y_d.data(),
                      Y_d.data(),
                      pw_emb_dists.data(),
                      n,
                      n,
                      model_params.dim,
                      raft::distance::DistanceType::L2Expanded,
                      false);
    handle.sync_stream(stream);

    // Compute theorical KL div
    results.kl_div_ref =
      get_kl_div(model_params, runner.COO_Matrix, pw_emb_dists.data(), n, stream);

    // Transfer embeddings
    float* embeddings_h = (float*)malloc(sizeof(float) * n * model_params.dim);
    assert(embeddings_h != NULL);
    raft::update_host(embeddings_h, Y_d.data(), n * model_params.dim, stream);
    handle.sync_stream(stream);
    // Move embeddings to host.
    // This can be used for printing if needed.
    int k = 0;
    float C_contiguous_embedding[n * model_params.dim];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < model_params.dim; j++)
        C_contiguous_embedding[k++] = embeddings_h[j * n + i];
    }
    // Move transposed embeddings back to device, as trustworthiness requires C contiguous format
    raft::update_device(Y_d.data(), C_contiguous_embedding, n * model_params.dim, stream);
    handle.sync_stream(stream);
    free(embeddings_h);

    // Produce trustworthiness score
    results.trustworthiness =
      trustworthiness_score<float, raft::distance::DistanceType::L2SqrtUnexpanded>(
        handle, X_d.data(), Y_d.data(), n, p, model_params.dim, 5);

    return results;
  }

  void basicTest()
  {
    std::cout << "Running BH:" << std::endl;
    score_bh = runTest(TSNE_ALGORITHM::BARNES_HUT);
    std::cout << "Running EXACT:" << std::endl;
    score_exact = runTest(TSNE_ALGORITHM::EXACT);
    std::cout << "Running FFT:" << std::endl;
    score_fft = runTest(TSNE_ALGORITHM::FFT);

    std::cout << "Running KNN BH:" << std::endl;
    knn_score_bh = runTest(TSNE_ALGORITHM::BARNES_HUT, true);
    std::cout << "Running KNN EXACT:" << std::endl;
    knn_score_exact = runTest(TSNE_ALGORITHM::EXACT, true);
    std::cout << "Running KNN FFT:" << std::endl;
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

  struct TSNEResults {
    double trustworthiness;
    double kl_div_ref;
    double kl_div;
  };

  TSNEResults score_bh;
  TSNEResults score_exact;
  TSNEResults score_fft;
  TSNEResults knn_score_bh;
  TSNEResults knn_score_exact;
  TSNEResults knn_score_fft;
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
  assert_results("BH", score_bh);
  assert_results("EXACT", score_exact);
  assert_results("FFT", score_fft);
  assert_results("KNN BH", knn_score_bh);
  assert_results("KNN EXACT", knn_score_exact);
  assert_results("KNN FFT", knn_score_fft);
}

INSTANTIATE_TEST_CASE_P(TSNETests, TSNETestF, ::testing::ValuesIn(inputs));
