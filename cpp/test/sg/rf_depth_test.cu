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

#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <cuml/ensemble/randomforest.hpp>
#include <queue>
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>
#include <random>

namespace ML {

template <typename T>  // template useless for now.
struct RfInputs {
  int n_rows;
  int n_cols;
  int n_trees;
  float max_features;
  float max_samples;
  int max_depth;
  int max_leaves;
  bool bootstrap;
  bool bootstrap_features;
  int n_bins;
  int split_algo;
  int min_samples_leaf;
  int min_samples_split;
  float min_impurity_decrease;
  int n_streams;
  CRITERION split_criterion;
};

template <typename T>
class RfClassifierDepthTest : public ::testing::TestWithParam<int> {
 protected:
  void basicTest() {
    const int max_depth = ::testing::TestWithParam<int>::GetParam();
    params = RfInputs<T>{10000,
                         10,
                         1,
                         1.0f,
                         1.0f,
                         max_depth,
                         -1,
                         false,
                         false,
                         8,
                         SPLIT_ALGO::GLOBAL_QUANTILE,
                         2,
                         2,
                         0.0,
                         2,
                         CRITERION::ENTROPY};

    RF_params rf_params;
    rf_params = set_rf_params(
      params.max_depth, params.max_leaves, params.max_features, params.n_bins,
      params.split_algo, params.min_samples_leaf, params.min_samples_split,
      params.min_impurity_decrease, params.bootstrap_features, params.bootstrap,
      params.n_trees, params.max_samples, 0, params.split_criterion,
      params.n_streams, true, 128);

    int data_len = params.n_rows * params.n_cols;
    raft::allocate(data, data_len);
    raft::allocate(labels, params.n_rows);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Populate data (assume Col major)
    std::mt19937 gen(0);
    std::vector<T> data_h(data_len);
    std::normal_distribution<> d{0, 1};
    for (int col = 0; col < params.n_cols; ++col) {
      for (int row = 0; row < params.n_rows; ++row) {
        data_h[row + col * params.n_rows] = d(gen);
      }
    }
    raft::update_device(data, data_h.data(), data_len, stream);

    // Populate labels
    labels_h.resize(params.n_rows);
    for (int row = 0; row < params.n_rows; ++row) {
      labels_h[row] =
        (data_h[row + 2 * params.n_rows] * data_h[row + 3 * params.n_rows] >
         0.5);
    }
    preprocess_labels(params.n_rows, labels_h, labels_map);
    raft::update_device(labels, labels_h.data(), params.n_rows, stream);

    forest = new typename ML::RandomForestMetaData<T, int>;
    null_trees_ptr(forest);

    raft::handle_t handle(rf_params.n_streams);
    handle.set_stream(stream);

    fit(handle, forest, data, params.n_rows, params.n_cols, labels,
        labels_map.size(), rf_params);

    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {
    labels_h.clear();
    labels_map.clear();

    CUDA_CHECK(cudaFree(labels));
    CUDA_CHECK(cudaFree(data));
    delete forest;
  }

 protected:
  RfInputs<T> params;
  T* data;
  int* labels;
  std::vector<int> labels_h;
  std::map<int, int> labels_map;
  // unique map of labels to int vals starting from 0

  RandomForestMetaData<T, int>* forest;
};

template <typename T>
class RfRegressorDepthTest : public ::testing::TestWithParam<int> {
 protected:
  void basicTest() {
    const int max_depth = ::testing::TestWithParam<int>::GetParam();
    params = RfInputs<T>{5000,
                         10,
                         1,
                         1.0f,
                         1.0f,
                         max_depth,
                         -1,
                         false,
                         false,
                         8,
                         SPLIT_ALGO::GLOBAL_QUANTILE,
                         2,
                         2,
                         0.0,
                         2,
                         CRITERION::MSE};

    RF_params rf_params;
    rf_params = set_rf_params(
      params.max_depth, params.max_leaves, params.max_features, params.n_bins,
      params.split_algo, params.min_samples_leaf, params.min_samples_split,
      params.min_impurity_decrease, params.bootstrap_features, params.bootstrap,
      params.n_trees, params.max_samples, 0, params.split_criterion,
      params.n_streams, true, 128);

    int data_len = params.n_rows * params.n_cols;
    raft::allocate(data, data_len);
    raft::allocate(labels, params.n_rows);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Populate data (assume Col major)
    std::mt19937 gen(0);
    std::vector<T> data_h(data_len);
    std::normal_distribution<> d{0, 1};
    for (int col = 0; col < params.n_cols; ++col) {
      for (int row = 0; row < params.n_rows; ++row) {
        data_h[row + col * params.n_rows] = d(gen);
      }
    }
    raft::update_device(data, data_h.data(), data_len, stream);

    // Populate labels
    labels_h.resize(params.n_rows);
    for (int row = 0; row < params.n_rows; ++row) {
      labels_h[row] =
        (data_h[row + 2 * params.n_rows] * data_h[row + 3 * params.n_rows]);
    }
    raft::update_device(labels, labels_h.data(), params.n_rows, stream);

    forest = new typename ML::RandomForestMetaData<T, T>;
    null_trees_ptr(forest);

    raft::handle_t handle(rf_params.n_streams);
    handle.set_stream(stream);

    fit(handle, forest, data, params.n_rows, params.n_cols, labels, rf_params);

    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {
    labels_h.clear();

    CUDA_CHECK(cudaFree(labels));
    CUDA_CHECK(cudaFree(data));
    delete forest;
  }

 protected:
  RfInputs<T> params;
  T* data;
  T* labels;
  std::vector<T> labels_h;

  RandomForestMetaData<T, T>* forest;
};

template <typename L, typename T>
int MaxDepthOfDecisionTree(const DecisionTree::TreeMetaDataNode<T, L>* tree) {
  const auto& node_array = tree->sparsetree;
  std::queue<std::pair<int, int>> q;  // (node ID, depth)
  // Traverse the tree breadth-first
  int initial_depth = 0;
  q.emplace(0, initial_depth);
  int max_depth = initial_depth;
  while (!q.empty()) {
    int node_id, depth;
    std::tie(node_id, depth) = q.front();
    q.pop();
    max_depth = std::max(depth, max_depth);
    const SparseTreeNode<T, L>& node = node_array.at(node_id);
    if (node.colid != -1) {
      q.emplace(node.left_child_id, depth + 1);
      q.emplace(node.left_child_id + 1, depth + 1);
    }
  }
  return max_depth;
}

typedef RfClassifierDepthTest<float> RfClassifierDepthTestF;
TEST_P(RfClassifierDepthTestF, Fit) {
  CUML_LOG_INFO("Param max_depth = %d", params.max_depth);
  for (int i = 0; i < forest->rf_params.n_trees; i++) {
    int actual_max_depth = MaxDepthOfDecisionTree(&(forest->trees[i]));
    ASSERT_EQ(actual_max_depth, params.max_depth);
    ASSERT_EQ(actual_max_depth, forest->trees[i].depth_counter);
  }
}

typedef RfClassifierDepthTest<double> RfClassifierDepthTestD;
TEST_P(RfClassifierDepthTestD, Fit) {
  CUML_LOG_INFO("Param max_depth = %d", params.max_depth);
  for (int i = 0; i < forest->rf_params.n_trees; i++) {
    int actual_max_depth = MaxDepthOfDecisionTree(&(forest->trees[i]));
    ASSERT_EQ(actual_max_depth, params.max_depth);
    ASSERT_EQ(actual_max_depth, forest->trees[i].depth_counter);
  }
}

INSTANTIATE_TEST_CASE_P(RfClassifierDepthTests, RfClassifierDepthTestF,
                        ::testing::Range(0, 19));

INSTANTIATE_TEST_CASE_P(RfClassifierDepthTests, RfClassifierDepthTestD,
                        ::testing::Range(0, 19));

typedef RfRegressorDepthTest<float> RfRegressorDepthTestF;
TEST_P(RfRegressorDepthTestF, Fit) {
  CUML_LOG_INFO("Param max_depth = %d", params.max_depth);
  for (int i = 0; i < forest->rf_params.n_trees; i++) {
    int actual_max_depth = MaxDepthOfDecisionTree(&(forest->trees[i]));
    ASSERT_EQ(actual_max_depth, params.max_depth);
    ASSERT_EQ(actual_max_depth, forest->trees[i].depth_counter);
  }
}

typedef RfRegressorDepthTest<double> RfRegressorDepthTestD;
TEST_P(RfRegressorDepthTestD, Fit) {
  CUML_LOG_INFO("Param max_depth = %d", params.max_depth);
  for (int i = 0; i < forest->rf_params.n_trees; i++) {
    int actual_max_depth = MaxDepthOfDecisionTree(&(forest->trees[i]));
    ASSERT_EQ(actual_max_depth, params.max_depth);
    ASSERT_EQ(actual_max_depth, forest->trees[i].depth_counter);
  }
}

INSTANTIATE_TEST_CASE_P(RfRegressorDepthTests, RfRegressorDepthTestF,
                        ::testing::Range(0, 19));

INSTANTIATE_TEST_CASE_P(RfRegressorDepthTests, RfRegressorDepthTestD,
                        ::testing::Range(0, 19));

}  // end namespace ML
