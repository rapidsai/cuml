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

#include <cuda_utils.h>
#include <gtest/gtest.h>
#include <test_utils.h>
#include "ml_utils.h"
#include "randomforest/randomforest.hpp"

namespace ML {

using namespace MLCommon;

template <typename T>  // template useless for now.
struct RfInputs {
  int n_rows;
  int n_cols;
  int n_trees;
  float max_features;
  float rows_sample;
  int n_inference_rows;
  int max_depth;
  int max_leaves;
  bool bootstrap;
  bool bootstrap_features;
  int n_bins;
  int split_algo;
  int min_rows_per_node;
  int n_streams;
  CRITERION split_criterion;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const RfInputs<T>& dims) {
  return os;
}

template <typename T>
class RfClassifierTest : public ::testing::TestWithParam<RfInputs<T>> {
 protected:
  void basicTest() {
    params = ::testing::TestWithParam<RfInputs<T>>::GetParam();

    DecisionTree::DecisionTreeParams tree_params;
    set_tree_params(tree_params, params.max_depth, params.max_leaves,
                    params.max_features, params.n_bins, params.split_algo,
                    params.min_rows_per_node, params.bootstrap_features,
                    params.split_criterion, false);
    RF_params rf_params;
    set_all_rf_params(rf_params, params.n_trees, params.bootstrap,
                      params.rows_sample, params.n_streams, tree_params);
    //print(rf_params);

    //--------------------------------------------------------
    // Random Forest
    //--------------------------------------------------------

    int data_len = params.n_rows * params.n_cols;
    allocate(data, data_len);
    allocate(labels, params.n_rows);
    allocate(predicted_labels, params.n_inference_rows);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Populate data (assume Col major)
    std::vector<T> data_h = {30.0, 1.0, 2.0, 0.0, 10.0, 20.0, 10.0, 40.0};
    data_h.resize(data_len);
    updateDevice(data, data_h.data(), data_len, stream);

    // Populate labels
    labels_h = {0, 1, 0, 4};
    labels_h.resize(params.n_rows);
    preprocess_labels(params.n_rows, labels_h, labels_map);
    updateDevice(labels, labels_h.data(), params.n_rows, stream);

    forest = new typename ML::RandomForestMetaData<T, int>;
    null_trees_ptr(forest);

    cumlHandle handle;
    handle.setStream(stream);

    fit(handle, forest, data, params.n_rows, params.n_cols, labels,
        labels_map.size(), rf_params);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    //print_rf_detailed(forest);
    // Inference data: same as train, but row major
    int inference_data_len = params.n_inference_rows * params.n_cols;
    inference_data_h = {30.0, 10.0, 1.0, 20.0, 2.0, 10.0, 0.0, 40.0};
    inference_data_h.resize(inference_data_len);
    allocate(inference_data_d, inference_data_len);
    updateDevice(inference_data_d, inference_data_h.data(), data_len, stream);

    // Predict and compare against known labels
    RF_metrics tmp =
      score(handle, forest, inference_data_d, labels, params.n_inference_rows,
            params.n_cols, predicted_labels, false);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    printf("accuracy here %f\n", accuracy);
    accuracy = tmp.accuracy;
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {
    accuracy = -1.0f;  // reset accuracy
    postprocess_labels(params.n_rows, labels_h, labels_map);
    inference_data_h.clear();
    labels_h.clear();
    labels_map.clear();

    CUDA_CHECK(cudaFree(labels));
    CUDA_CHECK(cudaFree(predicted_labels));
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(inference_data_d));
    delete[] forest->trees;
    delete forest;
  }

 protected:
  RfInputs<T> params;
  T *data, *inference_data_d;
  int* labels;
  std::vector<T> inference_data_h;
  std::vector<int> labels_h;
  std::map<int, int>
    labels_map;  //unique map of labels to int vals starting from 0

  RandomForestMetaData<T, int>* forest;
  float accuracy = -1.0f;  // overriden in each test SetUp and TearDown

  int* predicted_labels;
};

//-------------------------------------------------------------------------------------------------------------------------------------

template <typename T>
class RfRegressorTest : public ::testing::TestWithParam<RfInputs<T>> {
 protected:
  void basicTest() {
    params = ::testing::TestWithParam<RfInputs<T>>::GetParam();

    DecisionTree::DecisionTreeParams tree_params;
    set_tree_params(tree_params, params.max_depth, params.max_leaves,
                    params.max_features, params.n_bins, params.split_algo,
                    params.min_rows_per_node, params.bootstrap_features,
                    params.split_criterion, false);
    RF_params rf_params;
    set_all_rf_params(rf_params, params.n_trees, params.bootstrap,
                      params.rows_sample, params.n_streams, tree_params);
    //print(rf_params);

    //--------------------------------------------------------
    // Random Forest
    //--------------------------------------------------------

    int data_len = params.n_rows * params.n_cols;
    allocate(data, data_len);
    allocate(labels, params.n_rows);
    allocate(predicted_labels, params.n_inference_rows);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Populate data (assume Col major)
    std::vector<T> data_h = {0.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 40.0};
    data_h.resize(data_len);
    updateDevice(data, data_h.data(), data_len, stream);

    // Populate labels
    labels_h = {1.0, 2.0, 3.0, 4.0};
    labels_h.resize(params.n_rows);
    updateDevice(labels, labels_h.data(), params.n_rows, stream);

    forest = new typename ML::RandomForestMetaData<T, T>;
    null_trees_ptr(forest);

    cumlHandle handle;
    handle.setStream(stream);

    fit(handle, forest, data, params.n_rows, params.n_cols, labels, rf_params);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Inference data: same as train, but row major
    int inference_data_len = params.n_inference_rows * params.n_cols;
    inference_data_h = {0.0, 10.0, 0.0, 20.0, 0.0, 30.0, 0.0, 40.0};
    inference_data_h.resize(inference_data_len);
    allocate(inference_data_d, inference_data_len);
    updateDevice(inference_data_d, inference_data_h.data(), data_len, stream);

    // Predict and compare against known labels
    RF_metrics tmp =
      score(handle, forest, inference_data_d, labels, params.n_inference_rows,
            params.n_cols, predicted_labels, false);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    mse = tmp.mean_squared_error;
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {
    mse = -1.0f;  // reset mse
    inference_data_h.clear();
    labels_h.clear();

    CUDA_CHECK(cudaFree(labels));
    CUDA_CHECK(cudaFree(predicted_labels));
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(inference_data_d));
    delete[] forest->trees;
    delete forest;
  }

 protected:
  RfInputs<T> params;
  T *data, *inference_data_d;
  T* labels;
  std::vector<T> inference_data_h;
  std::vector<T> labels_h;

  RandomForestMetaData<T, T>* forest;
  float mse = -1.0f;  // overriden in each test SetUp and TearDown

  T* predicted_labels;
};
//-------------------------------------------------------------------------------------------------------------------------------------

const std::vector<RfInputs<float>> inputsf2_clf = {
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::GINI},  // single tree forest, bootstrap false, depth 8, 4 bins
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::GINI},  // single tree forest, bootstrap false, depth of 8, 4 bins
  {4, 2, 10, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::
     GINI},  //forest with 10 trees, all trees should produce identical predictions (no bootstrapping or column subsampling)
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::
     GINI},  //forest with 10 trees, with bootstrap and column subsampling enabled, 3 bins
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::GLOBAL_QUANTILE,
   2, 2,
   CRITERION::
     CRITERION_END},  //forest with 10 trees, with bootstrap and column subsampling enabled, 3 bins, different split algorithm
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::ENTROPY},
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::ENTROPY},
  {4, 2, 10, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::ENTROPY},
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::ENTROPY},
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::GLOBAL_QUANTILE,
   2, 2, CRITERION::ENTROPY}};

const std::vector<RfInputs<double>> inputsd2_clf = {  // Same as inputsf2_clf
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::GINI},
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::GINI},
  {4, 2, 10, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::GINI},
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::GINI},
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::GLOBAL_QUANTILE,
   2, 2, CRITERION::CRITERION_END},
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::ENTROPY},
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::ENTROPY},
  {4, 2, 10, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::ENTROPY},
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::ENTROPY},
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::GLOBAL_QUANTILE,
   2, 2, CRITERION::ENTROPY}};

typedef RfClassifierTest<float> RfClassifierTestF;
TEST_P(RfClassifierTestF, Fit) {
  //print_rf_detailed(forest);  // Prints all trees in the forest. Leaf nodes use the remapped values from labels_map.
  if (!params.bootstrap && (params.max_features == 1.0f)) {
    ASSERT_TRUE(accuracy == 1.0f);
  } else {
    ASSERT_TRUE(accuracy >= 0.75f);  // Empirically derived accuracy range
  }
}

typedef RfClassifierTest<double> RfClassifierTestD;
TEST_P(RfClassifierTestD, Fit) {
  if (!params.bootstrap && (params.max_features == 1.0f)) {
    ASSERT_TRUE(accuracy == 1.0f);
  } else {
    ASSERT_TRUE(accuracy >= 0.75f);
  }
}

INSTANTIATE_TEST_CASE_P(RfClassifierTests, RfClassifierTestF,
                        ::testing::ValuesIn(inputsf2_clf));

INSTANTIATE_TEST_CASE_P(RfClassifierTests, RfClassifierTestD,
                        ::testing::ValuesIn(inputsd2_clf));

typedef RfRegressorTest<float> RfRegressorTestF;
TEST_P(RfRegressorTestF, Fit) {
  //print_rf_detailed(forest);  // Prints all trees in the forest.
  if (!params.bootstrap && (params.max_features == 1.0f)) {
    ASSERT_TRUE(mse == 0.0f);
  } else {
    ASSERT_TRUE(mse <= 0.2f);
  }
}

typedef RfRegressorTest<double> RfRegressorTestD;
TEST_P(RfRegressorTestD, Fit) {
  if (!params.bootstrap && (params.max_features == 1.0f)) {
    ASSERT_TRUE(mse == 0.0f);
  } else {
    ASSERT_TRUE(mse <= 0.2f);
  }
}

const std::vector<RfInputs<float>> inputsf2_reg = {
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::MSE},
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::MSE},
  {4, 2, 5, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::
     CRITERION_END},  // CRITERION_END uses the default criterion (GINI for classification, MSE for regression)
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::MAE},
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::GLOBAL_QUANTILE,
   2, 2, CRITERION::MAE},
  {4, 2, 5, 1.0f, 1.0f, 4, 8, -1, true, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::CRITERION_END}};

const std::vector<RfInputs<double>> inputsd2_reg = {  // Same as inputsf2_reg
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::MSE},
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::MSE},
  {4, 2, 5, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::CRITERION_END},
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::MAE},
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::GLOBAL_QUANTILE,
   2, 2, CRITERION::MAE},
  {4, 2, 5, 1.0f, 1.0f, 4, 8, -1, true, false, 4, SPLIT_ALGO::HIST, 2, 2,
   CRITERION::CRITERION_END}};

INSTANTIATE_TEST_CASE_P(RfRegressorTests, RfRegressorTestF,
                        ::testing::ValuesIn(inputsf2_reg));
INSTANTIATE_TEST_CASE_P(RfRegressorTests, RfRegressorTestD,
                        ::testing::ValuesIn(inputsd2_reg));

}  // end namespace ML
