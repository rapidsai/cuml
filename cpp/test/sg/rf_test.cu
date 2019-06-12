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
#include <data_generators.h>
#include <gtest/gtest.h>
#include <test_utils.h>
#include "ml_utils.h"
#include "randomforest/randomforest.h"

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
  bool cross_validate;
};

static const bool VERBOSE_TEST = false;

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const RfInputs<T>& dims) {
  return os;
}

template <typename T>
class RfTest : public ::testing::TestWithParam<RfInputs<T>> {
 protected:
  void basicTest() {
    params = ::testing::TestWithParam<RfInputs<T>>::GetParam();

    DecisionTree::DecisionTreeParams tree_params(
      params.max_depth, params.max_leaves, params.max_features, params.n_bins,
      params.split_algo, params.min_rows_per_node, params.bootstrap_features);
    RF_params rf_params(params.bootstrap, params.bootstrap_features,
                        params.n_trees, params.rows_sample, tree_params);
    //rf_params.print();

    //--------------------------------------------------------
    // Random Forest
    //--------------------------------------------------------

    int data_len = params.n_rows * params.n_cols;
    allocate(data, data_len);
    allocate(labels, params.n_rows);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<T> data_h;

    // Create the dataset and transfer it to device
    unsigned n_informative = min(params.n_cols, 4);
    unsigned n_redundant = min(max(params.n_cols - 4, 0), 4);
    makeClassificationDataHost(data_h, labels_h, params.n_rows, params.n_cols,
                               5, n_informative, n_redundant);
    updateDevice(data, data_h.data(), data_len, stream);
    preprocess_labels(params.n_rows, labels_h, labels_map);
    updateDevice(labels, labels_h.data(), params.n_rows, stream);

    rf_classifier = new typename rfClassifier<T>::rfClassifier(rf_params);

    cumlHandle handle;
    handle.setStream(stream);

    fit(handle, rf_classifier, data, params.n_rows, params.n_cols, labels,
        labels_map.size());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // Inference data: same as train, but row major
    int inference_data_len = params.n_inference_rows * params.n_cols;
    inference_data_h.resize(inference_data_len);
    for (int i = 0; i < params.n_inference_rows; i++) {
      for (int j = 0; j < params.n_cols; j++) {
        inference_data_h[i * params.n_cols + j] = data_h[j * params.n_rows + i];
      }
    }

    // Predict and compare against known labels
    predicted_labels.resize(params.n_inference_rows);

    if (!params.cross_validate) {
      rf_classifier->predict(handle, inference_data_h.data(),
                             params.n_inference_rows, params.n_cols,
                             predicted_labels.data(), VERBOSE_TEST);

      float num_correct = 0.0;
      for (int i = 0; i < params.n_inference_rows; i++) {
        num_correct += (predicted_labels[i] == labels_h[i]);
      }
      accuracy = num_correct / params.n_inference_rows;
    } else {
      RF_metrics tmp =
        cross_validate(handle, rf_classifier, inference_data_h.data(),
                       labels_h.data(), params.n_inference_rows, params.n_cols,
                       predicted_labels.data(), VERBOSE_TEST);
      accuracy = tmp.accuracy;
    }
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {
    accuracy = -1.0f;  // reset accuracy
    postprocess_labels(params.n_rows, labels_h, labels_map);

    inference_data_h.clear();
    labels_h.clear();
    labels_map.clear();
    predicted_labels.clear();

    CUDA_CHECK(cudaFree(labels));
    CUDA_CHECK(cudaFree(data));
    delete rf_classifier;
  }

 protected:
  RfInputs<T> params;
  T* data;
  int* labels;
  std::vector<T> inference_data_h;
  std::vector<int> labels_h;
  std::map<int, int>
    labels_map;  //unique map of labels to int vals starting from 0

  rfClassifier<T>* rf_classifier;
  float accuracy = -1.0f;  // overriden in each test SetUp and TearDown

  std::vector<int> predicted_labels;
};

const std::vector<RfInputs<float>> float_input_params = {
  {4, 2, 1, 1.0f, 1.0f, 4, -1, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   false},  // single tree forest, bootstrap false, unlimited depth, 4 bins
  {4, 2, 1, 1.0f, 1.0f, 4, -1, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   true},  // same but with cross-validation
  {50000, 200, 250, 1.0f, 1.0f, 4, -1, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   false},  // larger data more trees, bootstrap false, unlimited depth,4 bins
  {5003, 11, 11, 1.0f, 1.0f, 4, -1, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   false},  // Odd numbers
  {10, 650, 10, 1.0f, 1.0f, 4, -1, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   false},  // Short and wide (if you set width to 700, it will crash)
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   false},  // single tree forest, bootstrap false, depth of 8, 4 bins
  {4, 2, 10, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   false},  //forest with 10 trees, all trees should produce identical predictions (no bootstrapping or column subsampling)
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::HIST, 2,
   false},  // forest with 10 trees, with bootstrap and column subsampling enabled, 3 bins
  {50000, 200, 250, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::HIST, 2,
   false},  // Large forest with bootstrap and subsampling
  {50000, 200, 250, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::HIST, 2,
   true},  // Same but with cross-validation
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::GLOBAL_QUANTILE,
   2,
   false}  // with bootstrap and column subsampling enabled, 3 bins, different split algorithm
};

const std::vector<RfInputs<double>> double_input_params = {
  // Same as float_input_params
  {4, 2, 1, 1.0f, 1.0f, 4, -1, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   false},  // single tree forest, bootstrap false, unlimited depth, 4 bins
  {4, 2, 1, 1.0f, 1.0f, 4, -1, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   true},  // same but with cross-validation
  {50000, 200, 250, 1.0f, 1.0f, 4, -1, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   false},  // larger data more trees, bootstrap false, unlimited depth,4 bins
  {5003, 11, 11, 1.0f, 1.0f, 4, -1, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   false},  // Odd numbers
  {10, 500, 10, 1.0f, 1.0f, 4, -1, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   false},  // Short and wide (if you set width to 700, it will crash)
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   false},  // single tree forest, bootstrap false, depth of 8, 4 bins
  {4, 2, 10, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   false},  //forest with 10 trees, all trees should produce identical predictions (no bootstrapping or column subsampling)
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::HIST, 2,
   false},  // forest with 10 trees, with bootstrap and column subsampling enabled, 3 bins
  {50000, 200, 250, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::HIST, 2,
   false},  // Large forest with bootstrap and subsampling
  {50000, 200, 250, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::HIST, 2,
   true},  // Same but with cross-validation
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::GLOBAL_QUANTILE,
   2,
   false}  // with bootstrap and column subsampling enabled, 3 bins, different split algorithm
};

typedef RfTest<float> RfTestF;
TEST_P(RfTestF, Fit) {
  //rf_classifier->print_rf_detailed(); // Prints all trees in the forest. Leaf nodes use the remapped values from labels_map.
  if (!params.bootstrap && (params.max_features == 1.0f)) {
    ASSERT_GE(accuracy, 1.0f);
  } else {
    ASSERT_GE(accuracy, 0.75f);  // Empirically derived accuracy range
  }
}

typedef RfTest<double> RfTestD;
TEST_P(RfTestD, Fit) {
  if (!params.bootstrap && (params.max_features == 1.0f)) {
    ASSERT_GE(accuracy, 1.0f);
  } else {
    ASSERT_GE(accuracy, 0.75f);
  }
}

INSTANTIATE_TEST_CASE_P(RfTests, RfTestF,
                        ::testing::ValuesIn(float_input_params));

INSTANTIATE_TEST_CASE_P(RfTests, RfTestD,
                        ::testing::ValuesIn(double_input_params));

}  // end namespace ML
