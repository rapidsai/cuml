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

/*
 * Fills in data and labels with an easy-to-predict classification dataset.
 * Both vectors will be resized to hold datasets of size nrows * ncols.
 * Data will be laid out in row-major format.
 * The dataset contains some relevant data, some repeated data, and some
 * irrelevant data.
 * The labels are computed as max(all_relevant_cols), scaled to
 * map to nclasses integer outputs.
 */
template <typename T>
void makeClassificationDataHost(std::vector<T>& data, std::vector<int>& labels,
                                int nrows, int ncols, int nclasses) {
  data.resize(nrows * ncols);
  labels.resize(nrows * ncols);

  const int N_INFORMATIVE = 4;
  const int N_REPEATED = 4;
  const T MAX_RELEVANT_VAL = 10.0 + nclasses;  // Data pattern respects this max

  // first NI columns are informative, next NR are identical, rest are junk
  for (int i = 0; i < nrows; i++) {
    T max_relevant_row = -1e6;

    for (int j = 0; j < ncols; j++) {
      if (j < N_INFORMATIVE) {
        // Totally arbitrary data pattern that spreads out a bit
        T val = 10.0 * ((i + 1) / (float)nrows) + ((i + j) % nclasses);
        max_relevant_row = max(max_relevant_row, val);
        data[(j * nrows) + i] = val;
      } else if (j < N_INFORMATIVE + N_REPEATED) {
        data[(j * nrows) + i] = data[((j - N_INFORMATIVE) * nrows) + i];
      } else {
        // Totally junk data (irrelevant distractors)
        data[(j * nrows) + i] = j + ((j + i) % 2) * -1;
      }
    }

    labels[i] = (int)nclasses * (max_relevant_row / MAX_RELEVANT_VAL);
  }

  if (VERBOSE_TEST) {
    std::cout << "Labels: " << std::endl;
    for (int i = 0; i < nrows; i++) {
      std::cout << labels[i] << ", " << std::endl;
    }
    std::cout << "Done with labels" << std::endl;
  }
}

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
    makeClassificationDataHost(data_h, labels_h, params.n_rows, params.n_cols,
                               5);
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

      int num_correct;
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
  {10, 680, 10, 1.0f, 1.0f, 4, -1, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   false},  // Short and wide (if you set width to 700, it will crash)
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   false},  // single tree forest, bootstrap false, depth of 8, 4 bins
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 1,
   false},  // same but min_rows_per_node=1 --> fails
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
  {10, 680, 10, 1.0f, 1.0f, 4, -1, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   false},  // Short and wide (if you set width to 700, it will crash)
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   false},  // single tree forest, bootstrap false, depth of 8, 4 bins
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 1,
   false},  // same but min_rows_per_node=1 --> fails
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
