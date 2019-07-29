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
#include <treelite/c_api.h>
#include <treelite/c_api_runtime.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include "decisiontree/decisiontree_impl.h"
#include "ml_utils.h"
#include "randomforest/randomforest.hpp"

/** check for system errors and assert accordingly */
#define SYSTEMCALL_CHECK(call)                                    \
  do {                                                            \
    int status = call;                                            \
    ASSERT(status == 0, "SYSTEM CALL FAIL: call='%s'.\n", #call); \
  } while (0)

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
  CRITERION split_criterion;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const RfInputs<T>& dims) {
  return os;
}

template <typename T>
class RfTreeliteTestCommon : public ::testing::TestWithParam<RfInputs<T>> {
 protected:
  void commonSetUp() {
    params = ::testing::TestWithParam<RfInputs<T>>::GetParam();

    DecisionTree::DecisionTreeParams tree_params;
    set_tree_params(tree_params, params.max_depth, params.max_leaves,
                    params.max_features, params.n_bins, params.split_algo,
                    params.min_rows_per_node, params.bootstrap_features,
                    params.split_criterion, false);
    set_all_rf_params(rf_params, params.n_trees, params.bootstrap,
                      params.rows_sample, tree_params);
    // print(rf_params);

    data_len = params.n_rows * params.n_cols;
    inference_data_len = params.n_inference_rows * params.n_cols;

    allocate(data_d, data_len);
    allocate(inference_data_d, inference_data_len);

    treelite_predicted_labels.resize(params.n_inference_rows);
    ref_predicted_labels.resize(params.n_inference_rows);

    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.setStream(stream);
  }

  void convertToTreelite(ModelHandle model) {
    CompilerHandle compiler;
    // "ast_navive" is the default compiler treelite used in their Python code.
    TREELITE_CHECK(TreeliteCompilerCreate("ast_native", &compiler));
    int verbose = 0;
    // Generate C code in the directory specified below.
    TREELITE_CHECK(TreeliteCompilerGenerateCode(compiler, model, verbose,
                                                "./test_treelite"));
    TREELITE_CHECK(TreeliteCompilerFree(compiler));

    // Options copied from https://github.com/dmlc/treelite/blob/528d883f8f39eb5dd633e929b95915b63e210b39/python/treelite/contrib/__init__.py.
    const char* obj_cmd =
      "gcc -c -O3 -o ./test_treelite/main.o ./test_treelite/main.c -fPIC "
      "-std=c99 -lm";
    const char* lib_cmd =
      "gcc -shared -O3 -o ./test_treelite/treelite_model.so "
      "./test_treelite/main.o -std=c99 -lm";

    SYSTEMCALL_CHECK(system(obj_cmd));
    SYSTEMCALL_CHECK(system(lib_cmd));

    PredictorHandle predictor;
    const char* lib_path = "./test_treelite/treelite_model.so";

    // -1 means use maximum possible worker threads.
    int worker_thread = -1;
    TREELITE_CHECK(TreelitePredictorLoad(lib_path, worker_thread, &predictor));

    DenseBatchHandle dense_batch;
    // Current RF dosen't seem to support missing value, put FLT_MAX to be safe.
    float missing_value = FLT_MAX;
    TREELITE_CHECK(TreeliteAssembleDenseBatch(
      inference_data_h.data(), missing_value, params.n_inference_rows,
      params.n_cols, &dense_batch));

    // Use dense batch so batch_sparse is 0.
    // pred_margin = true means to produce raw margins rather than transformed probability.
    int batch_sparse = 0;
    bool pred_margin = false;
    // Allocate larger array for treelite predicted label with using multi-class classification to aviod seg faults.
    // Altough later we only use first params.n_inference_rows elements.
    size_t treelite_predicted_labels_size;

    TREELITE_CHECK(TreelitePredictorPredictBatch(
      predictor, dense_batch, batch_sparse, verbose, pred_margin,
      treelite_predicted_labels.data(), &treelite_predicted_labels_size));

    TREELITE_CHECK(TreeliteDeleteDenseBatch(dense_batch));
    TREELITE_CHECK(TreelitePredictorFree(predictor));
  }

  void epsilonCheck() {
    float epsilon = std::numeric_limits<float>::epsilon();

    diff_elements = 0;
    // To convert the probablily of binary classification to class index
    // The number in treelite_label is the probability of selecting class 1 instead of class 0
    // So we could say if the treelite_label is larger than 0.5, then it predicts class 1
    // There is a problem when probability equals to 0.5 (there is a tie between two classes)
    // Here return class 1 in this case but in Rapids RF, we use the class which first reachs the half votes as the final class:
    // https://gitlab-master.nvidia.com/RAPIDS/cuml/blob/branch-0.9/cpp/src/randomforest/randomforest_impl.cuh#L291
    for (int i = 0; i < params.n_inference_rows; i++) {
      if (is_classification) {
        treelite_predicted_labels[i] =
          treelite_predicted_labels[i] >= 0.5 ? 1 : 0;
      }

      if (ref_predicted_labels[i] - treelite_predicted_labels[i] > epsilon) {
        diff_elements++;
      }
    }
  }

  void SetUp() override { commonSetUp(); }

  void TearDown() override {
    CUDA_CHECK(cudaStreamDestroy(stream));

    data_h.clear();
    inference_data_h.clear();
    treelite_predicted_labels.clear();
    ref_predicted_labels.clear();

    CUDA_CHECK(cudaFree(data_d));
    CUDA_CHECK(cudaFree(inference_data_d));
  }

 protected:
  RfInputs<T> params;
  RF_params rf_params;

  T *data_d, *inference_data_d;
  std::vector<T> data_h;
  std::vector<T> inference_data_h;

  int diff_elements = 0;
  // Set to 1 for regression and 2 for binary classification
  // #class for multi-classification
  int task_category;
  int is_classification;

  int data_len;
  int inference_data_len;

  cudaStream_t stream;
  cumlHandle handle;

  std::vector<float> treelite_predicted_labels;
  std::vector<float> ref_predicted_labels;
};

template <typename T>
class RfTreeliteTestClf : public RfTreeliteTestCommon<T> {
 protected:
  void testClassifier() {
    allocate(labels_d, this->params.n_rows);
    allocate(predicted_labels_d, this->params.n_inference_rows);

    // Populate data (assume Col major)
    this->data_h = {30.0, 1.0, 2.0, 0.0, 10.0, 20.0, 10.0, 40.0};
    this->data_h.resize(this->data_len);
    updateDevice(this->data_d, this->data_h.data(), this->data_len,
                 this->stream);

    // Populate labels
    labels_h = {0, 1, 1, 0};
    labels_h.resize(this->params.n_rows);
    preprocess_labels(this->params.n_rows, labels_h, labels_map);
    updateDevice(labels_d, labels_h.data(), this->params.n_rows, this->stream);

    forest = new typename ML::RandomForestMetaData<T, int>;
    null_trees_ptr(forest);

    fit(this->handle, forest, this->data_d, this->params.n_rows,
        this->params.n_cols, labels_d, labels_map.size(), this->rf_params);

    CUDA_CHECK(cudaStreamSynchronize(this->stream));

    // Inference data: same as train, but row major
    this->inference_data_h = {30.0, 10.0, 1.0, 20.0, 2.0, 10.0, 0.0, 40.0};
    this->inference_data_h.resize(this->inference_data_len);
    updateDevice(this->inference_data_d, this->inference_data_h.data(),
                 this->data_len, this->stream);

    // Predict and compare against known labels
    RF_metrics tmp = score(this->handle, forest, this->inference_data_d,
                           labels_d, this->params.n_inference_rows,
                           this->params.n_cols, predicted_labels_d, false);
    CUDA_CHECK(cudaStreamSynchronize(this->stream));

    predicted_labels_h.resize(this->params.n_inference_rows);
    CUDA_CHECK(cudaMemcpy(predicted_labels_h.data(), predicted_labels_d,
                          sizeof(int) * this->params.n_inference_rows,
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < this->params.n_inference_rows; i++) {
      this->ref_predicted_labels[i] = static_cast<float>(predicted_labels_h[i]);
    }

    // Test the implementation for converting fitted forest into treelite format.
    ModelHandle model;
    this->task_category = 2;
    build_treelite_forest(&model, forest, this->params.n_cols,
                          this->task_category);
    this->convertToTreelite(model);
    TREELITE_CHECK(TreeliteFreeModel(model));

    this->is_classification = 1;
    this->epsilonCheck();

    postprocess_labels(this->params.n_rows, labels_h, labels_map);
    delete[] forest->trees;
    delete forest;
    CUDA_CHECK(cudaFree(labels_d));
    CUDA_CHECK(cudaFree(predicted_labels_d));
    labels_h.clear();
    predicted_labels_h.clear();
    labels_map.clear();
  }

 protected:
  int *labels_d, *predicted_labels_d;
  std::vector<int> labels_h;
  std::vector<int> predicted_labels_h;

  std::map<int, int>
    labels_map;  //unique map of labels to int vals starting from 0
  RandomForestMetaData<T, int>* forest;
};

// //-------------------------------------------------------------------------------------------------------------------------------------
template <typename T>
class RfTreeliteTestReg : public RfTreeliteTestCommon<T> {
 protected:
  void testRegressor() {
    allocate(labels_d, this->params.n_rows);
    allocate(predicted_labels_d, this->params.n_inference_rows);

    // Populate data (assume Col major)
    this->data_h = {0.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 40.0};
    this->data_h.resize(this->data_len);
    updateDevice(this->data_d, this->data_h.data(), this->data_len,
                 this->stream);

    // Populate labels
    labels_h = {1.0, 2.0, 3.0, 4.0};
    labels_h.resize(this->params.n_rows);
    updateDevice(labels_d, labels_h.data(), this->params.n_rows, this->stream);

    forest = new typename ML::RandomForestMetaData<T, T>;
    null_trees_ptr(forest);

    fit(this->handle, forest, this->data_d, this->params.n_rows,
        this->params.n_cols, labels_d, this->rf_params);

    CUDA_CHECK(cudaStreamSynchronize(this->stream));

    // Inference data: same as train, but row major
    this->inference_data_h = {0.0, 10.0, 0.0, 20.0, 0.0, 30.0, 0.0, 40.0};
    this->inference_data_h.resize(this->inference_data_len);
    updateDevice(this->inference_data_d, this->inference_data_h.data(),
                 this->data_len, this->stream);

    // Predict and compare against known labels
    RF_metrics tmp = score(this->handle, forest, this->inference_data_d,
                           labels_d, this->params.n_inference_rows,
                           this->params.n_cols, predicted_labels_d, false);
    CUDA_CHECK(cudaStreamSynchronize(this->stream));

    predicted_labels_h.resize(this->params.n_inference_rows);
    CUDA_CHECK(cudaMemcpy(predicted_labels_h.data(), predicted_labels_d,
                          sizeof(T) * this->params.n_inference_rows,
                          cudaMemcpyDeviceToHost));

    this->ref_predicted_labels = predicted_labels_h;

    // Test the implementation for converting fitted forest into treelite format.
    ModelHandle model;
    this->task_category = 1;
    build_treelite_forest(&model, forest, this->params.n_cols,
                          this->task_category);
    this->convertToTreelite(model);
    TREELITE_CHECK(TreeliteFreeModel(model));

    this->is_classification = 0;
    this->epsilonCheck();

    delete[] forest->trees;
    delete forest;
    CUDA_CHECK(cudaFree(labels_d));
    CUDA_CHECK(cudaFree(predicted_labels_d));
    labels_h.clear();
    predicted_labels_h.clear();
  }

 protected:
  T *labels_d, *predicted_labels_d;
  std::vector<T> labels_h;
  std::vector<T> predicted_labels_h;

  RandomForestMetaData<T, T>* forest;
};

// //-------------------------------------------------------------------------------------------------------------------------------------

const std::vector<RfInputs<float>> inputsf2_clf = {
  {4, 2, 1, 1.0f, 1.0f, 4, -1, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::
     GINI},  // single tree forest, bootstrap false, unlimited depth, 4 bins
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::GINI},  // single tree forest, bootstrap false, depth of 8, 4 bins
  {4, 2, 10, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::
     GINI},  //forest with 10 trees, all trees should produce identical predictions (no bootstrapping or column subsampling)
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::HIST, 2,
   CRITERION::
     GINI},  //forest with 10 trees, with bootstrap and column subsampling enabled, 3 bins
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::GLOBAL_QUANTILE,
   2,
   CRITERION::
     CRITERION_END},  //forest with 10 trees, with bootstrap and column subsampling enabled, 3 bins, different split algorithm
  {4, 2, 1, 1.0f, 1.0f, 4, -1, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::ENTROPY},
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::ENTROPY},
  {4, 2, 10, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::ENTROPY},
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::HIST, 2,
   CRITERION::ENTROPY},
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::GLOBAL_QUANTILE,
   2, CRITERION::ENTROPY}};

typedef RfTreeliteTestClf<float> RfBinaryClassifierTreeliteTestF;
TEST_P(RfBinaryClassifierTreeliteTestF, Convert) {
  testClassifier();
  ASSERT_TRUE(diff_elements == 0);
}

INSTANTIATE_TEST_CASE_P(RfBinaryClassifierTreeliteTests,
                        RfBinaryClassifierTreeliteTestF,
                        ::testing::ValuesIn(inputsf2_clf));

const std::vector<RfInputs<float>> inputsf2_reg = {
  {4, 2, 1, 1.0f, 1.0f, 4, -1, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::MSE},
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::MSE},
  {4, 2, 5, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::
     CRITERION_END},  // CRITERION_END uses the default criterion (GINI for classification, MSE for regression)
  {4, 2, 1, 1.0f, 1.0f, 4, -1, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::MAE},
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::GLOBAL_QUANTILE,
   2, CRITERION::MAE},
  {4, 2, 5, 1.0f, 1.0f, 4, 8, -1, true, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::CRITERION_END}};

typedef RfTreeliteTestReg<float> RfRegressorTreeliteTestF;
TEST_P(RfRegressorTreeliteTestF, Convert) {
  testRegressor();
  ASSERT_TRUE(diff_elements == 0);
}

INSTANTIATE_TEST_CASE_P(RfRegressorTreeliteTests, RfRegressorTreeliteTestF,
                        ::testing::ValuesIn(inputsf2_reg));
}  // end namespace ML
