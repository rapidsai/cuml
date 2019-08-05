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
#include <sys/stat.h>
#include <test_utils.h>
#include <treelite/c_api.h>
#include <treelite/c_api_runtime.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include "decisiontree/decisiontree_impl.h"
#include "linalg/gemv.h"
#include "linalg/transpose.h"
#include "ml_utils.h"
#include "random/rng.h"
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
  CRITERION split_criterion;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const RfInputs<T> &dims) {
  return os;
}

template <typename T, typename L>
class RfTreeliteTestCommon : public ::testing::TestWithParam<RfInputs<T>> {
 protected:
  void convertToTreelite() {
    // Test the implementation for converting fitted forest into treelite format.
    ModelHandle model;
    build_treelite_forest(&model, forest, params.n_cols, task_category);

    std::string test_name =
      ::testing::UnitTest::GetInstance()->current_test_info()->name();
    // Get the test index from Google current_test_info.
    // The test index is the string after '/' in test_name.
    std::string index_str =
      test_name.substr(test_name.find("/") + 1, test_name.length());

    // Create a directory if the test is the first one in the test case.
    int mkdir_ret = mkdir(test_dir.c_str(), 0700);
    if (mkdir_ret != 0) {
      // Ignore the error if the error is caused by EEXIST.
      // Treelite will generate errors when the directory is not accessible.
      ASSERT(errno == EEXIST, "Call mkdir %s fails.", test_dir.c_str());
    }

    // Create a sub-directory for the test case.
    dir_name = test_dir + index_str;

    CompilerHandle compiler;
    // "ast_navive" is the default compiler treelite used in their Python code.
    TREELITE_CHECK(TreeliteCompilerCreate("ast_native", &compiler));

    int verbose = 0;
    // Generate C code in the directory specified below.
    // The parallel comilplation is disabled. To enable it, one needs to specify parallel_comp of CompilerHandle.
    // Treelite will create a directory if it doesn't exist.
    TREELITE_CHECK(
      TreeliteCompilerGenerateCode(compiler, model, verbose, dir_name.c_str()));
    TREELITE_CHECK(TreeliteCompilerFree(compiler));

    // Options copied from
    // https://github.com/dmlc/treelite/blob/528d883f8f39eb5dd633e929b95915b63e210b39/python/treelite/contrib/__init__.py.
    std::string obj_cmd = "gcc -c -O3 -o " + dir_name + "/main.o " + dir_name +
                          "/main.c -fPIC "
                          "-std=c99 -lm";

    std::string lib_cmd = "gcc -shared -O3 -o " + dir_name +
                          "/treelite_model.so " + dir_name +
                          "/main.o -std=c99 -lm";

    ASSERT(system(obj_cmd.c_str()) == 0, "Call %s fails.", obj_cmd.c_str());
    ASSERT(system(lib_cmd.c_str()) == 0, "Call %s fails.", lib_cmd.c_str());

    PredictorHandle predictor;
    std::string lib_path = dir_name + "/treelite_model.so";

    // -1 means use maximum possible worker threads.
    int worker_thread = -1;
    TREELITE_CHECK(
      TreelitePredictorLoad(lib_path.c_str(), worker_thread, &predictor));

    DenseBatchHandle dense_batch;
    // Current RF dosen't seem to support missing value, put NaN to be safe.
    float missing_value = std::numeric_limits<double>::quiet_NaN();
    TREELITE_CHECK(TreeliteAssembleDenseBatch(
      inference_data_h.data(), missing_value, params.n_inference_rows,
      params.n_cols, &dense_batch));

    // Use dense batch so batch_sparse is 0.
    // pred_margin = true means to produce raw margins rather than transformed probability.
    int batch_sparse = 0;
    bool pred_margin = false;
    // Allocate larger array for treelite predicted label with using multi-class classification to avoid seg faults.
    // Altough later we only use first params.n_inference_rows elements.
    size_t treelite_predicted_labels_size;

    TREELITE_CHECK(TreelitePredictorPredictBatch(
      predictor, dense_batch, batch_sparse, verbose, pred_margin,
      treelite_predicted_labels.data(), &treelite_predicted_labels_size));

    TREELITE_CHECK(TreeliteDeleteDenseBatch(dense_batch));
    TREELITE_CHECK(TreelitePredictorFree(predictor));
    TREELITE_CHECK(TreeliteFreeModel(model));
  }

  void getResultAndCheck() {
    // Predict and compare against known labels
    RF_metrics tmp =
      score(handle, forest, inference_data_d, labels_d, params.n_inference_rows,
            params.n_cols, predicted_labels_d, false);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    predicted_labels_h.resize(params.n_inference_rows);
    ref_predicted_labels.resize(params.n_inference_rows);

    updateHost(predicted_labels_h.data(), predicted_labels_d,
               params.n_inference_rows, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int i = 0; i < params.n_inference_rows; i++) {
      if (is_classification) {
        ref_predicted_labels[i] = static_cast<float>(predicted_labels_h[i]);
        treelite_predicted_labels[i] =
          treelite_predicted_labels[i] >= 0.5 ? 1 : 0;
      } else {
        ref_predicted_labels[i] = static_cast<float>(predicted_labels_h[i]);
      }
    }

    EXPECT_TRUE(devArrMatchHost(
      ref_predicted_labels.data(), treelite_predicted_labels.data(),
      params.n_inference_rows, Compare<float>(), stream));
  }

  void SetUp() override {
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

    allocate(labels_d, params.n_rows);
    allocate(predicted_labels_d, params.n_inference_rows);

    treelite_predicted_labels.resize(params.n_inference_rows);
    ref_predicted_labels.resize(params.n_inference_rows);

    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.setStream(stream);

    forest = new typename ML::RandomForestMetaData<T, L>;
    null_trees_ptr(forest);

    data_h.resize(data_len);
    inference_data_h.resize(inference_data_len);

    // Random number generator.
    Random::Rng r1(1234ULL);
    // Generate data_d is in column major order.
    r1.uniform(data_d, data_len, T(0.0), T(10.0), stream);
    Random::Rng r2(4321ULL);
    // Generate inference_data_d which is in row major order.
    r2.uniform(inference_data_d, inference_data_len, T(0.0), T(10.0), stream);

    updateHost(data_h.data(), data_d, data_len, stream);
    updateHost(inference_data_h.data(), inference_data_d, inference_data_len,
               stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(data_d));
    CUDA_CHECK(cudaFree(inference_data_d));
    CUDA_CHECK(cudaFree(labels_d));
    CUDA_CHECK(cudaFree(predicted_labels_d));

    delete[] forest->trees;
    delete forest;
    labels_h.clear();
    predicted_labels_h.clear();
    data_h.clear();
    inference_data_h.clear();
    treelite_predicted_labels.clear();
    ref_predicted_labels.clear();
  }

 protected:
  RfInputs<T> params;
  RF_params rf_params;

  T *data_d, *inference_data_d;
  std::vector<T> data_h;
  std::vector<T> inference_data_h;

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

  std::string test_dir;
  std::string dir_name;

  L *labels_d, *predicted_labels_d;
  std::vector<L> labels_h;
  std::vector<L> predicted_labels_h;

  RandomForestMetaData<T, L> *forest;
};  // namespace ML

template <typename T, typename L>
class RfTreeliteTestClf : public RfTreeliteTestCommon<T, L> {
 protected:
  void testClassifier() {
    this->test_dir = "./treelite_test_clf/";
    this->is_classification = 1;
    // task_category - 1 for regression, 2 for binary classification
    // #class for multi-class classification
    this->task_category = 2;

    float *weight, *temp_label_d, *temp_data_d;
    std::vector<float> temp_label_h;

    allocate(weight, this->params.n_cols);
    allocate(temp_label_d, this->params.n_rows);
    allocate(temp_data_d, this->data_len);

    Random::Rng r(1234ULL);

    // Generate weight for each feature.
    r.uniform(weight, this->params.n_cols, T(0.0), T(1.0), this->stream);
    // Generate noise.
    r.uniform(temp_label_d, this->params.n_rows, T(0.0), T(10.0), this->stream);

    LinAlg::transpose<float>(
      this->data_d, temp_data_d, this->params.n_rows, this->params.n_cols,
      this->handle.getImpl().getCublasHandle(), this->stream);

    LinAlg::gemv<float>(temp_data_d, this->params.n_cols, this->params.n_rows,
                        weight, temp_label_d, true, 1.f, 1.f,
                        this->handle.getImpl().getCublasHandle(), this->stream);

    temp_label_h.resize(this->params.n_rows);
    updateHost(temp_label_h.data(), temp_label_d, this->params.n_rows,
               this->stream);
    CUDA_CHECK(cudaStreamSynchronize(this->stream));

    int value;
    for (int i = 0; i < this->params.n_rows; i++) {
      // The value of temp_label is between 0 to 10*n_cols+noise_level(10).
      // Choose half of that as the theshold to balance two classes.
      if (temp_label_h[i] >= (10 * this->params.n_cols + 10) / 2.0) {
        value = 1;
      } else {
        value = 0;
      }
      this->labels_h.push_back(value);
    }

    updateDevice(this->labels_d, this->labels_h.data(), this->params.n_rows,
                 this->stream);

    preprocess_labels(this->params.n_rows, this->labels_h, labels_map);

    fit(this->handle, this->forest, this->data_d, this->params.n_rows,
        this->params.n_cols, this->labels_d, labels_map.size(),
        this->rf_params);

    CUDA_CHECK(cudaStreamSynchronize(this->stream));

    this->convertToTreelite();
    this->getResultAndCheck();

    postprocess_labels(this->params.n_rows, this->labels_h, this->labels_map);
    
    labels_map.clear();
    temp_label_h.clear(); 
    CUDA_CHECK(cudaFree(weight));
    CUDA_CHECK(cudaFree(temp_label_d));
    CUDA_CHECK(cudaFree(temp_data_d));
  }

 protected:
  std::map<int, int>
    labels_map;  //unique map of labels to int vals starting from 0
};

//-------------------------------------------------------------------------------------------------------------------------------------
template <typename T, typename L>
class RfTreeliteTestReg : public RfTreeliteTestCommon<T, L> {
 protected:
  void testRegressor() {
    this->test_dir = "./treelite_test_reg/";
    this->is_classification = 0;
    // task_category - 1 for regression, 2 for binary classification
    // #class for multi-class classification
    this->task_category = 1;

    float *weight, *temp_data_d;
    allocate(weight, this->params.n_cols);
    allocate(temp_data_d, this->data_len);

    Random::Rng r(1234ULL);

    // Generate weight for each feature.
    r.uniform(weight, this->params.n_cols, T(0.0), T(1.0), this->stream);
    // Generate noise.
    r.uniform(this->labels_d, this->params.n_rows, T(0.0), T(10.0),
              this->stream);

    LinAlg::transpose<float>(
      this->data_d, temp_data_d, this->params.n_rows, this->params.n_cols,
      this->handle.getImpl().getCublasHandle(), this->stream);

    LinAlg::gemv<float>(temp_data_d, this->params.n_cols, this->params.n_rows,
                        weight, this->labels_d, true, 1.f, 1.f,
                        this->handle.getImpl().getCublasHandle(), this->stream);

    this->labels_h.resize(this->params.n_rows);
    updateHost(this->labels_h.data(), this->labels_d, this->params.n_rows,
               this->stream);
    CUDA_CHECK(cudaStreamSynchronize(this->stream));

    fit(this->handle, this->forest, this->data_d, this->params.n_rows,
        this->params.n_cols, this->labels_d, this->rf_params);

    CUDA_CHECK(cudaStreamSynchronize(this->stream));

    this->convertToTreelite();
    this->getResultAndCheck();

    CUDA_CHECK(cudaFree(weight));
    CUDA_CHECK(cudaFree(temp_data_d));
  }
};

// //-------------------------------------------------------------------------------------------------------------------------------------

const std::vector<RfInputs<float>> inputsf2_clf = {
  {4, 2, 1, 1.0f, 1.0f, 4, -1, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::
     GINI},  // single tree forest, bootstrap false, unlimited depth, 4 bins
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::GINI},  // single tree forest, bootstrap false, depth of 8, 4 bins
  {4, 2, 11, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::
     GINI},  //forest with 11 trees, all trees should produce identical predictions (no bootstrapping or column subsampling)
  {40, 20, 11, 0.8f, 0.8f, 40, 8, -1, true, false, 3, SPLIT_ALGO::HIST, 2,
   CRITERION::
     GINI},  //forest with 11 trees, with bootstrap and column subsampling enabled, 3 bins
  {40, 20, 11, 0.8f, 0.8f, 40, 8, -1, true, false, 3,
   SPLIT_ALGO::GLOBAL_QUANTILE, 2,
   CRITERION::
     CRITERION_END},  //forest with 11 trees, with bootstrap and column subsampling enabled, 3 bins, different split algorithm
  {40, 20, 1, 1.0f, 1.0f, 40, -1, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::ENTROPY},
  {400, 200, 1, 1.0f, 1.0f, 400, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::ENTROPY},
  {400, 200, 11, 1.0f, 1.0f, 400, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::ENTROPY},
  {40, 20, 11, 0.8f, 0.8f, 40, 8, -1, true, false, 3, SPLIT_ALGO::HIST, 2,
   CRITERION::ENTROPY},
  {40, 20, 11, 0.8f, 0.8f, 40, 8, -1, true, false, 3,
   SPLIT_ALGO::GLOBAL_QUANTILE, 2, CRITERION::ENTROPY}};

typedef RfTreeliteTestClf<float, int> RfBinaryClassifierTreeliteTestF;
TEST_P(RfBinaryClassifierTreeliteTestF, Convert_Clf) { testClassifier(); }

INSTANTIATE_TEST_CASE_P(RfBinaryClassifierTreeliteTests,
                        RfBinaryClassifierTreeliteTestF,
                        ::testing::ValuesIn(inputsf2_clf));

const std::vector<RfInputs<float>> inputsf2_reg = {
  {4, 2, 1, 1.0f, 1.0f, 4, -1, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::MSE},
  {40, 20, 1, 1.0f, 1.0f, 40, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::MSE},
  {40, 20, 5, 1.0f, 1.0f, 40, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::
     CRITERION_END},  // CRITERION_END uses the default criterion (GINI for classification, MSE for regression)
  {40, 20, 1, 1.0f, 1.0f, 40, -1, -1, false, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::MAE},
  {400, 200, 1, 1.0f, 1.0f, 400, 8, -1, false, false, 4,
   SPLIT_ALGO::GLOBAL_QUANTILE, 2, CRITERION::MAE},
  {400, 200, 5, 1.0f, 1.0f, 400, 8, -1, true, false, 4, SPLIT_ALGO::HIST, 2,
   CRITERION::CRITERION_END}};

typedef RfTreeliteTestReg<float, float> RfRegressorTreeliteTestF;
TEST_P(RfRegressorTreeliteTestF, Convert_Reg) { testRegressor(); }

INSTANTIATE_TEST_CASE_P(RfRegressorTreeliteTests, RfRegressorTreeliteTestF,
                        ::testing::ValuesIn(inputsf2_reg));
}  // end namespace ML
