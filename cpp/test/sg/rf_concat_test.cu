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
#include "cuml/ensemble/randomforest.hpp"
#include "decisiontree/decisiontree_impl.h"
#include "linalg/gemv.h"
#include "linalg/transpose.h"
#include "ml_utils.h"
#include "random/rng.h"

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
  float min_impurity_decrease;
  int n_streams;
  CRITERION split_criterion;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const RfInputs<T> &dims) {
  return os;
}

template <typename T, typename L>
class RfTreeliteTestCommon : public ::testing::TestWithParam<RfInputs<T>> {
 protected:
  void ConcatenateTreeliteModels() {
    // Test the implementation for converting fitted forest into treelite format.
    ModelHandle concatenated_forest_handle;
    concatenated_forest_handle = concatenate_trees(treelite_indiv_handels);
    compare_concat_forest_to_subforests(concatenated_forest_handle,
                                        treelite_indiv_handels);
    TREELITE_CHECK(TreeliteFreeModel(concatenated_forest_handle));
  }

  void SetUp() override {
    params = ::testing::TestWithParam<RfInputs<T>>::GetParam();

    DecisionTree::DecisionTreeParams tree_params;
    set_tree_params(tree_params, params.max_depth, params.max_leaves,
                    params.max_features, params.n_bins, params.split_algo,
                    params.min_rows_per_node, params.min_impurity_decrease,
                    params.bootstrap_features, params.split_criterion, false);
    set_all_rf_params(rf_params, params.n_trees, params.bootstrap,
                      params.rows_sample, -1, params.n_streams, tree_params);
    handle.reset(new cumlHandle(rf_params.n_streams));

    data_len = params.n_rows * params.n_cols;
    inference_data_len = params.n_inference_rows * params.n_cols;

    allocate(data_d, data_len);
    allocate(inference_data_d, inference_data_len);

    allocate(labels_d, params.n_rows);
    allocate(predicted_labels_d, params.n_inference_rows);

    treelite_predicted_labels.resize(params.n_inference_rows);
    ref_predicted_labels.resize(params.n_inference_rows);

    CUDA_CHECK(cudaStreamCreate(&stream));
    handle->setStream(stream);

    forest = new typename ML::RandomForestMetaData<T, L>;
    null_trees_ptr(forest);
    forest_2 = new typename ML::RandomForestMetaData<T, L>;
    null_trees_ptr(forest_2);
    forest_3 = new typename ML::RandomForestMetaData<T, L>;
    null_trees_ptr(forest_3);
    all_forest_info = {forest, forest_2, forest_3};
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
    CUDA_CHECK(cudaFree(labels_d));

    delete[] forest->trees;
    delete forest;
    delete[] forest_2->trees;
    delete forest_2;
    delete[] forest_3->trees;
    delete forest_3;
    all_forest_info.clear();
    labels_h.clear();
    predicted_labels_h.clear();
    data_h.clear();
    inference_data_h.clear();
    treelite_predicted_labels.clear();
    ref_predicted_labels.clear();
    treelite_indiv_handels.clear();
  }

 protected:
  RfInputs<T> params;
  RF_params rf_params;
  T *data_d, *inference_data_d;
  std::vector<T> data_h;
  std::vector<T> inference_data_h;
  std::vector<ModelHandle> treelite_indiv_handels;

  // Set to 1 for regression and 2 for binary classification
  // #class for multi-classification
  int task_category;
  int is_classification;

  int data_len;
  int inference_data_len;

  cudaStream_t stream;
  std::shared_ptr<cumlHandle> handle;
  std::vector<float> treelite_predicted_labels;
  std::vector<float> ref_predicted_labels;
  std::vector<ML::RandomForestMetaData<T, L>*> all_forest_info;
  std::string test_dir;
  std::string dir_name;

  L *labels_d, *predicted_labels_d;
  std::vector<L> labels_h;
  std::vector<L> predicted_labels_h;

  RandomForestMetaData<T, L> *forest;
  RandomForestMetaData<T, L> *forest_2;
  RandomForestMetaData<T, L> *forest_3;
};  // namespace ML

template <typename T, typename L>
class RfConcatTestClf : public RfTreeliteTestCommon<T, L> {
 protected:
  void testClassifier() {
    this->test_dir = "./treelite_test_clf/";
    this->is_classification = 1;
    //task_category - 1 for regression, 2 for binary classification
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
      this->handle->getImpl().getCublasHandle(), this->stream);

    LinAlg::gemv<float>(temp_data_d, this->params.n_cols, this->params.n_rows,
                        weight, temp_label_d, true, 1.f, 1.f,
                        this->handle->getImpl().getCublasHandle(),
                        this->stream);
    
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

    for (int i = 0; i < 3; i++){
      ModelHandle model;
      std::vector<unsigned char> vec_data;
      this->rf_params.n_trees = this->rf_params.n_trees + i;

      fit(*(this->handle), this->all_forest_info[i], this->data_d, this->params.n_rows,
          this->params.n_cols, this->labels_d, labels_map.size(),
          this->rf_params);
      build_treelite_forest(&model, this->all_forest_info[i], this->params.n_cols, this->task_category,
                            vec_data);
      this->treelite_indiv_handels.push_back(model);
    }

    CUDA_CHECK(cudaStreamSynchronize(this->stream));
    
    this->ConcatenateTreeliteModels();

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
class RfConcatTestReg : public RfTreeliteTestCommon<T, L> {
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
      this->handle->getImpl().getCublasHandle(), this->stream);

    LinAlg::gemv<float>(temp_data_d, this->params.n_cols, this->params.n_rows,
                        weight, this->labels_d, true, 1.f, 1.f,
                        this->handle->getImpl().getCublasHandle(),
                        this->stream);

    this->labels_h.resize(this->params.n_rows);
    updateHost(this->labels_h.data(), this->labels_d, this->params.n_rows,
               this->stream);
    CUDA_CHECK(cudaStreamSynchronize(this->stream));

    for (int i = 0; i < 3; i++){
      ModelHandle model;
      std::vector<unsigned char> vec_data;
      this->rf_params.n_trees = this->rf_params.n_trees + i;
      
      fit(*(this->handle), this->all_forest_info[i], this->data_d, this->params.n_rows,
          this->params.n_cols, this->labels_d, this->rf_params);
      build_treelite_forest(&model, this->all_forest_info[i], this->params.n_cols, this->task_category,
                            vec_data);
      CUDA_CHECK(cudaStreamSynchronize(this->stream));
      this->treelite_indiv_handels.push_back(model);
    }

    this->ConcatenateTreeliteModels();

    CUDA_CHECK(cudaFree(weight));
    CUDA_CHECK(cudaFree(temp_data_d));
  }
};

// //-------------------------------------------------------------------------------------------------------------------------------------
const std::vector<RfInputs<float>> inputsf2_clf = {
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 0.0, 2,
   CRITERION::GINI},  // single tree forest, bootstrap false, depth 8, 4 bins
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 0.0, 2,
   CRITERION::GINI},  // single tree forest, bootstrap false, depth of 8, 4 bins
  {4, 2, 10, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 0.0, 2,
   CRITERION::
     GINI},  //forest with 10 trees, all trees should produce identical predictions (no bootstrapping or column subsampling)
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::HIST, 2, 0.0, 2,
   CRITERION::
     GINI},  //forest with 10 trees, with bootstrap and column subsampling enabled, 3 bins
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::GLOBAL_QUANTILE,
   2, 0.0, 2,
   CRITERION::
     CRITERION_END},  //forest with 10 trees, with bootstrap and column subsampling enabled, 3 bins, different split algorithm
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 0.0, 2,
   CRITERION::ENTROPY},
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 0.0, 2,
   CRITERION::ENTROPY},
  {4, 2, 10, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 0.0, 2,
   CRITERION::ENTROPY},
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::HIST, 2, 0.0, 2,
   CRITERION::ENTROPY},
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, false, 3, SPLIT_ALGO::GLOBAL_QUANTILE,
   2, 0.0, 2, CRITERION::ENTROPY}};

typedef RfConcatTestClf<float, int> RfClassifierConcatTestF;
TEST_P(RfClassifierConcatTestF, Convert_Clf) { testClassifier(); }

INSTANTIATE_TEST_CASE_P(RfBinaryClassifierConcatTests,
                        RfClassifierConcatTestF,
                        ::testing::ValuesIn(inputsf2_clf));

const std::vector<RfInputs<float>> inputsf2_reg = {
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 0.0, 2,
   CRITERION::MSE},
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 0.0, 2,
   CRITERION::MSE},
  {4, 2, 5, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 0.0, 2,
   CRITERION::
     CRITERION_END},  // CRITERION_END uses the default criterion (GINI for classification, MSE for regression)
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::HIST, 2, 0.0, 2,
   CRITERION::MAE},
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, false, 4, SPLIT_ALGO::GLOBAL_QUANTILE,
   2, 0.0, 2, CRITERION::MAE},
  {4, 2, 5, 1.0f, 1.0f, 4, 8, -1, true, false, 4, SPLIT_ALGO::HIST, 2, 0.0, 2,
   CRITERION::CRITERION_END}};

typedef RfConcatTestReg<float, float> RfRegressorConcatTestF;
TEST_P(RfRegressorConcatTestF, Convert_Reg) { testRegressor(); }

INSTANTIATE_TEST_CASE_P(RfRegressorConcatTests, RfRegressorConcatTestF,
                        ::testing::ValuesIn(inputsf2_reg));
}  // end namespace ML
