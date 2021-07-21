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

#include <test_utils.h>

#include <decisiontree/treelite_util.h>
#include <decisiontree/decisiontree.cuh>

#include <cuml/ensemble/randomforest.hpp>

#include <raft/cudart_utils.h>
#include <raft/linalg/gemv.h>
#include <raft/linalg/transpose.h>
#include <raft/cuda_utils.cuh>
#include <raft/random/rng.cuh>

#include <treelite/c_api.h>
#include <treelite/c_api_runtime.h>

#include <gtest/gtest.h>

#include <sys/stat.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

namespace ML {

using namespace MLCommon;

template <typename T>  // template useless for now.
struct RfInputs {
  int n_rows;
  int n_cols;
  int n_trees;
  float max_features;
  float max_samples;
  int n_inference_rows;
  int max_depth;
  int max_leaves;
  bool bootstrap;
  int n_bins;
  int min_samples_leaf;
  int min_samples_split;
  float min_impurity_decrease;
  int n_streams;
  CRITERION split_criterion;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const RfInputs<T>& dims)
{
  return os;
}

template <typename T, typename L>
class RfTreeliteTestCommon : public ::testing::TestWithParam<RfInputs<T>> {
 protected:
  void ConcatenateTreeliteModels()
  {
    // Test the implementation for converting fitted forest into treelite format.
    ModelHandle concatenated_forest_handle;
    concatenated_forest_handle = concatenate_trees(treelite_indiv_handles);
    compare_concat_forest_to_subforests(concatenated_forest_handle, treelite_indiv_handles);
    std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    // Get the test index from Google current_test_info.
    // The test index is the string after '/' in test_name.
    std::string index_str = test_name.substr(test_name.find("/") + 1, test_name.length());

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
    // The parallel comilplation is disabled. To enable it, one needs to specify parallel_comp of
    // CompilerHandle. Treelite will create a directory if it doesn't exist.
    TREELITE_CHECK(
      TreeliteCompilerGenerateCode(compiler, treelite_indiv_handles[0], verbose, dir_name.c_str()));
    TREELITE_CHECK(TreeliteCompilerFree(compiler));

    // Options copied from
    // https://github.com/dmlc/treelite/blob/528d883f8f39eb5dd633e929b95915b63e210b39/python/treelite/contrib/__init__.py.
    std::string obj_cmd = "gcc -c -O3 -o " + dir_name + "/main.o " + dir_name +
                          "/main.c -fPIC "
                          "-std=c99 -lm";

    std::string lib_cmd =
      "gcc -shared -O3 -o " + dir_name + "/treelite_model.so " + dir_name + "/main.o -std=c99 -lm";

    ASSERT(system(obj_cmd.c_str()) == 0, "Call %s fails.", obj_cmd.c_str());
    ASSERT(system(lib_cmd.c_str()) == 0, "Call %s fails.", lib_cmd.c_str());

    PredictorHandle predictor;
    std::string lib_path = dir_name + "/treelite_model.so";

    // -1 means use maximum possible worker threads.
    int worker_thread = -1;
    TREELITE_CHECK(TreelitePredictorLoad(lib_path.c_str(), worker_thread, &predictor));

    DMatrixHandle dmat;
    // Current RF doesn't seem to support missing value, put NaN to be safe.
    T missing_value = std::numeric_limits<T>::quiet_NaN();
    TREELITE_CHECK(TreeliteDMatrixCreateFromMat(inference_data_h.data(),
                                                ML::DT::TreeliteType<T>::value,
                                                params.n_inference_rows,
                                                params.n_cols,
                                                &missing_value,
                                                &dmat));

    // Use dense batch so batch_sparse is 0.
    // pred_margin = true means to produce raw margins rather than transformed probability.
    bool pred_margin = false;
    // Allocate larger array for treelite predicted label with using multi-class classification to
    // avoid seg faults. Altough later we only use first params.n_inference_rows elements.
    size_t treelite_predicted_labels_size;

    TREELITE_CHECK(TreelitePredictorPredictBatch(predictor,
                                                 dmat,
                                                 verbose,
                                                 pred_margin,
                                                 treelite_predicted_labels.data(),
                                                 &treelite_predicted_labels_size));

    TREELITE_CHECK(TreeliteDMatrixFree(dmat));
    TREELITE_CHECK(TreelitePredictorFree(predictor));
    TREELITE_CHECK(TreeliteFreeModel(concatenated_forest_handle));
    TREELITE_CHECK(TreeliteFreeModel(treelite_indiv_handles[0]));
    TREELITE_CHECK(TreeliteFreeModel(treelite_indiv_handles[1]));
    TREELITE_CHECK(TreeliteFreeModel(treelite_indiv_handles[2]));
  }

  void getResultAndCheck()
  {
    // Predict and compare against known labels
    predict(*handle,
            forest,
            inference_data_d,
            params.n_inference_rows,
            params.n_cols,
            predicted_labels_d);
    score(*handle, forest, labels_d, params.n_inference_rows, predicted_labels_d);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    predicted_labels_h.resize(params.n_inference_rows);
    ref_predicted_labels.resize(params.n_inference_rows);

    raft::update_host(
      predicted_labels_h.data(), predicted_labels_d, params.n_inference_rows, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int i = 0; i < params.n_inference_rows; i++) {
      if (is_classification) {
        ref_predicted_labels[i]      = static_cast<float>(predicted_labels_h[i]);
        treelite_predicted_labels[i] = treelite_predicted_labels[i] >= 0.5 ? 1 : 0;
      } else {
        ref_predicted_labels[i] = static_cast<float>(predicted_labels_h[i]);
      }
    }

    EXPECT_TRUE(raft::devArrMatchHost(ref_predicted_labels.data(),
                                      treelite_predicted_labels.data(),
                                      params.n_inference_rows,
                                      raft::Compare<float>(),
                                      stream));
  }

  void SetUp() override
  {
    params = ::testing::TestWithParam<RfInputs<T>>::GetParam();

    rf_params = set_rf_params(params.max_depth,
                              params.max_leaves,
                              params.max_features,
                              params.n_bins,
                              params.min_samples_leaf,
                              params.min_samples_split,
                              params.min_impurity_decrease,
                              params.bootstrap,
                              params.n_trees,
                              params.max_samples,
                              0,
                              params.split_criterion,
                              params.n_streams,
                              128);

    handle.reset(new raft::handle_t(rf_params.n_streams));

    data_len           = params.n_rows * params.n_cols;
    inference_data_len = params.n_inference_rows * params.n_cols;

    raft::allocate(data_d, data_len);
    raft::allocate(inference_data_d, inference_data_len);

    raft::allocate(labels_d, params.n_rows);
    raft::allocate(predicted_labels_d, params.n_inference_rows);

    treelite_predicted_labels.resize(params.n_inference_rows);
    ref_predicted_labels.resize(params.n_inference_rows);

    CUDA_CHECK(cudaStreamCreate(&stream));
    handle->set_stream(stream);

    forest          = new typename ML::RandomForestMetaData<T, L>;
    forest_2        = new typename ML::RandomForestMetaData<T, L>;
    forest_3        = new typename ML::RandomForestMetaData<T, L>;
    all_forest_info = {forest, forest_2, forest_3};
    data_h.resize(data_len);
    inference_data_h.resize(inference_data_len);

    // Random number generator.
    raft::random::Rng r1(1234ULL);
    // Generate data_d is in column major order.
    r1.uniform(data_d, data_len, T(0.0), T(10.0), stream);
    raft::random::Rng r2(4321ULL);
    // Generate inference_data_d which is in row major order.
    r2.uniform(inference_data_d, inference_data_len, T(0.0), T(10.0), stream);

    raft::update_host(data_h.data(), data_d, data_len, stream);
    raft::update_host(inference_data_h.data(), inference_data_d, inference_data_len, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown() override
  {
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(data_d));
    CUDA_CHECK(cudaFree(inference_data_d));
    CUDA_CHECK(cudaFree(labels_d));
    CUDA_CHECK(cudaFree(predicted_labels_d));

    delete forest;
    delete forest_2;
    delete forest_3;
    all_forest_info.clear();
    labels_h.clear();
    predicted_labels_h.clear();
    data_h.clear();
    inference_data_h.clear();
    treelite_predicted_labels.clear();
    ref_predicted_labels.clear();
    treelite_indiv_handles.clear();
  }

 protected:
  RfInputs<T> params;
  RF_params rf_params;
  T *data_d, *inference_data_d;
  std::vector<T> data_h;
  std::vector<T> inference_data_h;
  std::vector<ModelHandle> treelite_indiv_handles;

  // Set to 1 for regression and 2 for binary classification
  // #class for multi-classification
  int task_category;
  int is_classification;

  int data_len;
  int inference_data_len;

  cudaStream_t stream;
  std::shared_ptr<raft::handle_t> handle;
  std::vector<float> treelite_predicted_labels;
  std::vector<float> ref_predicted_labels;
  std::vector<ML::RandomForestMetaData<T, L>*> all_forest_info;
  std::string test_dir;
  std::string dir_name;

  L *labels_d, *predicted_labels_d;
  std::vector<L> labels_h;
  std::vector<L> predicted_labels_h;

  RandomForestMetaData<T, L>* forest;
  RandomForestMetaData<T, L>* forest_2;
  RandomForestMetaData<T, L>* forest_3;
};  // namespace ML

template <typename T, typename L>
class RfConcatTestClf : public RfTreeliteTestCommon<T, L> {
 protected:
  void testClassifier()
  {
    this->test_dir          = "./concat_test_clf/";
    this->is_classification = 1;
    // task_category - 1 for regression, 2 for binary classification
    // #class for multi-class classification
    this->task_category = 2;

    float *weight, *temp_label_d, *temp_data_d;
    std::vector<float> temp_label_h;

    raft::allocate(weight, this->params.n_cols);
    raft::allocate(temp_label_d, this->params.n_rows);
    raft::allocate(temp_data_d, this->data_len);

    raft::random::Rng r(1234ULL);

    // Generate weight for each feature.
    r.uniform(weight, this->params.n_cols, T(0.0), T(1.0), this->stream);
    // Generate noise.
    r.uniform(temp_label_d, this->params.n_rows, T(0.0), T(10.0), this->stream);

    raft::linalg::transpose<float>(*(this->handle),
                                   this->data_d,
                                   temp_data_d,
                                   this->params.n_rows,
                                   this->params.n_cols,
                                   this->stream);

    raft::linalg::gemv<float>(*(this->handle),
                              temp_data_d,
                              this->params.n_cols,
                              this->params.n_rows,
                              weight,
                              temp_label_d,
                              true,
                              1.f,
                              1.f,
                              this->stream);

    temp_label_h.resize(this->params.n_rows);
    raft::update_host(temp_label_h.data(), temp_label_d, this->params.n_rows, this->stream);

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

    raft::update_device(this->labels_d, this->labels_h.data(), this->params.n_rows, this->stream);

    preprocess_labels(this->params.n_rows, this->labels_h, labels_map);

    for (int i = 0; i < 3; i++) {
      ModelHandle model;

      this->rf_params.n_trees = this->rf_params.n_trees + i;

      fit(*(this->handle),
          this->all_forest_info[i],
          this->data_d,
          this->params.n_rows,
          this->params.n_cols,
          this->labels_d,
          labels_map.size(),
          this->rf_params);
      build_treelite_forest(
        &model, this->all_forest_info[i], this->params.n_cols, this->task_category);
      this->treelite_indiv_handles.push_back(model);
    }

    CUDA_CHECK(cudaStreamSynchronize(this->stream));

    this->ConcatenateTreeliteModels();
    this->getResultAndCheck();

    postprocess_labels(this->params.n_rows, this->labels_h, this->labels_map);

    labels_map.clear();
    temp_label_h.clear();
    CUDA_CHECK(cudaFree(weight));
    CUDA_CHECK(cudaFree(temp_label_d));
    CUDA_CHECK(cudaFree(temp_data_d));
  }

 protected:
  std::map<int, int> labels_map;  // unique map of labels to int vals starting from 0
};

//-------------------------------------------------------------------------------------------------------------------------------------
template <typename T, typename L>
class RfConcatTestReg : public RfTreeliteTestCommon<T, L> {
 protected:
  void testRegressor()
  {
    this->test_dir          = "./concat_test_reg/";
    this->is_classification = 0;
    // task_category - 1 for regression, 2 for binary classification
    // #class for multi-class classification
    this->task_category = 1;

    float *weight, *temp_data_d;
    raft::allocate(weight, this->params.n_cols);
    raft::allocate(temp_data_d, this->data_len);

    raft::random::Rng r(1234ULL);

    // Generate weight for each feature.
    r.uniform(weight, this->params.n_cols, T(0.0), T(1.0), this->stream);
    // Generate noise.
    r.uniform(this->labels_d, this->params.n_rows, T(0.0), T(10.0), this->stream);

    raft::linalg::transpose<float>(*(this->handle),
                                   this->data_d,
                                   temp_data_d,
                                   this->params.n_rows,
                                   this->params.n_cols,
                                   this->stream);

    raft::linalg::gemv<float>(*(this->handle),
                              temp_data_d,
                              this->params.n_cols,
                              this->params.n_rows,
                              weight,
                              this->labels_d,
                              true,
                              1.f,
                              1.f,
                              this->stream);

    this->labels_h.resize(this->params.n_rows);
    raft::update_host(this->labels_h.data(), this->labels_d, this->params.n_rows, this->stream);
    CUDA_CHECK(cudaStreamSynchronize(this->stream));

    for (int i = 0; i < 3; i++) {
      ModelHandle model;

      this->rf_params.n_trees = this->rf_params.n_trees + i;

      fit(*(this->handle),
          this->all_forest_info[i],
          this->data_d,
          this->params.n_rows,
          this->params.n_cols,
          this->labels_d,
          this->rf_params);
      build_treelite_forest(
        &model, this->all_forest_info[i], this->params.n_cols, this->task_category);
      CUDA_CHECK(cudaStreamSynchronize(this->stream));
      this->treelite_indiv_handles.push_back(model);
    }

    this->ConcatenateTreeliteModels();
    this->getResultAndCheck();

    CUDA_CHECK(cudaFree(weight));
    CUDA_CHECK(cudaFree(temp_data_d));
  }
};

// //-------------------------------------------------------------------------------------------------------------------------------------
const std::vector<RfInputs<float>> inputsf2_clf = {
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, 4, 2, 2, 0.0, 2, CRITERION::GINI},  // single tree forest,
                                                                             // bootstrap false,
                                                                             // depth 8, 4 bins
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, 4, 2, 2, 0.0, 2, CRITERION::GINI},  // single tree forest,
                                                                             // bootstrap false,
                                                                             // depth of 8, 4 bins
  {4,
   2,
   10,
   1.0f,
   1.0f,
   4,
   8,
   -1,
   false,
   4,
   2,
   2,
   0.0,
   2,
   CRITERION::GINI},  // forest with 10 trees, all trees should produce identical predictions (no
                      // bootstrapping or column subsampling)
  {4,
   2,
   10,
   0.8f,
   0.8f,
   4,
   8,
   -1,
   true,
   3,
   2,
   2,
   0.0,
   2,
   CRITERION::GINI},  // forest with 10 trees, with bootstrap and column subsampling enabled, 3 bins
  {4,
   2,
   10,
   0.8f,
   0.8f,
   4,
   8,
   -1,
   true,
   3,
   2,
   2,
   0.0,
   2,
   CRITERION::CRITERION_END},  // forest with 10 trees, with bootstrap and column subsampling
                               // enabled, 3 bins, different split algorithm
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, 4, 2, 2, 0.0, 2, CRITERION::ENTROPY},
  {4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, 4, 2, 2, 0.0, 2, CRITERION::ENTROPY},
  {4, 2, 10, 1.0f, 1.0f, 4, 8, -1, false, 4, 2, 2, 0.0, 2, CRITERION::ENTROPY},
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, 3, 2, 2, 0.0, 2, CRITERION::ENTROPY},
  {4, 2, 10, 0.8f, 0.8f, 4, 8, -1, true, 3, 2, 2, 0.0, 2, CRITERION::ENTROPY}};

typedef RfConcatTestClf<float, int> RfClassifierConcatTestF;
TEST_P(RfClassifierConcatTestF, Convert_Clf) { testClassifier(); }

INSTANTIATE_TEST_CASE_P(RfBinaryClassifierConcatTests,
                        RfClassifierConcatTestF,
                        ::testing::ValuesIn(inputsf2_clf));

const std::vector<RfInputs<float>> inputsf2_reg = {
  {4, 2, 1, 1.0f, 1.0f, 4, 7, -1, false, 4, 2, 2, 0.0, 2, CRITERION::MSE},
  {4, 2, 1, 1.0f, 1.0f, 4, 7, -1, false, 4, 2, 2, 0.0, 2, CRITERION::MSE},
  {4,
   2,
   5,
   1.0f,
   1.0f,
   4,
   7,
   -1,
   false,
   4,
   2,
   2,
   0.0,
   2,
   CRITERION::CRITERION_END},  // CRITERION_END uses the default criterion (GINI for classification,
                               // MSE for regression)
  {4, 2, 5, 1.0f, 1.0f, 4, 7, -1, true, 4, 2, 2, 0.0, 2, CRITERION::CRITERION_END}};

typedef RfConcatTestReg<float, float> RfRegressorConcatTestF;
TEST_P(RfRegressorConcatTestF, Convert_Reg) { testRegressor(); }

INSTANTIATE_TEST_CASE_P(RfRegressorConcatTests,
                        RfRegressorConcatTestF,
                        ::testing::ValuesIn(inputsf2_reg));
}  // end namespace ML
