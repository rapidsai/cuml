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
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
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
    TREELITE_CHECK(TreeliteCompilerCreateV2("ast_native", "{}", &compiler));

    // Generate C code in the directory specified below.
    // The parallel comilplation is disabled. To enable it, one needs to specify parallel_comp of
    // CompilerHandle. Treelite will create a directory if it doesn't exist.
    TREELITE_CHECK(
      TreeliteCompilerGenerateCodeV2(compiler, treelite_indiv_handles[0], dir_name.c_str()));
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

    int verbose = 0;
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
    thrust::device_vector<L> predicted_labels_d(params.n_inference_rows);
    // Predict and compare against known labels
    predict(*handle,
            &all_forest_info[0],
            inference_data_d.data().get(),
            params.n_inference_rows,
            params.n_cols,
            predicted_labels_d.data().get());
    thrust::host_vector<L> predicted_labels_h(predicted_labels_d);

    for (int i = 0; i < params.n_inference_rows; i++) {
      EXPECT_EQ(predicted_labels_h[i], treelite_predicted_labels[i]);
    }
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

    auto data_len           = params.n_rows * params.n_cols;
    auto inference_data_len = params.n_inference_rows * params.n_cols;

    data_d.resize(data_len);
    inference_data_d.resize(inference_data_len);
    labels_d.resize(params.n_rows);

    treelite_predicted_labels.resize(params.n_inference_rows);

    all_forest_info.resize(3);
    data_h.resize(data_len);
    inference_data_h.resize(inference_data_len);

    // Random number generator.
    raft::random::Rng r1(1234ULL);
    // Generate data_d is in column major order.
    r1.uniform(data_d.data().get(), data_len, T(0.0), T(10.0), handle->get_stream());
    raft::random::Rng r2(4321ULL);
    // Generate inference_data_d which is in row major order.
    r2.uniform(
      inference_data_d.data().get(), inference_data_len, T(0.0), T(10.0), handle->get_stream());

    raft::update_host(data_h.data(), data_d.data().get(), data_len, handle->get_stream());
    raft::update_host(inference_data_h.data(),
                      inference_data_d.data().get(),
                      inference_data_len,
                      handle->get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle->get_stream()));
  }

 protected:
  RfInputs<T> params;
  RF_params rf_params;
  thrust::device_vector<T> data_d;
  thrust::device_vector<T> inference_data_d;
  std::vector<T> data_h;
  std::vector<T> inference_data_h;
  std::vector<ModelHandle> treelite_indiv_handles;
  std::string test_dir;
  std::string dir_name;

  std::shared_ptr<raft::handle_t> handle;
  std::vector<float> treelite_predicted_labels;
  std::vector<ML::RandomForestMetaData<T, L>> all_forest_info;

  thrust::device_vector<L> labels_d;
  std::vector<L> labels_h;

};  // namespace ML

template <typename T, typename L>
class RfConcatTestClf : public RfTreeliteTestCommon<T, L> {
 protected:
  void testClassifier()
  {
    this->test_dir = "./concat_test_clf/";

    std::vector<float> temp_label_h;

    rmm::device_uvector<float> weight(this->params.n_cols, this->handle->get_stream());
    rmm::device_uvector<float> temp_label_d(this->params.n_rows, this->handle->get_stream());
    rmm::device_uvector<float> temp_data_d(this->data_d.size(), this->handle->get_stream());

    raft::random::Rng r(1234ULL);

    // Generate weight for each feature.
    r.uniform(weight.data(), this->params.n_cols, T(0.0), T(1.0), this->handle->get_stream());
    // Generate noise.
    r.uniform(
      temp_label_d.data(), this->params.n_rows, T(0.0), T(10.0), this->handle->get_stream());

    raft::linalg::transpose<float>(*(this->handle),
                                   this->data_d.data().get(),
                                   temp_data_d.data(),
                                   this->params.n_rows,
                                   this->params.n_cols,
                                   this->handle->get_stream());

    raft::linalg::gemv<float>(*(this->handle),
                              temp_data_d.data(),
                              this->params.n_cols,
                              this->params.n_rows,
                              weight.data(),
                              temp_label_d.data(),
                              true,
                              1.f,
                              1.f,
                              this->handle->get_stream());

    temp_label_h.resize(this->params.n_rows);
    raft::update_host(
      temp_label_h.data(), temp_label_d.data(), this->params.n_rows, this->handle->get_stream());

    CUDA_CHECK(cudaStreamSynchronize(this->handle->get_stream()));

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

    raft::update_device(this->labels_d.data().get(),
                        this->labels_h.data(),
                        this->params.n_rows,
                        this->handle->get_stream());

    preprocess_labels(this->params.n_rows, this->labels_h, labels_map);

    for (int i = 0; i < 3; i++) {
      ModelHandle model;

      this->rf_params.n_trees = this->rf_params.n_trees + i;

      auto forest_ptr = &this->all_forest_info[i];
      fit(*(this->handle),
          forest_ptr,
          this->data_d.data().get(),
          this->params.n_rows,
          this->params.n_cols,
          this->labels_d.data().get(),
          labels_map.size(),
          this->rf_params);
      build_treelite_forest(&model, &this->all_forest_info[i], this->params.n_cols);
      this->treelite_indiv_handles.push_back(model);
    }

    CUDA_CHECK(cudaStreamSynchronize(this->handle->get_stream()));

    this->ConcatenateTreeliteModels();
    this->getResultAndCheck();
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
    this->test_dir = "./concat_test_reg/";

    rmm::device_uvector<float> weight(this->params.n_cols, this->handle->get_stream());
    rmm::device_uvector<float> temp_data_d(this->data_d.size(), this->handle->get_stream());

    raft::random::Rng r(1234ULL);

    // Generate weight for each feature.
    r.uniform(weight.data(), this->params.n_cols, T(0.0), T(1.0), this->handle->get_stream());
    // Generate noise.
    r.uniform(this->labels_d.data().get(),
              this->params.n_rows,
              T(0.0),
              T(10.0),
              this->handle->get_stream());

    raft::linalg::transpose<float>(*(this->handle),
                                   this->data_d.data().get(),
                                   temp_data_d.data(),
                                   this->params.n_rows,
                                   this->params.n_cols,
                                   this->handle->get_stream());

    raft::linalg::gemv<float>(*(this->handle),
                              temp_data_d.data(),
                              this->params.n_cols,
                              this->params.n_rows,
                              weight.data(),
                              this->labels_d.data().get(),
                              true,
                              1.f,
                              1.f,
                              this->handle->get_stream());

    this->labels_h.resize(this->params.n_rows);
    raft::update_host(this->labels_h.data(),
                      this->labels_d.data().get(),
                      this->params.n_rows,
                      this->handle->get_stream());
    CUDA_CHECK(cudaStreamSynchronize(this->handle->get_stream()));

    for (int i = 0; i < 3; i++) {
      ModelHandle model;

      this->rf_params.n_trees = this->rf_params.n_trees + i;

      auto forest_ptr = &this->all_forest_info[i];
      fit(*(this->handle),
          forest_ptr,
          this->data_d.data().get(),
          this->params.n_rows,
          this->params.n_cols,
          this->labels_d.data().get(),
          this->rf_params);
      build_treelite_forest(&model, &this->all_forest_info[i], this->params.n_cols);
      CUDA_CHECK(cudaStreamSynchronize(this->handle->get_stream()));
      this->treelite_indiv_handles.push_back(model);
    }

    this->ConcatenateTreeliteModels();
    this->getResultAndCheck();
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
