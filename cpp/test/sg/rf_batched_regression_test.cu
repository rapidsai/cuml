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

#include <raft/cudart_utils.h>

#include <gtest/gtest.h>
#include <raft/linalg/transpose.h>
#include <test_utils.h>
#include <cuml/datasets/make_blobs.hpp>
#include <cuml/datasets/make_regression.hpp>
#include <cuml/ensemble/randomforest.hpp>
#include <metrics/scores.cuh>

namespace ML {

using namespace MLCommon;

struct RfInputs {
  int n_rows;
  int n_cols;
  int n_trees;
  float max_features;
  float max_samples;
  int max_depth;
  int max_leaves;
  bool bootstrap;
  int n_bins;
  int min_samples_leaf;
  int min_samples_split;
  float min_impurity_decrease;
  int n_streams;
  CRITERION split_criterion;
  float min_expected_acc;
};

template <typename T>
class RFBatchedRegTest : public ::testing::TestWithParam<RfInputs> {
 protected:
  void basicTest()
  {
    params = ::testing::TestWithParam<RfInputs>::GetParam();

    RF_params rf_params;
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

    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.reset(new raft::handle_t(rf_params.n_streams));
    handle->set_stream(stream);
    auto allocator = handle->get_device_allocator();

    int data_len     = params.n_rows * params.n_cols;
    data             = (T*)allocator->allocate(data_len * sizeof(T), stream);
    data_row_major   = (T*)allocator->allocate(data_len * sizeof(T), stream);
    labels           = (T*)allocator->allocate(params.n_rows * sizeof(T), stream);
    predicted_labels = (T*)allocator->allocate(params.n_rows * sizeof(T), stream);

    Datasets::make_regression(*handle,
                              data_row_major,
                              labels,
                              params.n_rows,
                              params.n_cols,
                              params.n_cols,
                              nullptr,
                              1,
                              0.0f,
                              -1,
                              0.0,
                              0.0f,
                              false,
                              3536699ULL);

    cublasHandle_t cublas_h = handle->get_cublas_handle();
    raft::linalg::transpose(*handle, data_row_major, data, params.n_cols, params.n_rows, stream);

    // Training part
    forest = new typename ML::RandomForestMetaData<T, T>;
    null_trees_ptr(forest);
    fit(*handle, forest, data, params.n_rows, params.n_cols, labels, rf_params);

    // predict function expects row major lay out of data, so we need to
    // transpose the data first
    predict(*handle, forest, data_row_major, params.n_rows, params.n_cols, predicted_labels);
    accuracy = Score::r2_score(predicted_labels, labels, params.n_rows, stream);
  }

  void SetUp() override { basicTest(); }

  void TearDown() override
  {
    auto allocator = handle->get_device_allocator();
    allocator->deallocate(data, params.n_rows * params.n_cols * sizeof(T), stream);
    allocator->deallocate(data_row_major, params.n_rows * params.n_cols * sizeof(T), stream);
    allocator->deallocate(labels, params.n_rows * sizeof(T), stream);
    allocator->deallocate(predicted_labels, params.n_rows * sizeof(T), stream);
    delete forest;
    handle.reset();
  }

 protected:
  std::shared_ptr<raft::handle_t> handle;
  cudaStream_t stream;
  RfInputs params;
  RandomForestMetaData<T, T>* forest;
  float accuracy = -1.0f;  // overriden in each test SetUp and TearDown
  T *data, *data_row_major;
  T *labels, *predicted_labels;
};

//-------------------------------------------------------------------------------------------------------------------------------------
const std::vector<RfInputs> inputs = {
  RfInputs{5, 1, 1, 1.0f, 1.0f, 1, -1, false, 5, 1, 2, 0.0, 1, CRITERION::MSE, -5.0},
  // Small datasets to repro corner cases as in #3107 (test for crash)
  {101, 57, 2, 1.0f, 1.0f, 2, -1, false, 13, 2, 2, 0.0, 2, CRITERION::MSE, -10.0},

  // Larger datasets for accuracy
  {2000, 20, 20, 1.0f, 0.6f, 13, -1, true, 10, 2, 2, 0.0, 2, CRITERION::MSE, 0.68f}};

typedef RFBatchedRegTest<float> RFBatchedRegTestF;
TEST_P(RFBatchedRegTestF, Fit) { ASSERT_GT(accuracy, params.min_expected_acc); }

INSTANTIATE_TEST_CASE_P(RFBatchedRegTests, RFBatchedRegTestF, ::testing::ValuesIn(inputs));

typedef RFBatchedRegTest<double> RFBatchedRegTestD;
TEST_P(RFBatchedRegTestD, Fit) { ASSERT_GT(accuracy, params.min_expected_acc); }

INSTANTIATE_TEST_CASE_P(RFBatchedRegTests, RFBatchedRegTestD, ::testing::ValuesIn(inputs));

}  // end namespace ML
