/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <common/cudart_utils.h>
#include <gtest/gtest.h>
#include <linalg/transpose.h>
#include <test_utils.h>
#include <cuda_utils.cuh>
#include <cuml/datasets/make_blobs.hpp>
#include <cuml/ensemble/randomforest.hpp>

namespace ML {

using namespace MLCommon;

struct RfInputs {
  int n_rows;
  int n_cols;
  int n_trees;
  float max_features;
  float rows_sample;
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
class RFBatchedClsTest : public ::testing::TestWithParam<RfInputs> {
 protected:
  void basicTest() {
    params = ::testing::TestWithParam<RfInputs>::GetParam();

    DecisionTree::DecisionTreeParams tree_params;
    set_tree_params(tree_params, params.max_depth, params.max_leaves,
                    params.max_features, params.n_bins, params.split_algo,
                    params.min_rows_per_node, params.min_impurity_decrease,
                    params.bootstrap_features, params.split_criterion, false,
                    true);
    RF_params rf_params;
    set_all_rf_params(rf_params, params.n_trees, params.bootstrap,
                      params.rows_sample, -1, params.n_streams, tree_params);

    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.reset(new raft::handle_t(rf_params.n_streams));
    handle->set_stream(stream);
    auto allocator = handle->get_device_allocator();

    int data_len = params.n_rows * params.n_cols;
    data = (T*)allocator->allocate(data_len * sizeof(T), stream);
    labels = (int*)allocator->allocate(params.n_rows * sizeof(int), stream);
    predicted_labels =
      (int*)allocator->allocate(params.n_rows * sizeof(int), stream);

    Datasets::make_blobs(*handle, data, labels, params.n_rows, params.n_cols, 5,
                         false, nullptr, nullptr, T(0.1), false, T(-0.5),
                         T(0.5), 3536699ULL);

    labels_h.resize(params.n_rows);
    raft::update_host(labels_h.data(), labels, params.n_rows, stream);
    preprocess_labels(params.n_rows, labels_h, labels_map);
    raft::update_device(labels, labels_h.data(), params.n_rows, stream);

    // Training part
    forest = new typename ML::RandomForestMetaData<T, int>;
    null_trees_ptr(forest);
    fit(*handle, forest, data, params.n_rows, params.n_cols, labels,
        labels_map.size(), rf_params);

    // predict function expects row major lay out of data, so we need to
    // transpose the data first
    T* data_row_major;
    data_row_major = (T*)allocator->allocate(data_len * sizeof(T), stream);
    cublasHandle_t cublas_h = handle->get_cublas_handle();
    raft::linalg::transpose(*handle, data, data_row_major, params.n_rows,
                            params.n_cols, stream);

    predict(*handle, forest, data_row_major, params.n_rows, params.n_cols,
            predicted_labels);
    raft::update_host(labels_h.data(), predicted_labels, params.n_rows, stream);

    RF_metrics tmp =
      score(*handle, forest, labels, params.n_rows, predicted_labels);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    accuracy = tmp.accuracy;
    allocator->deallocate(data_row_major, data_len * sizeof(T), stream);
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {
    auto allocator = handle->get_device_allocator();
    accuracy = -1.0f;
    postprocess_labels(params.n_rows, labels_h, labels_map);
    labels_h.clear();
    labels_map.clear();

    allocator->deallocate(labels, params.n_rows * sizeof(int), stream);
    allocator->deallocate(predicted_labels, params.n_rows * sizeof(int),
                          stream);
    allocator->deallocate(data, params.n_rows * params.n_cols * sizeof(T),
                          stream);
    delete forest;
    handle.reset();
  }

 protected:
  std::shared_ptr<raft::handle_t> handle;
  cudaStream_t stream;
  RfInputs params;
  T* data;
  int* labels;
  std::vector<int> labels_h;
  std::map<int, int>
    labels_map;  //unique map of labels to int vals starting from 0

  RandomForestMetaData<T, int>* forest;
  float accuracy = -1.0f;  // overriden in each test SetUp and TearDown

  int* predicted_labels;
};

//-------------------------------------------------------------------------------------------------------------------------------------
const std::vector<RfInputs> inputsf2_clf = {
  {20000, 10, 25, 1.0f, 0.4f, 16, -1, true, false, 10,
   SPLIT_ALGO::GLOBAL_QUANTILE, 2, 0.0, 2, CRITERION::GINI},
  {20000, 10, 5, 1.0f, 0.4f, 14, -1, true, false, 10,
   SPLIT_ALGO::GLOBAL_QUANTILE, 2, 0.0, 2, CRITERION::ENTROPY}};

typedef RFBatchedClsTest<float> RFBatchedClsTestF;
TEST_P(RFBatchedClsTestF, Fit) {
  if (!params.bootstrap && (params.max_features == 1.0f)) {
    ASSERT_TRUE(accuracy == 1.0f);
  } else {
    ASSERT_TRUE(accuracy >= 0.75f);  // Empirically derived accuracy range
  }
}

INSTANTIATE_TEST_CASE_P(RFBatchedClsTests, RFBatchedClsTestF,
                        ::testing::ValuesIn(inputsf2_clf));

typedef RFBatchedClsTest<double> RFBatchedClsTestD;
TEST_P(RFBatchedClsTestD, Fit) {
  if (!params.bootstrap && (params.max_features == 1.0f)) {
    ASSERT_TRUE(accuracy == 1.0f);
  } else {
    ASSERT_TRUE(accuracy >= 0.75f);  // Empirically derived accuracy range
  }
}

INSTANTIATE_TEST_CASE_P(RFBatchedClsTests, RFBatchedClsTestD,
                        ::testing::ValuesIn(inputsf2_clf));

}  // end namespace ML
