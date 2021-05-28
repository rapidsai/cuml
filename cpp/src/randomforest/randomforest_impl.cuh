/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#ifndef _OPENMP
#define omp_get_thread_num() 0
#endif
#include <decisiontree/memory.h>
#include <decisiontree/treelite_util.h>
#include <raft/cudart_utils.h>
#include <cuml/common/logger.hpp>
#include <decisiontree/quantile/quantile.cuh>
#include <metrics/scores.cuh>
#include <raft/mr/device/allocator.hpp>
#include <raft/random/rng.cuh>
#include <random/permute.cuh>
#include "randomforest_impl.h"

#include <common/nvtx.hpp>

namespace ML {
/**
 * @brief Construct rf (random forest) object.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] cfg_rf_params: Random forest hyper-parameter struct.
 * @param[in] cfg_rf_type: Random forest type.
 */
template <typename T, typename L>
rf<T, L>::rf(RF_params cfg_rf_params, int cfg_rf_type)
  : rf_params(cfg_rf_params), rf_type(cfg_rf_type) {
  validity_check(rf_params);
}

/**
 * @brief Return number of trees in the forest.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 */
template <typename T, typename L>
int rf<T, L>::get_ntrees() {
  return rf_params.n_trees;
}

/**
 * @brief Sample row IDs for tree fitting and bootstrap if requested.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] tree_id: unique tree ID
 * @param[in] n_rows: total number of data samples.
 * @param[in] n_sampled_rows: number of rows used for training
 * @param[in, out] selected_rows: already allocated array w/ row IDs
 * @param[in] num_sms: No of SM in current GPU
 * @param[in] stream: Current cuda stream
 * @param[in] device_allocator: Current device allocator from cuml handle
 */
template <typename T, typename L>
void rf<T, L>::prepare_fit_per_tree(
  int tree_id, int n_rows, int n_sampled_rows, unsigned int* selected_rows,
  const int num_sms, const cudaStream_t stream,
  const std::shared_ptr<raft::mr::device::allocator> device_allocator) {
  ML::PUSH_RANGE("bootstrapping row IDs @randomforest_impl.cuh");
  int rs = tree_id;
  if (rf_params.seed != 0) rs = rf_params.seed + tree_id;

  raft::random::Rng rng(rs * 1000 | 0xFF00AA,
                        raft::random::GeneratorType::GenKiss99);
  if (rf_params.bootstrap) {
    // Use bootstrapped sample set
    rng.uniformInt<unsigned>(selected_rows, n_sampled_rows, 0, n_rows, stream);

  } else {
    // Use all the samples from the dataset
    thrust::sequence(thrust::cuda::par.on(stream), selected_rows,
                     selected_rows + n_sampled_rows);
  }
  ML::POP_RANGE();
}

template <typename T, typename L>
void rf<T, L>::error_checking(const T* input, L* predictions, int n_rows,
                              int n_cols, bool predict) const {
  if (predict) {
    ASSERT(predictions != nullptr,
           "Error! User has not allocated memory for predictions.");
  }
  ASSERT((n_rows > 0), "Invalid n_rows %d", n_rows);
  ASSERT((n_cols > 0), "Invalid n_cols %d", n_cols);

  bool input_is_dev_ptr = is_dev_ptr(input);
  bool preds_is_dev_ptr = is_dev_ptr(predictions);

  if (!input_is_dev_ptr || (input_is_dev_ptr != preds_is_dev_ptr)) {
    ASSERT(false,
           "RF Error: Expected both input and labels/predictions to be GPU "
           "pointers");
  }
}

/**
 * @brief Construct rfClassifier object.
 * @tparam T: data type for input data (float or double).
 * @param[in] cfg_rf_params: Random forest hyper-parameter struct.
 */
template <typename T>
rfClassifier<T>::rfClassifier(RF_params cfg_rf_params)
  : rf<T, int>::rf(cfg_rf_params, RF_type::CLASSIFICATION) {
  trees = new DecisionTree::DecisionTreeClassifier<T>[this->rf_params.n_trees];
};

/**
 * @brief Destructor for random forest classifier object.
 * @tparam T: data type for input data (float or double).
 */
template <typename T>
rfClassifier<T>::~rfClassifier() {
  delete[] trees;
}

/**
 * @brief Return a const pointer to decision tree classifiers.
 * @tparam T: data type for input data (float or double).
 */
template <typename T>
const DecisionTree::DecisionTreeClassifier<T>* rfClassifier<T>::get_trees_ptr()
  const {
  return trees;
}

/**
 * @brief Build (i.e., fit, train) random forest classifier for input data.
 * @tparam T: data type for input data (float or double).
 * @param[in] user_handle: raft::handle_t
 * @param[in] input: train data (n_rows samples, n_cols features) in column major format,
 *   excluding labels. Device pointer.
 * @param[in] n_rows: number of training data samples.
 * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
 * @param[in] labels: 1D array of target features (int only), with one label per training sample. Device pointer.
          Assumption: labels were preprocessed to map to ascending numbers from 0;
          needed for current gini impl in decision tree
 * @param[in] n_unique_labels: #unique label values (known during preprocessing)
 * @param[in] forest: CPU point to RandomForestMetaData struct.
 */
template <typename T>
void rfClassifier<T>::fit(const raft::handle_t& user_handle, const T* input,
                          int n_rows, int n_cols, int* labels,
                          int n_unique_labels,
                          RandomForestMetaData<T, int>*& forest) {
  ML::PUSH_RANGE("rfClassifer::fit @randomforest_impl.cuh");
  this->error_checking(input, labels, n_rows, n_cols, false);

  const raft::handle_t& handle = user_handle;
  int n_sampled_rows = 0;
  if (this->rf_params.bootstrap) {
    n_sampled_rows = std::round(this->rf_params.max_samples * n_rows);
  } else {
    if (this->rf_params.max_samples != 1.0) {
      CUML_LOG_WARN(
        "If bootstrap sampling is disabled, max_samples value is ignored and "
        "whole dataset is used for building each tree");
      this->rf_params.max_samples = 1.0;
    }
    n_sampled_rows = n_rows;
  }
  int n_streams = this->rf_params.n_streams;
  ASSERT(
    n_streams <= handle.get_num_internal_streams(),
    "rf_params.n_streams (=%d) should be <= raft::handle_t.n_streams (=%d)",
    n_streams, handle.get_num_internal_streams());

  // Select n_sampled_rows (with replacement) numbers from [0, n_rows) per tree.
  // selected_rows: randomly generated IDs for bootstrapped samples (w/ replacement); a device ptr.
  MLCommon::device_buffer<unsigned int>* selected_rows[n_streams];
  for (int i = 0; i < n_streams; i++) {
    auto s = handle.get_internal_stream(i);
    selected_rows[i] = new MLCommon::device_buffer<unsigned int>(
      handle.get_device_allocator(), s, n_sampled_rows);
  }

  std::shared_ptr<TemporaryMemory<T, int>> tempmem[n_streams];
  if (this->rf_params.tree_params.use_experimental_backend) {
    // TemporaryMemory is unused for batched (new) backend
    for (int i = 0; i < n_streams; i++) {
      tempmem[i] = nullptr;
    }
  } else {
    // Allocate TemporaryMemory for each stream
    for (int i = 0; i < n_streams; i++) {
      tempmem[i] = std::make_shared<TemporaryMemory<T, int>>(
        handle, handle.get_internal_stream(i), n_rows, n_cols, n_unique_labels,
        this->rf_params.tree_params);
    }
  }

  std::unique_ptr<MLCommon::device_buffer<T>> global_quantiles_buffer = nullptr;
  T* global_quantiles = nullptr;
  auto quantile_size = this->rf_params.tree_params.n_bins * n_cols;

  //Preprocess once only per forest
  if (this->rf_params.tree_params.use_experimental_backend) {
    // Using batched backend
    // allocate space for d_global_quantiles
    global_quantiles_buffer = std::make_unique<MLCommon::device_buffer<T>>(
      handle.get_device_allocator(), handle.get_stream(), quantile_size);
    global_quantiles = global_quantiles_buffer->data();
    DecisionTree::computeQuantiles(
      global_quantiles, this->rf_params.tree_params.n_bins, input, n_rows,
      n_cols, handle.get_device_allocator(), handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
  } else {
    if (this->rf_params.tree_params.split_algo == SPLIT_ALGO::GLOBAL_QUANTILE) {
      // Using level (old) backend
      global_quantiles = tempmem[0]->d_quantile->data();
      // compute global quantiles in first index of tempmem
      DecisionTree::computeQuantiles(
        global_quantiles, this->rf_params.tree_params.n_bins, input, n_rows,
        n_cols, handle.get_device_allocator(), handle.get_stream());
      CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
      // device to host for first index of tempmem
      CUDA_CHECK(cudaMemcpyAsync(
        (void*)tempmem[0]->h_quantile->data(), (void*)global_quantiles,
        this->rf_params.tree_params.n_bins * n_cols * sizeof(T),
        cudaMemcpyDeviceToHost, tempmem[0]->stream));
      // copy to rest of indices in tempmem from index:0
      for (int i = 1; i < n_streams; i++) {
        CUDA_CHECK(cudaMemcpyAsync(
          tempmem[i]->d_quantile->data(), global_quantiles,
          this->rf_params.tree_params.n_bins * n_cols * sizeof(T),
          cudaMemcpyDeviceToDevice, tempmem[i]->stream));
        memcpy((void*)(tempmem[i]->h_quantile->data()),
               (void*)(tempmem[0]->h_quantile->data()),
               this->rf_params.tree_params.n_bins * n_cols * sizeof(T));
      }
    }
  }

#pragma omp parallel for num_threads(n_streams)
  for (int i = 0; i < this->rf_params.n_trees; i++) {
    int stream_id = omp_get_thread_num();
    unsigned int* rowids = selected_rows[stream_id]->data();

    this->prepare_fit_per_tree(
      i, n_rows, n_sampled_rows, rowids, raft::getMultiProcessorCount(),
      handle.get_internal_stream(stream_id), handle.get_device_allocator());

    /* Build individual tree in the forest.
       - input is a pointer to orig data that have n_cols features and n_rows rows.
       - n_sampled_rows: # rows sampled for tree's bootstrap sample.
       - sorted_selected_rows: points to a list of row #s (w/ n_sampled_rows elements)
         used to build the bootstrapped sample.
         Expectation: Each tree node will contain (a) # n_sampled_rows and
         (b) a pointer to a list of row numbers w.r.t original data.
    */
    DecisionTree::TreeMetaDataNode<T, int>* tree_ptr = &(forest->trees[i]);
    tree_ptr->treeid = i;
    trees[i].fit(handle.get_device_allocator(), handle.get_host_allocator(),
                 handle.get_internal_stream(stream_id), input, n_cols, n_rows,
                 labels, rowids, n_sampled_rows, n_unique_labels, tree_ptr,
                 this->rf_params.tree_params, this->rf_params.seed,
                 global_quantiles, tempmem[stream_id]);
  }
  //Cleanup
  for (int i = 0; i < n_streams; i++) {
    auto s = handle.get_internal_stream(i);
    CUDA_CHECK(cudaStreamSynchronize(s));
    selected_rows[i]->release(s);
    delete selected_rows[i];
    if (!this->rf_params.tree_params.use_experimental_backend) {
      tempmem[i].reset();
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
  ML::POP_RANGE();
}

/**
 * @brief Predict target feature for input data; n-ary classification for single feature supported.
 * @tparam T: data type for input data (float or double).
 * @param[in] user_handle: raft::handle_t.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbosity: verbosity level for logging messages during execution
 */
template <typename T>
void rfClassifier<T>::predict(const raft::handle_t& user_handle, const T* input,
                              int n_rows, int n_cols, int* predictions,
                              const RandomForestMetaData<T, int>* forest,
                              int verbosity) const {
  ML::Logger::get().setLevel(verbosity);
  this->error_checking(input, predictions, n_rows, n_cols, true);
  std::vector<int> h_predictions(n_rows);
  cudaStream_t stream = user_handle.get_stream();

  std::vector<T> h_input(n_rows * n_cols);
  raft::update_host(h_input.data(), input, n_rows * n_cols, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int row_size = n_cols;

  ML::PatternSetter _("%v");
  for (int row_id = 0; row_id < n_rows; row_id++) {
    if (ML::Logger::get().shouldLogFor(CUML_LEVEL_DEBUG)) {
      std::stringstream ss;
      ss << "Predict for sample: ";
      for (int i = 0; i < n_cols; i++)
        ss << h_input[row_id * row_size + i] << ", ";
      CUML_LOG_DEBUG(ss.str().c_str());
    }

    std::map<int, int> prediction_to_cnt;
    std::pair<std::map<int, int>::iterator, bool> ret;
    int max_cnt_so_far = 0;
    int majority_prediction = -1;

    for (int i = 0; i < this->rf_params.n_trees; i++) {
      int prediction;
      trees[i].predict(user_handle, &forest->trees[i],
                       &h_input[row_id * row_size], 1, n_cols, &prediction,
                       verbosity);
      ret = prediction_to_cnt.insert(std::pair<int, int>(prediction, 1));
      if (!(ret.second)) {
        ret.first->second += 1;
      }
      if (max_cnt_so_far < ret.first->second) {
        max_cnt_so_far = ret.first->second;
        majority_prediction = ret.first->first;
      }
    }

    h_predictions[row_id] = majority_prediction;
  }

  raft::update_device(predictions, h_predictions.data(), n_rows, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

/**
 * @brief Predict target feature for input data; n-ary classification for single feature supported.
 * @tparam T: data type for input data (float or double).
 * @param[in] user_handle: raft::handle_t.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbosity: verbosity level for logging messages during execution
 */
template <typename T>
void rfClassifier<T>::predictGetAll(const raft::handle_t& user_handle,
                                    const T* input, int n_rows, int n_cols,
                                    int* predictions,
                                    const RandomForestMetaData<T, int>* forest,
                                    int verbosity) {
  ML::Logger::get().setLevel(verbosity);
  int num_trees = this->rf_params.n_trees;
  std::vector<int> h_predictions(n_rows * num_trees);

  std::vector<T> h_input(n_rows * n_cols);
  cudaStream_t stream = user_handle.get_stream();
  raft::update_host(h_input.data(), input, n_rows * n_cols, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int row_size = n_cols;
  int pred_id = 0;

  for (int row_id = 0; row_id < n_rows; row_id++) {
    if (ML::Logger::get().shouldLogFor(CUML_LEVEL_DEBUG)) {
      std::stringstream ss;
      ss << "Predict for sample: ";
      for (int i = 0; i < n_cols; i++)
        ss << h_input[row_id * row_size + i] << ", ";
      CUML_LOG_DEBUG(ss.str().c_str());
    }

    for (int i = 0; i < num_trees; i++) {
      int prediction;
      trees[i].predict(user_handle, &forest->trees[i],
                       &h_input[row_id * row_size], 1, n_cols, &prediction,
                       verbosity);
      h_predictions[pred_id] = prediction;
      pred_id++;
    }
  }

  raft::update_device(predictions, h_predictions.data(), n_rows * num_trees,
                      stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

/**
 * @brief Predict target feature for input data and validate against ref_labels.
 * @tparam T: data type for input data (float or double).
 * @param[in] user_handle: raft::handle_t.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] ref_labels: label values for cross validation (n_rows elements); GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbosity: verbosity level for logging messages during execution
 */
template <typename T>
RF_metrics rfClassifier<T>::score(const raft::handle_t& user_handle,
                                  const int* ref_labels, int n_rows,
                                  const int* predictions, int verbosity) {
  ML::Logger::get().setLevel(verbosity);
  cudaStream_t stream = user_handle.get_stream();
  auto d_alloc = user_handle.get_device_allocator();
  float accuracy = MLCommon::Score::accuracy_score(predictions, ref_labels,
                                                   n_rows, d_alloc, stream);
  RF_metrics stats = set_rf_metrics_classification(accuracy);
  if (ML::Logger::get().shouldLogFor(CUML_LEVEL_DEBUG)) print(stats);

  /* TODO: Potentially augment RF_metrics w/ more metrics (e.g., precision, F1, etc.).
     For non binary classification problems (i.e., one target and  > 2 labels), need avg.
     for each of these metrics */
  return stats;
}

/**
 * @brief Construct rfRegressor object.
 * @tparam T: data type for input data (float or double).
 * @param[in] cfg_rf_params: Random forest hyper-parameter struct.
 */
template <typename T>
rfRegressor<T>::rfRegressor(RF_params cfg_rf_params)
  : rf<T, T>::rf(cfg_rf_params, RF_type::REGRESSION) {
  trees = new DecisionTree::DecisionTreeRegressor<T>[this->rf_params.n_trees];
}

/**
 * @brief Destructor for random forest regressor object.
 * @tparam T: data type for input data (float or double).
 */
template <typename T>
rfRegressor<T>::~rfRegressor() {
  delete[] trees;
}

/**
 * @brief Return a const pointer to decision tree regressors.
 * @tparam T: data type for input data (float or double).
 */
template <typename T>
const DecisionTree::DecisionTreeRegressor<T>* rfRegressor<T>::get_trees_ptr()
  const {
  return trees;
}

/**
 * @brief Build (i.e., fit, train) random forest regressor for input data.
 * @tparam T: data type for input data (float or double).
 * @param[in] user_handle: raft::handle_t
 * @param[in] input: train data (n_rows samples, n_cols features) in column major format, excluding labels. Device pointer.
 * @param[in] n_rows: number of training data samples.
 * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
 * @param[in] labels: 1D array of target features (float or double), with one label per training sample. Device pointer.
 * @param[in, out] forest: CPU pointer to RandomForestMetaData struct
 */
template <typename T>
void rfRegressor<T>::fit(const raft::handle_t& user_handle, const T* input,
                         int n_rows, int n_cols, T* labels,
                         RandomForestMetaData<T, T>*& forest) {
  ML::PUSH_RANGE("rfRegressor::fit @randomforest_impl.cuh");
  this->error_checking(input, labels, n_rows, n_cols, false);

  const raft::handle_t& handle = user_handle;
  int n_sampled_rows = 0;
  if (this->rf_params.bootstrap) {
    n_sampled_rows = std::round(this->rf_params.max_samples * n_rows);
  } else {
    if (this->rf_params.max_samples != 1.0) {
      CUML_LOG_WARN(
        "If bootstrap sampling is disabled, max_samples value is ignored and "
        "whole dataset is used for building each tree");
      this->rf_params.max_samples = 1.0;
    }
    n_sampled_rows = n_rows;
  }
  int n_streams = this->rf_params.n_streams;
  ASSERT(
    n_streams <= handle.get_num_internal_streams(),
    "rf_params.n_streams (=%d) should be <= raft::handle_t.n_streams (=%d)",
    n_streams, handle.get_num_internal_streams());

  // Select n_sampled_rows (with replacement) numbers from [0, n_rows) per tree.
  // selected_rows: randomly generated IDs for bootstrapped samples (w/ replacement); a device ptr.
  MLCommon::device_buffer<unsigned int>* selected_rows[n_streams];
  for (int i = 0; i < n_streams; i++) {
    auto s = handle.get_internal_stream(i);
    selected_rows[i] = new MLCommon::device_buffer<unsigned int>(
      handle.get_device_allocator(), s, n_sampled_rows);
  }

  std::shared_ptr<TemporaryMemory<T, T>> tempmem[n_streams];
  if (this->rf_params.tree_params.use_experimental_backend) {
    // TemporaryMemory is unused for batched (new) backend
    for (int i = 0; i < n_streams; i++) {
      tempmem[i] = nullptr;
    }
  } else {
    // Allocate TemporaryMemory for each stream
    for (int i = 0; i < n_streams; i++) {
      tempmem[i] = std::make_shared<TemporaryMemory<T, T>>(
        handle, handle.get_internal_stream(i), n_rows, n_cols, 1,
        this->rf_params.tree_params);
    }
  }

  std::unique_ptr<MLCommon::device_buffer<T>> global_quantiles_buffer = nullptr;
  T* global_quantiles = nullptr;
  auto quantile_size = this->rf_params.tree_params.n_bins * n_cols;

  //Preprocess once only per forest
  if (this->rf_params.tree_params.use_experimental_backend) {
    // Using batched backend
    // allocate space for d_global_quantiles
    global_quantiles_buffer = std::make_unique<MLCommon::device_buffer<T>>(
      handle.get_device_allocator(), handle.get_stream(), quantile_size);
    global_quantiles = global_quantiles_buffer->data();
    DecisionTree::computeQuantiles(
      global_quantiles, this->rf_params.tree_params.n_bins, input, n_rows,
      n_cols, handle.get_device_allocator(), handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
  } else {
    if (this->rf_params.tree_params.split_algo == SPLIT_ALGO::GLOBAL_QUANTILE) {
      // Using level (old) backend
      global_quantiles = tempmem[0]->d_quantile->data();
      // compute global quantiles in first index of tempmem
      DecisionTree::computeQuantiles(
        global_quantiles, this->rf_params.tree_params.n_bins, input, n_rows,
        n_cols, handle.get_device_allocator(), handle.get_stream());
      CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
      // device to host for first index of tempmem
      CUDA_CHECK(cudaMemcpyAsync(
        (void*)tempmem[0]->h_quantile->data(), (void*)global_quantiles,
        this->rf_params.tree_params.n_bins * n_cols * sizeof(T),
        cudaMemcpyDeviceToHost, tempmem[0]->stream));
      // copy to rest of indices in tempmem from index:0
      for (int i = 1; i < n_streams; i++) {
        CUDA_CHECK(cudaMemcpyAsync(
          tempmem[i]->d_quantile->data(), global_quantiles,
          this->rf_params.tree_params.n_bins * n_cols * sizeof(T),
          cudaMemcpyDeviceToDevice, tempmem[i]->stream));
        memcpy((void*)(tempmem[i]->h_quantile->data()),
               (void*)(tempmem[0]->h_quantile->data()),
               this->rf_params.tree_params.n_bins * n_cols * sizeof(T));
      }
    }
  }
#pragma omp parallel for num_threads(n_streams)
  for (int i = 0; i < this->rf_params.n_trees; i++) {
    int stream_id = omp_get_thread_num();
    unsigned int* rowids = selected_rows[stream_id]->data();

    this->prepare_fit_per_tree(
      i, n_rows, n_sampled_rows, rowids, raft::getMultiProcessorCount(),
      handle.get_internal_stream(stream_id), handle.get_device_allocator());

    /* Build individual tree in the forest.
       - input is a pointer to orig data that have n_cols features and n_rows rows.
       - n_sampled_rows: # rows sampled for tree's bootstrap sample.
       - sorted_selected_rows: points to a list of row #s (w/ n_sampled_rows elements)
         used to build the bootstrapped sample. Expectation: Each tree node will contain
         (a) # n_sampled_rows and (b) a pointer to a list of row numbers w.r.t original data.
    */
    DecisionTree::TreeMetaDataNode<T, T>* tree_ptr = &(forest->trees[i]);
    tree_ptr->treeid = i;
    trees[i].fit(handle.get_device_allocator(), handle.get_host_allocator(),
                 handle.get_internal_stream(stream_id), input, n_cols, n_rows,
                 labels, rowids, n_sampled_rows, tree_ptr,
                 this->rf_params.tree_params, this->rf_params.seed,
                 global_quantiles, tempmem[stream_id]);
  }
  //Cleanup
  for (int i = 0; i < n_streams; i++) {
    auto s = handle.get_internal_stream(i);
    CUDA_CHECK(cudaStreamSynchronize(s));
    selected_rows[i]->release(s);
    delete selected_rows[i];
    if (!this->rf_params.tree_params.use_experimental_backend) {
      tempmem[i].reset();
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
  ML::POP_RANGE();
}

/**
 * @brief Predict target feature for input data; regression for single feature supported.
 * @tparam T: data type for input data (float or double).
 * @param[in] user_handle: raft::handle_t.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] forest: CPU pointer to RandomForestMetaData struct
 * @param[in] verbosity: verbosity level for logging messages during execution
 */
template <typename T>
void rfRegressor<T>::predict(const raft::handle_t& user_handle, const T* input,
                             int n_rows, int n_cols, T* predictions,
                             const RandomForestMetaData<T, T>* forest,
                             int verbosity) const {
  this->error_checking(input, predictions, n_rows, n_cols, true);

  std::vector<T> h_predictions(n_rows);
  cudaStream_t stream = user_handle.get_stream();

  std::vector<T> h_input(n_rows * n_cols);
  raft::update_host(h_input.data(), input, n_rows * n_cols, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int row_size = n_cols;

  for (int row_id = 0; row_id < n_rows; row_id++) {
    if (ML::Logger::get().shouldLogFor(CUML_LEVEL_DEBUG)) {
      std::stringstream ss;
      ss << "Predict for sample: ";
      for (int i = 0; i < n_cols; i++)
        ss << h_input[row_id * row_size + i] << ", ";
      CUML_LOG_DEBUG(ss.str().c_str());
    }

    T sum_predictions = 0;

    for (int i = 0; i < this->rf_params.n_trees; i++) {
      T prediction;
      trees[i].predict(user_handle, &forest->trees[i],
                       &h_input[row_id * row_size], 1, n_cols, &prediction,
                       verbosity);
      sum_predictions += prediction;
    }
    // Random forest's prediction is the arithmetic mean of all its decision tree predictions.
    h_predictions[row_id] = sum_predictions / this->rf_params.n_trees;
  }

  raft::update_device(predictions, h_predictions.data(), n_rows, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

/**
 * @brief Predict target feature for input data and validate against ref_labels.
 * @tparam T: data type for input data (float or double).
 * @param[in] user_handle: raft::handle_t.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] ref_labels: label values for cross validation (n_rows elements); GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] forest: CPU pointer to RandomForestMetaData struct
 * @param[in] verbosity: verbosity level for logging messages during execution
 */
template <typename T>
RF_metrics rfRegressor<T>::score(const raft::handle_t& user_handle,
                                 const T* ref_labels, int n_rows,
                                 const T* predictions, int verbosity) {
  ML::Logger::get().setLevel(verbosity);
  cudaStream_t stream = user_handle.get_stream();
  auto d_alloc = user_handle.get_device_allocator();

  double mean_abs_error, mean_squared_error, median_abs_error;
  MLCommon::Score::regression_metrics(predictions, ref_labels, n_rows, d_alloc,
                                      stream, mean_abs_error,
                                      mean_squared_error, median_abs_error);
  RF_metrics stats = set_rf_metrics_regression(
    mean_abs_error, mean_squared_error, median_abs_error);
  if (ML::Logger::get().shouldLogFor(CUML_LEVEL_DEBUG)) print(stats);

  return stats;
}

template class rf<float, int>;
template class rf<float, float>;
template class rf<double, int>;
template class rf<double, double>;

template class rfClassifier<float>;
template class rfClassifier<double>;

template class rfRegressor<float>;
template class rfRegressor<double>;

}  //End namespace ML
