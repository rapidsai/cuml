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

#pragma once

#include <common/nvtx.hpp>

#include <decisiontree/treelite_util.h>
#include <decisiontree/batched-levelalgo/quantile.cuh>
#include <decisiontree/decisiontree.cuh>

#include <cuml/common/device_buffer.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/ensemble/randomforest.hpp>

#include <metrics/scores.cuh>
#include <random/permute.cuh>

#include <raft/cudart_utils.h>
#include <raft/mr/device/allocator.hpp>
#include <raft/random/rng.cuh>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num()  0
#define omp_get_max_threads() 1
#endif

#include <map>

namespace ML {

template <class T, class L>
class RandomForest {
 protected:
  RF_params rf_params;  // structure containing RF hyperparameters
  int rf_type;          // 0 for classification 1 for regression

  void get_row_sample(int tree_id,
                      int n_rows,
                      MLCommon::device_buffer<unsigned int>* selected_rows,
                      const cudaStream_t stream,
                      const std::shared_ptr<raft::mr::device::allocator> device_allocator)
  {
    ML::PUSH_RANGE("bootstrapping row IDs @randomforest.cuh");
    // TODO(Rory): this is not a good way to set the seed
    // Incrementing seed changes only one tree in the whole ensemble
    int rs = rf_params.seed + tree_id;

    raft::random::Rng rng(rs * 1000 | 0xFF00AA, raft::random::GeneratorType::GenKiss99);
    if (rf_params.bootstrap) {
      // Use bootstrapped sample set
      rng.uniformInt<unsigned>(selected_rows->data(), selected_rows->size(), 0, n_rows, stream);

    } else {
      // Use all the samples from the dataset
      thrust::sequence(thrust::cuda::par.on(stream), selected_rows->begin(), selected_rows->end());
    }
    ML::POP_RANGE();
  }

  void error_checking(const T* input, L* predictions, int n_rows, int n_cols, bool predict) const
  {
    if (predict) {
      ASSERT(predictions != nullptr, "Error! User has not allocated memory for predictions.");
    }
    ASSERT((n_rows > 0), "Invalid n_rows %d", n_rows);
    ASSERT((n_cols > 0), "Invalid n_cols %d", n_cols);

    bool input_is_dev_ptr = DT::is_dev_ptr(input);
    bool preds_is_dev_ptr = DT::is_dev_ptr(predictions);

    if (!input_is_dev_ptr || (input_is_dev_ptr != preds_is_dev_ptr)) {
      ASSERT(false,
             "RF Error: Expected both input and labels/predictions to be GPU "
             "pointers");
    }
  }

 public:
  /**
   * @brief Construct RandomForest object.
   * @param[in] cfg_rf_params: Random forest hyper-parameter struct.
   * @param[in] cfg_rf_type: Task type: 0 for classification, 1 for regression
   */
  RandomForest(RF_params cfg_rf_params, int cfg_rf_type = RF_type::CLASSIFICATION)
    : rf_params(cfg_rf_params), rf_type(cfg_rf_type){};

  /**
   * @brief Build (i.e., fit, train) random forest for input data.
   * @param[in] user_handle: raft::handle_t
   * @param[in] input: train data (n_rows samples, n_cols features) in column major format,
   *   excluding labels. Device pointer.
   * @param[in] n_rows: number of training data samples.
   * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
   * @param[in] labels: 1D array of target predictions/labels. Device Pointer.
            For classification task, only labels of type int are supported.
              Assumption: labels were preprocessed to map to ascending numbers from 0;
              needed for current gini impl in decision tree
            For regression task, the labels (predictions) can be float or double data type.
  * @param[in] n_unique_labels: (meaningful only for classification) #unique label values (known
  during preprocessing)
  * @param[in] forest: CPU point to RandomForestMetaData struct.
  */
  void fit(const raft::handle_t& user_handle,
           const T* input,
           int n_rows,
           int n_cols,
           L* labels,
           int n_unique_labels,
           RandomForestMetaData<T, L>*& forest)
  {
    ML::PUSH_RANGE("RandomForest::fit @randomforest.cuh");
    this->error_checking(input, labels, n_rows, n_cols, false);
    const raft::handle_t& handle = user_handle;
    int n_sampled_rows           = 0;
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
    ASSERT(n_streams <= handle.get_num_internal_streams(),
           "rf_params.n_streams (=%d) should be <= raft::handle_t.n_streams (=%d)",
           n_streams,
           handle.get_num_internal_streams());

    // Select n_sampled_rows (with replacement) numbers from [0, n_rows) per tree.
    // selected_rows: randomly generated IDs for bootstrapped samples (w/ replacement); a device
    // ptr.
    // Use a deque instead of vector because it can be used on objects with a deleted copy
    // constructor
    std::deque<MLCommon::device_buffer<unsigned int>> selected_rows;
    for (int i = 0; i < n_streams; i++) {
      selected_rows.emplace_back(
        handle.get_device_allocator(), handle.get_internal_stream(i), n_sampled_rows);
    }

    // Preprocess once only per forest
    // Using batched backend
    // allocate space for d_global_quantiles
    auto global_quantiles =
      DT::computeQuantiles(this->rf_params.tree_params.n_bins, input, n_rows, n_cols, handle);
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

#pragma omp parallel for num_threads(n_streams)
    for (int i = 0; i < this->rf_params.n_trees; i++) {
      int stream_id        = omp_get_thread_num();
      unsigned int* rowids = selected_rows[stream_id].data();

      this->get_row_sample(i,
                           n_rows,
                           &selected_rows[stream_id],
                           handle.get_internal_stream(stream_id),
                           handle.get_device_allocator());

      /* Build individual tree in the forest.
        - input is a pointer to orig data that have n_cols features and n_rows rows.
        - n_sampled_rows: # rows sampled for tree's bootstrap sample.
        - sorted_selected_rows: points to a list of row #s (w/ n_sampled_rows elements)
          used to build the bootstrapped sample.
          Expectation: Each tree node will contain (a) # n_sampled_rows and
          (b) a pointer to a list of row numbers w.r.t original data.
      */

      forest->trees[i] = DT::DecisionTree::fit(handle,
                                               input,
                                               n_cols,
                                               n_rows,
                                               labels,
                                               rowids,
                                               n_sampled_rows,
                                               n_unique_labels,
                                               this->rf_params.tree_params,
                                               this->rf_params.seed,
                                               global_quantiles,
                                               i);
    }
    // Cleanup
    for (int i = 0; i < n_streams; i++) {
      auto s = handle.get_internal_stream(i);
      CUDA_CHECK(cudaStreamSynchronize(s));
    }
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
    ML::POP_RANGE();
  }

  /**
   * @brief Predict target feature for input data
   * @param[in] user_handle: raft::handle_t.
   * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
   * @param[in] n_rows: number of  data samples.
   * @param[in] n_cols: number of features (excluding target feature).
   * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
   * @param[in] verbosity: verbosity level for logging messages during execution
   */
  void predict(const raft::handle_t& user_handle,
               const T* input,
               int n_rows,
               int n_cols,
               L* predictions,
               const RandomForestMetaData<T, L>* forest,
               int verbosity) const
  {
    ML::Logger::get().setLevel(verbosity);
    this->error_checking(input, predictions, n_rows, n_cols, true);
    std::vector<L> h_predictions(n_rows);
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

      if (rf_type == RF_type::CLASSIFICATION) {  // classification task: use 'majority' prediction
        std::map<int, int> prediction_to_cnt;
        std::pair<std::map<int, int>::iterator, bool> ret;
        int max_cnt_so_far      = 0;
        int majority_prediction = -1;

        for (int i = 0; i < this->rf_params.n_trees; i++) {
          L prediction;
          DT::DecisionTree::predict(user_handle,
                                    &forest->trees[i],
                                    &h_input[row_id * row_size],
                                    1,
                                    n_cols,
                                    &prediction,
                                    verbosity);
          ret = prediction_to_cnt.insert(std::pair<int, int>(prediction, 1));
          if (!(ret.second)) { ret.first->second += 1; }
          if (max_cnt_so_far < ret.first->second) {
            max_cnt_so_far      = ret.first->second;
            majority_prediction = ret.first->first;
          }
        }

        h_predictions[row_id] = majority_prediction;
      } else {  // regression task: use 'average' prediction
        L sum_predictions = 0;
        for (int i = 0; i < this->rf_params.n_trees; i++) {
          L prediction;
          DT::DecisionTree::predict(user_handle,
                                    &forest->trees[i],
                                    &h_input[row_id * row_size],
                                    1,
                                    n_cols,
                                    &prediction,
                                    verbosity);
          sum_predictions += prediction;
        }
        // Random forest's prediction is the arithmetic mean of all its decision tree predictions.
        h_predictions[row_id] = sum_predictions / this->rf_params.n_trees;
      }
    }

    raft::update_device(predictions, h_predictions.data(), n_rows, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  /**
   * @brief Predict target feature for input data and score against ref_labels.
   * @param[in] user_handle: raft::handle_t.
   * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
   * @param[in] ref_labels: label values for cross validation (n_rows elements); GPU pointer.
   * @param[in] n_rows: number of  data samples.
   * @param[in] n_cols: number of features (excluding target feature).
   * @param[in] predictions: n_rows predicted labels. GPU pointer, user allocated.
   * @param[in] verbosity: verbosity level for logging messages during execution
   * @param[in] rf_type: task type: 0 for classification, 1 for regression
   */
  static RF_metrics score(const raft::handle_t& user_handle,
                          const L* ref_labels,
                          int n_rows,
                          const L* predictions,
                          int verbosity,
                          int rf_type = RF_type::CLASSIFICATION)
  {
    ML::Logger::get().setLevel(verbosity);
    cudaStream_t stream = user_handle.get_stream();
    auto d_alloc        = user_handle.get_device_allocator();
    RF_metrics stats;
    if (rf_type == RF_type::CLASSIFICATION) {  // task classifiation: get classification metrics
      float accuracy =
        MLCommon::Score::accuracy_score(predictions, ref_labels, n_rows, d_alloc, stream);
      stats = set_rf_metrics_classification(accuracy);
      if (ML::Logger::get().shouldLogFor(CUML_LEVEL_DEBUG)) print(stats);

      /* TODO: Potentially augment RF_metrics w/ more metrics (e.g., precision, F1, etc.).
        For non binary classification problems (i.e., one target and  > 2 labels), need avg.
        for each of these metrics */
    } else {  // regression task: get regression metrics
      double mean_abs_error, mean_squared_error, median_abs_error;
      MLCommon::Score::regression_metrics(predictions,
                                          ref_labels,
                                          n_rows,
                                          d_alloc,
                                          stream,
                                          mean_abs_error,
                                          mean_squared_error,
                                          median_abs_error);
      stats = set_rf_metrics_regression(mean_abs_error, mean_squared_error, median_abs_error);
      if (ML::Logger::get().shouldLogFor(CUML_LEVEL_DEBUG)) print(stats);
    }

    return stats;
  }
};

// class specializations
template class RandomForest<float, int>;
template class RandomForest<float, float>;
template class RandomForest<double, int>;
template class RandomForest<double, double>;

}  // End namespace ML
