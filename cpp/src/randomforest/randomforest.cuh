/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <cuml/ensemble/randomforest.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/random/permute.cuh>
#include <raft/random/rng.cuh>
#include <raft/stats/accuracy.cuh>
#include <raft/stats/regression_metrics.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <decisiontree/batched-levelalgo/quantiles.cuh>
#include <decisiontree/decisiontree.cuh>
#include <decisiontree/treelite_util.h>

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
 public:
  RF_type rf_type;  // CLASSIFICATION or REGRESSION

 protected:
  RF_params rf_params;  // structure containing RF hyperparameters

  void get_row_sample(int tree_id,
                      int n_rows,
                      rmm::device_uvector<int>* selected_rows,
                      std::vector<int>& oob_indices,
                      const cudaStream_t stream)
  {
    raft::common::nvtx::range fun_scope("bootstrapping row IDs @randomforest.cuh");

    // Hash these together so they are uncorrelated
    auto rs = DT::fnv1a32_basis;
    rs      = DT::fnv1a32(rs, rf_params.seed);
    rs      = DT::fnv1a32(rs, tree_id);
    raft::random::Rng rng(rs, raft::random::GenPhilox);
    if (rf_params.bootstrap) {
      // Use bootstrapped sample set
      rng.uniformInt<int>(selected_rows->data(), selected_rows->size(), 0, n_rows, stream);

      if (rf_params.oob_score) {
        std::vector<int> h_selected_rows(selected_rows->size());
        raft::update_host(
          h_selected_rows.data(), selected_rows->data(), selected_rows->size(), stream);
        RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

        std::vector<bool> selected(n_rows, false);
        for (int idx : h_selected_rows) {
          selected[idx] = true;
        }

        oob_indices.clear();
        for (int i = 0; i < n_rows; i++) {
          if (!selected[i]) { oob_indices.push_back(i); }
        }
      }

    } else {
      // Use all the samples from the dataset
      thrust::sequence(thrust::cuda::par.on(stream), selected_rows->begin(), selected_rows->end());
      oob_indices.clear();
    }
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
   * @param[in] cfg_rf_type: Task type: CLASSIFICATION or REGRESSION
   */
  RandomForest(RF_params cfg_rf_params, RF_type cfg_rf_type = RF_type::CLASSIFICATION)
    : rf_params(cfg_rf_params), rf_type(cfg_rf_type) {};

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
           RandomForestMetaData<T, L>* forest)
  {
    raft::common::nvtx::range fun_scope("RandomForest::fit @randomforest.cuh");
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
    ASSERT(static_cast<std::size_t>(n_streams) <= handle.get_stream_pool_size(),
           "rf_params.n_streams (=%d) should be <= raft::handle_t.n_streams (=%lu)",
           n_streams,
           handle.get_stream_pool_size());

    // computing the quantiles: last two return values are shared pointers to device memory
    // encapsulated by quantiles struct
    auto [quantiles, quantiles_array, n_bins_array] =
      DT::computeQuantiles(handle, input, this->rf_params.tree_params.max_n_bins, n_rows, n_cols);

    // n_streams should not be less than n_trees
    if (this->rf_params.n_trees < n_streams) n_streams = this->rf_params.n_trees;

    // Select n_sampled_rows (with replacement) numbers from [0, n_rows) per tree.
    // selected_rows: randomly generated IDs for bootstrapped samples (w/ replacement); a device
    // ptr.
    // Use a deque instead of vector because it can be used on objects with a deleted copy
    // constructor
    std::deque<rmm::device_uvector<int>> selected_rows;
    for (int i = 0; i < n_streams; i++) {
      selected_rows.emplace_back(n_sampled_rows, handle.get_stream_from_stream_pool(i));
    }

    forest->n_features                   = n_cols;
    forest->n_rows                       = n_rows;
    forest->rf_type                      = rf_type;
    forest->n_unique_labels              = n_unique_labels;
    forest->feature_importances_computed = false;
    forest->oob_score                    = -1.0;
    forest->feature_importances.clear();

    // Initialize OOB tracking if needed
    if (rf_params.oob_score && rf_params.bootstrap) {
      forest->oob_indices_per_tree.resize(rf_params.n_trees);
    }

#pragma omp parallel for num_threads(n_streams)
    for (int i = 0; i < this->rf_params.n_trees; i++) {
      int stream_id = omp_get_thread_num();
      auto s        = handle.get_stream_from_stream_pool(stream_id);

      std::vector<int> oob_indices;
      this->get_row_sample(i, n_rows, &selected_rows[stream_id], oob_indices, s);

      if (rf_params.oob_score && rf_params.bootstrap) {
        forest->oob_indices_per_tree[i] = std::move(oob_indices);
      }

      /* Build individual tree in the forest.
        - input is a pointer to orig data that have n_cols features and n_rows rows.
        - n_sampled_rows: # rows sampled for tree's bootstrap sample.
        - sorted_selected_rows: points to a list of row #s (w/ n_sampled_rows elements)
          used to build the bootstrapped sample.
          Expectation: Each tree node will contain (a) # n_sampled_rows and
          (b) a pointer to a list of row numbers w.r.t original data.
      */

      forest->trees[i] = DT::DecisionTree::fit(handle,
                                               s,
                                               input,
                                               n_cols,
                                               n_rows,
                                               labels,
                                               &selected_rows[stream_id],
                                               n_unique_labels,
                                               this->rf_params.tree_params,
                                               this->rf_params.seed,
                                               quantiles,
                                               i);
    }
    // Cleanup
    handle.sync_stream_pool();
    handle.sync_stream();

    // Compute OOB score if enabled (during training to avoid storing data)
    if (rf_params.oob_score && rf_params.bootstrap) {
      std::vector<std::vector<T>> oob_predictions(n_rows);
      std::vector<int> oob_counts(n_rows, 0);

      if (rf_type == RF_type::CLASSIFICATION) {
        for (int i = 0; i < n_rows; i++) {
          oob_predictions[i].resize(n_unique_labels, 0.0);
        }
      } else {
        for (int i = 0; i < n_rows; i++) {
          oob_predictions[i].resize(1, 0.0);
        }
      }

      // Copy training data to host for predictions
      std::vector<T> h_input(std::size_t(n_rows) * n_cols);
      raft::update_host(h_input.data(), input, std::size_t(n_rows) * n_cols, handle.get_stream());
      handle.sync_stream(handle.get_stream());

      for (int tree_idx = 0; tree_idx < rf_params.n_trees; tree_idx++) {
        const auto& oob_indices = forest->oob_indices_per_tree[tree_idx];
        const auto& tree        = forest->trees[tree_idx];

        for (int oob_idx : oob_indices) {
          std::vector<T> row_prediction(tree->num_outputs);
          DT::DecisionTree::predict(handle,
                                    *tree,
                                    &h_input[oob_idx * n_cols],
                                    1,
                                    n_cols,
                                    row_prediction.data(),
                                    tree->num_outputs,
                                    rapids_logger::level_enum::info);

          if (rf_type == RF_type::CLASSIFICATION) {
            for (int k = 0; k < tree->num_outputs; k++) {
              oob_predictions[oob_idx][k] += row_prediction[k];
            }
          } else {
            oob_predictions[oob_idx][0] += row_prediction[0];
          }
          oob_counts[oob_idx]++;
        }
      }

      std::vector<L> final_predictions(n_rows);
      int valid_predictions = 0;

      for (int i = 0; i < n_rows; i++) {
        if (oob_counts[i] > 0) {
          valid_predictions++;

          if (rf_type == RF_type::CLASSIFICATION) {
            int best_class = 0;
            T best_score   = 0.0;
            for (int k = 0; k < n_unique_labels; k++) {
              T score = oob_predictions[i][k] / oob_counts[i];
              if (score > best_score) {
                best_score = score;
                best_class = k;
              }
            }
            final_predictions[i] = best_class;
          } else {
            final_predictions[i] = oob_predictions[i][0] / oob_counts[i];
          }
        }
      }

      // Copy training labels to host
      std::vector<L> h_labels(n_rows);
      raft::update_host(h_labels.data(), labels, n_rows, handle.get_stream());
      handle.sync_stream(handle.get_stream());

      if (rf_type == RF_type::CLASSIFICATION) {
        int correct = 0;
        for (int i = 0; i < n_rows; i++) {
          if (oob_counts[i] > 0 && final_predictions[i] == h_labels[i]) { correct++; }
        }
        forest->oob_score = static_cast<double>(correct) / valid_predictions;
      } else {
        double sum_squared_errors = 0.0;
        double sum_squared_total  = 0.0;
        double mean_y             = 0.0;
        int count                 = 0;

        for (int i = 0; i < n_rows; i++) {
          if (oob_counts[i] > 0) {
            mean_y += h_labels[i];
            count++;
          }
        }
        mean_y /= count;

        for (int i = 0; i < n_rows; i++) {
          if (oob_counts[i] > 0) {
            double error = h_labels[i] - final_predictions[i];
            sum_squared_errors += error * error;
            double diff = h_labels[i] - mean_y;
            sum_squared_total += diff * diff;
          }
        }

        forest->oob_score = 1.0 - (sum_squared_errors / sum_squared_total);
      }
    }
    // Note: Feature importances are computed lazily when requested
  }

  /**
   * @brief Predict target feature for input data
   * @param[in] user_handle: raft::handle_t.
   * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU
   * pointer.
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
               rapids_logger::level_enum verbosity) const
  {
    ML::default_logger().set_level(verbosity);
    this->error_checking(input, predictions, n_rows, n_cols, true);
    std::vector<L> h_predictions(n_rows);
    cudaStream_t stream = user_handle.get_stream();

    std::vector<T> h_input(std::size_t(n_rows) * n_cols);
    raft::update_host(h_input.data(), input, std::size_t(n_rows) * n_cols, stream);
    user_handle.sync_stream(stream);

    int row_size = n_cols;

    default_logger().set_pattern("%v");
    for (int row_id = 0; row_id < n_rows; row_id++) {
      std::vector<T> row_prediction(forest->trees[0]->num_outputs);
      for (int i = 0; i < this->rf_params.n_trees; i++) {
        DT::DecisionTree::predict(user_handle,
                                  *forest->trees[i],
                                  &h_input[row_id * row_size],
                                  1,
                                  n_cols,
                                  row_prediction.data(),
                                  forest->trees[i]->num_outputs,
                                  verbosity);
      }
      for (int k = 0; k < forest->trees[0]->num_outputs; k++) {
        row_prediction[k] /= this->rf_params.n_trees;
      }
      if (rf_type == RF_type::CLASSIFICATION) {  // classification task: use 'majority' prediction
        L best_class = 0;
        T best_prob  = 0.0;
        for (int k = 0; k < forest->trees[0]->num_outputs; k++) {
          if (row_prediction[k] > best_prob) {
            best_class = k;
            best_prob  = row_prediction[k];
          }
        }

        h_predictions[row_id] = best_class;
      } else {
        h_predictions[row_id] = row_prediction[0];
      }
    }

    raft::update_device(predictions, h_predictions.data(), n_rows, stream);
    user_handle.sync_stream(stream);
    default_logger().set_pattern(default_pattern());
  }

  /**
   * @brief Predict target feature for input data and score against ref_labels.
   * @param[in] user_handle: raft::handle_t.
   * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU
   * pointer.
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
                          rapids_logger::level_enum verbosity,
                          int rf_type = RF_type::CLASSIFICATION)
  {
    ML::default_logger().set_level(verbosity);
    cudaStream_t stream = user_handle.get_stream();
    RF_metrics stats;
    if (rf_type == RF_type::CLASSIFICATION) {  // task classifiation: get classification metrics
      float accuracy = raft::stats::accuracy(predictions, ref_labels, n_rows, stream);
      stats          = set_rf_metrics_classification(accuracy);
      if (ML::default_logger().should_log(rapids_logger::level_enum::debug)) print(stats);

      /* TODO: Potentially augment RF_metrics w/ more metrics (e.g., precision, F1, etc.).
        For non binary classification problems (i.e., one target and  > 2 labels), need avg.
        for each of these metrics */
    } else {  // regression task: get regression metrics
      double mean_abs_error, mean_squared_error, median_abs_error;
      raft::stats::regression_metrics(predictions,
                                      ref_labels,
                                      n_rows,
                                      stream,
                                      mean_abs_error,
                                      mean_squared_error,
                                      median_abs_error);
      stats = set_rf_metrics_regression(mean_abs_error, mean_squared_error, median_abs_error);
      if (ML::default_logger().should_log(rapids_logger::level_enum::debug)) print(stats);
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
