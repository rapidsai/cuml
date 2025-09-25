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

#include <cuml/common/logger.hpp>
#include <cuml/ensemble/treelite_defs.hpp>
#include <cuml/tree/decisiontree.hpp>

#include <map>
#include <memory>

namespace raft {
class handle_t;  // forward decl
}

namespace ML {

enum RF_type {
  CLASSIFICATION,
  REGRESSION,
};

enum task_category { REGRESSION_MODEL = 1, CLASSIFICATION_MODEL = 2 };

struct RF_metrics {
  RF_type rf_type;

  // Classification metrics
  float accuracy;

  // Regression metrics
  double mean_abs_error;
  double mean_squared_error;
  double median_abs_error;
};

RF_metrics set_all_rf_metrics(RF_type rf_type,
                              float accuracy,
                              double mean_abs_error,
                              double mean_squared_error,
                              double median_abs_error);
RF_metrics set_rf_metrics_classification(float accuracy);
RF_metrics set_rf_metrics_regression(double mean_abs_error,
                                     double mean_squared_error,
                                     double median_abs_error);
void print(const RF_metrics rf_metrics);

struct RF_params {
  /**
   * Number of decision trees in the random forest.
   */
  int n_trees;
  /**
   * Control bootstrapping.
   * If bootstrapping is set to true, bootstrapped samples are used for building
   * each tree. Bootstrapped sampling is done by randomly drawing
   * round(max_samples * n_samples) number of samples with replacement. More on
   * bootstrapping:
   *     https://en.wikipedia.org/wiki/Bootstrap_aggregating
   * If bootstrapping is set to false, whole dataset is used to build each
   * tree.
   */
  bool bootstrap;
  /**
   * Ratio of dataset rows used while fitting each tree.
   */
  float max_samples;
  /**
   * Decision tree training hyper parameter struct.
   */
  /**
   * random seed
   */
  uint64_t seed;
  /**
   * Number of concurrent GPU streams for parallel tree building.
   * Each stream is independently managed by CPU thread.
   * N streams need N times RF workspace.
   */
  int n_streams;
  DT::DecisionTreeParams tree_params;
};

/* Update labels so they are unique from 0 to n_unique_vals.
   Create an old_label to new_label map per random forest.
*/
void preprocess_labels(int n_rows,
                       std::vector<int>& labels,
                       std::map<int, int>& labels_map,
                       rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

/* Revert preprocessing effect, if needed. */
void postprocess_labels(int n_rows,
                        std::vector<int>& labels,
                        std::map<int, int>& labels_map,
                        rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

template <class T, class L>
struct RandomForestMetaData {
  std::vector<std::shared_ptr<DT::TreeMetaDataNode<T, L>>> trees;
  RF_params rf_params;
};

template <class T, class L>
void delete_rf_metadata(RandomForestMetaData<T, L>* forest);

template <class T, class L>
std::string get_rf_summary_text(const RandomForestMetaData<T, L>* forest);

template <class T, class L>
std::string get_rf_detailed_text(const RandomForestMetaData<T, L>* forest);

template <class T, class L>
std::string get_rf_json(const RandomForestMetaData<T, L>* forest);

template <class T, class L>
void build_treelite_forest(TreeliteModelHandle* model,
                           const RandomForestMetaData<T, L>* forest,
                           int num_features);

// ----------------------------- Classification ----------------------------------- //

typedef RandomForestMetaData<float, int> RandomForestClassifierF;
typedef RandomForestMetaData<double, int> RandomForestClassifierD;

void fit(const raft::handle_t& user_handle,
         RandomForestClassifierF* forest,
         float* input,
         int n_rows,
         int n_cols,
         int* labels,
         int n_unique_labels,
         RF_params rf_params,
         rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);
void fit(const raft::handle_t& user_handle,
         RandomForestClassifierD* forest,
         double* input,
         int n_rows,
         int n_cols,
         int* labels,
         int n_unique_labels,
         RF_params rf_params,
         rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

template <typename T, typename L>
void fit_treelite(const raft::handle_t& user_handle,
                  TreeliteModelHandle* model,
                  T* input,
                  int n_rows,
                  int n_cols,
                  L* labels,
                  int n_unique_labels,
                  RF_params rf_params,
                  rapids_logger::level_enum verbosity);

void predict(const raft::handle_t& user_handle,
             const RandomForestClassifierF* forest,
             const float* input,
             int n_rows,
             int n_cols,
             int* predictions,
             rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);
void predict(const raft::handle_t& user_handle,
             const RandomForestClassifierD* forest,
             const double* input,
             int n_rows,
             int n_cols,
             int* predictions,
             rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

RF_metrics score(const raft::handle_t& user_handle,
                 const RandomForestClassifierF* forest,
                 const int* ref_labels,
                 int n_rows,
                 const int* predictions,
                 rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);
RF_metrics score(const raft::handle_t& user_handle,
                 const RandomForestClassifierD* forest,
                 const int* ref_labels,
                 int n_rows,
                 const int* predictions,
                 rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

RF_params set_rf_params(int max_depth,
                        int max_leaves,
                        float max_features,
                        int max_n_bins,
                        int min_samples_leaf,
                        int min_samples_split,
                        float min_impurity_decrease,
                        bool bootstrap,
                        int n_trees,
                        float max_samples,
                        uint64_t seed,
                        CRITERION split_criterion,
                        int cfg_n_streams,
                        int max_batch_size);

// ----------------------------- Regression ----------------------------------- //

typedef RandomForestMetaData<float, float> RandomForestRegressorF;
typedef RandomForestMetaData<double, double> RandomForestRegressorD;

void fit(const raft::handle_t& user_handle,
         RandomForestRegressorF* forest,
         float* input,
         int n_rows,
         int n_cols,
         float* labels,
         RF_params rf_params,
         rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);
void fit(const raft::handle_t& user_handle,
         RandomForestRegressorD* forest,
         double* input,
         int n_rows,
         int n_cols,
         double* labels,
         RF_params rf_params,
         rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

template <typename T, typename L>
void fit_treelite(const raft::handle_t& user_handle,
                  TreeliteModelHandle* model,
                  T* input,
                  int n_rows,
                  int n_cols,
                  L* labels,
                  RF_params rf_params,
                  rapids_logger::level_enum verbosity);

void predict(const raft::handle_t& user_handle,
             const RandomForestRegressorF* forest,
             const float* input,
             int n_rows,
             int n_cols,
             float* predictions,
             rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);
void predict(const raft::handle_t& user_handle,
             const RandomForestRegressorD* forest,
             const double* input,
             int n_rows,
             int n_cols,
             double* predictions,
             rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

RF_metrics score(const raft::handle_t& user_handle,
                 const RandomForestRegressorF* forest,
                 const float* ref_labels,
                 int n_rows,
                 const float* predictions,
                 rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);
RF_metrics score(const raft::handle_t& user_handle,
                 const RandomForestRegressorD* forest,
                 const double* ref_labels,
                 int n_rows,
                 const double* predictions,
                 rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);
};  // namespace ML
