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

#pragma once
#include <cuml/ensemble/treelite_defs.hpp>
#include <cuml/tree/decisiontree.hpp>
#include <map>

namespace ML {

enum RF_type {
  CLASSIFICATION,
  REGRESSION,
};

enum task_category {
  REGRESSION_CATEGORY = 1,
  CLASSIFICATION_CATEGORY = 2,
};

struct RF_metrics {
  RF_type rf_type;

  // Classification metrics
  float accuracy;

  // Regression metrics
  double mean_abs_error;
  double mean_squared_error;
  double median_abs_error;
};

RF_metrics set_all_rf_metrics(RF_type rf_type, float accuracy,
                              double mean_abs_error, double mean_squared_error,
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
   * Control bootstrapping. If set, each tree in the forest is built on a
   * bootstrapped sample with replacement.
   * If false, sampling without replacement is done.
   */
  bool bootstrap;
  /**
   * Ratio of dataset rows used while fitting each tree.
   */
  float rows_sample;
  /**
   * Decision tree training hyper parameter struct.
   */
  /**
   * random seed
   */
  int seed;
  /**
   * Number of concurrent GPU streams for parallel tree building.
   * Each stream is independently managed by CPU thread.
   * N streams need N times RF workspace.
   */
  int n_streams;
  DecisionTree::DecisionTreeParams tree_params;
};

void set_rf_params(RF_params& params, int cfg_n_trees = 1,
                   bool cfg_bootstrap = true, float cfg_rows_sample = 1.0f,
                   int cfg_seed = -1, int cfg_n_streams = 8);
void set_all_rf_params(RF_params& params, int cfg_n_trees, bool cfg_bootstrap,
                       float cfg_rows_sample, int cfg_seed, int cfg_n_streams,
                       DecisionTree::DecisionTreeParams cfg_tree_params);
void validity_check(const RF_params rf_params);
void print(const RF_params rf_params);

/* Update labels so they are unique from 0 to n_unique_vals.
   Create an old_label to new_label map per random forest.
*/
void preprocess_labels(int n_rows, std::vector<int>& labels,
                       std::map<int, int>& labels_map, bool verbose = false);

/* Revert preprocessing effect, if needed. */
void postprocess_labels(int n_rows, std::vector<int>& labels,
                        std::map<int, int>& labels_map, bool verbose = false);

template <class T, class L>
struct RandomForestMetaData {
  DecisionTree::TreeMetaDataNode<T, L>* trees;
  RF_params rf_params;
  //TODO can add prepare, train time, if needed
};

template <class T, class L>
void null_trees_ptr(RandomForestMetaData<T, L>*& forest);

template <class T, class L>
void print_rf_summary(const RandomForestMetaData<T, L>* forest);

template <class T, class L>
void print_rf_detailed(const RandomForestMetaData<T, L>* forest);

template <class T, class L>
void build_treelite_forest(ModelHandle* model,
                           const RandomForestMetaData<T, L>* forest,
                           int num_features, int task_category,
                           std::vector<unsigned char>& data);

std::vector<unsigned char> save_model(ModelHandle model);

// ----------------------------- Classification ----------------------------------- //

typedef RandomForestMetaData<float, int> RandomForestClassifierF;
typedef RandomForestMetaData<double, int> RandomForestClassifierD;

void fit(const cumlHandle& user_handle, RandomForestClassifierF*& forest,
         float* input, int n_rows, int n_cols, int* labels, int n_unique_labels,
         RF_params rf_params);
void fit(const cumlHandle& user_handle, RandomForestClassifierD*& forest,
         double* input, int n_rows, int n_cols, int* labels,
         int n_unique_labels, RF_params rf_params);

void predict(const cumlHandle& user_handle,
             const RandomForestClassifierF* forest, const float* input,
             int n_rows, int n_cols, int* predictions, bool verbose = false);
void predict(const cumlHandle& user_handle,
             const RandomForestClassifierD* forest, const double* input,
             int n_rows, int n_cols, int* predictions, bool verbose = false);

void predictGetAll(const cumlHandle& user_handle,
                   const RandomForestClassifierF* forest, const float* input,
                   int n_rows, int n_cols, int* predictions,
                   bool verbose = false);
void predictGetAll(const cumlHandle& user_handle,
                   const RandomForestClassifierD* forest, const double* input,
                   int n_rows, int n_cols, int* predictions,
                   bool verbose = false);

RF_metrics score(const cumlHandle& user_handle,
                 const RandomForestClassifierF* forest, const int* ref_labels,
                 int n_rows, const int* predictions, bool verbose = false);
RF_metrics score(const cumlHandle& user_handle,
                 const RandomForestClassifierD* forest, const int* ref_labels,
                 int n_rows, const int* predictions, bool verbose = false);

RF_params set_rf_class_obj(int max_depth, int max_leaves, float max_features,
                           int n_bins, int split_algo, int min_rows_per_node,
                           float min_impurity_decrease, bool bootstrap_features,
                           bool bootstrap, int n_trees, float rows_sample,
                           int seed, CRITERION split_criterion,
                           bool quantile_per_tree, int cfg_n_streams);

// ----------------------------- Regression ----------------------------------- //

typedef RandomForestMetaData<float, float> RandomForestRegressorF;
typedef RandomForestMetaData<double, double> RandomForestRegressorD;

void fit(const cumlHandle& user_handle, RandomForestRegressorF*& forest,
         float* input, int n_rows, int n_cols, float* labels,
         RF_params rf_params);
void fit(const cumlHandle& user_handle, RandomForestRegressorD*& forest,
         double* input, int n_rows, int n_cols, double* labels,
         RF_params rf_params);

void predict(const cumlHandle& user_handle,
             const RandomForestRegressorF* forest, const float* input,
             int n_rows, int n_cols, float* predictions, bool verbose = false);
void predict(const cumlHandle& user_handle,
             const RandomForestRegressorD* forest, const double* input,
             int n_rows, int n_cols, double* predictions, bool verbose = false);

RF_metrics score(const cumlHandle& user_handle,
                 const RandomForestRegressorF* forest, const float* ref_labels,
                 int n_rows, const float* predictions, bool verbose = false);
RF_metrics score(const cumlHandle& user_handle,
                 const RandomForestRegressorD* forest, const double* ref_labels,
                 int n_rows, const double* predictions, bool verbose = false);
};  // namespace ML
