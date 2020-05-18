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

#include <cuml/tree/flatnode.h>
#include <cuml/tree/decisiontree.hpp>
#include "decisiontree_impl.cuh"

namespace ML {
namespace DecisionTree {

void set_tree_params(DecisionTreeParams &params, int cfg_max_depth,
                     int cfg_max_leaves, float cfg_max_features, int cfg_n_bins,
                     int cfg_split_algo, int cfg_min_rows_per_node,
                     float cfg_min_impurity_decrease,
                     bool cfg_bootstrap_features, CRITERION cfg_split_criterion,
                     bool cfg_quantile_per_tree, bool cfg_shuffle_features) {
  params.max_depth = cfg_max_depth;
  params.max_leaves = cfg_max_leaves;
  params.max_features = cfg_max_features;
  params.n_bins = cfg_n_bins;
  params.split_algo = cfg_split_algo;
  params.min_rows_per_node = cfg_min_rows_per_node;
  params.bootstrap_features = cfg_bootstrap_features;
  params.split_criterion = cfg_split_criterion;
  params.quantile_per_tree = cfg_quantile_per_tree;
  params.shuffle_features = cfg_shuffle_features;
  params.min_impurity_decrease = cfg_min_impurity_decrease;
}

void validity_check(const DecisionTreeParams params) {
  ASSERT((params.max_depth > 0), "Invalid max depth %d", params.max_depth);
  ASSERT((params.max_leaves == -1) || (params.max_leaves > 0),
         "Invalid max leaves %d", params.max_leaves);
  ASSERT((params.max_features > 0) && (params.max_features <= 1.0),
         "max_features value %f outside permitted (0, 1] range",
         params.max_features);
  ASSERT((params.n_bins > 0), "Invalid n_bins %d", params.n_bins);
  ASSERT((params.split_algo >= 0) &&
           (params.split_algo < SPLIT_ALGO::SPLIT_ALGO_END),
         "split_algo value %d outside permitted [0, %d) range",
         params.split_algo, SPLIT_ALGO::SPLIT_ALGO_END);
  ASSERT((params.min_rows_per_node >= 2),
         "Invalid min # rows per node value %d. Should be >= 2.",
         params.min_rows_per_node);
}

void print(const DecisionTreeParams params) {
  CUML_LOG_DEBUG("max_depth: %d", params.max_depth);
  CUML_LOG_DEBUG("max_leaves: %d", params.max_leaves);
  CUML_LOG_DEBUG("max_features: %f", params.max_features);
  CUML_LOG_DEBUG("n_bins: %d", params.n_bins);
  CUML_LOG_DEBUG("split_algo: %d", params.split_algo);
  CUML_LOG_DEBUG("min_rows_per_node: %d", params.min_rows_per_node);
  CUML_LOG_DEBUG("bootstrap_features: %d", params.bootstrap_features);
  CUML_LOG_DEBUG("split_criterion: %d", params.split_criterion);
  CUML_LOG_DEBUG("quantile_per_tree: %d", params.quantile_per_tree);
  CUML_LOG_DEBUG("shuffle_features: %d", params.shuffle_features);
}

template <class T, class L>
void print_tree_summary(const TreeMetaDataNode<T, L> *tree) {
  CUML_LOG_DEBUG(" Decision Tree depth --> %d and n_leaves --> %d",
                 tree->depth_counter, tree->leaf_counter);
  CUML_LOG_DEBUG(" Tree Fitting - Overall time --> %lf s",
                 tree->prepare_time + tree->train_time);
  CUML_LOG_DEBUG("   - preparing for fit time: %lf s", tree->prepare_time);
  CUML_LOG_DEBUG("   - tree growing time: %lf s", tree->train_time);
}

template <class T, class L>
void print_tree(const TreeMetaDataNode<T, L> *tree) {
  print_tree_summary<T, L>(tree);
  print_node<T, L>("", tree->sparsetree, 0, false);
}

void decisionTreeClassifierFit(const ML::cumlHandle &handle,
                               TreeClassifierF *&tree, float *data,
                               const int ncols, const int nrows, int *labels,
                               unsigned int *rowids, const int n_sampled_rows,
                               int unique_labels,
                               DecisionTree::DecisionTreeParams tree_params) {
  std::shared_ptr<DecisionTreeClassifier<float>> dt_classifier =
    std::make_shared<DecisionTreeClassifier<float>>();
  dt_classifier->fit(handle, data, ncols, nrows, labels, rowids, n_sampled_rows,
                     unique_labels, tree, tree_params);
}

void decisionTreeClassifierFit(const ML::cumlHandle &handle,
                               TreeClassifierD *&tree, double *data,
                               const int ncols, const int nrows, int *labels,
                               unsigned int *rowids, const int n_sampled_rows,
                               int unique_labels,
                               DecisionTree::DecisionTreeParams tree_params) {
  std::shared_ptr<DecisionTreeClassifier<double>> dt_classifier =
    std::make_shared<DecisionTreeClassifier<double>>();
  dt_classifier->fit(handle, data, ncols, nrows, labels, rowids, n_sampled_rows,
                     unique_labels, tree, tree_params);
}

void decisionTreeClassifierPredict(const ML::cumlHandle &handle,
                                   const TreeClassifierF *tree,
                                   const float *rows, const int n_rows,
                                   const int n_cols, int *predictions,
                                   int verbosity) {
  std::shared_ptr<DecisionTreeClassifier<float>> dt_classifier =
    std::make_shared<DecisionTreeClassifier<float>>();
  dt_classifier->predict(handle, tree, rows, n_rows, n_cols, predictions,
                         verbosity);
}

void decisionTreeClassifierPredict(const ML::cumlHandle &handle,
                                   const TreeClassifierD *tree,
                                   const double *rows, const int n_rows,
                                   const int n_cols, int *predictions,
                                   int verbosity) {
  std::shared_ptr<DecisionTreeClassifier<double>> dt_classifier =
    std::make_shared<DecisionTreeClassifier<double>>();
  dt_classifier->predict(handle, tree, rows, n_rows, n_cols, predictions,
                         verbosity);
}

// ----------------------------- Regression ----------------------------------- //

void decisionTreeRegressorFit(const ML::cumlHandle &handle,
                              TreeRegressorF *&tree, float *data,
                              const int ncols, const int nrows, float *labels,
                              unsigned int *rowids, const int n_sampled_rows,
                              DecisionTree::DecisionTreeParams tree_params) {
  std::shared_ptr<DecisionTreeRegressor<float>> dt_regressor =
    std::make_shared<DecisionTreeRegressor<float>>();
  dt_regressor->fit(handle, data, ncols, nrows, labels, rowids, n_sampled_rows,
                    tree, tree_params);
}

void decisionTreeRegressorFit(const ML::cumlHandle &handle,
                              TreeRegressorD *&tree, double *data,
                              const int ncols, const int nrows, double *labels,
                              unsigned int *rowids, const int n_sampled_rows,
                              DecisionTree::DecisionTreeParams tree_params) {
  std::shared_ptr<DecisionTreeRegressor<double>> dt_regressor =
    std::make_shared<DecisionTreeRegressor<double>>();
  dt_regressor->fit(handle, data, ncols, nrows, labels, rowids, n_sampled_rows,
                    tree, tree_params);
}

void decisionTreeRegressorPredict(const ML::cumlHandle &handle,
                                  const TreeRegressorF *tree, const float *rows,
                                  const int n_rows, const int n_cols,
                                  float *predictions, int verbosity) {
  std::shared_ptr<DecisionTreeRegressor<float>> dt_regressor =
    std::make_shared<DecisionTreeRegressor<float>>();
  dt_regressor->predict(handle, tree, rows, n_rows, n_cols, predictions,
                        verbosity);
}

void decisionTreeRegressorPredict(const ML::cumlHandle &handle,
                                  const TreeRegressorD *tree,
                                  const double *rows, const int n_rows,
                                  const int n_cols, double *predictions,
                                  int verbosity) {
  std::shared_ptr<DecisionTreeRegressor<double>> dt_regressor =
    std::make_shared<DecisionTreeRegressor<double>>();
  dt_regressor->predict(handle, tree, rows, n_rows, n_cols, predictions,
                        verbosity);
}

// Functions' specializations
template void print_tree_summary<float, int>(const TreeClassifierF *tree);
template void print_tree_summary<double, int>(const TreeClassifierD *tree);
template void print_tree_summary<float, float>(const TreeRegressorF *tree);
template void print_tree_summary<double, double>(const TreeRegressorD *tree);

template void print_tree<float, int>(const TreeClassifierF *tree);
template void print_tree<double, int>(const TreeClassifierD *tree);
template void print_tree<float, float>(const TreeRegressorF *tree);
template void print_tree<double, double>(const TreeRegressorD *tree);

}  // End namespace DecisionTree
}  //End namespace ML
