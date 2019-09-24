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

#include "decisiontree.hpp"
#include "decisiontree_impl.cuh"
#include "flatnode.h"

namespace ML {
namespace DecisionTree {

/**
 * @brief Set all DecisionTreeParams members.
 * @param[in,out] params: update with tree parameters
 * @param[in] cfg_max_depth: maximum tree depth; default -1
 * @param[in] cfg_max_leaves: maximum leaves; default -1
 * @param[in] cfg_max_features: maximum number of features; default 1.0f
 * @param[in] cfg_n_bins: number of bins; default 8
 * @param[in] cfg_split_algo: split algorithm; default SPLIT_ALGO::HIST
 * @param[in] cfg_min_rows_per_node: min. rows per node; default 2
 * @param[in] cfg_bootstrap_features: bootstrapping for features; default false
 * @param[in] cfg_split_criterion: split criterion; default CRITERION_END,
 *            i.e., GINI for classification or MSE for regression
 * @param[in] cfg_quantile_per_tree: compute quantile per tree; default false
 */
void set_tree_params(DecisionTreeParams &params, int cfg_max_depth,
                     int cfg_max_leaves, float cfg_max_features, int cfg_n_bins,
                     int cfg_split_algo, int cfg_min_rows_per_node,
                     bool cfg_bootstrap_features, CRITERION cfg_split_criterion,
                     bool cfg_quantile_per_tree) {
  params.max_depth = cfg_max_depth;
  params.max_leaves = cfg_max_leaves;
  params.max_features = cfg_max_features;
  params.n_bins = cfg_n_bins;
  params.split_algo = cfg_split_algo;
  params.min_rows_per_node = cfg_min_rows_per_node;
  params.bootstrap_features = cfg_bootstrap_features;
  params.split_criterion = cfg_split_criterion;
  params.quantile_per_tree = cfg_quantile_per_tree;
}

/**
 * @brief Check validity of all decision tree hyper-parameters.
 * @param[in] params: decision tree hyper-parameters.
 */
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
  if (params.split_algo == SPLIT_ALGO::GLOBAL_QUANTILE) {
    ASSERT((params.max_depth <= 32),
           "For GLOBAL_QUANTILE algorithm, only max depth of 32 is currently "
           "supported");
  }
}

/**
 * @brief Print all decision tree hyper-parameters.
 * @param[in] params: decision tree hyper-parameters.
 */
void print(const DecisionTreeParams params) {
  std::cout << "max_depth: " << params.max_depth << std::endl;
  std::cout << "max_leaves: " << params.max_leaves << std::endl;
  std::cout << "max_features: " << params.max_features << std::endl;
  std::cout << "n_bins: " << params.n_bins << std::endl;
  std::cout << "split_algo: " << params.split_algo << std::endl;
  std::cout << "min_rows_per_node: " << params.min_rows_per_node << std::endl;
  std::cout << "bootstrap_features: " << params.bootstrap_features << std::endl;
  std::cout << "split_criterion: " << params.split_criterion << std::endl;
  std::cout << "quantile_per_tree: " << params.quantile_per_tree << std::endl;
}

/**
 * @brief Print high-level tree information.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] tree: CPU pointer to TreeMetaDataNode
 */
template <class T, class L>
void print_tree_summary(const TreeMetaDataNode<T, L> *tree) {
  std::cout << " Decision Tree depth --> " << tree->depth_counter
            << " and n_leaves --> " << tree->leaf_counter << std::endl;
  std::cout << " Tree Fitting - Overall time --> "
            << tree->prepare_time + tree->train_time << " seconds" << std::endl;
  std::cout << "   - preparing for fit time: " << tree->prepare_time
            << " seconds" << std::endl;
  std::cout << "   - tree growing time: " << tree->train_time << " seconds"
            << std::endl;
}

/**
 * @brief Print detailed tree information.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] tree: CPU pointer to TreeMetaDataNode
 */
template <class T, class L>
void print_tree(const TreeMetaDataNode<T, L> *tree) {
  print_tree_summary<T, L>(tree);
  print_node<T, L>("", tree->sparsetree, 0, false);
}

/**
 * @defgroup Decision Tree Classifier - Fit function
 * @brief Build (i.e., fit, train) Decision Tree classifier for input data.
 * @param[in] handle: cumlHandle
 * @param[in, out] tree: CPU pointer to TreeMetaDataNode. User allocated.
 * @param[in] data: train data (nrows samples, ncols features) in column major format,
 *    excluding labels. Device pointer.
 * @param[in] ncols: number of features (i.e., columns) excluding target feature.
 * @param[in] nrows: number of training data samples of the whole unsampled dataset.
 * @param[in] labels: 1D array of target features (int only). One label per training
 *    sample. Device pointer.
 *    Assumption: labels need to be preprocessed to map to ascending numbers from 0;
 *    needed for current gini impl. in decision tree.
 * @param[in,out] rowids: array of n_sampled_rows integers in [0, nrows) range.
 *    Device pointer. The same array is then rearranged when splits are made,
 *    allowing us to construct trees without rearranging the actual dataset.
 * @param[in] n_sampled_rows: number of training samples, after sampling.
 *    If using decision tree directly over the whole dataset: n_sampled_rows = nrows
 * @param[in] n_unique_labels: #unique label values. Number of categories of classification.
 * @param[in] tree_params: Decision Tree training hyper parameter struct.
 * @{
 */
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
/** @} */

/**
 * @defgroup Decision Tree Classifier - Predict function
 * @brief Predict target feature for input data; n-ary classification for
 *   single feature supported. Inference of trees is CPU only for now.
 * @param[in] handle: cumlHandle (currently unused; API placeholder)
 * @param[in] tree: CPU pointer to TreeMetaDataNode.
 * @param[in] rows: test data (n_rows samples, n_cols features) in row major format.
 *    Current impl. expects a CPU pointer. TODO future API change.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in,out] predictions: n_rows predicted labels. Current impl. expects a
 *    CPU pointer, user allocated. TODO future API change.
 * @param[in] verbose: flag for debugging purposes.
 * @{
 */
void decisionTreeClassifierPredict(const ML::cumlHandle &handle,
                                   const TreeClassifierF *tree,
                                   const float *rows, const int n_rows,
                                   const int n_cols, int *predictions,
                                   bool verbose) {
  std::shared_ptr<DecisionTreeClassifier<float>> dt_classifier =
    std::make_shared<DecisionTreeClassifier<float>>();
  dt_classifier->predict(handle, tree, rows, n_rows, n_cols, predictions,
                         verbose);
}

void decisionTreeClassifierPredict(const ML::cumlHandle &handle,
                                   const TreeClassifierD *tree,
                                   const double *rows, const int n_rows,
                                   const int n_cols, int *predictions,
                                   bool verbose) {
  std::shared_ptr<DecisionTreeClassifier<double>> dt_classifier =
    std::make_shared<DecisionTreeClassifier<double>>();
  dt_classifier->predict(handle, tree, rows, n_rows, n_cols, predictions,
                         verbose);
}
/** @} */

// ----------------------------- Regression ----------------------------------- //

/**
 * @defgroup Decision Tree Regressor - Fit function
 * @brief Build (i.e., fit, train) Decision Tree regressor for input data.
 * @param[in] handle: cumlHandle
 * @param[in, out] tree: CPU pointer to TreeMetaDataNode. User allocated.
 * @param[in] data: train data (nrows samples, ncols features) in column major format,
 *   excluding labels. Device pointer.
 * @param[in] ncols: number of features (i.e., columns) excluding target feature.
 * @param[in] nrows: number of training data samples of the whole unsampled dataset.
 * @param[in] labels: 1D array of target features (float or double). One label per
 *    training sample. Device pointer.
 * @param[in,out] rowids: array of n_sampled_rows integers in [0, nrows) range.
 *   Device pointer. The same array is then rearranged when splits are made,
 *   allowing us to construct trees without rearranging the actual dataset.
 * @param[in] n_sampled_rows: number of training samples, after sampling. If using decision
 *   tree directly over the whole dataset: n_sampled_rows = nrows
 * @param[in] tree_params: Decision Tree training hyper parameter struct.
 * @{
 */
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
/** @} */

/**
 * @defgroup Decision Tree Regressor - Predict function
 * @brief Predict target feature for input data; regression for single feature supported.
 *   Inference of trees is CPU only for now.
 * @param[in] handle: cumlHandle (currently unused; API placeholder)
 * @param[in] tree: CPU pointer to TreeMetaDataNode.
 * @param[in] rows: test data (n_rows samples, n_cols features) in row major format.
 *   Current impl. expects a CPU pointer. TODO future API change.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in,out] predictions: n_rows predicted labels. Current impl. expects a CPU
 *   pointer, user allocated. TODO future API change.
 * @param[in] verbose: flag for debugging purposes.
 * @{
 */
void decisionTreeRegressorPredict(const ML::cumlHandle &handle,
                                  const TreeRegressorF *tree, const float *rows,
                                  const int n_rows, const int n_cols,
                                  float *predictions, bool verbose) {
  std::shared_ptr<DecisionTreeRegressor<float>> dt_regressor =
    std::make_shared<DecisionTreeRegressor<float>>();
  dt_regressor->predict(handle, tree, rows, n_rows, n_cols, predictions,
                        verbose);
}

void decisionTreeRegressorPredict(const ML::cumlHandle &handle,
                                  const TreeRegressorD *tree,
                                  const double *rows, const int n_rows,
                                  const int n_cols, double *predictions,
                                  bool verbose) {
  std::shared_ptr<DecisionTreeRegressor<double>> dt_regressor =
    std::make_shared<DecisionTreeRegressor<double>>();
  dt_regressor->predict(handle, tree, rows, n_rows, n_cols, predictions,
                        verbose);
}
/** @} */

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
