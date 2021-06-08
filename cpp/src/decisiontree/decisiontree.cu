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

#include <cuml/common/device_buffer.hpp>
#include <cuml/tree/decisiontree.hpp>
#include <raft/handle.hpp>

#include <cuml/tree/flatnode.h>
#include "decisiontree_impl.cuh"

namespace ML {
namespace DecisionTree {

/**
 * @brief Set all DecisionTreeParams members.
 * @param[in,out] params: update with tree parameters
 * @param[in] cfg_max_depth: maximum tree depth; default -1
 * @param[in] cfg_max_leaves: maximum leaves; default -1
 * @param[in] cfg_max_features: maximum number of features; default 1.0f
 * @param[in] cfg_n_bins: number of bins; default 8
 * @param[in] cfg_min_samples_leaf: min. rows in each leaf node; default 1
 * @param[in] cfg_min_samples_split: min. rows needed to split an internal node;
 *            default 2
 * @param[in] cfg_split_criterion: split criterion; default CRITERION_END,
 *            i.e., GINI for classification or MSE for regression
 * @param[in] cfg_max_batch_size: batch size for experimental backend
 */
void set_tree_params(DecisionTreeParams &params, int cfg_max_depth,
                     int cfg_max_leaves, float cfg_max_features, int cfg_n_bins,
                     int cfg_min_samples_leaf, int cfg_min_samples_split,
                     float cfg_min_impurity_decrease, CRITERION cfg_split_criterion,
                     int cfg_max_batch_size) {
  params.max_depth = cfg_max_depth;
  params.max_leaves = cfg_max_leaves;
  params.max_features = cfg_max_features;
  params.n_bins = cfg_n_bins;
  params.min_samples_leaf = cfg_min_samples_leaf;
  params.min_samples_split = cfg_min_samples_split;
  params.split_criterion = cfg_split_criterion;
  params.min_impurity_decrease = cfg_min_impurity_decrease;
  params.max_batch_size = cfg_max_batch_size;
}

void validity_check(const DecisionTreeParams params) {
  ASSERT((params.max_depth >= 0), "Invalid max depth %d", params.max_depth);
  ASSERT((params.max_leaves == -1) || (params.max_leaves > 0),
         "Invalid max leaves %d", params.max_leaves);
  ASSERT((params.max_features > 0) && (params.max_features <= 1.0),
         "max_features value %f outside permitted (0, 1] range",
         params.max_features);
  ASSERT((params.n_bins > 0), "Invalid n_bins %d", params.n_bins);
  ASSERT((params.split_criterion != 3), "MAE not supported.");
  ASSERT((params.min_samples_leaf >= 1),
         "Invalid value for min_samples_leaf %d. Should be >= 1.",
         params.min_samples_leaf);
  ASSERT((params.min_samples_split >= 2),
         "Invalid value for min_samples_split: %d. Should be >= 2.",
         params.min_samples_split);
}

void print(const DecisionTreeParams params) {
  CUML_LOG_DEBUG("max_depth: %d", params.max_depth);
  CUML_LOG_DEBUG("max_leaves: %d", params.max_leaves);
  CUML_LOG_DEBUG("max_features: %f", params.max_features);
  CUML_LOG_DEBUG("n_bins: %d", params.n_bins);
  CUML_LOG_DEBUG("min_samples_leaf: %d", params.min_samples_leaf);
  CUML_LOG_DEBUG("min_samples_split: %d", params.min_samples_split);
  CUML_LOG_DEBUG("split_criterion: %d", params.split_criterion);
  CUML_LOG_DEBUG("min_impurity_decrease: %f", params.min_impurity_decrease);
  CUML_LOG_DEBUG("max_batch_size: %d", params.max_batch_size);
}

template <class T, class L>
std::string get_tree_summary_text(const TreeMetaDataNode<T, L> *tree) {
  std::ostringstream oss;
  oss << " Decision Tree depth --> " << tree->depth_counter
      << " and n_leaves --> " << tree->leaf_counter << "\n"
      << " Tree Fitting - Overall time --> "
      << (tree->prepare_time + tree->train_time) << " s"
      << "\n"
      << "   - preparing for fit time: " << tree->prepare_time << " s"
      << "\n"
      << "   - tree growing time: " << tree->train_time << " s";
  return oss.str();
}

template <class T, class L>
std::string get_tree_text(const TreeMetaDataNode<T, L> *tree) {
  std::string summary = get_tree_summary_text<T, L>(tree);
  return summary + "\n" + get_node_text<T, L>("", tree->sparsetree, 0, false);
}

template <class T, class L>
std::string get_tree_json(const TreeMetaDataNode<T, L> *tree) {
  std::ostringstream oss;
  return get_node_json("", tree->sparsetree, 0);
}

void decisionTreeClassifierFit(const raft::handle_t &handle,
                               TreeClassifierF *&tree, float *data,
                               const int ncols, const int nrows, int *labels,
                               unsigned int *rowids, const int n_sampled_rows,
                               int unique_labels,
                               DecisionTree::DecisionTreeParams tree_params,
                               uint64_t seed) {
  std::shared_ptr<DecisionTreeClassifier<float>> dt_classifier =
    std::make_shared<DecisionTreeClassifier<float>>();
  auto quantile_size = tree_params.n_bins * ncols;
  MLCommon::device_buffer<float> global_quantiles_buffer(
    handle.get_device_allocator(), handle.get_stream(), quantile_size);
  DecisionTree::computeQuantiles(
    global_quantiles_buffer.data(), tree_params.n_bins, data, nrows, ncols,
    handle.get_device_allocator(), handle.get_stream());
  dt_classifier->fit(handle, data, ncols, nrows, labels, rowids, n_sampled_rows,
                     unique_labels, tree, tree_params, seed,
                     global_quantiles_buffer.data());
}

void decisionTreeClassifierFit(const raft::handle_t &handle,
                               TreeClassifierD *&tree, double *data,
                               const int ncols, const int nrows, int *labels,
                               unsigned int *rowids, const int n_sampled_rows,
                               int unique_labels,
                               DecisionTree::DecisionTreeParams tree_params,
                               uint64_t seed) {
  std::shared_ptr<DecisionTreeClassifier<double>> dt_classifier =
    std::make_shared<DecisionTreeClassifier<double>>();

  auto quantile_size = tree_params.n_bins * ncols;
  MLCommon::device_buffer<double> global_quantiles_buffer(
    handle.get_device_allocator(), handle.get_stream(), quantile_size);
  DecisionTree::computeQuantiles(
    global_quantiles_buffer.data(), tree_params.n_bins, data, nrows, ncols,
    handle.get_device_allocator(), handle.get_stream());
  dt_classifier->fit(handle, data, ncols, nrows, labels, rowids, n_sampled_rows,
                     unique_labels, tree, tree_params, seed,
                     global_quantiles_buffer.data());
}

void decisionTreeClassifierPredict(const raft::handle_t &handle,
                                   const TreeClassifierF *tree,
                                   const float *rows, const int n_rows,
                                   const int n_cols, int *predictions,
                                   int verbosity) {
  std::shared_ptr<DecisionTreeClassifier<float>> dt_classifier =
    std::make_shared<DecisionTreeClassifier<float>>();
  dt_classifier->predict(handle, tree, rows, n_rows, n_cols, predictions,
                         verbosity);
}

void decisionTreeClassifierPredict(const raft::handle_t &handle,
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

void decisionTreeRegressorFit(const raft::handle_t &handle,
                              TreeRegressorF *&tree, float *data,
                              const int ncols, const int nrows, float *labels,
                              unsigned int *rowids, const int n_sampled_rows,
                              DecisionTree::DecisionTreeParams tree_params,
                              uint64_t seed) {
  std::shared_ptr<DecisionTreeRegressor<float>> dt_regressor =
    std::make_shared<DecisionTreeRegressor<float>>();
  auto quantile_size = tree_params.n_bins * ncols;
  MLCommon::device_buffer<float> global_quantiles(
    handle.get_device_allocator(), handle.get_stream(), quantile_size);
  DecisionTree::computeQuantiles(
    global_quantiles.data(), tree_params.n_bins, data, nrows, ncols,
    handle.get_device_allocator(), handle.get_stream());
  dt_regressor->fit(handle, data, ncols, nrows, labels, rowids, n_sampled_rows,
                    tree, tree_params, seed, global_quantiles.data());
}

void decisionTreeRegressorFit(const raft::handle_t &handle,
                              TreeRegressorD *&tree, double *data,
                              const int ncols, const int nrows, double *labels,
                              unsigned int *rowids, const int n_sampled_rows,
                              DecisionTree::DecisionTreeParams tree_params,
                              uint64_t seed) {
  std::shared_ptr<DecisionTreeRegressor<double>> dt_regressor =
    std::make_shared<DecisionTreeRegressor<double>>();

  auto quantile_size = tree_params.n_bins * ncols;
  MLCommon::device_buffer<double> global_quantiles(
    handle.get_device_allocator(), handle.get_stream(), quantile_size);
  DecisionTree::computeQuantiles(
    global_quantiles.data(), tree_params.n_bins, data, nrows, ncols,
    handle.get_device_allocator(), handle.get_stream());
  dt_regressor->fit(handle, data, ncols, nrows, labels, rowids, n_sampled_rows,
                    tree, tree_params, seed, global_quantiles.data());
}

void decisionTreeRegressorPredict(const raft::handle_t &handle,
                                  const TreeRegressorF *tree, const float *rows,
                                  const int n_rows, const int n_cols,
                                  float *predictions, int verbosity) {
  std::shared_ptr<DecisionTreeRegressor<float>> dt_regressor =
    std::make_shared<DecisionTreeRegressor<float>>();
  dt_regressor->predict(handle, tree, rows, n_rows, n_cols, predictions,
                        verbosity);
}

void decisionTreeRegressorPredict(const raft::handle_t &handle,
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
template std::string get_tree_summary_text<float, int>(
  const TreeClassifierF *tree);
template std::string get_tree_summary_text<double, int>(
  const TreeClassifierD *tree);
template std::string get_tree_summary_text<float, float>(
  const TreeRegressorF *tree);
template std::string get_tree_summary_text<double, double>(
  const TreeRegressorD *tree);

template std::string get_tree_text<float, int>(const TreeClassifierF *tree);
template std::string get_tree_text<double, int>(const TreeClassifierD *tree);
template std::string get_tree_text<float, float>(const TreeRegressorF *tree);
template std::string get_tree_text<double, double>(const TreeRegressorD *tree);

template std::string get_tree_json<float, int>(const TreeClassifierF *tree);
template std::string get_tree_json<double, int>(const TreeClassifierD *tree);
template std::string get_tree_json<float, float>(const TreeRegressorF *tree);
template std::string get_tree_json<double, double>(const TreeRegressorD *tree);

}  // End namespace DecisionTree
}  //End namespace ML
