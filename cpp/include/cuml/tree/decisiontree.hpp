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
#include <vector>
#include "algo_helper.h"
#include "flatnode.h"

namespace raft {
class handle_t;
}

namespace ML {

namespace DecisionTree {

struct DecisionTreeParams {
  /**
   * Maximum tree depth. Unlimited (e.g., until leaves are pure), if -1.
   */
  int max_depth;
  /**
   * Maximum leaf nodes per tree. Soft constraint. Unlimited, if -1.
   */
  int max_leaves;
  /**
   * Ratio of number of features (columns) to consider per node split.
   */
  float max_features;
  /**
   * Number of bins used by the split algorithm.
   */
  int n_bins;
  /**
   * The minimum number of samples (rows) in each leaf node.
   */
  int min_samples_leaf;
  /**
   * The minimum number of samples (rows) needed to split an internal node.
   */
  int min_samples_split;
  /**
   * Node split criterion. GINI and Entropy for classification, MSE for regression.
   */
  CRITERION split_criterion;
  /**
   * Minimum impurity decrease required for spliting a node. If the impurity decrease is below this value, node is leafed out. Default is 0.0
   */
  float min_impurity_decrease = 0.0f;

  /**
   * Maximum number of nodes that can be processed in a given batch. This is
   * used only for batched-level algo
   */
  int max_batch_size;
};

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
 * @param[in] cfg_min_impurity_decrease: split a node only if its reduction in
 *                                       impurity is more than this value
 * @param[in] cfg_split_criterion: split criterion; default CRITERION_END,
 *            i.e., GINI for classification or MSE for regression
 * @param[in] cfg_max_batch_size: Maximum number of nodes that can be processed
              in a batch. This is used only for batched-level algo. Default
              value 128.
 */
void set_tree_params(DecisionTreeParams &params, int cfg_max_depth = -1,
                     int cfg_max_leaves = -1, float cfg_max_features = 1.0f,
                     int cfg_n_bins = 128, int cfg_min_samples_leaf = 1,
                     int cfg_min_samples_split = 2,
                     float cfg_min_impurity_decrease = 0.0f,
                     CRITERION cfg_split_criterion = CRITERION_END,
                     int cfg_max_batch_size = 128);

/**
 * @brief Check validity of all decision tree hyper-parameters.
 * @param[in] params: decision tree hyper-parameters.
 */
void validity_check(const DecisionTreeParams params);

/**
 * @brief Print all decision tree hyper-parameters.
 * @param[in] params: decision tree hyper-parameters.
 */
void print(const DecisionTreeParams params);

template <class T, class L>
struct TreeMetaDataNode {
  int treeid;
  int depth_counter;
  int leaf_counter;
  double prepare_time;
  double train_time;
  std::vector<SparseTreeNode<T, L>> sparsetree;
};

/**
 * @brief Obtain high-level tree information.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] tree: CPU pointer to TreeMetaDataNode
 * @return High-level tree information as string
 */
template <class T, class L>
std::string get_tree_summary_text(const TreeMetaDataNode<T, L> *tree);

/**
 * @brief Obtain detailed tree information.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] tree: CPU pointer to TreeMetaDataNode
 * @return Detailed tree information as string
 */
template <class T, class L>
std::string get_tree_text(const TreeMetaDataNode<T, L> *tree);

/**
 * @brief Export tree as a JSON string
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] tree: CPU pointer to TreeMetaDataNode
 * @return Tree structure as JSON stsring
 */
template <class T, class L>
std::string get_tree_json(const TreeMetaDataNode<T, L> *tree);

// ----------------------------- Classification ----------------------------------- //

typedef TreeMetaDataNode<float, int> TreeClassifierF;
typedef TreeMetaDataNode<double, int> TreeClassifierD;

/**
 * @defgroup DecisionTreeClassifierFit Fit functions
 * @brief Build (i.e., fit, train) Decision Tree classifier for input data.
 * @param[in] handle: raft::handle_t
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
 * @param[in] n_unique_labels: number of unique label values. Number of
 *                             categories of classification.
 * @param[in] tree_params: Decision Tree training hyper parameter struct.
 * @param[in] seed: Controls the randomness in tree fitting/growing algorithm.
 * @{
 */
void decisionTreeClassifierFit(const raft::handle_t &handle,
                               TreeClassifierF *&tree, float *data,
                               const int ncols, const int nrows, int *labels,
                               unsigned int *rowids, const int n_sampled_rows,
                               int unique_labels,
                               DecisionTree::DecisionTreeParams tree_params,
                               uint64_t seed);
void decisionTreeClassifierFit(const raft::handle_t &handle,
                               TreeClassifierD *&tree, double *data,
                               const int ncols, const int nrows, int *labels,
                               unsigned int *rowids, const int n_sampled_rows,
                               int unique_labels,
                               DecisionTree::DecisionTreeParams tree_params,
                               uint64_t seed);
/** @} */

/**
 * @defgroup DecisionTreeClassifierPredict Predict functions
 * @brief Predict target feature for input data; n-ary classification for
 *   single feature supported. Inference of trees is CPU only for now.
 * @param[in] handle: raft::handle_t (currently unused; API placeholder)
 * @param[in] tree: CPU pointer to TreeMetaDataNode.
 * @param[in] rows: test data (n_rows samples, n_cols features) in row major format.
 *    Current impl. expects a CPU pointer. TODO future API change.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in,out] predictions: n_rows predicted labels. Current impl. expects a
 *    CPU pointer, user allocated. TODO future API change.
 * @param[in] verbosity: verbosity level for logging messages during execution.
 *                       A negative value means to not perform an explicit
 *                       `setLevel()` call, but to continue with the level that
 *                       the caller itself might have set.
 * @{
 */
void decisionTreeClassifierPredict(const raft::handle_t &handle,
                                   const TreeClassifierF *tree,
                                   const float *rows, const int n_rows,
                                   const int n_cols, int *predictions,
                                   int verbosity = -1);
void decisionTreeClassifierPredict(const raft::handle_t &handle,
                                   const TreeClassifierD *tree,
                                   const double *rows, const int n_rows,
                                   const int n_cols, int *predictions,
                                   int verbosity = -1);
/** @} */

// ----------------------------- Regression ----------------------------------- //

typedef TreeMetaDataNode<float, float> TreeRegressorF;
typedef TreeMetaDataNode<double, double> TreeRegressorD;

/**
 * @defgroup DecisionTreeRegressorFit Fit functions
 * @brief Build (i.e., fit, train) Decision Tree regressor for input data.
 * @param[in] handle: raft::handle_t
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
 * @param[in] seed: Controls the randomness in tree fitting/growing algorithm.
 * @{
 */
void decisionTreeRegressorFit(const raft::handle_t &handle,
                              TreeRegressorF *&tree, float *data,
                              const int ncols, const int nrows, float *labels,
                              unsigned int *rowids, const int n_sampled_rows,
                              DecisionTree::DecisionTreeParams tree_params,
                              uint64_t seed);
void decisionTreeRegressorFit(const raft::handle_t &handle,
                              TreeRegressorD *&tree, double *data,
                              const int ncols, const int nrows, double *labels,
                              unsigned int *rowids, const int n_sampled_rows,
                              DecisionTree::DecisionTreeParams tree_params,
                              uint64_t seed);
/** @} */

/**
 * @defgroup DecisionTreeRegressorPredict Predict functions
 * @brief Predict target feature for input data; regression for single feature supported.
 *   Inference of trees is CPU only for now.
 * @param[in] handle: raft::handle_t (currently unused; API placeholder)
 * @param[in] tree: CPU pointer to TreeMetaDataNode.
 * @param[in] rows: test data (n_rows samples, n_cols features) in row major format.
 *   Current impl. expects a CPU pointer. TODO future API change.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in,out] predictions: n_rows predicted labels. Current impl. expects a CPU
 *   pointer, user allocated. TODO future API change.
 * @param[in] verbosity: verbosity level for logging messages during execution.
 *                       A negative value means to not perform an explicit
 *                       `setLevel()` call, but to continue with the level that
 *                       the caller itself might have set.
 * @{
 */
void decisionTreeRegressorPredict(const raft::handle_t &handle,
                                  const TreeRegressorF *tree, const float *rows,
                                  const int n_rows, const int n_cols,
                                  float *predictions, int verbosity = -1);
void decisionTreeRegressorPredict(const raft::handle_t &handle,
                                  const TreeRegressorD *tree,
                                  const double *rows, const int n_rows,
                                  const int n_cols, double *predictions,
                                  int verbosity = -1);
/** @} */

}  // End namespace DecisionTree
}  //End namespace ML
