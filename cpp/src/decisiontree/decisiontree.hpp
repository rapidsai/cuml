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
#include <common/cumlHandle.hpp>
#include "algo_helper.h"

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
   * TODO SKL's default is sqrt(n_cols)
   */
  float max_features;
  /**
   * Number of bins used by the split algorithm.
   */
  int n_bins;
  /**
   * The split algorithm: HIST or GLOBAL_QUANTILE.
   */
  int split_algo;
  /**
   * The minimum number of samples (rows) needed to split a node.
   */
  int min_rows_per_node;
  /**
   * Control bootstrapping for features. If features are drawn with or without replacement
   */
  bool bootstrap_features;
  /**
   * Whether a quantile needs to be computed for individual trees in RF.
   * Default: compute quantiles once per RF. Only affects GLOBAL_QUANTILE split_algo.
   */
  bool quantile_per_tree;
  /**
   * Node split criterion. GINI and Entropy for classification, MSE or MAE for regression.
   */
  CRITERION split_criterion;
  /**
   * Depth level algorithm
   */
  bool levelalgo = false;
};

void set_tree_params(DecisionTreeParams &params, int cfg_max_depth = -1,
                     int cfg_max_leaves = -1, float cfg_max_features = 1.0f,
                     int cfg_n_bins = 8, int cfg_split_algo = SPLIT_ALGO::HIST,
                     int cfg_min_rows_per_node = 2,
                     bool cfg_bootstrap_features = false,
                     CRITERION cfg_split_criterion = CRITERION_END,
                     bool cfg_quantile_per_tree = false);
void validity_check(const DecisionTreeParams params);
void print(const DecisionTreeParams params);

template <class T>
struct Question {
  int column;
  T value;
};

template <class T, class L>
struct TreeNode {
  TreeNode<T, L> *left;
  TreeNode<T, L> *right;
  L prediction;
  Question<T> question;
  T split_metric_val;
};

template <class T, class L>
struct TreeMetaDataNode {
  int depth_counter;
  int leaf_counter;
  double prepare_time;
  double train_time;
  TreeNode<T, L> *root;
};

template <class T, class L>
void print_tree_summary(const TreeMetaDataNode<T, L> *tree);

template <class T, class L>
void print_tree(const TreeMetaDataNode<T, L> *tree);

// ----------------------------- Classification ----------------------------------- //

typedef TreeMetaDataNode<float, int> TreeClassifierF;
typedef TreeMetaDataNode<double, int> TreeClassifierD;

void decisionTreeClassifierFit(const ML::cumlHandle &handle,
                               TreeClassifierF *&tree, float *data,
                               const int ncols, const int nrows, int *labels,
                               unsigned int *rowids, const int n_sampled_rows,
                               int unique_labels,
                               DecisionTree::DecisionTreeParams tree_params);

void decisionTreeClassifierFit(const ML::cumlHandle &handle,
                               TreeClassifierD *&tree, double *data,
                               const int ncols, const int nrows, int *labels,
                               unsigned int *rowids, const int n_sampled_rows,
                               int unique_labels,
                               DecisionTree::DecisionTreeParams tree_params);

void decisionTreeClassifierPredict(const ML::cumlHandle &handle,
                                   const TreeClassifierF *tree,
                                   const float *rows, const int n_rows,
                                   const int n_cols, int *predictions,
                                   bool verbose = false);

void decisionTreeClassifierPredict(const ML::cumlHandle &handle,
                                   const TreeClassifierD *tree,
                                   const double *rows, const int n_rows,
                                   const int n_cols, int *predictions,
                                   bool verbose = false);

// ----------------------------- Regression ----------------------------------- //

typedef TreeMetaDataNode<float, float> TreeRegressorF;
typedef TreeMetaDataNode<double, double> TreeRegressorD;

void decisionTreeRegressorFit(const ML::cumlHandle &handle,
                              TreeRegressorF *&tree, float *data,
                              const int ncols, const int nrows, float *labels,
                              unsigned int *rowids, const int n_sampled_rows,
                              DecisionTree::DecisionTreeParams tree_params);

void decisionTreeRegressorFit(const ML::cumlHandle &handle,
                              TreeRegressorD *&tree, double *data,
                              const int ncols, const int nrows, double *labels,
                              unsigned int *rowids, const int n_sampled_rows,
                              DecisionTree::DecisionTreeParams tree_params);

void decisionTreeRegressorPredict(const ML::cumlHandle &handle,
                                  const TreeRegressorF *tree, const float *rows,
                                  const int n_rows, const int n_cols,
                                  float *predictions, bool verbose = false);

void decisionTreeRegressorPredict(const ML::cumlHandle &handle,
                                  const TreeRegressorD *tree,
                                  const double *rows, const int n_rows,
                                  const int n_cols, double *predictions,
                                  bool verbose = false);

}  // End namespace DecisionTree
}  //End namespace ML
