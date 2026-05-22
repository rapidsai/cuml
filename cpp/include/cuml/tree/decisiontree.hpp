/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "algo_helper.h"
#include "flatnode.h"

#include <cuml/common/export.hpp>

#include <string>
#include <vector>

namespace CUML_EXPORT ML {

namespace DT {

/**
 * Selects the per-node split-finding strategy. SPLITTER_BEST evaluates every
 * candidate bin edge and picks the highest-gain split (RandomForest behavior).
 * SPLITTER_RANDOM draws one uniformly random bin edge per candidate feature
 * and scores only that draw (ExtraTrees behavior).
 */
enum Splitter {
  SPLITTER_BEST   = 0,
  SPLITTER_RANDOM = 1,
};

struct DecisionTreeParams {
  /**
   * Maximum tree depth. Set to INT32_MAX for unlimited depth
   * (i.e., until leaves are pure or other stopping criteria are met).
   */
  int max_depth;
  /**
   * Maximum leaf nodes per tree. Soft constraint. Unlimited, If `-1`.
   */
  int max_leaves;
  /**
   * Ratio of number of features (columns) to consider per node split.
   */
  float max_features;
  /**
   * maximum number of bins used by the split algorithm per feature.
   */
  int max_n_bins;
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
   * Minimum impurity decrease required for splitting a node. If the impurity decrease is below this
   * value, node is leafed out. Default is 0.0
   */
  float min_impurity_decrease = 0.0f;

  /**
   * Maximum number of nodes that can be processed in a given batch. This is
   * used only for batched-level algo
   */
  int max_batch_size;
  /**
   * Per-node split-finding strategy. Defaults to SPLITTER_BEST to preserve
   * byte-identical RandomForest behavior; SPLITTER_RANDOM selects the
   * ExtraTrees random-split path.
   */
  Splitter splitter = SPLITTER_BEST;
};

/**
 * @brief Set all DecisionTreeParams members.
 * @param[in,out] params: update with tree parameters
 * @param[in] cfg_max_depth: maximum tree depth; default -1
 * @param[in] cfg_max_leaves: maximum leaves; default -1
 * @param[in] cfg_max_features: maximum number of features; default 1.0f
 * @param[in] cfg_max_n_bins: maximum number of bins; default 128
 * @param[in] cfg_min_samples_leaf: min. rows in each leaf node; default 1
 * @param[in] cfg_min_samples_split: min. rows needed to split an internal node;
 *            default 2
 * @param[in] cfg_min_impurity_decrease: split a node only if its reduction in
 *                                       impurity is more than this value
 * @param[in] cfg_split_criterion: split criterion; default CRITERION_END,
 *            i.e., GINI for classification or MSE for regression
 * @param[in] cfg_max_batch_size: Maximum number of nodes that can be processed
              in a batch. This is used only for batched-level algo. Default
              value 4096.
 * @param[in] cfg_splitter: per-node split-finding strategy. Default
              SPLITTER_BEST preserves RandomForest behavior; SPLITTER_RANDOM
              selects the ExtraTrees random-split path.
 */
void set_tree_params(DecisionTreeParams& params,
                     int cfg_max_depth               = -1,
                     int cfg_max_leaves              = -1,
                     float cfg_max_features          = 1.0f,
                     int cfg_max_n_bins              = 128,
                     int cfg_min_samples_leaf        = 1,
                     int cfg_min_samples_split       = 2,
                     float cfg_min_impurity_decrease = 0.0f,
                     CRITERION cfg_split_criterion   = CRITERION_END,
                     int cfg_max_batch_size          = 4096,
                     Splitter cfg_splitter           = SPLITTER_BEST);

template <class T, class L>
struct TreeMetaDataNode {
  int treeid;
  int depth_counter;
  int leaf_counter;
  double train_time;
  std::vector<T> vector_leaf;
  std::vector<SparseTreeNode<T, L>> sparsetree;
  // Per-node sum of sample_weight, parallel to sparsetree. Empty for
  // unweighted fits; populated only when sample_weight is provided so the
  // unweighted tree-build path stays byte-identical to the no-weight build.
  // Used internally by compute_feature_importances to normalize per-node
  // BestMetric by weighted mass; not serialized to JSON or Treelite by
  // design (Treelite carries inference structure only; JSON is debug-only).
  std::vector<double> weighted_node_count;
  int num_outputs;
};

/**
 * @brief Obtain high-level tree information.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] tree: CPU pointer to TreeMetaDataNode
 * @return High-level tree information as string
 */
template <class T, class L>
std::string get_tree_summary_text(const TreeMetaDataNode<T, L>* tree);

/**
 * @brief Obtain detailed tree information.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] tree: CPU pointer to TreeMetaDataNode
 * @return Detailed tree information as string
 */
template <class T, class L>
std::string get_tree_text(const TreeMetaDataNode<T, L>* tree);

/**
 * @brief Export tree as a JSON string
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] tree: CPU pointer to TreeMetaDataNode
 * @return Tree structure as JSON stsring
 */
template <class T, class L>
std::string get_tree_json(const TreeMetaDataNode<T, L>* tree);

typedef TreeMetaDataNode<float, int> TreeClassifierF;
typedef TreeMetaDataNode<double, int> TreeClassifierD;
typedef TreeMetaDataNode<float, float> TreeRegressorF;
typedef TreeMetaDataNode<double, double> TreeRegressorD;

}  // End namespace DT
}  // End namespace CUML_EXPORT ML
