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
#include <common/Timer.h>
#include <algorithm>
#include <climits>
#include <common/cumlHandle.hpp>
#include <map>
#include <numeric>
#include <vector>
#include "algo_helper.h"
#include "kernels/metric_def.h"

namespace ML {

bool is_dev_ptr(const void *p);

namespace DecisionTree {

template <class T>
struct Question {
  int column;
  T value;
  void update(const MetricQuestion<T> &ques);
};

template <class T, class L>
struct TreeNode {
  TreeNode *left = nullptr;
  TreeNode *right = nullptr;
  L prediction;
  Question<T> question;
  T split_metric_val;

  void print(std::ostream &os) const;
};

struct DataInfo {
  unsigned int NLocalrows;
  unsigned int NGlobalrows;
  unsigned int Ncols;
};

struct DecisionTreeParams {
  /**
   * Maximum tree depth. Unlimited (e.g., until leaves are pure), if -1.
   */
  int max_depth = -1;
  /**
   * Maximum leaf nodes per tree. Soft constraint. Unlimited, if -1.
   */
  int max_leaves = -1;
  /**
   * Ratio of number of features (columns) to consider per node split.
   * TODO SKL's default is sqrt(n_cols)
   */
  float max_features = 1.0;
  /**
   * Number of bins used by the split algorithm.
   */
  int n_bins = 8;
  /**
   * The split algorithm: HIST or GLOBAL_QUANTILE.
   */
  int split_algo = SPLIT_ALGO::HIST;
  /**
   * The minimum number of samples (rows) needed to split a node.
   */
  int min_rows_per_node = 2;
  /**
   * Whether to bootstrap columns with or without replacement.
   */
  bool bootstrap_features = false;
  /**
   * Whether a quantile needs to be computed for individual trees in RF.
   * Default: compute quantiles once per RF. Only affects GLOBAL_QUANTILE split_algo.
   **/
  bool quantile_per_tree = false;
  /**
   * Node split criterion. GINI and Entropy for classification, MSE or MAE for regression.
   */
  CRITERION split_criterion = CRITERION_END;
  bool levelalgo = false;
  DecisionTreeParams();
  DecisionTreeParams(int cfg_max_depth, int cfg_max_leaves,
                     float cfg_max_features, int cfg_n_bins, int cfg_split_aglo,
                     int cfg_min_rows_per_node, bool cfg_bootstrap_features,
                     CRITERION cfg_split_criterion, bool cfg_quantile_per_tree);
  void validity_check() const;
  void print() const;
};

template <class T, class L>
class DecisionTreeBase {
 protected:
  int split_algo;
  TreeNode<T, L> *root = nullptr;
  int nbins;
  DataInfo dinfo;
  int treedepth;
  int depth_counter = 0;
  int maxleaves;
  int leaf_counter = 0;
  std::vector<std::shared_ptr<TemporaryMemory<T, L>>> tempmem;
  size_t total_temp_mem;
  const int MAXSTREAMS = 1;
  size_t max_shared_mem;
  size_t shmem_used = 0;
  int n_unique_labels = -1;  // number of unique labels in dataset
  double construct_time;
  int min_rows_per_node;
  bool bootstrap_features;
  CRITERION split_criterion;
  std::vector<unsigned int> feature_selector;
  
  void print_node(const std::string &prefix, const TreeNode<T, L> *const node,
                  bool isLeft) const;
  void split_branch(T *data, MetricQuestion<T> &ques, const int n_sampled_rows,
                    int &nrowsleft, int &nrowsright, unsigned int *rowids);
  
  void plant(const cumlHandle_impl &handle, T *data, const int ncols,
             const int nrows, L *labels, unsigned int *rowids,
             const int n_sampled_rows, int unique_labels, int maxdepth = -1,
             int max_leaf_nodes = -1, const float colper = 1.0, int n_bins = 8,
             int split_algo_flag = SPLIT_ALGO::HIST,
             int cfg_min_rows_per_node = 2, bool cfg_bootstrap_features = false,
             CRITERION cfg_split_criterion = CRITERION::CRITERION_END,
             bool cfg_quantile_per_tree = false,
             std::shared_ptr<TemporaryMemory<T, L>> in_tempmem = nullptr,
             bool levelalgo = false);
  void init_depth_zero(const L *labels, std::vector<unsigned int> &colselector,
                       const unsigned int *rowids, const int n_sampled_rows,
                       const std::shared_ptr<TemporaryMemory<T, L>> tempmem);
  TreeNode<T, L> *grow_tree(T *data, const float colper, L *labels, int depth,
                            unsigned int *rowids, const int n_sampled_rows,
                            MetricInfo<T> prev_split_info);
  virtual void find_best_fruit_all(T *data, L *labels, const float colper,
                                   MetricQuestion<T> &ques, float &gain,
                                   unsigned int *rowids,
                                   const int n_sampled_rows,
                                   MetricInfo<T> split_info[3], int depth) = 0;
  
  virtual TreeNode<T, L> *grow_deep_tree_member(
						const ML::cumlHandle_impl &handle, T *data, L *labels, unsigned int *rowids,
						const int n_sampled_rows, const int ncols, const int nrows) = 0;
  
  void base_fit(const ML::cumlHandle &handle, T *data, const int ncols,
                const int nrows, L *labels, unsigned int *rowids,
                const int n_sampled_rows, int unique_labels,
                DecisionTreeParams &tree_params, bool is_classifier,
                std::shared_ptr<TemporaryMemory<T, L>> in_tempmem);
  
 public:
  // Printing utility for high level tree info.
  void print_tree_summary() const;

  // Printing utility for debug and looking at nodes and leaves.
  void print() const;

  // Predict labels for n_rows rows, with n_cols features each, for a given tree. rows in row-major format.
  void predict(const ML::cumlHandle &handle, const T *rows, const int n_rows,
               const int n_cols, L *predictions, bool verbose = false) const;
  void predict_all(const T *rows, const int n_rows, const int n_cols, L *preds,
                   bool verbose = false) const;
  L predict_one(const T *row, const TreeNode<T, L> *const node,
                bool verbose = false) const;

};  // End DecisionTreeBase Class

template <class T>
class DecisionTreeClassifier : public DecisionTreeBase<T, int> {
 public:
  // Expects column major T dataset, integer labels
  // data, labels are both device ptr.
  // Assumption: labels are all mapped to contiguous numbers starting from 0 during preprocessing. Needed for gini hist impl.
  void fit(const ML::cumlHandle &handle, T *data, const int ncols,
           const int nrows, int *labels, unsigned int *rowids,
           const int n_sampled_rows, const int unique_labels,
           DecisionTreeParams tree_params,
           std::shared_ptr<TemporaryMemory<T, int>> in_tempmem = nullptr);

 private:
  /* depth is used to distinguish between root and other tree nodes for computations */
  void find_best_fruit_all(T *data, int *labels, const float colper,
                           MetricQuestion<T> &ques, float &gain,
                           unsigned int *rowids, const int n_sampled_rows,
                           MetricInfo<T> split_info[3], int depth);
  TreeNode<T, int> *grow_deep_tree_member(const ML::cumlHandle_impl &handle,
                                          T *data, int *labels,
                                          unsigned int *rowids,
                                          const int n_sampled_rows,
                                          const int ncols, const int nrows);

};  // End DecisionTreeClassifier Class

template <class T>
class DecisionTreeRegressor : public DecisionTreeBase<T, T> {
 public:
  void fit(const ML::cumlHandle &handle, T *data, const int ncols,
           const int nrows, T *labels, unsigned int *rowids,
           const int n_sampled_rows, DecisionTreeParams tree_params,
           std::shared_ptr<TemporaryMemory<T, T>> in_tempmem = nullptr);

 private:
  /* depth is used to distinguish between root and other tree nodes for computations */
  void find_best_fruit_all(T *data, T *labels, const float colper,
                           MetricQuestion<T> &ques, float &gain,
                           unsigned int *rowids, const int n_sampled_rows,
                           MetricInfo<T> split_info[3], int depth);
};  // End DecisionTreeRegressor Class

}  //End namespace DecisionTree

// Stateless API functions

// ----------------------------- Classification ----------------------------------- //

void fit(const ML::cumlHandle &handle,
         DecisionTree::DecisionTreeClassifier<float> *dt_classifier,
         float *data, const int ncols, const int nrows, int *labels,
         unsigned int *rowids, const int n_sampled_rows, int unique_labels,
         DecisionTree::DecisionTreeParams tree_params);

void fit(const ML::cumlHandle &handle,
         DecisionTree::DecisionTreeClassifier<double> *dt_classifier,
         double *data, const int ncols, const int nrows, int *labels,
         unsigned int *rowids, const int n_sampled_rows, int unique_labels,
         DecisionTree::DecisionTreeParams tree_params);

void predict(const ML::cumlHandle &handle,
             const DecisionTree::DecisionTreeClassifier<float> *dt_classifier,
             const float *rows, const int n_rows, const int n_cols,
             int *predictions, bool verbose = false);
void predict(const ML::cumlHandle &handle,
             const DecisionTree::DecisionTreeClassifier<double> *dt_classifier,
             const double *rows, const int n_rows, const int n_cols,
             int *predictions, bool verbose = false);

// ----------------------------- Regression ----------------------------------- //

void fit(const ML::cumlHandle &handle,
         DecisionTree::DecisionTreeRegressor<float> *dt_regressor, float *data,
         const int ncols, const int nrows, float *labels, unsigned int *rowids,
         const int n_sampled_rows,
         DecisionTree::DecisionTreeParams tree_params);

void fit(const ML::cumlHandle &handle,
         DecisionTree::DecisionTreeRegressor<double> *dt_regressor,
         double *data, const int ncols, const int nrows, double *labels,
         unsigned int *rowids, const int n_sampled_rows,
         DecisionTree::DecisionTreeParams tree_params);

void predict(const ML::cumlHandle &handle,
             const DecisionTree::DecisionTreeRegressor<float> *dt_regressor,
             const float *rows, const int n_rows, const int n_cols,
             float *predictions, bool verbose = false);
void predict(const ML::cumlHandle &handle,
             const DecisionTree::DecisionTreeRegressor<double> *dt_regressor,
             const double *rows, const int n_rows, const int n_cols,
             double *predictions, bool verbose = false);

}  //End namespace ML
