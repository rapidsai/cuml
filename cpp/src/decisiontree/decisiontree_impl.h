
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
#include "decisiontree.hpp"
#include "kernels/metric_def.h"

namespace ML {

bool is_dev_ptr(const void *p);

namespace DecisionTree {

template <class T, class L>
void null_tree_node_child_ptrs(TreeNode<T, L> &node);

template <class T, class L>
void print(const TreeNode<T, L> &node, std::ostream &os);

template <class T, class L>
void print_node(const std::string &prefix, const TreeNode<T, L> *const node,
                bool isLeft);

struct DataInfo {
  unsigned int NLocalrows;
  unsigned int NGlobalrows;
  unsigned int Ncols;
};

template <class T, class L>
class DecisionTreeBase {
 protected:
  int split_algo;
  int nbins;
  DataInfo dinfo;
  int treedepth;
  int depth_counter = 0;
  int maxleaves;
  int leaf_counter = 0;
  std::shared_ptr<TemporaryMemory<T, L>> tempmem;
  size_t total_temp_mem;
  const int MAXSTREAMS = 1;
  size_t max_shared_mem;
  size_t shmem_used = 0;
  int n_unique_labels = -1;  // number of unique labels in dataset
  double prepare_time = 0;
  double train_time = 0;
  int min_rows_per_node;
  bool bootstrap_features;
  CRITERION split_criterion;
  std::vector<unsigned int> feature_selector;
  MLCommon::TimerCPU prepare_fit_timer;

  void split_branch(const T *data, MetricQuestion<T> &ques,
                    const int n_sampled_rows, int &nrowsleft, int &nrowsright,
                    unsigned int *rowids);

  void plant(const cumlHandle_impl &handle, TreeNode<T, L> *&root,
             const T *data, const int ncols, const int nrows, const L *labels,
             unsigned int *rowids, const int n_sampled_rows, int unique_labels,
             int maxdepth = -1, int max_leaf_nodes = -1,
             const float colper = 1.0, int n_bins = 8,
             int split_algo_flag = SPLIT_ALGO::HIST,
             int cfg_min_rows_per_node = 2, bool cfg_bootstrap_features = false,
             CRITERION cfg_split_criterion = CRITERION::CRITERION_END,
             bool cfg_quantile_per_tree = false,
             std::shared_ptr<TemporaryMemory<T, L>> in_tempmem = nullptr);
  void init_depth_zero(const L *labels, std::vector<unsigned int> &colselector,
                       const unsigned int *rowids, const int n_sampled_rows,
                       const std::shared_ptr<TemporaryMemory<T, L>> tempmem);
  TreeNode<T, L> *grow_tree(const T *data, const float colper, const L *labels,
                            int depth, unsigned int *rowids,
                            const int n_sampled_rows,
                            MetricInfo<T> prev_split_info);
  virtual void find_best_fruit_all(const T *data, const L *labels,
                                   const float colper, MetricQuestion<T> &ques,
                                   float &gain, unsigned int *rowids,
                                   const int n_sampled_rows,
                                   MetricInfo<T> split_info[3], int depth) = 0;

  virtual TreeNode<T, L> *grow_deep_tree(
    const ML::cumlHandle_impl &handle, const T *data, const L *labels,
    unsigned int *rowids, const std::vector<unsigned int> &feature_selector,
    const int n_sampled_rows, const int ncols, const int nrows,
    std::shared_ptr<TemporaryMemory<T, L>> tempmem) = 0;

  void base_fit(const ML::cumlHandle &handle, const T *data, const int ncols,
                const int nrows, const L *labels, unsigned int *rowids,
                const int n_sampled_rows, int unique_labels,
                TreeNode<T, L> *&root, DecisionTreeParams &tree_params,
                bool is_classifier,
                std::shared_ptr<TemporaryMemory<T, L>> in_tempmem);

 public:
  // Printing utility for high level tree info.
  void print_tree_summary() const;

  // Printing utility for debug and looking at nodes and leaves.
  void print(const TreeNode<T, L> *root) const;

  // Predict labels for n_rows rows, with n_cols features each, for a given tree. rows in row-major format.
  void predict(const ML::cumlHandle &handle, const TreeMetaDataNode<T, L> *tree,
               const T *rows, const int n_rows, const int n_cols,
               L *predictions, bool verbose = false) const;
  void predict_all(const TreeMetaDataNode<T, L> *tree, const T *rows,
                   const int n_rows, const int n_cols, L *preds,
                   bool verbose = false) const;
  L predict_one(const T *row, const TreeNode<T, L> *const node,
                bool verbose = false) const;

  void set_metadata(TreeMetaDataNode<T, L> *&tree);

};  // End DecisionTreeBase Class

template <class T>
class DecisionTreeClassifier : public DecisionTreeBase<T, int> {
 public:
  // Expects column major T dataset, integer labels
  // data, labels are both device ptr.
  // Assumption: labels are all mapped to contiguous numbers starting from 0 during preprocessing. Needed for gini hist impl.
  void fit(const ML::cumlHandle &handle, const T *data, const int ncols,
           const int nrows, const int *labels, unsigned int *rowids,
           const int n_sampled_rows, const int unique_labels,
           TreeMetaDataNode<T, int> *&tree, DecisionTreeParams tree_params,
           std::shared_ptr<TemporaryMemory<T, int>> in_tempmem = nullptr);

 private:
  /* depth is used to distinguish between root and other tree nodes for computations */
  void find_best_fruit_all(const T *data, const int *labels, const float colper,
                           MetricQuestion<T> &ques, float &gain,
                           unsigned int *rowids, const int n_sampled_rows,
                           MetricInfo<T> split_info[3], int depth);
  TreeNode<T, int> *grow_deep_tree(
    const ML::cumlHandle_impl &handle, const T *data, const int *labels,
    unsigned int *rowids, const std::vector<unsigned int> &feature_selector,
    const int n_sampled_rows, const int ncols, const int nrows,
    std::shared_ptr<TemporaryMemory<T, int>> tempmem);

};  // End DecisionTreeClassifier Class

template <class T>
class DecisionTreeRegressor : public DecisionTreeBase<T, T> {
 public:
  void fit(const ML::cumlHandle &handle, const T *data, const int ncols,
           const int nrows, const T *labels, unsigned int *rowids,
           const int n_sampled_rows, TreeMetaDataNode<T, T> *&tree,
           DecisionTreeParams tree_params,
           std::shared_ptr<TemporaryMemory<T, T>> in_tempmem = nullptr);

 private:
  /* depth is used to distinguish between root and other tree nodes for computations */
  void find_best_fruit_all(const T *data, const T *labels, const float colper,
                           MetricQuestion<T> &ques, float &gain,
                           unsigned int *rowids, const int n_sampled_rows,
                           MetricInfo<T> split_info[3], int depth);
  TreeNode<T, T> *grow_deep_tree(
    const ML::cumlHandle_impl &handle, const T *data, const T *labels,
    unsigned int *rowids, const std::vector<unsigned int> &feature_selector,
    const int n_sampled_rows, const int ncols, const int nrows,
    std::shared_ptr<TemporaryMemory<T, T>> tempmem);

};  // End DecisionTreeRegressor Class

}  //End namespace DecisionTree

}  //End namespace ML
