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

#include <utils.h>
#include <queue>
#include <type_traits>
#include "decisiontree_impl.h"
#include "levelalgo/levelfunc_classifier.cuh"
#include "levelalgo/levelfunc_regressor.cuh"
#include "levelalgo/metric.cuh"
#include "memory.cuh"
#include "quantile/quantile.cuh"

namespace ML {

bool is_dev_ptr(const void *p) {
  cudaPointerAttributes pointer_attr;
  cudaError_t err = cudaPointerGetAttributes(&pointer_attr, p);
  if (err == cudaSuccess) {
    return pointer_attr.devicePointer;
  } else {
    err = cudaGetLastError();
    return false;
  }
}

namespace DecisionTree {

template <class T, class L>
void print(const SparseTreeNode<T, L> &node, std::ostream &os) {
  if (node.colid == -1) {
    os << "(leaf, " << node.prediction << ", " << node.best_metric_val << ")";
  } else {
    os << "(" << node.colid << ", " << node.quesval << ", "
       << node.best_metric_val << ")";
  }
  return;
}

template <class T, class L>
void print_node(const std::string &prefix,
                const std::vector<SparseTreeNode<T, L>> &sparsetree, int idx,
                bool isLeft) {
  const SparseTreeNode<T, L> &node = sparsetree[idx];
  std::cout << prefix;

  std::cout << (isLeft ? "├" : "└");

  // print the value of the node
  std::cout << node << std::endl;

  if ((node.colid != -1)) {
    // enter the next tree level - left and right branch
    print_node(prefix + (isLeft ? "│   " : "    "), sparsetree,
               node.left_child_id, true);
    print_node(prefix + (isLeft ? "│   " : "    "), sparsetree,
               node.left_child_id + 1, false);
  }
}

template <typename T, typename L>
std::ostream &operator<<(std::ostream &os, const SparseTreeNode<T, L> &node) {
  DecisionTree::print(node, os);
  return os;
}

template <typename T, typename L>
struct Node_ID_info {
  const SparseTreeNode<T, L> &node;
  int unique_node_id;

  Node_ID_info(const SparseTreeNode<T, L> &cfg_node, int cfg_unique_node_id)
    : node(cfg_node), unique_node_id(cfg_unique_node_id) {}
};

template <class T, class L>
void build_treelite_tree(TreeBuilderHandle tree_builder,
                         DecisionTree::TreeMetaDataNode<T, L> *tree_ptr,
                         int num_output_group) {
  int node_id = 0;
  TREELITE_CHECK(TreeliteTreeBuilderCreateNode(tree_builder, node_id));
  TREELITE_CHECK(TreeliteTreeBuilderSetRootNode(tree_builder, node_id));

  std::queue<Node_ID_info<T, L>> cur_level_queue;
  std::queue<Node_ID_info<T, L>> next_level_queue;

  cur_level_queue.push(Node_ID_info<T, L>(tree_ptr->sparsetree[0], 0));
  node_id = -1;

  while (!cur_level_queue.empty()) {
    int cur_level_size = cur_level_queue.size();
    node_id += cur_level_size;

    for (int i = 0; i < cur_level_size; i++) {
      Node_ID_info<T, L> q_node = cur_level_queue.front();
      cur_level_queue.pop();

      bool is_leaf_node = q_node.node.colid == -1;

      if (!is_leaf_node) {
        // Push left child to next_level queue.
        next_level_queue.push(Node_ID_info<T, L>(
          tree_ptr->sparsetree[q_node.node.left_child_id], node_id + 1));
        TREELITE_CHECK(
          TreeliteTreeBuilderCreateNode(tree_builder, node_id + 1));

        // Push right child to next_level deque.
        next_level_queue.push(Node_ID_info<T, L>(
          tree_ptr->sparsetree[q_node.node.left_child_id + 1], node_id + 2));
        TREELITE_CHECK(
          TreeliteTreeBuilderCreateNode(tree_builder, node_id + 2));

        // Set node from current level as numerical node. Children IDs known.
        TREELITE_CHECK(TreeliteTreeBuilderSetNumericalTestNode(
          tree_builder, q_node.unique_node_id, q_node.node.colid,
          "<=", q_node.node.quesval, 1, node_id + 1, node_id + 2));

        node_id += 2;
      } else {
        if (num_output_group == 1) {
          TREELITE_CHECK(TreeliteTreeBuilderSetLeafNode(
            tree_builder, q_node.unique_node_id, q_node.node.prediction));
        } else {
          std::vector<double> leaf_vector(num_output_group);
          for (int j = 0; j < num_output_group; j++) {
            if (q_node.node.prediction == j) {
              leaf_vector[j] = 1;
            } else {
              leaf_vector[j] = 0;
            }
          }
          TREELITE_CHECK(TreeliteTreeBuilderSetLeafVectorNode(
            tree_builder, q_node.unique_node_id, leaf_vector.data(),
            num_output_group));
          leaf_vector.clear();
        }
      }
    }

    // The cur_level_queue is empty here, as all the elements are already poped out.
    cur_level_queue.swap(next_level_queue);
  }
}

/**
 * @brief Print high-level tree information.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 */
template <typename T, typename L>
void DecisionTreeBase<T, L>::print_tree_summary() const {
  std::cout << " Decision Tree depth --> " << depth_counter
            << " and n_leaves --> " << leaf_counter << std::endl;
  std::cout << " Total temporary memory usage--> "
            << ((double)total_temp_mem / (1024 * 1024)) << "  MB" << std::endl;
  std::cout << " Shared memory used --> " << shmem_used << "  bytes "
            << std::endl;
  std::cout << " Tree Fitting - Overall time --> " << prepare_time + train_time
            << " seconds" << std::endl;
  std::cout << "   - preparing for fit time: " << prepare_time << " seconds"
            << std::endl;
  std::cout << "   - tree growing time: " << train_time << " seconds"
            << std::endl;
}

/**
 * @brief Print detailed tree information.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] root: pointer to tree's root node
 */
template <typename T, typename L>
void DecisionTreeBase<T, L>::print(
  const std::vector<SparseTreeNode<T, L>> &sparsetree) const {
  DecisionTreeBase<T, L>::print_tree_summary();
  print_node<T, L>("", sparsetree, 0, false);
}

template <typename T, typename L>
void DecisionTreeBase<T, L>::plant(
  std::vector<SparseTreeNode<T, L>> &sparsetree, const T *data, const int ncols,
  const int nrows, const L *labels, unsigned int *rowids,
  const int n_sampled_rows, int unique_labels, int maxdepth, int max_leaf_nodes,
  const float colper, int n_bins, int split_algo_flag,
  int cfg_min_rows_per_node, bool cfg_bootstrap_features,
  CRITERION cfg_split_criterion, bool quantile_per_tree) {
  split_algo = split_algo_flag;
  dinfo.NLocalrows = nrows;
  dinfo.NGlobalrows = nrows;
  dinfo.Ncols = ncols;
  nbins = n_bins;
  treedepth = maxdepth;
  maxleaves = max_leaf_nodes;
  n_unique_labels = unique_labels;
  min_rows_per_node = cfg_min_rows_per_node;
  bootstrap_features = cfg_bootstrap_features;
  split_criterion = cfg_split_criterion;
  //Bootstrap features
  feature_selector.resize(dinfo.Ncols);
  if (bootstrap_features) {
    srand(n_bins);
    for (int i = 0; i < dinfo.Ncols; i++) {
      feature_selector.push_back(rand() % dinfo.Ncols);
    }
  } else {
    std::iota(feature_selector.begin(), feature_selector.end(), 0);
  }

  std::random_shuffle(feature_selector.begin(), feature_selector.end());
  feature_selector.resize((int)(colper * dinfo.Ncols));
  if (split_algo == SPLIT_ALGO::GLOBAL_QUANTILE && quantile_per_tree) {
    preprocess_quantile(data, rowids, n_sampled_rows, ncols, dinfo.NLocalrows,
                        n_bins, tempmem);
  }
  CUDA_CHECK(cudaStreamSynchronize(
    tempmem->stream));  // added to ensure accurate measurement

  prepare_time = prepare_fit_timer.getElapsedSeconds();

  total_temp_mem = tempmem->totalmem;
  MLCommon::TimerCPU timer;
  grow_deep_tree(data, labels, rowids, feature_selector, n_sampled_rows, ncols,
                 dinfo.NLocalrows, sparsetree, tempmem);
  train_time = timer.getElapsedSeconds();
  tempmem.reset();
}
template <typename T, typename L>
void DecisionTreeBase<T, L>::predict(const ML::cumlHandle &handle,
                                     const TreeMetaDataNode<T, L> *tree,
                                     const T *rows, const int n_rows,
                                     const int n_cols, L *predictions,
                                     bool verbose) const {
  ASSERT(!is_dev_ptr(rows) && !is_dev_ptr(predictions),
         "DT Error: Current impl. expects both input and predictions to be CPU "
         "pointers.\n");

  ASSERT(tree && (tree->sparsetree.size() != 0),
         "Cannot predict w/ empty tree, tree size %zu",
         tree->sparsetree.size());
  ASSERT((n_rows > 0), "Invalid n_rows %d", n_rows);
  ASSERT((n_cols > 0), "Invalid n_cols %d", n_cols);

  predict_all(tree, rows, n_rows, n_cols, predictions, verbose);
}

template <typename T, typename L>
void DecisionTreeBase<T, L>::predict_all(const TreeMetaDataNode<T, L> *tree,
                                         const T *rows, const int n_rows,
                                         const int n_cols, L *preds,
                                         bool verbose) const {
  for (int row_id = 0; row_id < n_rows; row_id++) {
    preds[row_id] =
      predict_one(&rows[row_id * n_cols], tree->sparsetree, 0, verbose);
  }
}

template <typename T, typename L>
L DecisionTreeBase<T, L>::predict_one(
  const T *row, const std::vector<SparseTreeNode<T, L>> sparsetree, int idx,
  bool verbose) const {
  int colid = sparsetree[idx].colid;
  T quesval = sparsetree[idx].quesval;
  int leftchild = sparsetree[idx].left_child_id;
  if (colid == -1) {
    if (verbose) {
      std::cout << "Leaf node. Predicting " << sparsetree[idx].prediction
                << std::endl;
    }
    return sparsetree[idx].prediction;
  } else if (row[colid] <= quesval) {
    if (verbose) {
      std::cout << "Classifying Left @ node w/ column " << colid
                << " and value " << quesval << std::endl;
    }
    return predict_one(row, sparsetree, leftchild, verbose);
  } else {
    if (verbose) {
      std::cout << "Classifying Right @ node w/ column " << colid
                << " and value " << quesval << std::endl;
    }
    return predict_one(row, sparsetree, leftchild + 1, verbose);
  }
}

template <typename T, typename L>
void DecisionTreeBase<T, L>::set_metadata(TreeMetaDataNode<T, L> *&tree) {
  tree->depth_counter = depth_counter;
  tree->leaf_counter = leaf_counter;
  tree->train_time = train_time;
  tree->prepare_time = prepare_time;
}

template <typename T, typename L>
void DecisionTreeBase<T, L>::base_fit(
  const ML::cumlHandle &handle, const T *data, const int ncols, const int nrows,
  const L *labels, unsigned int *rowids, const int n_sampled_rows,
  int unique_labels, std::vector<SparseTreeNode<T, L>> &sparsetree,
  DecisionTreeParams &tree_params, bool is_classifier,
  std::shared_ptr<TemporaryMemory<T, L>> in_tempmem) {
  prepare_fit_timer.reset();
  const char *CRITERION_NAME[] = {"GINI", "ENTROPY", "MSE", "MAE", "END"};
  CRITERION default_criterion =
    (is_classifier) ? CRITERION::GINI : CRITERION::MSE;
  CRITERION last_criterion =
    (is_classifier) ? CRITERION::ENTROPY : CRITERION::MAE;

  validity_check(tree_params);
  if (tree_params.n_bins > n_sampled_rows) {
    std::cout << "Warning! Calling with number of bins > number of rows! ";
    std::cout << "Resetting n_bins to " << n_sampled_rows << "." << std::endl;
    tree_params.n_bins = n_sampled_rows;
  }

  if (
    tree_params.split_criterion ==
    CRITERION::
      CRITERION_END) {  // Set default to GINI (classification) or MSE (regression)
    tree_params.split_criterion = default_criterion;
  }
  ASSERT((tree_params.split_criterion >= default_criterion) &&
           (tree_params.split_criterion <= last_criterion),
         "Unsupported criterion %s\n",
         CRITERION_NAME[tree_params.split_criterion]);

  if (in_tempmem != nullptr) {
    tempmem = in_tempmem;
  } else {
    tempmem = std::make_shared<TemporaryMemory<T, L>>(
      handle.getImpl(), nrows, ncols, unique_labels, tree_params.n_bins,
      tree_params.split_algo, tree_params.max_depth);
    tree_params.quantile_per_tree = true;
  }

  plant(sparsetree, data, ncols, nrows, labels, rowids, n_sampled_rows,
        unique_labels, tree_params.max_depth, tree_params.max_leaves,
        tree_params.max_features, tree_params.n_bins, tree_params.split_algo,
        tree_params.min_rows_per_node, tree_params.bootstrap_features,
        tree_params.split_criterion, tree_params.quantile_per_tree);
}

template <typename T>
void DecisionTreeClassifier<T>::fit(
  const ML::cumlHandle &handle, const T *data, const int ncols, const int nrows,
  const int *labels, unsigned int *rowids, const int n_sampled_rows,
  const int unique_labels, TreeMetaDataNode<T, int> *&tree,
  DecisionTreeParams tree_params,
  std::shared_ptr<TemporaryMemory<T, int>> in_tempmem) {
  this->base_fit(handle, data, ncols, nrows, labels, rowids, n_sampled_rows,
                 unique_labels, tree->sparsetree, tree_params, true,
                 in_tempmem);
  this->set_metadata(tree);
}

template <typename T>
void DecisionTreeRegressor<T>::fit(
  const ML::cumlHandle &handle, const T *data, const int ncols, const int nrows,
  const T *labels, unsigned int *rowids, const int n_sampled_rows,
  TreeMetaDataNode<T, T> *&tree, DecisionTreeParams tree_params,
  std::shared_ptr<TemporaryMemory<T, T>> in_tempmem) {
  this->base_fit(handle, data, ncols, nrows, labels, rowids, n_sampled_rows, 1,
                 tree->sparsetree, tree_params, false, in_tempmem);
  this->set_metadata(tree);
}
template <typename T>
void DecisionTreeClassifier<T>::grow_deep_tree(
  const T *data, const int *labels, unsigned int *rowids,
  const std::vector<unsigned int> &feature_selector, const int n_sampled_rows,
  const int ncols, const int nrows,
  std::vector<SparseTreeNode<T, int>> &sparsetree,
  std::shared_ptr<TemporaryMemory<T, int>> tempmem) {
  int leaf_cnt = 0;
  int depth_cnt = 0;
  grow_deep_tree_classification(
    data, labels, rowids, feature_selector, n_sampled_rows, nrows,
    this->n_unique_labels, this->nbins, this->treedepth, this->maxleaves,
    this->min_rows_per_node, this->split_criterion, this->split_algo, depth_cnt,
    leaf_cnt, sparsetree, tempmem);
  this->depth_counter = depth_cnt;
  this->leaf_counter = leaf_cnt;
}

template <typename T>
void DecisionTreeRegressor<T>::grow_deep_tree(
  const T *data, const T *labels, unsigned int *rowids,
  const std::vector<unsigned int> &feature_selector, const int n_sampled_rows,
  const int ncols, const int nrows,
  std::vector<SparseTreeNode<T, T>> &sparsetree,
  std::shared_ptr<TemporaryMemory<T, T>> tempmem) {
  int leaf_cnt = 0;
  int depth_cnt = 0;
  grow_deep_tree_regression(data, labels, rowids, feature_selector,
                            n_sampled_rows, nrows, this->nbins, this->treedepth,
                            this->maxleaves, this->min_rows_per_node,
                            this->split_criterion, this->split_algo, depth_cnt,
                            leaf_cnt, sparsetree, tempmem);
  this->depth_counter = depth_cnt;
  this->leaf_counter = leaf_cnt;
}

//Class specializations
template class DecisionTreeBase<float, int>;
template class DecisionTreeBase<float, float>;
template class DecisionTreeBase<double, int>;
template class DecisionTreeBase<double, double>;

template class DecisionTreeClassifier<float>;
template class DecisionTreeClassifier<double>;

template class DecisionTreeRegressor<float>;
template class DecisionTreeRegressor<double>;

template void build_treelite_tree<float, int>(
  TreeBuilderHandle tree_builder,
  DecisionTree::TreeMetaDataNode<float, int> *tree_ptr, int num_output_group);
template void build_treelite_tree<double, int>(
  TreeBuilderHandle tree_builder,
  DecisionTree::TreeMetaDataNode<double, int> *tree_ptr, int num_output_group);
template void build_treelite_tree<float, float>(
  TreeBuilderHandle tree_builder,
  DecisionTree::TreeMetaDataNode<float, float> *tree_ptr, int num_output_group);
template void build_treelite_tree<double, double>(
  TreeBuilderHandle tree_builder,
  DecisionTree::TreeMetaDataNode<double, double> *tree_ptr,
  int num_output_group);
}  //End namespace DecisionTree

}  //End namespace ML
