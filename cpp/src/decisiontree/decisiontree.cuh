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

#include <common/Timer.h>

#include <cuml/tree/algo_helper.h>
#include <cuml/tree/flatnode.h>
#include <cuml/common/logger.hpp>
#include <cuml/tree/decisiontree.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>

#include <treelite/c_api.h>
#include <treelite/tree.h>

#include <algorithm>
#include <climits>
#include <common/iota.cuh>
#include <common/nvtx.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/tree/decisiontree.hpp>
#include <iomanip>
#include <locale>
#include <map>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>
#include "batched-levelalgo/builder.cuh"
#include "quantile/quantile.h"
#include "treelite_util.h"

/** check for treelite runtime API errors and assert accordingly */
#define TREELITE_CHECK(call)                                                                     \
  do {                                                                                           \
    int status = call;                                                                           \
    ASSERT(status >= 0, "TREELITE FAIL: call='%s'. Reason:%s\n", #call, TreeliteGetLastError()); \
  } while (0)

namespace ML {

namespace tl = treelite;

namespace DT {

inline bool is_dev_ptr(const void* p)
{
  cudaPointerAttributes pointer_attr;
  cudaError_t err = cudaPointerGetAttributes(&pointer_attr, p);
  if (err == cudaSuccess) {
    return pointer_attr.devicePointer;
  } else {
    err = cudaGetLastError();
    return false;
  }
}

template <class T, class L>
std::string get_node_text(const std::string& prefix,
                          const std::vector<SparseTreeNode<T, L>>& sparsetree,
                          int idx,
                          bool isLeft)
{
  const SparseTreeNode<T, L>& node = sparsetree[idx];

  std::ostringstream oss;

  // print the value of the node
  std::stringstream ss;
  ss << prefix.c_str();
  ss << (isLeft ? "├" : "└");
  ss << node;

  oss << ss.str();

  if ((node.colid != -1)) {
    // enter the next tree level - left and right branch
    oss << "\n"
        << get_node_text(prefix + (isLeft ? "│   " : "    "), sparsetree, node.left_child_id, true)
        << "\n"
        << get_node_text(
             prefix + (isLeft ? "│   " : "    "), sparsetree, node.left_child_id + 1, false);
  }
  return oss.str();
}

template <typename T>
std::string to_string_high_precision(T x)
{
  static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
                "T must be float, double, or integer");
  std::ostringstream oss;
  oss.imbue(std::locale::classic());  // use C locale
  if (std::is_floating_point<T>::value) {
    oss << std::setprecision(std::numeric_limits<T>::max_digits10) << x;
  } else {
    oss << x;
  }
  return oss.str();
}

template <class T, class L>
std::string get_node_json(const std::string& prefix,
                          const std::vector<SparseTreeNode<T, L>>& sparsetree,
                          int idx)
{
  const SparseTreeNode<T, L>& node = sparsetree[idx];

  std::ostringstream oss;
  if ((node.colid != -1)) {
    oss << prefix << "{\"nodeid\": " << idx << ", \"split_feature\": " << node.colid
        << ", \"split_threshold\": " << to_string_high_precision(node.quesval)
        << ", \"gain\": " << to_string_high_precision(node.best_metric_val);
    if (node.instance_count != UINT32_MAX) {
      oss << ", \"instance_count\": " << node.instance_count;
    }
    oss << ", \"yes\": " << node.left_child_id << ", \"no\": " << (node.left_child_id + 1)
        << ", \"children\": [\n";
    // enter the next tree level - left and right branch
    oss << get_node_json(prefix + "  ", sparsetree, node.left_child_id) << ",\n"
        << get_node_json(prefix + "  ", sparsetree, node.left_child_id + 1) << "\n"
        << prefix << "]}";
  } else {
    oss << prefix << "{\"nodeid\": " << idx
        << ", \"leaf_value\": " << to_string_high_precision(node.prediction);
    if (node.instance_count != UINT32_MAX) {
      oss << ", \"instance_count\": " << node.instance_count;
    }
    oss << "}";
  }
  return oss.str();
}

template <typename T, typename L>
std::ostream& operator<<(std::ostream& os, const SparseTreeNode<T, L>& node)
{
  if (node.colid == -1) {
    os << "(leaf, "
       << "prediction: " << node.prediction << ", best_metric_val: " << node.best_metric_val << ")";
  } else {
    os << "("
       << "colid: " << node.colid << ", quesval: " << node.quesval
       << ", best_metric_val: " << node.best_metric_val << ")";
  }
  return os;
}

template <class T, class L>
tl::Tree<T, T> build_treelite_tree(const DT::TreeMetaDataNode<T, L>& rf_tree,
                                   unsigned int num_class,
                                   std::vector<Node_ID_info<T, L>>& cur_level_queue,
                                   std::vector<Node_ID_info<T, L>>& next_level_queue)
{
  tl::Tree<T, T> tl_tree;
  tl_tree.Init();

  // Track head and tail of bounded "queues" (implemented as vectors for
  // performance)
  size_t cur_front  = 0;
  size_t cur_end    = 0;
  size_t next_front = 0;
  size_t next_end   = 0;

  cur_level_queue.resize(std::max<size_t>(cur_level_queue.size(), 1));
  cur_level_queue[0] = Node_ID_info<T, L>(rf_tree.sparsetree[0], 0);
  ++cur_end;

  while (cur_front != cur_end) {
    size_t cur_level_size = cur_end - cur_front;
    next_level_queue.resize(std::max(2 * cur_level_size, next_level_queue.size()));

    for (size_t i = 0; i < cur_level_size; ++i) {
      Node_ID_info<T, L> q_node = cur_level_queue[cur_front];
      ++cur_front;

      bool is_leaf_node = q_node.node->colid == -1;
      int node_id       = q_node.unique_node_id;

      if (!is_leaf_node) {
        tl_tree.AddChilds(node_id);

        // Push left child to next_level queue.
        next_level_queue[next_end] = Node_ID_info<T, L>(
          rf_tree.sparsetree[q_node.node->left_child_id], tl_tree.LeftChild(node_id));
        ++next_end;

        // Push right child to next_level queue.
        next_level_queue[next_end] = Node_ID_info<T, L>(
          rf_tree.sparsetree[q_node.node->left_child_id + 1], tl_tree.RightChild(node_id));
        ++next_end;

        // Set node from current level as numerical node. Children IDs known.
        tl_tree.SetNumericalSplit(
          node_id, q_node.node->colid, q_node.node->quesval, true, tl::Operator::kLE);

      } else {
        if (num_class == 1) {
          tl_tree.SetLeaf(node_id, static_cast<T>(q_node.node->prediction));
        } else {
          std::vector<T> leaf_vector(num_class, 0);
          leaf_vector[q_node.node->prediction] = 1;
          tl_tree.SetLeafVector(node_id, leaf_vector);
        }
      }
    }

    cur_level_queue.swap(next_level_queue);
    cur_front  = next_front;
    cur_end    = next_end;
    next_front = 0;
    next_end   = 0;
  }
  return tl_tree;
}

struct DataInfo {
  unsigned int NLocalrows;
  unsigned int NGlobalrows;
  unsigned int Ncols;
};

template <class T, class L>
class DecisionTree {
 protected:
  DataInfo dinfo;
  int depth_counter   = 0;
  int leaf_counter    = 0;
  int n_unique_labels = -1;  // number of unique labels in dataset
  double prepare_time = 0;
  double train_time   = 0;
  MLCommon::TimerCPU prepare_fit_timer;
  DecisionTreeParams tree_params;

 public:
  /**
   * @brief Fits a DecisionTree on given input data and labels
   * @param[in] handle             cuML handle
   * @param[in] data               pointer to input training data
   * @param[in] ncols              number of features (columns)
   * @param[in] nrows              number of samples (rows)
   * @param[in] labels             pointer to label data
   * @param[in] rowids             pointer to array of row indices mapping to data
   * @param[in] n_sampled_rows     count of rows sampled
   * @param[in] unique_labels      count of unique labels
   * @param[in] is_classifier      true if task is classification, else false
   * @param[in,out] tree           pointer to tree structure
   * @param[in] tree_parameters    structure of tree parameters
   * @param[in] seed               random seed
   * @param[in] d_global_quantiles device pointer to global quantiles
   */
  void fit(const raft::handle_t& handle,
           const T* data,
           const int ncols,
           const int nrows,
           const L* labels,
           unsigned int* rowids,
           const int n_sampled_rows,
           int unique_labels,
           DT::TreeMetaDataNode<T, L>*& tree,
           DecisionTreeParams tree_parameters,
           uint64_t seed,
           T* d_global_quantiles)
  {
    this->tree_params = tree_parameters;
    this->prepare_fit_timer.reset();
    const char* CRITERION_NAME[] = {"GINI", "ENTROPY", "MSE", "MAE", "POISSON", "END"};
    CRITERION default_criterion =
      (std::numeric_limits<L>::is_integer) ? CRITERION::GINI : CRITERION::MSE;
    CRITERION last_criterion =
      (std::numeric_limits<L>::is_integer) ? CRITERION::ENTROPY : CRITERION::POISSON;

    validity_check(tree_params);

    if (tree_params.split_criterion ==
        CRITERION::CRITERION_END) {  // Set default to GINI (classification) or MSE (regression)
      tree_params.split_criterion = default_criterion;
    }
    ASSERT((tree_params.split_criterion >= default_criterion) &&
             (tree_params.split_criterion <= last_criterion),
           "Unsupported criterion %s\n",
           CRITERION_NAME[tree_params.split_criterion]);

    dinfo.NLocalrows   = nrows;
    dinfo.NGlobalrows  = nrows;
    dinfo.Ncols        = ncols;
    n_unique_labels    = unique_labels;
    this->prepare_time = this->prepare_fit_timer.getElapsedMilliseconds();
    prepare_fit_timer.reset();
    grow_tree(handle,
              data,
              tree->treeid,
              seed,
              ncols,
              nrows,
              labels,
              d_global_quantiles,
              (int*)rowids,
              n_sampled_rows,
              unique_labels,
              tree_params,
              tree->sparsetree,
              this->leaf_counter,
              this->depth_counter);
    this->train_time = this->prepare_fit_timer.getElapsedMilliseconds();
    this->set_metadata(tree);
  }

  /**
   * @brief Print high-level tree information.
   */
  void print_tree_summary() const
  {
    PatternSetter _("%v");
    CUML_LOG_DEBUG(" Decision Tree depth --> %d and n_leaves --> %d", depth_counter, leaf_counter);
    CUML_LOG_DEBUG(" Tree Fitting - Overall time --> %lf milliseconds", prepare_time + train_time);
    CUML_LOG_DEBUG("   - preparing for fit time: %lf milliseconds", prepare_time);
    CUML_LOG_DEBUG("   - tree growing time: %lf milliseconds", train_time);
  }

  /**
   * @brief Print detailed tree information.
   * @param[in] sparsetree: Sparse tree strcut
   */
  void print(const std::vector<SparseTreeNode<T, L>>& sparsetree) const
  {
    DecisionTree<T, L>::print_tree_summary();
    get_node_text<T, L>("", sparsetree, 0, false);
  }

  void predict(const raft::handle_t& handle,
               const DT::TreeMetaDataNode<T, L>* tree,
               const T* rows,
               const int n_rows,
               const int n_cols,
               L* predictions,
               int verbosity) const
  {
    if (verbosity >= 0) { ML::Logger::get().setLevel(verbosity); }
    ASSERT(!is_dev_ptr(rows) && !is_dev_ptr(predictions),
           "DT Error: Current impl. expects both input and predictions to be CPU "
           "pointers.\n");

    ASSERT(tree && (tree->sparsetree.size() != 0),
           "Cannot predict w/ empty tree, tree size %zu",
           tree->sparsetree.size());
    ASSERT((n_rows > 0), "Invalid n_rows %d", n_rows);
    ASSERT((n_cols > 0), "Invalid n_cols %d", n_cols);

    predict_all(tree, rows, n_rows, n_cols, predictions);
  }

  void predict_all(const DT::TreeMetaDataNode<T, L>* tree,
                   const T* rows,
                   const int n_rows,
                   const int n_cols,
                   L* preds) const
  {
    for (int row_id = 0; row_id < n_rows; row_id++) {
      preds[row_id] = predict_one(&rows[row_id * n_cols], tree->sparsetree, 0);
    }
  }

  L predict_one(const T* row, const std::vector<SparseTreeNode<T, L>> sparsetree, int idx) const
  {
    int colid     = sparsetree[idx].colid;
    T quesval     = sparsetree[idx].quesval;
    int leftchild = sparsetree[idx].left_child_id;
    if (colid == -1) {
      CUML_LOG_DEBUG("Leaf node. Predicting %f", (float)sparsetree[idx].prediction);
      return sparsetree[idx].prediction;
    } else if (row[colid] <= quesval) {
      CUML_LOG_DEBUG("Classifying Left @ node w/ column %d and value %f", colid, (float)quesval);
      return predict_one(row, sparsetree, leftchild);
    } else {
      CUML_LOG_DEBUG("Classifying Right @ node w/ column %d and value %f", colid, (float)quesval);
      return predict_one(row, sparsetree, leftchild + 1);
    }
  }

  void set_metadata(DT::TreeMetaDataNode<T, L>*& tree)
  {
    tree->depth_counter = depth_counter;
    tree->leaf_counter  = leaf_counter;
    tree->train_time    = train_time;
    tree->prepare_time  = prepare_time;
  }

};  // End DecisionTree Class

// Class specializations
template class DecisionTree<float, int>;
template class DecisionTree<float, float>;
template class DecisionTree<double, int>;
template class DecisionTree<double, double>;

template tl::Tree<float, float> build_treelite_tree<float, int>(
  const DT::TreeMetaDataNode<float, int>& rf_tree,
  unsigned int num_class,
  std::vector<Node_ID_info<float, int>>& working_queue_1,
  std::vector<Node_ID_info<float, int>>& working_queue_2);
template tl::Tree<double, double> build_treelite_tree<double, int>(
  const DT::TreeMetaDataNode<double, int>& rf_tree,
  unsigned int num_class,
  std::vector<Node_ID_info<double, int>>& working_queue_1,
  std::vector<Node_ID_info<double, int>>& working_queue_2);
template tl::Tree<float, float> build_treelite_tree<float, float>(
  const DT::TreeMetaDataNode<float, float>& rf_tree,
  unsigned int num_class,
  std::vector<Node_ID_info<float, float>>& working_queue_1,
  std::vector<Node_ID_info<float, float>>& working_queue_2);
template tl::Tree<double, double> build_treelite_tree<double, double>(
  const DT::TreeMetaDataNode<double, double>& rf_tree,
  unsigned int num_class,
  std::vector<Node_ID_info<double, double>>& working_queue_1,
  std::vector<Node_ID_info<double, double>>& working_queue_2);

}  // End namespace DT

}  // End namespace ML
