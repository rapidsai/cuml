/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cuml/common/logger.hpp>
#include <cuml/tree/flatnode.h>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>

#include "treelite_util.h"
#include <treelite/c_api.h>
#include <treelite/tree.h>

#include <algorithm>
#include <climits>
#include <iomanip>
#include <locale>
#include <map>
#include <numeric>
#include <raft/common/nvtx.hpp>
#include <random>
#include <vector>

#include "batched-levelalgo/builder.cuh"
#include "batched-levelalgo/quantiles.cuh"

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
std::string get_node_text(const std::string& prefix,
                          const TreeMetaDataNode<T, L>* tree,
                          int idx,
                          bool isLeft)
{
  const SparseTreeNode<T, L>& node = tree->sparsetree[idx];

  std::ostringstream oss;

  // print the value of the node
  oss << prefix.c_str();
  oss << (isLeft ? "├" : "└");

  if (node.IsLeaf()) {
    oss << "(leaf, "
        << "prediction: [";

    for (int k = 0; k < tree->num_outputs - 1; k++) {
      oss << tree->vector_leaf[idx * tree->num_outputs + k] << ", ";
    }
    oss << tree->vector_leaf[idx * tree->num_outputs + tree->num_outputs - 1];

    oss << "], best_metric_val: " << node.BestMetric() << ")";
  } else {
    oss << "("
        << "colid: " << node.ColumnId() << ", quesval: " << node.QueryValue()
        << ", best_metric_val: " << node.BestMetric() << ")";
  }

  if (!node.IsLeaf()) {
    // enter the next tree level - left and right branch
    oss << "\n"
        << get_node_text(prefix + (isLeft ? "│   " : "    "), tree, node.LeftChildId(), true)
        << "\n"
        << get_node_text(prefix + (isLeft ? "│   " : "    "), tree, node.RightChildId(), false);
  }
  return oss.str();
}

template <class T, class L>
std::string get_node_json(const std::string& prefix, const TreeMetaDataNode<T, L>* tree, int idx)
{
  const SparseTreeNode<T, L>& node = tree->sparsetree[idx];

  std::ostringstream oss;
  if (!node.IsLeaf()) {
    oss << prefix << "{\"nodeid\": " << idx << ", \"split_feature\": " << node.ColumnId()
        << ", \"split_threshold\": " << to_string_high_precision(node.QueryValue())
        << ", \"gain\": " << to_string_high_precision(node.BestMetric());
    oss << ", \"instance_count\": " << node.InstanceCount();
    oss << ", \"yes\": " << node.LeftChildId() << ", \"no\": " << (node.RightChildId())
        << ", \"children\": [\n";
    // enter the next tree level - left and right branch
    oss << get_node_json(prefix + "  ", tree, node.LeftChildId()) << ",\n"
        << get_node_json(prefix + "  ", tree, node.RightChildId()) << "\n"
        << prefix << "]}";
  } else {
    oss << prefix << "{\"nodeid\": " << idx << ", \"leaf_value\": [";
    for (int k = 0; k < tree->num_outputs - 1; k++) {
      oss << to_string_high_precision(tree->vector_leaf[idx * tree->num_outputs + k]) << ", ";
    }
    oss << to_string_high_precision(
      tree->vector_leaf[idx * tree->num_outputs + tree->num_outputs - 1]);
    oss << "], \"instance_count\": " << node.InstanceCount();
    oss << "}";
  }
  return oss.str();
}

template <class T, class L>
tl::Tree<T, T> build_treelite_tree(const DT::TreeMetaDataNode<T, L>& rf_tree,
                                   unsigned int num_class)
{
  // First index refers to the cuml node id
  // Seccond refers to the tl node id
  using kv = std::pair<std::size_t, std::size_t>;
  std::vector<kv> cur_level_queue;
  std::vector<kv> next_level_queue;

  tl::Tree<T, T> tl_tree;
  tl_tree.Init();

  // Track head and tail of bounded "queues" (implemented as vectors for
  // performance)
  size_t cur_front  = 0;
  size_t cur_end    = 0;
  size_t next_front = 0;
  size_t next_end   = 0;

  cur_level_queue.resize(std::max<size_t>(cur_level_queue.size(), 1));
  cur_level_queue[0] = {0, 0};
  ++cur_end;

  while (cur_front != cur_end) {
    size_t cur_level_size = cur_end - cur_front;
    next_level_queue.resize(std::max(2 * cur_level_size, next_level_queue.size()));

    for (size_t i = 0; i < cur_level_size; ++i) {
      auto cuml_node_id                  = cur_level_queue[cur_front].first;
      const SparseTreeNode<T, L>& q_node = rf_tree.sparsetree[cuml_node_id];
      auto tl_node_id                    = cur_level_queue[cur_front].second;
      ++cur_front;

      if (!q_node.IsLeaf()) {
        tl_tree.AddChilds(tl_node_id);

        // Push left child to next_level queue.
        next_level_queue[next_end] = {q_node.LeftChildId(), tl_tree.LeftChild(tl_node_id)};
        ++next_end;

        // Push right child to next_level queue.
        next_level_queue[next_end] = {q_node.RightChildId(), tl_tree.RightChild(tl_node_id)};
        ++next_end;

        // Set node from current level as numerical node. Children IDs known.
        tl_tree.SetNumericalSplit(
          tl_node_id, q_node.ColumnId(), q_node.QueryValue(), true, tl::Operator::kLE);

      } else {
        auto leaf_begin = rf_tree.vector_leaf.begin() + cuml_node_id * num_class;
        if (num_class == 1) {
          tl_tree.SetLeaf(tl_node_id, *leaf_begin);
        } else {
          std::vector<T> leaf_vector(leaf_begin, leaf_begin + num_class);
          tl_tree.SetLeafVector(tl_node_id, leaf_vector);
        }
      }
      tl_tree.SetDataCount(tl_node_id, q_node.InstanceCount());
    }

    cur_level_queue.swap(next_level_queue);
    cur_front  = next_front;
    cur_end    = next_end;
    next_front = 0;
    next_end   = 0;
  }
  return tl_tree;
}

class DecisionTree {
 public:
  template <class DataT, class LabelT>
  static std::shared_ptr<DT::TreeMetaDataNode<DataT, LabelT>> fit(
    const raft::handle_t& handle,
    const cudaStream_t s,
    const DataT* data,
    const int ncols,
    const int nrows,
    const LabelT* labels,
    rmm::device_uvector<int>* row_ids,
    int unique_labels,
    DecisionTreeParams params,
    uint64_t seed,
    const Quantiles<DataT, int>& quantiles,
    int treeid)
  {
    if (params.split_criterion ==
        CRITERION::CRITERION_END) {  // Set default to GINI (classification) or MSE (regression)
      CRITERION default_criterion =
        (std::numeric_limits<LabelT>::is_integer) ? CRITERION::GINI : CRITERION::MSE;
      params.split_criterion = default_criterion;
    }
    using IdxT = int;
    // Dispatch objective
    if (not std::is_same<DataT, LabelT>::value and params.split_criterion == CRITERION::GINI) {
      return Builder<GiniObjectiveFunction<DataT, LabelT, IdxT>>(handle,
                                                                 s,
                                                                 treeid,
                                                                 seed,
                                                                 params,
                                                                 data,
                                                                 labels,
                                                                 nrows,
                                                                 ncols,
                                                                 row_ids,
                                                                 unique_labels,
                                                                 quantiles)
        .train();
    } else if (not std::is_same<DataT, LabelT>::value and
               params.split_criterion == CRITERION::ENTROPY) {
      return Builder<EntropyObjectiveFunction<DataT, LabelT, IdxT>>(handle,
                                                                    s,
                                                                    treeid,
                                                                    seed,
                                                                    params,
                                                                    data,
                                                                    labels,
                                                                    nrows,
                                                                    ncols,
                                                                    row_ids,
                                                                    unique_labels,
                                                                    quantiles)
        .train();
    } else if (std::is_same<DataT, LabelT>::value and params.split_criterion == CRITERION::MSE) {
      return Builder<MSEObjectiveFunction<DataT, LabelT, IdxT>>(handle,
                                                                s,
                                                                treeid,
                                                                seed,
                                                                params,
                                                                data,
                                                                labels,
                                                                nrows,
                                                                ncols,
                                                                row_ids,
                                                                unique_labels,
                                                                quantiles)
        .train();
    } else if (std::is_same<DataT, LabelT>::value and
               params.split_criterion == CRITERION::POISSON) {
      return Builder<PoissonObjectiveFunction<DataT, LabelT, IdxT>>(handle,
                                                                    s,
                                                                    treeid,
                                                                    seed,
                                                                    params,
                                                                    data,
                                                                    labels,
                                                                    nrows,
                                                                    ncols,
                                                                    row_ids,
                                                                    unique_labels,
                                                                    quantiles)
        .train();
    } else if (std::is_same<DataT, LabelT>::value and params.split_criterion == CRITERION::GAMMA) {
      return Builder<GammaObjectiveFunction<DataT, LabelT, IdxT>>(handle,
                                                                  s,
                                                                  treeid,
                                                                  seed,
                                                                  params,
                                                                  data,
                                                                  labels,
                                                                  nrows,
                                                                  ncols,
                                                                  row_ids,
                                                                  unique_labels,
                                                                  quantiles)
        .train();
    } else if (std::is_same<DataT, LabelT>::value and
               params.split_criterion == CRITERION::INVERSE_GAUSSIAN) {
      return Builder<InverseGaussianObjectiveFunction<DataT, LabelT, IdxT>>(handle,
                                                                            s,
                                                                            treeid,
                                                                            seed,
                                                                            params,
                                                                            data,
                                                                            labels,
                                                                            nrows,
                                                                            ncols,
                                                                            row_ids,
                                                                            unique_labels,
                                                                            quantiles)
        .train();
    } else {
      ASSERT(false, "Unknown split criterion.");
    }
  }

  template <class DataT, class LabelT>
  static void predict(const raft::handle_t& handle,
                      const DT::TreeMetaDataNode<DataT, LabelT>& tree,
                      const DataT* rows,
                      std::size_t n_rows,
                      std::size_t n_cols,
                      DataT* predictions,
                      int num_outputs,
                      int verbosity)
  {
    if (verbosity >= 0) { ML::Logger::get().setLevel(verbosity); }
    ASSERT(!is_dev_ptr(rows) && !is_dev_ptr(predictions),
           "DT Error: Current impl. expects both input and predictions to be CPU "
           "pointers.\n");

    ASSERT(tree.sparsetree.size() != 0,
           "Cannot predict w/ empty tree, tree size %zu",
           tree.sparsetree.size());

    predict_all(tree, rows, n_rows, n_cols, predictions, num_outputs);
  }

  template <class DataT, class LabelT>
  static void predict_all(const DT::TreeMetaDataNode<DataT, LabelT>& tree,
                          const DataT* rows,
                          std::size_t n_rows,
                          std::size_t n_cols,
                          DataT* preds,
                          int num_outputs)
  {
    for (std::size_t row_id = 0; row_id < n_rows; row_id++) {
      predict_one(&rows[row_id * n_cols], tree, preds + row_id * num_outputs, num_outputs);
    }
  }

  template <class DataT, class LabelT>
  static void predict_one(const DataT* row,
                          const DT::TreeMetaDataNode<DataT, LabelT>& tree,
                          DataT* preds_out,
                          int num_outputs)
  {
    std::size_t idx = 0;
    auto n          = tree.sparsetree[idx];
    while (!n.IsLeaf()) {
      if (row[n.ColumnId()] <= n.QueryValue()) {
        idx = n.LeftChildId();
      } else {
        idx = n.RightChildId();
      }
      n = tree.sparsetree[idx];
    }
    for (int i = 0; i < num_outputs; i++) {
      preds_out[i] += tree.vector_leaf[idx * num_outputs + i];
    }
  }

};  // End DecisionTree Class

// Class specializations
template tl::Tree<float, float> build_treelite_tree<float, int>(
  const DT::TreeMetaDataNode<float, int>& rf_tree, unsigned int num_class);
template tl::Tree<double, double> build_treelite_tree<double, int>(
  const DT::TreeMetaDataNode<double, int>& rf_tree, unsigned int num_class);
template tl::Tree<float, float> build_treelite_tree<float, float>(
  const DT::TreeMetaDataNode<float, float>& rf_tree, unsigned int num_class);
template tl::Tree<double, double> build_treelite_tree<double, double>(
  const DT::TreeMetaDataNode<double, double>& rf_tree, unsigned int num_class);
}  // End namespace DT

}  // End namespace ML
