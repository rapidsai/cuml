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

#include <cuml/tree/flatnode.h>
#include "split.cuh"

namespace ML {
namespace DecisionTree {

/**
 * @brief All info pertaining to a node in the decision tree.
 *
 * @tparam DataT  data type
 * @tparam LabelT label type
 * @tparam IdxT   indexing type
 */
template <typename DataT, typename LabelT, typename IdxT>
struct Node {
  typedef Node<DataT, LabelT, IdxT> NodeT;
  typedef Split<DataT, IdxT> SplitT;

  /** special value to represent the leaf node */
  static constexpr IdxT Leaf = static_cast<IdxT>(-1);

  /** node related public information */
  SparseTreeNode<DataT, LabelT, IdxT> info;
  /** start of sampled rows belonging to this node */
  IdxT start;
  /** number of sampled rows belonging to this node */
  IdxT count;
  /** depth of this node */
  IdxT depth;

  /**
   * @brief Initialize the underlying sparse tree node struct
   */
  HDI void initSpNode() volatile {
    info.prediction = LabelT(0);
    info.colid = Leaf;
    info.quesval = DataT(0);
    info.best_metric_val = DataT(0);
    info.left_child_id = Leaf;
  }

  /**
   * @brief Makes this node as a leaf. Side effect of this is that it atomically
   *        updates the number of leaves counter
   *
   * @param[inout] n_leaves global memory location tracking the total number of
   *                        leaves created in the tree so far
   * @param[in]    pred     the prediction for this leaf node
   *
   * @note to be called only by one thread across all participating threadblocks
   */
  DI void makeLeaf(IdxT* n_leaves, LabelT pred) volatile {
    info.prediction = pred;
    info.colid = Leaf;
    info.quesval = DataT(0);          // don't care for leaf nodes
    info.best_metric_val = DataT(0);  // don't care for leaf nodes
    info.left_child_id = Leaf;
    atomicAdd(n_leaves, 1);
    __threadfence();
  }

  /**
   * @brief create left/right child nodes
   *
   * @param[inout] n_nodes     number of nodes created in current kernel launch
   * @param[in]    total_nodes total nodes created so far across all levels
   * @param[out]   nodes       the list of nodes
   * @param[in]    splits      split info for current node
   * @param[inout] n_depth     max depth of the created tree so far
   *
   * @return the position of the left child node in the above list
   *
   * @note to be called only by one thread across all participating threadblocks
   */
  DI IdxT makeChildNodes(IdxT* n_nodes, IdxT total_nodes, volatile NodeT* nodes,
                         const SplitT& split, IdxT* n_depth) volatile {
    IdxT pos = atomicAdd(n_nodes, 2);
    // current
    info.prediction = LabelT(0);  // don't care for non-leaf nodes
    info.colid = split.colid;
    info.quesval = split.quesval;
    info.best_metric_val = split.best_metric_val;
    info.left_child_id = total_nodes + pos;
    // left
    nodes[pos].initSpNode();
    nodes[pos].depth = depth + 1;
    nodes[pos].start = start;
    nodes[pos].count = split.nLeft;
    nodes[pos].info.unique_id = 2 * info.unique_id + 1;
    // right
    ++pos;
    nodes[pos].initSpNode();
    nodes[pos].depth = depth + 1;
    nodes[pos].start = start + split.nLeft;
    nodes[pos].count = count - split.nLeft;
    nodes[pos].info.unique_id = 2 * info.unique_id + 2;
    // update depth
    auto val = atomicMax(n_depth, depth + 1);
    __threadfence();
    return pos;
  }
};  // end Node

template <typename DataT, typename LabelT, typename IdxT, int TPB = 256>
void printNodes(Node<DataT, LabelT, IdxT>* nodes, IdxT len, cudaStream_t s) {
  auto op = [] __device__(Node<DataT, LabelT, IdxT> * ptr, IdxT idx) {
    printf(
      "prediction = %d, colid = %d, quesval = %f, best_metric_val = %f, "
      "left_child_id = %d, start = %d, count = %d, depth = %d\n",
      ptr->info.prediction, ptr->info.colid, ptr->info.quesval,
      ptr->info.best_metric_val, ptr->info.left_child_id, ptr->start,
      ptr->count, ptr->depth);
  };
  raft::linalg::writeOnlyUnaryOp<Node<DataT, LabelT, IdxT>, decltype(op), IdxT,
                                 TPB>(nodes, len, op, s);
  CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace DecisionTree
}  // namespace ML
