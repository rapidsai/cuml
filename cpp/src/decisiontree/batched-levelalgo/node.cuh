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

#include <cuml/tree/flatnode.h>
#include "split.cuh"

namespace ML {
namespace DecisionTree {

/**
 * All info pertaining to a node in the decision tree.
 * @tparam DataT data type
 * @tparam LabelT label type
 * @tparam IdxT indexing type
 */
template <typename DataT, typename LabelT, typename IdxT>
struct Node {
  typedef Node<DataT, LabelT, IdxT> NodeT;
  typedef Split<DataT, IdxT> SplitT;

  /** special value to represent the leaf node */
  static constexpr IdxT Leaf = static_cast<IdxT>(-1);

  /** node related public information */
  SparseTreeNode<DataT, LabelT, IdxT> info;
  /** parent gain */
  DataT parentGain;
  /** start of sampled rows belonging to this node */
  IdxT start;
  /** number of sampled rows belonging to this node */
  IdxT end;
  /** depth of this node */
  IdxT depth;

  /**
   * @brief Makes this node as a leaf. Side effect of this is that it atomically
   *        updates the number of leaves counter
   * @param n_leaves global memory location tracking the total number of leaves
   *                        created in the tree so far
   * @param pred the prediction for this leaf node
   * @note to be called only by one thread across all participating threadblocks
   */
  DI void makeLeaf(IdxT* n_leaves, LabelT pred) volatile {
    info.left_child_id = Leaf;
    info.colid = Leaf;
    info.prediction = pred;
    atomicAdd(n_leaves, 1);
    __threadfence();
  }

  /**
   * @brief create left/right child nodes
   * @param n_nodes number of nodes created in current kernel launch
   * @param total_nodes total nodes created so far across all levels
   * @param nodes the list of nodes
   * @param splits split info for current node
   * @param n_depth max depth of the created tree so far
   * @return the position of the left child node in the above list
   * @note to be called only by one thread across all participating threadblocks
   */
  DI IdxT makeChildNodes(IdxT* n_nodes, IdxT total_nodes, volatile NodeT* nodes,
                         const SplitT& split, IdxT* n_depth) volatile {
    IdxT pos = atomicAdd(n_nodes, 2);
    // current
    info.colid = split.colid;
    info.quesval = split.quesval;
    info.best_metric_val = split.best_metric_val;
    info.left_child_id = total_nodes + pos;
    // left
    nodes[pos].parentGain = split.best_metric_val;
    nodes[pos].depth = depth + 1;
    nodes[pos].start = start;
    nodes[pos].end = split.nLeft;
    // right
    ++pos;
    nodes[pos].parentGain = split.best_metric_val;
    nodes[pos].depth = depth + 1;
    nodes[pos].start = start + split.nLeft;
    nodes[pos].end = end - split.nLeft;
    // update depth
    auto val = atomicMax(n_depth, depth + 1);
    __threadfence();
    return pos;
  }
};  // end Node

}  // namespace DecisionTree
}  // namespace ML
