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
namespace DT {

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

  Node() = default;  // to be removed
  Node(IdxT row_start, IdxT row_count, IdxT depth)
    : start(row_start), count(row_count), depth(depth)
  {
  }

  static Node CreateChild(IdxT depth, IdxT start_sample_range, IdxT sample_count, IdxT unique_id)
  {
    Node n;
    n.depth          = depth;
    n.start          = start_sample_range;
    n.count          = sample_count;
    n.info.unique_id = unique_id;
    return n;
  }

  static Node CreateSplit(IdxT colid, IdxT query_value, IdxT best_metric_val, IdxT left_child_id)
  {
    Node n;
    n.info.prediction      = LabelT(0);  // don't care for non-leaf nodes
    n.info.colid           = colid;
    n.info.quesval         = query_value;
    n.info.best_metric_val = best_metric_val;
    n.info.left_child_id   = left_child_id;
    return n;
  }

  HDI void makeLeaf(LabelT pred)
  {
    info.prediction      = pred;
    info.colid           = Leaf;
    info.quesval         = DataT(0);  // don't care for leaf nodes
    info.best_metric_val = DataT(0);  // don't care for leaf nodes
    info.left_child_id   = Leaf;
  }

  HDI bool IsLeaf() { return info.left_child_id == -1; }
};  // end Node

template <typename DataT, typename LabelT, typename IdxT, int TPB = 256>
void printNodes(Node<DataT, LabelT, IdxT>* nodes, IdxT len, cudaStream_t s)
{
  auto op = [] __device__(Node<DataT, LabelT, IdxT> * ptr, IdxT idx) {
    printf(
      "prediction = %d, colid = %d, quesval = %f, best_metric_val = %f, "
      "left_child_id = %d, start = %d, count = %d, depth = %d\n",
      ptr->info.prediction,
      ptr->info.colid,
      ptr->info.quesval,
      ptr->info.best_metric_val,
      ptr->info.left_child_id,
      ptr->start,
      ptr->count,
      ptr->depth);
  };
  raft::linalg::writeOnlyUnaryOp<Node<DataT, LabelT, IdxT>, decltype(op), IdxT, TPB>(
    nodes, len, op, s);
  CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace DT
}  // namespace ML
