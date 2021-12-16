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

// We want to define some functions as usable on device
// But need to guard against this file being compiled by a host compiler
#ifdef __CUDACC__
#define FLATNODE_HD __host__ __device__
#else
#define FLATNODE_HD
#endif

/**
 * A node in Decision Tree.
 * @tparam T data type
 * @tparam L label type
 * @tparam IdxT type used for indexing operations
 */
template <typename DataT, typename LabelT, typename IdxT = int>
struct SparseTreeNode {
 private:
  IdxT colid            = 0;
  DataT quesval         = DataT(0);
  DataT best_metric_val = DataT(0);
  IdxT left_child_id    = -1;
  IdxT instance_count   = 0;
  FLATNODE_HD SparseTreeNode(
    IdxT colid, DataT quesval, DataT best_metric_val, int64_t left_child_id, IdxT instance_count)
    : colid(colid),
      quesval(quesval),
      best_metric_val(best_metric_val),
      left_child_id(left_child_id),
      instance_count(instance_count)
  {
  }

 public:
  FLATNODE_HD IdxT ColumnId() const { return colid; }
  FLATNODE_HD DataT QueryValue() const { return quesval; }
  FLATNODE_HD DataT BestMetric() const { return best_metric_val; }
  FLATNODE_HD int64_t LeftChildId() const { return left_child_id; }
  FLATNODE_HD int64_t RightChildId() const { return left_child_id + 1; }
  FLATNODE_HD IdxT InstanceCount() const { return instance_count; }

  FLATNODE_HD static SparseTreeNode CreateSplitNode(
    IdxT colid, DataT quesval, DataT best_metric_val, int64_t left_child_id, IdxT instance_count)
  {
    return SparseTreeNode<DataT, LabelT>{
      colid, quesval, best_metric_val, left_child_id, instance_count};
  }
  FLATNODE_HD static SparseTreeNode CreateLeafNode(IdxT instance_count)
  {
    return SparseTreeNode<DataT, LabelT>{0, 0, 0, -1, instance_count};
  }
  FLATNODE_HD bool IsLeaf() const { return left_child_id == -1; }
  bool operator==(const SparseTreeNode& other) const
  {
    return (this->colid == other.colid) && (this->quesval == other.quesval) &&
           (this->best_metric_val == other.best_metric_val) &&
           (this->left_child_id == other.left_child_id) &&
           (this->instance_count == other.instance_count);
  }
};
