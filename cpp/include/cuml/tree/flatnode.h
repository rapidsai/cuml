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
/**
 * A node in Decision Tree.
 * This however uses an index instead of pointer to left child. Right child
 * index is assumed to be `left_child_id + 1`
 * @tparam T data type
 * @tparam L label type
 * @tparam IdxT type used for indexing operations
 */
template <typename DataT, typename LabelT, typename IdxT = int>
struct SparseTreeNode {
  LabelT prediction       = LabelT(0);
  IdxT colid              = IdxT(-1);
  DataT quesval           = DataT(0);
  DataT best_metric_val   = DataT(0);
  IdxT left_child_id      = IdxT(-1);
  uint32_t instance_count = UINT32_MAX;  // UINT32_MAX indicates n/a
  bool IsLeaf() const { return left_child_id == -1; }
};

template <typename DataT, typename LabelT, typename IdxT>
bool operator==(const SparseTreeNode<DataT, LabelT, IdxT>& lhs,
                const SparseTreeNode<DataT, LabelT, IdxT>& rhs)
{
  return (lhs.prediction == rhs.prediction) && (lhs.colid == rhs.colid) &&
         (lhs.quesval == rhs.quesval) && (lhs.best_metric_val == rhs.best_metric_val) &&
         (lhs.left_child_id == rhs.left_child_id) && (lhs.instance_count == rhs.instance_count);
}
