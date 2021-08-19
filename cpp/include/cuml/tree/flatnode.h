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
 */
template <typename DataT, typename LabelT>
struct SparseTreeNode {
  LabelT prediction          = LabelT(0);
  std::size_t colid          = 0;
  DataT quesval              = DataT(0);
  DataT best_metric_val      = DataT(0);
  int64_t left_child_id      = -1;
  std::size_t instance_count = 0;
  bool IsLeaf() const { return left_child_id == -1; }
};

template <typename DataT, typename LabelT>
bool operator==(const SparseTreeNode<DataT, LabelT>& lhs, const SparseTreeNode<DataT, LabelT>& rhs)
{
  return (lhs.prediction == rhs.prediction) && (lhs.colid == rhs.colid) &&
         (lhs.quesval == rhs.quesval) && (lhs.best_metric_val == rhs.best_metric_val) &&
         (lhs.left_child_id == rhs.left_child_id) && (lhs.instance_count == rhs.instance_count);
}
