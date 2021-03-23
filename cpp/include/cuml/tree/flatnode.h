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
  LabelT prediction;
  IdxT colid = IdxT(-1);
  DataT quesval;
  DataT best_metric_val;
  IdxT left_child_id = IdxT(-1);
  uint32_t unique_id = UINT32_MAX;
  uint32_t instance_count = UINT32_MAX;  // UINT32_MAX indicates n/a
};

template <typename T, typename L>
struct Node_ID_info {
  const SparseTreeNode<T, L>* node;
  int unique_node_id;

  Node_ID_info() : node(nullptr), unique_node_id(-1) {}
  Node_ID_info(const SparseTreeNode<T, L>& cfg_node, int cfg_unique_node_id)
    : node(&cfg_node), unique_node_id(cfg_unique_node_id) {}
};
