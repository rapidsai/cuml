/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <stddef.h>
#include <cuml/experimental/fil/detail/raft_proto/gpu_support.hpp>
#include <cuml/experimental/fil/detail/index_type.hpp>
#include <cuml/experimental/fil/detail/node.hpp>

namespace ML {
namespace experimental {
namespace fil {

/* A collection of trees which together form a forest model
 */
template <tree_layout layout_v, typename threshold_t, typename index_t, typename metadata_storage_t, typename offset_t>
struct forest {
  using node_type = node<layout_v, threshold_t, index_t, metadata_storage_t, offset_t>;
  using io_type = threshold_t;

  HOST DEVICE forest(node_type* forest_nodes, index_type* forest_root_indexes, index_type num_trees) :
    nodes_{forest_nodes}, root_node_indexes_{forest_root_indexes}, num_trees_{num_trees} {}

  /* Return pointer to the root node of the indicated tree */
  HOST DEVICE auto* get_tree_root(index_type tree_index) const {
    return nodes_ + root_node_indexes_[tree_index];
  }

  /* Return the number of trees in this forest */
  HOST DEVICE auto tree_count() const {
    return num_trees_;
  }
 private:
  node_type* nodes_;
  index_type* root_node_indexes_;
  index_type num_trees_;
};

}
}
}
