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
#include <cstddef>
#include <type_traits>
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
  template <typename vector_output_t>
  using raw_output_type = std::conditional_t<
      !std::is_same_v<vector_output_t, std::nullptr_t>,
      std::remove_pointer_t<vector_output_t>,
      typename node_type::threshold_type
  >;

  HOST DEVICE forest(
      node_type* forest_nodes,
      index_type* forest_root_indexes,
      index_type* node_id_mapping,
      index_type num_trees) :
    nodes_{forest_nodes},
    root_node_indexes_{forest_root_indexes},
    node_id_mapping_{node_id_mapping},
    num_trees_{num_trees} {}

  /* Return pointer to the root node of the indicated tree */
  HOST DEVICE auto* get_tree_root(index_type tree_index) const {
    return nodes_ + root_node_indexes_[tree_index];
  }

  HOST DEVICE auto get_node_id(const node_type* node) const {
    return node_id_mapping_[node - nodes_];
  }

  /* Return the number of trees in this forest */
  HOST DEVICE auto tree_count() const {
    return num_trees_;
  }
 private:
  node_type* nodes_;
  index_type* root_node_indexes_;
  index_type* node_id_mapping_;
  index_type num_trees_;
};

}
}
}
