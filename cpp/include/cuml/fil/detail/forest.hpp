/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
#include <cuml/fil/detail/index_type.hpp>
#include <cuml/fil/detail/node.hpp>
#include <cuml/fil/detail/raft_proto/gpu_support.hpp>

#include <stddef.h>

#include <type_traits>

namespace ML {
namespace fil {

/* A collection of trees which together form a forest model
 */
template <tree_layout layout_v,
          typename threshold_t,
          typename index_t,
          typename metadata_storage_t,
          typename offset_t>
struct forest {
  using node_type = node<layout_v, threshold_t, index_t, metadata_storage_t, offset_t>;
  using io_type   = threshold_t;
  template <typename vector_output_t>
  using raw_output_type = std::conditional_t<!std::is_same_v<vector_output_t, std::nullptr_t>,
                                             std::remove_pointer_t<vector_output_t>,
                                             typename node_type::threshold_type>;

  HOST DEVICE forest(node_type* forest_nodes,
                     index_type* forest_root_indexes,
                     index_type* node_id_mapping,
                     index_type num_trees,
                     index_type num_outputs)
    : nodes_{forest_nodes},
      root_node_indexes_{forest_root_indexes},
      node_id_mapping_{node_id_mapping},
      num_trees_{num_trees},
      num_outputs_{num_outputs}
  {
  }

  /* Return pointer to the root node of the indicated tree */
  HOST DEVICE auto* get_tree_root(index_type tree_index) const
  {
    return nodes_ + root_node_indexes_[tree_index];
  }

  /* Return pointer to the mapping from internal node IDs to final node ID outputs.
   * Only used when infer_type == infer_kind::leaf_id */
  HOST DEVICE const auto* get_node_id_mapping() const { return node_id_mapping_; }

  /* Return the number of trees in this forest */
  HOST DEVICE auto tree_count() const { return num_trees_; }

  /* Return the number of outputs per row for default evaluation of this
   * forest */
  HOST DEVICE auto num_outputs() const { return num_outputs_; }

 private:
  node_type* nodes_;
  index_type* root_node_indexes_;
  index_type* node_id_mapping_;
  index_type num_trees_;
  index_type num_outputs_;
};

}  // namespace fil
}  // namespace ML
