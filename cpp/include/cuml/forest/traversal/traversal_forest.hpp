/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include <cuml/forest/traversal/traversal_node.hpp>
#include <cuml/forest/traversal/traversal_order.hpp>

#include <cstddef>
#include <queue>
#include <stack>

namespace ML {
namespace forest {

namespace detail {
/** A template for storing nodes in order to traverse them in the
 * indicated order */
template <forest_order order, typename T>
struct traversal_container {
  using backing_container_t =
    std::conditional_t<order == forest_order::depth_first, std::stack<T>, std::queue<T>>;
  void add(T const& val) { data_.push(val); }
  void add(T const& hot, T const& distant)
  {
    if constexpr (order == forest_order::depth_first) {
      data_.push(distant);
      data_.push(hot);
    } else {
      data_.push(hot);
      data_.push(distant);
    }
  }
  auto next()
  {
    if constexpr (std::is_same_v<backing_container_t, std::stack<T>>) {
      auto result = data_.top();
      data_.pop();
      return result;
    } else {
      auto result = data_.front();
      data_.pop();
      return result;
    }
  }
  auto peek()
  {
    if constexpr (std::is_same_v<backing_container_t, std::stack<T>>) {
      return data_.top();
    } else {
      return data_.front();
    }
  }
  [[nodiscard]] auto empty() { return data_.empty(); }
  auto size() { return data_.size(); }

 private:
  backing_container_t data_;
};
}  // namespace detail
   //

template <typename node_t = traversal_node<std::size_t>, typename tree_id_t = std::size_t>
struct traversal_forest {
  using node_type     = node_t;
  using node_id_type  = typename node_type::id_type;
  using tree_id_type  = tree_id_t;
  using node_uid_type = std::pair<tree_id_type, node_id_type>;
  using index_type    = std::size_t;

  virtual node_type get_node(tree_id_type tree_id, node_id_type node_id) const = 0;

  traversal_forest(std::vector<node_uid_type>&& root_node_uids) : root_node_uids_{root_node_uids} {}

  template <forest_order order, typename lambda_t>
  void for_each(lambda_t&& lambda) const
  {
    auto to_be_visited = detail::traversal_container<
      order,
      std::conditional_t<order == forest_order::layered_children_segregated ||
                           order == forest_order::layered_children_together,
                         // Layered traversals can track current depth without storing
                         // alongside each node. This can also be done with depth-first
                         // traversals, but we exchange memory footprint of the depth-first
                         // case for simplified code. By storing depth for both depth-first
                         // and breadth-first, we can make the code for each identical.
                         node_uid_type,
                         std::pair<node_uid_type, index_type>>>{};
    auto parent_indices = detail::traversal_container<order, index_type>{};
    auto cur_index      = index_type{};
    if constexpr (order == forest_order::depth_first || order == forest_order::breadth_first) {
      for (auto const& root_node_uid : root_node_uids_) {
        to_be_visited.add(std::make_pair(root_node_uid, std::size_t{}));
        parent_indices.add(cur_index);
        while (!to_be_visited.empty()) {
          auto [node_uid, depth] = to_be_visited.next();
          auto parent_index      = parent_indices.next();
          auto node              = get_node(node_uid);
          lambda(node_uid.first, node, depth, parent_index);
          if (!node.is_leaf()) {
            auto hot_uid     = std::make_pair(std::make_pair(node_uid.first, node.hot_child()),
                                          depth + index_type{1});
            auto distant_uid = std::make_pair(std::make_pair(node_uid.first, node.distant_child()),
                                              depth + index_type{1});
            to_be_visited.add(hot_uid, distant_uid);
            parent_indices.add(cur_index, cur_index);
          }
          ++cur_index;
        }
      }
    } else if constexpr (order == forest_order::layered_children_segregated) {
      for (auto const& root_node_uid : root_node_uids_) {
        to_be_visited.add(root_node_uid);
        parent_indices.add(cur_index++);
      }
      cur_index  = index_type{};
      auto depth = index_type{};
      while (!to_be_visited.empty()) {
        auto layer_node_uids      = std::vector<node_uid_type>{};
        auto layer_parent_indices = std::vector<index_type>{};
        while (!to_be_visited.empty()) {
          layer_node_uids.push_back(to_be_visited.next());
          layer_parent_indices.push_back(parent_indices.next());
        }
        for (auto layer_index = index_type{}; layer_index < layer_node_uids.size(); ++layer_index) {
          auto node_uid     = layer_node_uids[layer_index];
          auto parent_index = layer_parent_indices[layer_index];
          auto node         = get_node(node_uid);
          lambda(node_uid.first, node, depth, parent_index);
          if (!node.is_leaf()) {
            auto hot_uid = std::make_pair(node_uid.first, node.hot_child());
            to_be_visited.add(hot_uid);
            parent_indices.add(cur_index);
          }
          ++cur_index;
        }
        // Reset cur_index before iterating through distant nodes
        cur_index -= layer_node_uids.size();
        for (auto layer_index = index_type{}; layer_index < layer_node_uids.size(); ++layer_index) {
          auto node_uid = layer_node_uids[layer_index];
          auto node     = get_node(node_uid);
          if (!node.is_leaf()) {
            auto distant_uid = std::make_pair(node_uid.first, node.distant_child());
            to_be_visited.add(distant_uid);
            parent_indices.add(cur_index);
          }
          ++cur_index;
        }
        ++depth;
      }
    } else if constexpr (order == forest_order::layered_children_together) {
      for (auto const& root_node_uid : root_node_uids_) {
        to_be_visited.add(root_node_uid);
        parent_indices.add(cur_index++);
      }
      cur_index  = index_type{};
      auto depth = index_type{};
      while (!to_be_visited.empty()) {
        auto layer_node_uids      = std::vector<node_uid_type>{};
        auto layer_parent_indices = std::vector<index_type>{};
        while (!to_be_visited.empty()) {
          layer_node_uids.push_back(to_be_visited.next());
          layer_parent_indices.push_back(parent_indices.next());
        }
        for (auto layer_index = index_type{}; layer_index < layer_node_uids.size(); ++layer_index) {
          auto node_uid     = layer_node_uids[layer_index];
          auto parent_index = layer_parent_indices[layer_index];
          auto node         = get_node(node_uid);
          lambda(node_uid.first, node, depth, parent_index);
          if (!node.is_leaf()) {
            auto hot_uid     = std::make_pair(node_uid.first, node.hot_child());
            auto distant_uid = std::make_pair(node_uid.first, node.distant_child());
            to_be_visited.add(hot_uid, distant_uid);
            parent_indices.add(cur_index, cur_index);
          }
          ++cur_index;
        }
        ++depth;
      }
    }
  }

 private:
  auto get_node(node_uid_type node_uid) const { return get_node(node_uid.first, node_uid.second); }

  std::vector<node_uid_type> root_node_uids_{};
};

}  // namespace forest
}  // namespace ML
