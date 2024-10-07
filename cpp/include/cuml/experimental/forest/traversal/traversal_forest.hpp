/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <queue>
#include <stack>
#include <cuml/experimental/forest/traversal/traversal_order.hpp>
#include <cuml/experimental/forest/traversal/traversal_node.hpp>

namespace ML {
namespace experimental {
namespace forest {

/** A template for storing nodes in order to traverse them in the
 * indicated order */
template <forest_order order, typename T>
struct traversal_container {
  using backing_container_t = std::conditional_t<
    order == forest_order::depth_first,
     std::stack<T>,
     std::queue<T>
  >;
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

template <typename node_t = traversal_node<std::size_t>, typename tree_id_t = std::size_t>
struct traversal_forest {
 public:
  using node_type = node_t;
  using node_id_type = typename node_type::id_type;
  using tree_id_type = tree_id_t;
  virtual node_type const& get_node(tree_id_type tree_id, node_id_type node_id) const {
    // Default assumption is that the node id represents its index in the
    // overall forest.
    return nodes_[node_id];
  }

  template <forest_order order, typename lambda_t>
  void for_each(lambda_t&& lambda) {
    auto to_be_visited = traversal_container<
      order,
      node_uid_type
    >{};
    if constexpr (
      order == forest_order::depth_first ||
      order == forest_order::breadth_first
    ) {
      for (auto const& root_node_uid : root_node_uids_) {
        to_be_visited.add(root_node_uid);
        while(!to_be_visited.empty()) {
          auto node_uid = to_be_visited.next();
          auto const& node = get_node(node_uid);
          lambda(node_uid.first, node);
          if (!node.is_leaf()) {
            auto hot_uid = std::make_pair(
              node_uid.first,
              node.hot_child()
            );
            auto distant_uid = std::make_pair(
              node_uid.first,
              node.distant_child()
            );
            to_be_visited.add(hot_uid, distant_uid);
          }
        }
      }
    } else if constexpr (order == forest_order::layered_children_segregated) {
      for (auto const& root_node_uid : root_node_uids_) {
        to_be_visited.add(root_node_uid);
      }
      auto depth = std::size_t{};
      while(!to_be_visited.empty()) {
        auto layer_node_uids = std::vector<node_uid_type>{};
        while(!to_be_visited.empty()) {
          layer_node_uids.push_back(to_be_visited.next());
        }
        ++depth;
        for (auto node_uid : layer_node_uids) {
          auto const& node = get_node(node_uid);
          lambda(node_uid.first, node, depth);
          if (!node.is_leaf()) {
            auto hot_uid = std::make_pair(
              node_uid.first,
              node.hot_child()
            );
            to_be_visited.add(hot_uid);
          }
        }
        for (auto node_uid : layer_node_uids) {
          auto const& node = get_node(node_uid);
          if (!node.is_leaf()) {
            auto distant_uid = std::make_pair(
              node_uid.first,
              node.distant_child()
            );
            to_be_visited.add(distant_uid);
          }
        }
      }
    } else if constexpr (order == forest_order::layered_children_together) {
      for (auto const& root_node_uid : root_node_uids_) {
        to_be_visited.add(root_node_uid);
      }
      auto depth = std::size_t{};
      while(!to_be_visited.empty()) {
        auto layer_node_uids = std::vector<node_uid_type>{};
        while(!to_be_visited.empty()) {
          layer_node_uids.push_back(to_be_visited.next());
        }
        ++depth;
        for (auto node_uid : layer_node_uids) {
          auto const& node = get_node(node_uid);
          lambda(node_uid.first, node, depth);
          if (!node.is_leaf()) {
            auto hot_uid = std::make_pair(
              node_uid.first,
              node.hot_child()
            );
            auto distant_uid = std::make_pair(
              node_uid.first,
              node.distant_child()
            );
            to_be_visited.add(hot_uid, distant_uid);
          }
        }
      }
    }
  }
 protected:
  using node_uid_type = std::pair<tree_id_type, node_id_type>;

  auto const& get_node(node_uid_type node_uid) const {
    return get_node(node_uid.first, node_uid.second);
  }

  std::vector<node_type> nodes_{};
  std::vector<node_uid_type> root_node_uids_{};
};

}  // namespace forest
}  // namespace experimental
}  // namespace ML
