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
#include <cuml/forest/exceptions.hpp>
#include <cuml/forest/traversal/traversal_forest.hpp>
#include <cuml/forest/traversal/traversal_node.hpp>

#include <treelite/tree.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace ML {
namespace forest {

using TREELITE_NODE_ID_T = int;

template <typename tl_threshold_t, typename tl_output_t>
struct treelite_traversal_node : public traversal_node<TREELITE_NODE_ID_T> {
  treelite_traversal_node(treelite::Tree<tl_threshold_t, tl_output_t> const& tl_tree,
                          id_type node_id)
    : traversal_node{}, tl_tree_{tl_tree}, node_id_{node_id}
  {
  }

  bool is_leaf() const override { return tl_tree_.IsLeaf(node_id_); }

  id_type hot_child() const override
  {
    auto result = id_type{};
    if (left_is_hot()) {
      result = tl_tree_.LeftChild(node_id_);
    } else {
      result = tl_tree_.RightChild(node_id_);
    }
    return result;
  }

  id_type distant_child() const override
  {
    auto result = id_type{};
    if (left_is_hot()) {
      result = tl_tree_.RightChild(node_id_);
    } else {
      result = tl_tree_.LeftChild(node_id_);
    }
    return result;
  }

  auto default_distant() const { return tl_tree_.DefaultChild(node_id_) == distant_child(); }

  auto get_feature() const { return tl_tree_.SplitIndex(node_id_); }

  auto is_inclusive() const
  {
    auto tl_operator = tl_tree_.ComparisonOp(node_id_);
    return tl_operator == treelite::Operator::kGT || tl_operator == treelite::Operator::kLE;
  }

  auto is_categorical() const
  {
    return tl_tree_.NodeType(node_id_) == treelite::TreeNodeType::kCategoricalTestNode;
  }

  auto get_categories() const { return tl_tree_.CategoryList(node_id_); }

  auto threshold() const { return tl_tree_.Threshold(node_id_); }

  auto max_num_categories() const
  {
    auto result = std::remove_const_t<std::remove_reference_t<decltype(get_categories()[0])>>{};
    if (is_categorical()) {
      auto categories = get_categories();
      if (categories.size() != 0) {
        result = *std::max_element(std::begin(categories), std::end(categories)) + 1;
      }
    }
    return result;
  }

  auto get_output() const
  {
    auto result = std::vector<tl_output_t>{};
    if (tl_tree_.HasLeafVector(node_id_)) {
      result = tl_tree_.LeafVector(node_id_);
    } else {
      result.push_back(tl_tree_.LeafValue(node_id_));
    }
    return result;
  }

  auto get_treelite_id() const { return node_id_; }

 private:
  treelite::Tree<tl_threshold_t, tl_output_t> const& tl_tree_;
  id_type node_id_;

  auto left_is_hot() const
  {
    auto result = false;
    if (is_categorical()) {
      if (tl_tree_.CategoryListRightChild(node_id_)) { result = true; }
    } else {
      auto tl_operator = tl_tree_.ComparisonOp(node_id_);
      if (tl_operator == treelite::Operator::kLT || tl_operator == treelite::Operator::kLE) {
        result = false;
      } else if (tl_operator == treelite::Operator::kGT || tl_operator == treelite::Operator::kGE) {
        result = true;
      } else {
        throw traversal_exception("Unrecognized Treelite operator");
      }
    }
    return result;
  }
};

template <typename tl_threshold_t, typename tl_output_t>
struct treelite_traversal_forest
  : public traversal_forest<treelite_traversal_node<tl_threshold_t, tl_output_t>> {
 private:
  using base_type = traversal_forest<treelite_traversal_node<tl_threshold_t, tl_output_t>>;

 public:
  using node_type     = typename base_type::node_type;
  using node_id_type  = typename base_type::node_id_type;
  using tree_id_type  = typename base_type::tree_id_type;
  using node_uid_type = typename base_type::node_uid_type;

  treelite_traversal_forest(treelite::ModelPreset<tl_threshold_t, tl_output_t> const& tl_model)
    : traversal_forest<treelite_traversal_node<tl_threshold_t, tl_output_t>>{[&tl_model]() {
        auto result = std::vector<node_uid_type>{};
        result.reserve(tl_model.GetNumTree());
        for (auto i = std::size_t{}; i < tl_model.GetNumTree(); ++i) {
          result.push_back(std::make_pair(i, TREELITE_NODE_ID_T{}));
        }
        return result;
      }()},
      tl_model_{tl_model}
  {
  }

  node_type get_node(tree_id_type tree_id, node_id_type node_id) const override
  {
    return node_type{tl_model_.trees[tree_id], node_id};
  }

 private:
  treelite::ModelPreset<tl_threshold_t, tl_output_t> const& tl_model_;
};

template <typename lambda_t>
void tree_for_each(treelite::Model const& tl_model, lambda_t&& lambda)
{
  std::visit(
    [&lambda](auto&& concrete_tl_model) {
      std::for_each(std::begin(concrete_tl_model.trees), std::end(concrete_tl_model.trees), lambda);
    },
    tl_model.variant_);
}

template <typename iter_t, typename lambda_t>
void tree_transform(treelite::Model const& tl_model, iter_t out_iter, lambda_t&& lambda)
{
  std::visit(
    [&lambda, out_iter](auto&& concrete_tl_model) {
      std::transform(
        std::begin(concrete_tl_model.trees), std::end(concrete_tl_model.trees), out_iter, lambda);
    },
    tl_model.variant_);
}

template <typename T, typename lambda_t>
auto tree_accumulate(treelite::Model const& tl_model, T init, lambda_t&& lambda)
{
  return std::visit(
    [&lambda, init](auto&& concrete_tl_model) {
      return std::accumulate(
        std::begin(concrete_tl_model.trees), std::end(concrete_tl_model.trees), init, lambda);
    },
    tl_model.variant_);
}

template <forest_order order, typename lambda_t>
void node_for_each(treelite::Model const& tl_model, lambda_t&& lambda)
{
  std::visit(
    [&lambda](auto&& concrete_tl_model) {
      treelite_traversal_forest{concrete_tl_model}.template for_each<order>(lambda);
    },
    tl_model.variant_);
}

template <forest_order order, typename iter_t, typename lambda_t>
void node_transform(treelite::Model const& tl_model, iter_t output_iter, lambda_t&& lambda)
{
  node_for_each<order>(
    tl_model,
    [&output_iter, &lambda](auto&& tree_id, auto&& node, auto&& depth, auto&& parent_index) {
      *output_iter = lambda(tree_id, node, depth, parent_index);
      ++output_iter;
    });
}

template <forest_order order, typename T, typename lambda_t>
auto node_accumulate(treelite::Model const& tl_model, T init, lambda_t&& lambda)
{
  auto result = init;
  node_for_each<order>(
    tl_model, [&result, &lambda](auto&& tree_id, auto&& node, auto&& depth, auto&& parent_index) {
      result = lambda(result, tree_id, node, depth, parent_index);
    });
  return result;
}

}  // namespace forest
}  // namespace ML
