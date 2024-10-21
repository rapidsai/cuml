/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuml/experimental/forest/traversal/traversal_order.hpp>
#include <cuml/experimental/forest/integrations/treelite.hpp>
#include <treelite/enum/task_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/model_builder.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cstddef>

namespace ML {
namespace experimental {
namespace forest {

/* |Test Forest
 * |-----------------------------------|--------------------------------------|
 * |TREE 0                             |KEY                                   |
 * |               A-0-3               |* Non-leaf nodes: Label-Fea-Split     |
 * |                / \                |* Cat nodes: Label-Fea-{cat0, cat1...}|
 * |               /   \               |* Leaf nodes: Label above outputs     | 
 * |              /     \              |  - Regression output                 |
 * |             /       \             |  - Binary classification output      |
 * |          B-1-2     C-2-1          |  - Multiclass output                 |
 * |           /\         0            |--------------------------------------|
 * |          /  \                     |
 * |         D  E-3-0                  |
 * |         1    /\                   |
 * |             F  G                  |
 * |             2  3                  |
 * |-----------------------------------|
 * |TREE 1                             |
 * |               H-0-0               |
 * |                / \                |
 * |               /   \               |
 * |              /     \              |
 * |             /       \             |
 * |            I       J-1-1          |
 * |            4         /\           |
 * |                     /  \          |
 * |                    K    L         |
 * |                    5    6         |
 * |-----------------------------------|
 * |TREE 2                             |
 * |               M-5-{0, 1, 3}       |
 * |                / \                |
 * |               /   \               |
 * |              /     \              |
 * |             /       \             |
 * |            N         O            |
 * |            7         8            |
 * |-----------------------------------|
 * |TREE 3                             |
 * |                 P                 |
 * |                 9                 |
 * |                                   |
 * |                                   |
 * |                                   |
 * |                                   |
 * |-----------------------------------|
 * |TREE 4                             |
 * |               Q-5-{0, 7}          |
 * |                / \                |
 * |               /   \               |
 * |              /     \              |
 * |             /       \             |
 * |            R         S            |
 * |           10         11           |
 * |-----------------------------------|
 * |TREE 5                             |
 * |               T-6-{1, 3}          |
 * |                / \                |
 * |               /   \               |
 * |              /     \              |
 * |             /       \             |
 * |            U       V-6-{1, 4}     |
 * |           12         /\           |
 * |                     /  \          |
 * |                W-6-{3}  X         |
 * |                   /\    13        |
 * |                  Y  Z             |
 * |                 14  15            |
 * |-----------------------------------|
 */
auto static const SAMPLE_COL_COUNT = 7;
auto static const SAMPLE_TREE_COUNT = 6;
auto static const SAMPLE_CLASS_COUNT = 3;
auto static const SAMPLE_FOREST = []() {
  auto metadata = treelite::model_builder::Metadata{
    SAMPLE_COL_COUNT,
    treelite::TaskType::kRegressor,
    true,
    1,
    {1},
    {1, 1},
  };
  auto tree_annotation = treelite::model_builder::TreeAnnotation{
    SAMPLE_TREE_COUNT,
    std::vector<std::int32_t>(SAMPLE_TREE_COUNT, 0),
    std::vector<std::int32_t>{0, 0, 0, 0, 0, 0}
  };
  auto model_builder = treelite::model_builder::GetModelBuilder(
    treelite::TypeInfo::kFloat32,
    treelite::TypeInfo::kFloat32,
    metadata,
    tree_annotation,
    treelite::model_builder::PostProcessorFunc{"identity_multiclass"},
    std::vector<double>(1, 0.0f)
  );
  // TREE 0
  model_builder->StartTree();
  // Node A
  model_builder->StartNode(0);
  // For numerical splits, the right child is "hot" if the operator is kLT or
  // kLE. For categorical splits, whichever child corresponds to
  // out-of-category is the "hot" node.
  // feature index, threshold, default left, operator, left child, right child
  model_builder->NumericalTest(0, 3.0, true, treelite::Operator::kGE, 1, 2);
  model_builder->EndNode();
  // Node B
  model_builder->StartNode(1);
  model_builder->NumericalTest(1, 2.0, false, treelite::Operator::kLT, 4, 3);
  model_builder->EndNode();
  // Node C
  model_builder->StartNode(2);
  model_builder->LeafScalar(0.0);
  model_builder->EndNode();
  // Node D
  model_builder->StartNode(3);
  model_builder->LeafScalar(1.0);
  model_builder->EndNode();
  // Node E
  model_builder->StartNode(4);
  model_builder->NumericalTest(3, 0.0, true, treelite::Operator::kGT, 5, 6);
  model_builder->EndNode();
  // Node F
  model_builder->StartNode(5);
  model_builder->LeafScalar(2.0);
  model_builder->EndNode();
  // Node G
  model_builder->StartNode(6);
  model_builder->LeafScalar(3.0);
  model_builder->EndNode();
  model_builder->EndTree();

  // TREE 1
  model_builder->StartTree();
  // Node H
  model_builder->StartNode(0);
  model_builder->NumericalTest(0, 0.0, true, treelite::Operator::kGE, 1, 2);
  model_builder->EndNode();
  // Node I
  model_builder->StartNode(1);
  model_builder->LeafScalar(4.0);
  model_builder->EndNode();
  // Node J
  model_builder->StartNode(2);
  model_builder->NumericalTest(0, 0.0, true, treelite::Operator::kGE, 3, 4);
  model_builder->EndNode();
  // Node K
  model_builder->StartNode(3);
  model_builder->LeafScalar(5.0);
  model_builder->EndNode();
  // Node L
  model_builder->StartNode(4);
  model_builder->LeafScalar(6.0);
  model_builder->EndNode();
  model_builder->EndTree();

  // TREE 2
  model_builder->StartTree();
  // Node M
  model_builder->StartNode(0);
  model_builder->CategoricalTest(5, true, std::vector<std::uint32_t>{0, 1, 3}, true, 1, 2);
  model_builder->EndNode();
  // Node N
  model_builder->StartNode(1);
  model_builder->LeafScalar(7.0);
  model_builder->EndNode();
  // Node O
  model_builder->StartNode(2);
  model_builder->LeafScalar(8.0);
  model_builder->EndNode();
  model_builder->EndTree();

  // TREE 3
  model_builder->StartTree();
  // Node P
  model_builder->StartNode(0);
  model_builder->LeafScalar(9.0);
  model_builder->EndNode();
  model_builder->EndTree();

  // TREE 4
  model_builder->StartTree();
  // Node Q
  model_builder->StartNode(0);
  model_builder->CategoricalTest(5, true, std::vector<std::uint32_t>{0, 7}, false, 2, 1);
  model_builder->EndNode();
  // Node R
  model_builder->StartNode(1);
  model_builder->LeafScalar(10.0);
  model_builder->EndNode();
  // Node S
  model_builder->StartNode(2);
  model_builder->EndNode();
  model_builder->LeafScalar(11.0);
  model_builder->EndTree();

  // TREE 5
  model_builder->StartTree();
  // Node T
  model_builder->StartNode(0);
  model_builder->CategoricalTest(6, true, std::vector<std::uint32_t>{1, 3}, true, 1, 2);
  model_builder->EndNode();
  // Node U
  model_builder->StartNode(1);
  model_builder->LeafScalar(12.0);
  model_builder->EndNode();
  // Node V
  model_builder->StartNode(2);
  model_builder->CategoricalTest(6, true, std::vector<std::uint32_t>{1, 4}, true, 3, 4);
  model_builder->EndNode();
  // Node W
  model_builder->StartNode(3);
  model_builder->CategoricalTest(6, true, std::vector<std::uint32_t>{3}, true, 5, 6);
  model_builder->EndNode();
  // Node X
  model_builder->StartNode(4);
  model_builder->LeafScalar(13.0);
  model_builder->EndNode();
  // Node Y
  model_builder->StartNode(5);
  model_builder->LeafScalar(14.0);
  model_builder->EndNode();
  // Node Z
  model_builder->StartNode(6);
  model_builder->LeafScalar(15.0);
  model_builder->EndNode();
  model_builder->EndTree();
  return model_builder->CommitModel();
}();

struct treelite_traversal_results {
  std::vector<double> feature_or_output;
  std::vector<std::size_t> depth;
  std::vector<std::size_t> parents;
  std::vector<std::size_t> tree_indices;
};

auto static const TRAVERSAL_RESULTS = std::vector<std::pair<forest_order, treelite_traversal_results>>{
  std::make_pair(
    forest_order::depth_first,
    treelite_traversal_results{
      std::vector<double>{},
      std::vector<std::size_t>{0, 1, 2, 2, 3, 3, 1, 0, 1, 1, 2, 2, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 2, 3, 3, 2},
      std::vector<std::size_t>{0, 0, 1, 1, 3, 3, 0, 7, 7, 7, 9, 9, 12, 12, 12, 15, 16, 16, 16, 19, 19, 19, 21, 22, 22, 21},
      std::vector<std::size_t>{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5}
    }
  ),
  std::make_pair(
    forest_order::breadth_first,
    treelite_traversal_results{
      std::vector<double>{},
      std::vector<std::size_t>{0, 1, 1, 2, 2, 3, 3, 0, 1, 1, 2, 2, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 2, 2, 3, 3},
      std::vector<std::size_t>{0, 0, 0, 1, 1, 4, 4, 7, 7, 7, 9, 9, 12, 12, 12, 15, 16, 16, 16, 19, 19, 19, 21, 21, 22, 22},
      std::vector<std::size_t>{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5}
    }
  ),
  std::make_pair(
    forest_order::layered_children_together,
    treelite_traversal_results{
      std::vector<double>{},
      std::vector<std::size_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3},
      std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 0, 0, 1, 1, 2, 2, 4, 4, 5, 5, 6, 6, 9, 9, 15, 15, 17, 17, 20, 20},
      std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 0, 0, 1, 1, 2, 2, 4, 4, 5, 5, 0, 0, 1, 1, 5, 5, 0, 0, 5, 5}
    }
  ),
  std::make_pair(
    forest_order::layered_children_segregated,
    treelite_traversal_results{
      std::vector<double>{},
      std::vector<std::size_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3},
      std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 0, 1, 2, 4, 5, 0, 1, 2, 4, 5, 6, 12, 15, 6, 12, 15, 18, 19, 18, 19},
      std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 0, 1, 2, 4, 5, 0, 1, 2, 4, 5, 0, 1, 5, 0, 1, 5, 5, 0, 5, 0}
    }
  ),
};

template<forest_order order>
auto get_expected_for_each_result() {
  return std::find_if(
    std::begin(TRAVERSAL_RESULTS),
    std::end(TRAVERSAL_RESULTS),
    [](auto&& pair) {
      return pair.first == order;
    }
  )->second;
}

template<forest_order order>
auto get_feature_or_outputs() {
  auto result = std::vector<double>{};
  node_transform<order>(
    SAMPLE_FOREST,
    std::back_inserter(result),
    [](auto&& tree_id, auto&& node, auto&& depth, auto&& parent_index) {
      auto feature_or_output = double{};
      if(node.is_leaf()) {
        feature_or_output = node.threshold();
      } else {
        feature_or_output = node.get_feature();
      }
      return feature_or_output;
    }
  );
  return result;
}

template<forest_order order>
auto get_for_each_depth() {
  auto result = std::vector<std::size_t>{};
  test_forest{}.for_each<order>(
    [&result](auto&& tree_id, auto&& node, auto&& depth, auto&& parent_index) {
      result.push_back(depth);
    }
  );
  return result;
}

template<forest_order order>
auto get_for_each_parent() {
  auto result = std::vector<std::size_t>{};
  test_forest{}.for_each<order>(
    [&result](auto&& tree_id, auto&& node, auto&& depth, auto&& parent_index) {
      result.push_back(parent_index);
    }
  );
  return result;
}

template<forest_order order>
auto get_for_each_tree() {
  auto result = std::vector<std::size_t>{};
  test_forest{}.for_each<order>(
    [&result](auto&& tree_id, auto&& node, auto&& depth, auto&& parent_index) {
      result.push_back(tree_id);
    }
  );
  return result;
}

TEST(ForestTraversal, depth_first)
{
  auto order = get_for_each_order<forest_order::depth_first>();
  auto depths = get_for_each_depth<forest_order::depth_first>();
  auto parents = get_for_each_parent<forest_order::depth_first>();
  auto trees = get_for_each_tree<forest_order::depth_first>();
  auto expected = get_expected_for_each_result<forest_order::depth_first>();
  EXPECT_EQ(order, expected.order);
  for (auto i = std::size_t{}; i < expected.depth.size(); ++i) {
    EXPECT_EQ(depths[i], expected.depth[i]);
  }
  for (auto i = std::size_t{}; i < expected.parents.size(); ++i) {
    EXPECT_EQ(parents[i], expected.parents[i]);
  }
  for (auto i = std::size_t{}; i < expected.tree_indices.size(); ++i) {
    EXPECT_EQ(trees[i], expected.tree_indices[i]);
  }
}

}  // namespace forest
}  // namespace experimental
}  // namespace ML
