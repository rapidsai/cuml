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

#include <cuml/forest/traversal/traversal_forest.hpp>
#include <cuml/forest/traversal/traversal_node.hpp>
#include <cuml/forest/traversal/traversal_order.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>

namespace ML {
namespace forest {

struct test_node : traversal_node<> {
  test_node() : label_{}, hot_child_{}, distant_child_{} {}
  test_node(char label,
            std::optional<std::size_t> hot_child,
            std::optional<std::size_t> distant_child)
    : label_{label}, hot_child_{hot_child}, distant_child_{distant_child}
  {
  }
  auto get_label() { return label_; }
  bool is_leaf() const override { return !hot_child_.has_value(); }
  std::size_t hot_child() const override { return hot_child_.value_or(std::size_t{}); }
  std::size_t distant_child() const override { return distant_child_.value_or(std::size_t{}); }

 private:
  char label_;
  std::optional<std::size_t> hot_child_{};
  std::optional<std::size_t> distant_child_{};
};

/* |Test Forest
 * |-----------------------------------|
 * |TREE 0                             |
 * |                 A                 |
 * |                / \                |
 * |               /   \               |
 * |              /     \              |
 * |             /       \             |
 * |            B         C            |
 * |           /\                      |
 * |          /  \                     |
 * |         D    E                    |
 * |              /\                   |
 * |             F  G                  |
 * |-----------------------------------|
 * |TREE 1                             |
 * |                 H                 |
 * |                / \                |
 * |               /   \               |
 * |              /     \              |
 * |             /       \             |
 * |            I         J            |
 * |                      /\           |
 * |                     /  \          |
 * |                    K    L         |
 * |-----------------------------------|
 * |TREE 2                             |
 * |                 M                 |
 * |                / \                |
 * |               /   \               |
 * |              /     \              |
 * |             /       \             |
 * |            N         O            |
 * |-----------------------------------|
 * |TREE 3                             |
 * |                 P                 |
 * |                                   |
 * |                                   |
 * |                                   |
 * |                                   |
 * |                                   |
 * |-----------------------------------|
 * |TREE 4                             |
 * |                 Q                 |
 * |                / \                |
 * |               /   \               |
 * |              /     \              |
 * |             /       \             |
 * |            R         S            |
 * |-----------------------------------|
 * |TREE 5                             |
 * |                 T                 |
 * |                / \                |
 * |               /   \               |
 * |              /     \              |
 * |             /       \             |
 * |            U         V            |
 * |                      /\           |
 * |                     /  \          |
 * |                    W    X         |
 * |                   /\              |
 * |                  Y  Z             |
 * |-----------------------------------|
 */

struct test_forest : traversal_forest<test_node> {
  test_forest()
    : traversal_forest<test_node>{std::vector<std::pair<std::size_t, std::size_t>>{
        std::make_pair(std::size_t{}, std::size_t{}),
        std::make_pair(std::size_t{1}, std::size_t{7}),
        std::make_pair(std::size_t{2}, std::size_t{12}),
        std::make_pair(std::size_t{3}, std::size_t{15}),
        std::make_pair(std::size_t{4}, std::size_t{16}),
        std::make_pair(std::size_t{5}, std::size_t{19}),
      }}
  {
  }
  test_node get_node(std::size_t tree_id, std::size_t node_id) const override
  {
    return nodes_[node_id];
  }

 private:
  std::vector<test_node> nodes_{
    test_node{'A', std::size_t{1}, std::size_t{2}},                              // 0
    test_node{'B', std::size_t{3}, std::size_t{4}},                              // 1
    test_node{'C', std::optional<std::size_t>{}, std::optional<std::size_t>{}},  // 2
    test_node{'D', std::optional<std::size_t>{}, std::optional<std::size_t>{}},  // 3
    test_node{'E', std::size_t{5}, std::size_t{6}},                              // 4
    test_node{'F', std::optional<std::size_t>{}, std::optional<std::size_t>{}},  // 5
    test_node{'G', std::optional<std::size_t>{}, std::optional<std::size_t>{}},  // 6
    test_node{'H', std::size_t{8}, std::size_t{9}},                              // 7
    test_node{'I', std::optional<std::size_t>{}, std::optional<std::size_t>{}},  // 8
    test_node{'J', std::size_t{10}, std::size_t{11}},                            // 9
    test_node{'K', std::optional<std::size_t>{}, std::optional<std::size_t>{}},  // 10
    test_node{'L', std::optional<std::size_t>{}, std::optional<std::size_t>{}},  // 11
    test_node{'M', std::size_t{13}, std::size_t{14}},                            // 12
    test_node{'N', std::optional<std::size_t>{}, std::optional<std::size_t>{}},  // 13
    test_node{'O', std::optional<std::size_t>{}, std::optional<std::size_t>{}},  // 14
    test_node{'P', std::optional<std::size_t>{}, std::optional<std::size_t>{}},  // 15
    test_node{'Q', std::size_t{17}, std::size_t{18}},                            // 16
    test_node{'R', std::optional<std::size_t>{}, std::optional<std::size_t>{}},  // 17
    test_node{'S', std::optional<std::size_t>{}, std::optional<std::size_t>{}},  // 18
    test_node{'T', std::size_t{20}, std::size_t{21}},                            // 19
    test_node{'U', std::optional<std::size_t>{}, std::optional<std::size_t>{}},  // 20
    test_node{'V', std::size_t{22}, std::size_t{23}},                            // 21
    test_node{'W', std::size_t{24}, std::size_t{25}},                            // 22
    test_node{'X', std::optional<std::size_t>{}, std::optional<std::size_t>{}},  // 23
    test_node{'Y', std::optional<std::size_t>{}, std::optional<std::size_t>{}},  // 24
    test_node{'Z', std::optional<std::size_t>{}, std::optional<std::size_t>{}}   // 25
  };
};

struct traversal_forest_results {
  std::string order;
  std::vector<std::size_t> depth;
  std::vector<std::size_t> parents;
  std::vector<std::size_t> tree_indices;
};

auto static const TRAVERSAL_RESULTS =
  std::vector<std::pair<forest_order, traversal_forest_results>>{
    std::make_pair(forest_order::depth_first,
                   traversal_forest_results{
                     "ABDEFGCHIJKLMNOPQRSTUVWYZX",
                     std::vector<std::size_t>{0, 1, 2, 2, 3, 3, 1, 0, 1, 1, 2, 2, 0,
                                              1, 1, 0, 0, 1, 1, 0, 1, 1, 2, 3, 3, 2},
                     std::vector<std::size_t>{0,  0,  1,  1,  3,  3,  0,  7,  7,  7,  9,  9,  12,
                                              12, 12, 15, 16, 16, 16, 19, 19, 19, 21, 22, 22, 21},
                     std::vector<std::size_t>{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2,
                                              2, 2, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5}}),
    std::make_pair(forest_order::breadth_first,
                   traversal_forest_results{
                     "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                     std::vector<std::size_t>{0, 1, 1, 2, 2, 3, 3, 0, 1, 1, 2, 2, 0,
                                              1, 1, 0, 0, 1, 1, 0, 1, 1, 2, 2, 3, 3},
                     std::vector<std::size_t>{0,  0,  0,  1,  1,  4,  4,  7,  7,  7,  9,  9,  12,
                                              12, 12, 15, 16, 16, 16, 19, 19, 19, 21, 21, 22, 22},
                     std::vector<std::size_t>{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2,
                                              2, 2, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5}}),
    std::make_pair(forest_order::layered_children_together,
                   traversal_forest_results{
                     "AHMPQTBCIJNORSUVDEKLWXFGYZ",
                     std::vector<std::size_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                                              1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3},
                     std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 0, 0,  1,  1,  2,  2,  4,
                                              4, 5, 5, 6, 6, 9, 9, 15, 15, 17, 17, 20, 20},
                     std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 0, 0, 1, 1, 2, 2, 4,
                                              4, 5, 5, 0, 0, 1, 1, 5, 5, 0, 0, 5, 5}}),
    std::make_pair(forest_order::layered_children_segregated,
                   traversal_forest_results{
                     "AHMPQTBINRUCJOSVDKWELXYFZG",
                     std::vector<std::size_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                                              1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3},
                     std::vector<std::size_t>{0, 1, 2, 3, 4,  5,  0, 1,  2,  4,  5,  0,  1,
                                              2, 4, 5, 6, 12, 15, 6, 12, 15, 18, 19, 18, 19},
                     std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 0, 1, 2, 4, 5, 0, 1,
                                              2, 4, 5, 0, 1, 5, 0, 1, 5, 5, 0, 5, 0}}),
  };

template <forest_order order>
auto get_expected_for_each_result()
{
  return std::find_if(std::begin(TRAVERSAL_RESULTS),
                      std::end(TRAVERSAL_RESULTS),
                      [](auto&& pair) { return pair.first == order; })
    ->second;
}

template <forest_order order>
auto get_for_each_order()
{
  auto result = std::vector<char>{};
  test_forest{}.for_each<order>(
    [&result](auto&& tree_id, auto&& node, auto&& depth, auto&& parent_index) {
      result.push_back(node.get_label());
    });
  return std::string(std::begin(result), std::end(result));
}

template <forest_order order>
auto get_for_each_depth()
{
  auto result = std::vector<std::size_t>{};
  test_forest{}.for_each<order>(
    [&result](auto&& tree_id, auto&& node, auto&& depth, auto&& parent_index) {
      result.push_back(depth);
    });
  return result;
}

template <forest_order order>
auto get_for_each_parent()
{
  auto result = std::vector<std::size_t>{};
  test_forest{}.for_each<order>(
    [&result](auto&& tree_id, auto&& node, auto&& depth, auto&& parent_index) {
      result.push_back(parent_index);
    });
  return result;
}

template <forest_order order>
auto get_for_each_tree()
{
  auto result = std::vector<std::size_t>{};
  test_forest{}.for_each<order>(
    [&result](auto&& tree_id, auto&& node, auto&& depth, auto&& parent_index) {
      result.push_back(tree_id);
    });
  return result;
}

TEST(ForestTraversal, depth_first)
{
  auto order    = get_for_each_order<forest_order::depth_first>();
  auto depths   = get_for_each_depth<forest_order::depth_first>();
  auto parents  = get_for_each_parent<forest_order::depth_first>();
  auto trees    = get_for_each_tree<forest_order::depth_first>();
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

TEST(ForestTraversal, breadth_first)
{
  auto order    = get_for_each_order<forest_order::breadth_first>();
  auto depths   = get_for_each_depth<forest_order::breadth_first>();
  auto parents  = get_for_each_parent<forest_order::breadth_first>();
  auto trees    = get_for_each_tree<forest_order::breadth_first>();
  auto expected = get_expected_for_each_result<forest_order::breadth_first>();
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

TEST(ForestTraversal, layered_children_together)
{
  auto order    = get_for_each_order<forest_order::layered_children_together>();
  auto depths   = get_for_each_depth<forest_order::layered_children_together>();
  auto parents  = get_for_each_parent<forest_order::layered_children_together>();
  auto trees    = get_for_each_tree<forest_order::layered_children_together>();
  auto expected = get_expected_for_each_result<forest_order::layered_children_together>();
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

TEST(ForestTraversal, layered_children_segregated)
{
  auto order    = get_for_each_order<forest_order::layered_children_segregated>();
  auto depths   = get_for_each_depth<forest_order::layered_children_segregated>();
  auto parents  = get_for_each_parent<forest_order::layered_children_segregated>();
  auto trees    = get_for_each_tree<forest_order::layered_children_segregated>();
  auto expected = get_expected_for_each_result<forest_order::layered_children_segregated>();
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
}  // namespace ML
