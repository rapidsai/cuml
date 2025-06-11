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

#include <cuml/fil/detail/raft_proto/device_type.hpp>
#include <cuml/fil/postproc_ops.hpp>
#include <cuml/fil/tree_layout.hpp>
#include <cuml/fil/treelite_importer.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/model_builder.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ML {
namespace fil {

/* |Test Forest
 * |-----------------------------------|--------------------------------------|
 * |TREE 0                             |KEY                                   |
 * |               A-0-3               |* Non-leaf nodes: Label-Fea-Split     |
 * |                / \                |* Cat nodes: Label-Fea-{cat0, cat1...}|
 * |               /   \               |* Leaf nodes: Label above outputs     |
 * |              /     \              |  - Regression output                 |
 * |             /       \             |  - Binary classification output      |
 * |          B-1-2       C            |  - Multiclass output                 |
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
auto static constexpr const SAMPLE_COL_COUNT  = 7;
auto static constexpr const SAMPLE_TREE_COUNT = 6;
auto static const SAMPLE_FOREST               = []() {
  auto metadata = treelite::model_builder::Metadata{
    SAMPLE_COL_COUNT,
    treelite::TaskType::kRegressor,
    true,
    1,
                  {1},
                  {1, 1},
  };
  auto tree_annotation =
    treelite::model_builder::TreeAnnotation{SAMPLE_TREE_COUNT,
                                            std::vector<std::int32_t>(SAMPLE_TREE_COUNT, 0),
                                            std::vector<std::int32_t>{0, 0, 0, 0, 0, 0}};
  auto model_builder = treelite::model_builder::GetModelBuilder(
    treelite::TypeInfo::kFloat32,
    treelite::TypeInfo::kFloat32,
    metadata,
    tree_annotation,
    treelite::model_builder::PostProcessorFunc{"identity_multiclass"},
    std::vector<double>(1, 0.0f));
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
  model_builder->NumericalTest(1, 0.0, true, treelite::Operator::kGE, 3, 4);
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
  model_builder->LeafScalar(11.0);
  model_builder->EndNode();
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

TEST(TreeliteImporter, depth_first)
{
  auto fil_model = import_from_treelite_model(*SAMPLE_FOREST, tree_layout::depth_first);
  ASSERT_EQ(fil_model.num_features(), 7);
  ASSERT_EQ(fil_model.num_outputs(), 1);
  ASSERT_EQ(fil_model.num_trees(), 6);
  ASSERT_FALSE(fil_model.has_vector_leaves());
  ASSERT_EQ(fil_model.row_postprocessing(), row_op::disable);
  ASSERT_EQ(fil_model.elem_postprocessing(), element_op::disable);
  ASSERT_EQ(fil_model.memory_type(), raft_proto::device_type::cpu);
  ASSERT_EQ(fil_model.device_index(), 0);
  ASSERT_FALSE(fil_model.is_double_precision());
}

TEST(TreeliteImporter, breadth_first)
{
  auto fil_model = import_from_treelite_model(*SAMPLE_FOREST, tree_layout::breadth_first);
  ASSERT_EQ(fil_model.num_features(), 7);
  ASSERT_EQ(fil_model.num_outputs(), 1);
  ASSERT_EQ(fil_model.num_trees(), 6);
  ASSERT_FALSE(fil_model.has_vector_leaves());
  ASSERT_EQ(fil_model.row_postprocessing(), row_op::disable);
  ASSERT_EQ(fil_model.elem_postprocessing(), element_op::disable);
  ASSERT_EQ(fil_model.memory_type(), raft_proto::device_type::cpu);
  ASSERT_EQ(fil_model.device_index(), 0);
  ASSERT_FALSE(fil_model.is_double_precision());
}

TEST(TreeliteImporter, layered_children_together)
{
  auto fil_model =
    import_from_treelite_model(*SAMPLE_FOREST, tree_layout::layered_children_together);
  ASSERT_EQ(fil_model.num_features(), 7);
  ASSERT_EQ(fil_model.num_outputs(), 1);
  ASSERT_EQ(fil_model.num_trees(), 6);
  ASSERT_FALSE(fil_model.has_vector_leaves());
  ASSERT_EQ(fil_model.row_postprocessing(), row_op::disable);
  ASSERT_EQ(fil_model.elem_postprocessing(), element_op::disable);
  ASSERT_EQ(fil_model.memory_type(), raft_proto::device_type::cpu);
  ASSERT_EQ(fil_model.device_index(), 0);
  ASSERT_FALSE(fil_model.is_double_precision());
}

template <bool use_leaf_vector, typename leaf_t>
auto make_degenerate_tree(const leaf_t& leaf)
{
  auto task_type        = treelite::TaskType{};
  auto num_class        = std::int32_t{};
  auto class_annotation = std::vector<std::int32_t>{};
  if constexpr (use_leaf_vector) {
    task_type        = treelite::TaskType::kMultiClf;
    num_class        = leaf.size();
    class_annotation = {-1};
  } else {
    task_type        = treelite::TaskType::kBinaryClf;
    num_class        = 1;
    class_annotation = {0};
  }
  auto metadata = treelite::model_builder::Metadata{
    1,
    task_type,
    false,
    1,
    {num_class},
    {1, num_class},
  };
  auto tree_annotation = treelite::model_builder::TreeAnnotation{1, {0}, class_annotation};
  auto model_builder   = treelite::model_builder::GetModelBuilder(
    treelite::TypeInfo::kFloat64,
    treelite::TypeInfo::kFloat64,
    metadata,
    tree_annotation,
    treelite::model_builder::PostProcessorFunc{"identity_multiclass"},
    std::vector<double>(num_class, 0.0));
  model_builder->StartTree();
  model_builder->StartNode(0);
  if constexpr (use_leaf_vector) {
    model_builder->LeafVector(leaf);
  } else {
    model_builder->LeafScalar(leaf);
  }
  model_builder->EndNode();
  model_builder->EndTree();
  return model_builder->CommitModel();
}

TEST(TreeliteImporter, DegenerateTree)
{
  auto tl_model  = make_degenerate_tree<false>(1.0);
  auto fil_model = import_from_treelite_model(*tl_model, tree_layout::breadth_first);
  ASSERT_FALSE(fil_model.has_vector_leaves());

  auto handle         = raft::handle_t{};
  auto X              = std::vector<double>{0.0};
  auto preds          = std::vector<double>(1, 0.0);
  auto expected_preds = std::vector<double>{1.0};
  fil_model.predict(handle,
                    preds.data(),
                    X.data(),
                    1,
                    raft_proto::device_type::cpu,
                    raft_proto::device_type::cpu,
                    ML::fil::infer_kind::default_kind,
                    1);
  ASSERT_EQ(preds, expected_preds);
}

TEST(TreeliteImporter, DegenerateTreeWithVectorLeaf)
{
  auto tl_model  = make_degenerate_tree<true>(std::vector<double>{0.5, 0.5});
  auto fil_model = import_from_treelite_model(*tl_model, tree_layout::breadth_first);
  ASSERT_TRUE(fil_model.has_vector_leaves());

  auto handle         = raft::handle_t{};
  auto X              = std::vector<double>{0.0};
  auto preds          = std::vector<double>(2, 0.0);
  auto expected_preds = std::vector<double>{0.5, 0.5};
  fil_model.predict(handle,
                    preds.data(),
                    X.data(),
                    1,
                    raft_proto::device_type::cpu,
                    raft_proto::device_type::cpu,
                    ML::fil::infer_kind::default_kind,
                    1);
  ASSERT_EQ(preds, expected_preds);
}

}  // namespace fil
}  // namespace ML
