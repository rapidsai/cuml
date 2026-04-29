/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/fil/exceptions.hpp>
#include <cuml/fil/tree_layout.hpp>
#include <cuml/fil/treelite_importer.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/model_builder.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace ML {
namespace fil {

TEST(TreeliteImporter, large_category_value)
{
  // For tree models with 32-bit data storage,
  // attempting to use UINT32_MAX in a categorical test node
  // must throw an exception.

  auto metadata = treelite::model_builder::Metadata{
    1,
    treelite::TaskType::kRegressor,
    false,
    1,
    {1},
    {1, 1},
  };
  auto tree_annotation = treelite::model_builder::TreeAnnotation{1, {0}, {0}};
  auto model_builder =
    treelite::model_builder::GetModelBuilder(treelite::TypeInfo::kFloat32,
                                             treelite::TypeInfo::kFloat32,
                                             metadata,
                                             tree_annotation,
                                             treelite::model_builder::PostProcessorFunc{"identity"},
                                             {0.0});

  model_builder->StartTree();
  model_builder->StartNode(0);
  model_builder->CategoricalTest(
    0, false, {std::numeric_limits<std::uint32_t>::max()}, false, 1, 2);
  model_builder->EndNode();

  model_builder->StartNode(1);
  model_builder->LeafScalar(1.0);
  model_builder->EndNode();

  model_builder->StartNode(2);
  model_builder->LeafScalar(-1.0);
  model_builder->EndNode();

  model_builder->EndTree();

  auto tl_model = model_builder->CommitModel();

  auto expected_error_msg = std::string{"Tree 0, Node 0: Category index must be at most "} +
                            std::to_string(std::numeric_limits<std::uint32_t>::max() - 1);

  ASSERT_THAT([&]() { import_from_treelite_model(*tl_model, tree_layout::breadth_first); },
              testing::ThrowsMessage<model_import_error>(testing::HasSubstr(expected_error_msg)));
}

TEST(TreeliteImporter, large_category_value2)
{
  // For tree models with 32-bit data storage,
  // it should be possible to use (UINT32_MAX - 1) in a categorical test node

  auto metadata = treelite::model_builder::Metadata{
    1,
    treelite::TaskType::kRegressor,
    false,
    1,
    {1},
    {1, 1},
  };
  auto tree_annotation = treelite::model_builder::TreeAnnotation{1, {0}, {0}};
  auto model_builder =
    treelite::model_builder::GetModelBuilder(treelite::TypeInfo::kFloat32,
                                             treelite::TypeInfo::kFloat32,
                                             metadata,
                                             tree_annotation,
                                             treelite::model_builder::PostProcessorFunc{"identity"},
                                             {0.0});

  model_builder->StartTree();
  model_builder->StartNode(0);
  model_builder->CategoricalTest(
    0, false, {std::numeric_limits<std::uint32_t>::max() - 1}, false, 1, 2);
  model_builder->EndNode();

  model_builder->StartNode(1);
  model_builder->LeafScalar(1.0);
  model_builder->EndNode();

  model_builder->StartNode(2);
  model_builder->LeafScalar(-1.0);
  model_builder->EndNode();

  model_builder->EndTree();

  auto tl_model = model_builder->CommitModel();
  ASSERT_NO_THROW(import_from_treelite_model(*tl_model, tree_layout::breadth_first));
}

}  // namespace fil
}  // namespace ML
