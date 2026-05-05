/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/fil/decision_forest.hpp>
#include <cuml/fil/detail/decision_forest_builder.hpp>
#include <cuml/fil/exceptions.hpp>
#include <cuml/fil/tree_layout.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace ML {
namespace fil {
namespace detail {

using test_forest_t =
  decision_forest<tree_layout::breadth_first, float, std::uint32_t, std::uint16_t, std::uint16_t>;

TEST(DecisionForestBuilder, CategoricalStorageOffsetOutOfBounds)
{
  auto builder = decision_forest_builder<test_forest_t>(
    /*max_num_categories=*/std::uint32_t{33}, /*align_bytes=*/std::uint32_t{0});

  // Construct a malformed categorical node that references non-local category
  // storage at an out-of-range offset. This should be rejected by the invariant
  // checks in get_decision_forest().
  builder.add_node(std::uint32_t{1234},
                   /*tl_node_id=*/0,
                   /*depth=*/0,
                   /*is_leaf_node=*/false,
                   /*default_to_distant_child=*/false,
                   /*is_categorical_node=*/true,
                   /*feature=*/0,
                   /*offset=*/1);

  ASSERT_THAT(
    [&] { builder.get_decision_forest(/*num_feature=*/1, /*num_class=*/1); },
    testing::ThrowsMessage<model_import_error>(testing::HasSubstr("storage offset out of bounds")));
}

TEST(DecisionForestBuilder, CategoricalBitsetExtentOutOfBounds)
{
  auto builder = decision_forest_builder<test_forest_t>(
    /*max_num_categories=*/std::uint32_t{33}, /*align_bytes=*/std::uint32_t{0});

  // Create a valid categorical node first, which allocates non-local storage:
  // categorical_storage_ = [num_categories, packed_bin_data]
  std::array<std::uint32_t, 1> categories{0};
  builder.add_categorical_node(categories.begin(),
                               categories.end(),
                               /*tl_node_id=*/0,
                               /*depth=*/0,
                               /*default_to_distant_child=*/false,
                               /*feature=*/0,
                               /*offset=*/1);

  // Construct another categorical node that points at offset=1, i.e. the first
  // packed bin entry rather than the metadata entry. The value at offset=1 is
  // interpreted as stored_num_cats, and the resulting bins_required exceeds the
  // available headroom.
  builder.add_node(std::uint32_t{1},
                   /*tl_node_id=*/1,
                   /*depth=*/1,
                   /*is_leaf_node=*/false,
                   /*default_to_distant_child=*/false,
                   /*is_categorical_node=*/true,
                   /*feature=*/0,
                   /*offset=*/1);

  ASSERT_THAT([&] { builder.get_decision_forest(/*num_feature=*/1, /*num_class=*/1); },
              testing::ThrowsMessage<model_import_error>(
                testing::HasSubstr("bitset extends past categorical_storage end")));
}

}  // namespace detail
}  // namespace fil
}  // namespace ML
