/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuml/fil/tree_layout.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <variant>

namespace ML {
namespace fil {
namespace detail {

/*
 * A template used solely to help manage the types which will be compiled in
 * standard cuML FIL
 *
 * The relatively simple and human-readable template parameters of this
 * template are translated into the specific types and values required
 * to instantiate more complex templates and compile-time checks.
 *
 * @tparam layout_v The layout of trees within a model
 * @tparam double_precision Whether this model should use double-precision
 * for floating-point evaluation and 64-bit integers for indexes
 * @tparam large_trees Whether this forest expects more than 2**(16 -3) - 1 =
 * 8191 features or contains nodes whose child is offset more than 2**16 - 1 = 65535 nodes away.
 */
template <tree_layout layout_v, bool double_precision, bool large_trees>
struct specialization_types {
  /* The node threshold type to be used based on the template parameters
   */
  using threshold_type = std::conditional_t<double_precision, double, float>;
  /* The type required for specifying indexes to vector leaf outputs or
   * non-local categorical data.
   */
  using index_type = std::conditional_t<double_precision, std::uint64_t, std::uint32_t>;
  /* The type used to provide metadata storage for nodes */
  using metadata_type = std::conditional_t<large_trees, std::uint32_t, std::uint16_t>;
  /* The type used to provide metadata storage for nodes */
  using offset_type = std::conditional_t<large_trees, std::uint32_t, std::uint16_t>;
  /* The tree layout (alias for layout_v)*/
  auto static constexpr const layout = layout_v;
  /* Whether or not this tree requires double precision (alias for
   * double_precision)
   */
  auto static constexpr const is_double_precision = double_precision;
  /* Whether or not this forest contains large trees (alias for
   * large_trees)
   */
  auto static constexpr const has_large_trees = large_trees;
};

/* A variant holding information on all specialization types compiled
 * in standard cuML FIL
 */
using specialization_variant =
  std::variant<specialization_types<tree_layout::depth_first, false, false>,
               specialization_types<tree_layout::depth_first, false, true>,
               specialization_types<tree_layout::depth_first, true, false>,
               specialization_types<tree_layout::depth_first, true, true>,
               specialization_types<tree_layout::breadth_first, false, false>,
               specialization_types<tree_layout::breadth_first, false, true>,
               specialization_types<tree_layout::breadth_first, true, false>,
               specialization_types<tree_layout::breadth_first, true, true>,
               specialization_types<tree_layout::layered_children_together, false, false>,
               specialization_types<tree_layout::layered_children_together, false, true>,
               specialization_types<tree_layout::layered_children_together, true, false>,
               specialization_types<tree_layout::layered_children_together, true, true>>;

}  // namespace detail
}  // namespace fil
}  // namespace ML
