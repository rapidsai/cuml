#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <variant>
#include <cuml/experimental/kayak/tree_layout.hpp>
namespace ML {
namespace experimental {
namespace fil {
namespace detail {

template <kayak::tree_layout layout_v, bool double_precision, bool large_trees>
struct specialization_types {
  using threshold_type = std::conditional_t<
    double_precision, double, float
  >;
  using index_type = std::conditional_t<
    double_precision, std::uint64_t, std::uint32_t
  >;
  using metadata_type = std::conditional_t<
    large_trees, std::uint32_t, std::uint16_t
  >;
  using offset_type = std::conditional_t<
    large_trees, std::uint32_t, std::uint16_t
  >;
  auto static constexpr const layout = layout_v;
  auto static constexpr const is_double_precision = double_precision;
  auto static constexpr const has_large_trees = large_trees;
};

using specialization_variant = std::variant<
  specialization_types<kayak::tree_layout::depth_first, false, false>,
  specialization_types<kayak::tree_layout::depth_first, false, true>,
  specialization_types<kayak::tree_layout::depth_first, true, false>,
  specialization_types<kayak::tree_layout::depth_first, true, true>,
  specialization_types<kayak::tree_layout::breadth_first, false, false>,
  specialization_types<kayak::tree_layout::breadth_first, false, true>,
  specialization_types<kayak::tree_layout::breadth_first, true, false>,
  specialization_types<kayak::tree_layout::breadth_first, true, true>
>;

}
}
}
}
