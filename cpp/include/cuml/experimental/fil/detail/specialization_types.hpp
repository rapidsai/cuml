#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <variant>
namespace ML {
namespace experimental {
namespace fil {
namespace detail {

template <bool double_precision, bool large_trees>
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
};

using specialization_variant = std::variant<
  specialization_types<false, false>,
  specialization_types<false, true>,
  specialization_types<true, false>,
  specialization_types<true, true>
>;

}
}
}
}
