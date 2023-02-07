#pragma once
#include <variant>
#include <cuml/experimental/fil/constants.hpp>
#include <cuml/experimental/fil/detail/specialization_types.hpp>
#include <cuml/experimental/fil/detail/forest.hpp>
#include <cuml/experimental/kayak/tree_layout.hpp>

#define HERRING_SPEC(variant_index) std::variant_alternative_t<variant_index, fil::detail::specialization_variant>

#define HERRING_FOREST(variant_index) forest< HERRING_SPEC(variant_index)::layout, typename HERRING_SPEC(variant_index)::threshold_type, typename HERRING_SPEC(variant_index)::index_type, typename HERRING_SPEC(variant_index)::metadata_type, typename HERRING_SPEC(variant_index)::offset_type>
