#pragma once
#include <variant>
#include <herring3/constants.hpp>
#include <herring3/detail/specialization_types.hpp>
#include <herring3/detail/forest.hpp>

#define HERRING_SPEC(variant_index) typename std::variant_alternative_t<variant_index, herring::detail::specialization_variant>

#define HERRING_FOREST(variant_index) forest<preferred_tree_layout, HERRING_SPEC(variant_index)::threshold_type, HERRING_SPEC(variant_index)::index_type, HERRING_SPEC(variant_index)::metadata_type, HERRING_SPEC(variant_index)::offset_type>
