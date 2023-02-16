#pragma once
#include <kayak/tree_layout.hpp>

namespace herring {
  auto constexpr static const preferred_tree_layout = kayak::tree_layout::depth_first;
  auto constexpr static const reserved_node_metadata_bits = 3;
}
