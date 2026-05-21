/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/common/export.hpp>
namespace CUML_EXPORT ML {
namespace fil {
enum class tree_layout : unsigned char {
  depth_first   = 0,
  breadth_first = 1,
  // Traverse forest by proceeding through the root nodes of each tree first,
  // followed by the hot and distant children of those root nodes for each tree,
  // and so forth. This traversal order ensures that all nodes of a
  // particular tree at a particular depth are traversed together.
  layered_children_together = 2
};

}
}  // namespace CUML_EXPORT ML
