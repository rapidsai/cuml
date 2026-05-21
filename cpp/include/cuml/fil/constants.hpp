/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/common/export.hpp>
#include <cuml/fil/tree_layout.hpp>

namespace CUML_EXPORT ML {
namespace fil {
/**
 * The default memory layout for FIL trees if not otherwise specified
 */
auto constexpr static const preferred_tree_layout = tree_layout::breadth_first;
/**
 * The number of bits used for flags in node metadata
 *
 * Each node in a FIL tree must specify the feature used for its split in
 * addition to some other basic information. The feature ID is "packed"
 * with a few flags in order to reduce the size of the node. This constant
 * indicates how many leading bits are reserved for flags to allow import
 * functions to assess how much space is required for the whole metadata
 * field.
 */
auto constexpr static const reserved_node_metadata_bits = 3;

}  // namespace fil
}  // namespace CUML_EXPORT ML
