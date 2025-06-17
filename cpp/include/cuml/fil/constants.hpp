/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <cuml/fil/tree_layout.hpp>

namespace ML {
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
}  // namespace ML
