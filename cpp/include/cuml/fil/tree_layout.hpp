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
namespace ML {
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
}  // namespace ML
