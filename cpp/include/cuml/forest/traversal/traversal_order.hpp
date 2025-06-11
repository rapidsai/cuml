/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
namespace forest {

/* A class used to specify the order in which nodes of a forest should be
 * traversed
 *
 * Because the meaning of "left" and "right" vary by convention, we refer to the two children of a
 * node as "hot" or "distant" rather than left or right. The "hot" child is the one which is
 * traversed soonest after the parent, and the "distant" child is traversed latest.
 */
enum class forest_order : unsigned char {
  // Traverse forest by proceeding depth-first through each tree
  // consecutively
  depth_first = 0,
  // Traverse forest by proceeding breadth-first through each tree
  // consecutively
  breadth_first = 1,
  // Traverse forest by proceeding through the root nodes of each tree first,
  // followed by the hot and distant children of those root nodes for each tree,
  // and so forth. This traversal order ensures that all nodes of a
  // particular tree at a particular depth are traversed together.
  layered_children_together = 2,
  // Traverse forest by proceeding through the root nodes of each tree first,
  // followed by all of the hot children of those root nodes, then all of
  // the distant children of those root nodes, and so forth. This
  // traversal order ensures that all hot children at a particular depth
  // are traversed together, followed by all distant children.
  layered_children_segregated = 3
};

}  // namespace forest
}  // namespace ML
