/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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
namespace Dbscan {
namespace AdjGraph {

template <typename Index_ = int>
struct Pack {
  /**
   * vertex degree array
   * Last position is the sum of all elements in this array (excluding it)
   * Hence, its length is one more than the number of poTypes
   */
  Index_* vd;
  /** the adjacency matrix */
  bool* adj;
  /** the adjacency graph */
  Index_* adj_graph;

  Index_ adjnnz;

  /** exculusive scan generated from vd */
  Index_* ex_scan;
  /** number of points in the dataset */
  Index_ N;
};

}  // namespace AdjGraph
}  // namespace Dbscan
}  // namespace ML
