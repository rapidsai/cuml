/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
