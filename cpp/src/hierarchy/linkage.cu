/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/cluster/linkage.hpp>
#include <cuml/common/distance_type.hpp>

#include <raft/core/handle.hpp>

#include <cuvs/cluster/agglomerative.hpp>

namespace ML {
namespace linkage {

void single_linkage(const raft::handle_t& handle,
                    const float* X,
                    int n_rows,
                    int n_cols,
                    size_t n_clusters,
                    ML::distance::DistanceType metric,
                    int* children,
                    int* labels,
                    bool use_knn,
                    int c)
{
  auto X_view = raft::make_device_matrix_view<const float, int, raft::row_major>(X, n_rows, n_cols);
  auto children_view =
    raft::make_device_matrix_view<int, int, raft::row_major>(children, n_rows - 1, 2);
  auto labels_view = raft::make_device_vector_view<int, int>(labels, n_rows);
  auto linkage     = (use_knn ? cuvs::cluster::agglomerative::Linkage::KNN_GRAPH
                              : cuvs::cluster::agglomerative::Linkage::PAIRWISE);
  cuvs::cluster::agglomerative::single_linkage(handle,
                                               X_view,
                                               children_view,
                                               labels_view,
                                               static_cast<cuvs::distance::DistanceType>(metric),
                                               n_clusters,
                                               linkage,
                                               use_knn ? c : 0);
}

};  // end namespace linkage
};  // end namespace ML
