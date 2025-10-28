/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/distance_type.hpp>

#include <raft/core/handle.hpp>

namespace ML {
namespace linkage {

/**
 * @brief Computes single-linkage hierarchical clustering on a dense input
 * feature matrix and outputs the labels, dendrogram, and minimum spanning tree.
 *
 * @param[in] handle: raft handle to encapsulate expensive resources
 * @param[in] X: dense feature matrix on device, C contiguous
 * @param[in] n_rows: number of rows in X
 * @param[in] n_cols: number of columns in X
 * @param[in] n_clusters: the number of clusters to fit.
 * @param[in] metric: distance metric to use. Must be supported by the
 *            dense pairwise distances API.
 * @param[out] children: the output dendrogram, shape=(n_rows - 1, 2), C contiguous
 * @param[out] labels: the output labels, shape=(n_rows,)
 * @param[in] use_knn: whether to construct a knn graph instead of the full
 *            n^2 pairwise distance matrix. This can be faster for very large
 *            datasets or in cases where lower memory usage is required.
 * @param[in] c: tunes the number of neighbors when `use_knn` is true, where
 *            `n_neighbors=log(n_rows) + c`.
 */
void single_linkage(const raft::handle_t& handle,
                    const float* X,
                    int n_rows,
                    int n_cols,
                    size_t n_clusters,
                    ML::distance::DistanceType metric,
                    int* children,
                    int* labels,
                    bool use_knn = false,
                    int c        = 15);

};  // namespace linkage
};  // namespace ML
