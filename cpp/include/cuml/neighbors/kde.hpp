/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/distance_type.hpp>

namespace raft {
class handle_t;
}

namespace ML::KDE {

enum class KernelType : int {
  Gaussian     = 0,
  Tophat       = 1,
  Epanechnikov = 2,
  Exponential  = 3,
  Linear       = 4,
  Cosine       = 5
};

/**
 * @brief Compute log-density estimates for query points using kernel density estimation.
 *
 * Fuses pairwise distance computation, kernel evaluation, logsumexp reduction,
 * and normalization into a single CUDA kernel pass. O(N+M) memory usage.
 *
 * @param[in]  handle      RAFT handle for stream management
 * @param[in]  query       Query points, row-major (n_query, n_features)
 * @param[in]  train       Training points, row-major (n_train, n_features)
 * @param[in]  weights     Sample weights (n_train,), or nullptr for uniform
 * @param[out] output      Log-density estimates (n_query,)
 * @param[in]  n_query     Number of query points
 * @param[in]  n_train     Number of training points
 * @param[in]  n_features  Dimensionality
 * @param[in]  bandwidth   Kernel bandwidth
 * @param[in]  sum_weights Sum of sample weights (or n_train if uniform)
 * @param[in]  kernel      Kernel function type
 * @param[in]  metric      Distance metric type
 * @param[in]  metric_arg  Metric parameter (e.g. p for Minkowski)
 */
void score_samples(const raft::handle_t& handle,
                   const float* query,
                   const float* train,
                   const float* weights,
                   float* output,
                   int n_query,
                   int n_train,
                   int n_features,
                   float bandwidth,
                   float sum_weights,
                   KernelType kernel,
                   ML::distance::DistanceType metric,
                   float metric_arg);

void score_samples(const raft::handle_t& handle,
                   const double* query,
                   const double* train,
                   const double* weights,
                   double* output,
                   int n_query,
                   int n_train,
                   int n_features,
                   double bandwidth,
                   double sum_weights,
                   KernelType kernel,
                   ML::distance::DistanceType metric,
                   double metric_arg);

}  // namespace ML::KDE
