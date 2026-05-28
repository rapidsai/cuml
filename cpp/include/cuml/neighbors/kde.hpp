/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuml/common/distance_type.hpp>
#include <cuml/common/export.hpp>

#include <raft/core/resources.hpp>

#include <cstdint>

namespace CUML_EXPORT ML {
namespace KDE {

enum class DensityKernelType : int {
  Gaussian     = 0,
  Tophat       = 1,
  Epanechnikov = 2,
  Exponential  = 3,
  Linear       = 4,
  Cosine       = 5
};

/**
 * @brief Compute normalized log-density scores for query samples.
 *
 * The query and training arrays must be dense row-major (C-contiguous)
 * device arrays with shapes `(n_query, n_features)` and
 * `(n_train, n_features)`, respectively.
 *
 * @tparam T floating point type, either float or double
 * @param[in] handle raft resources used to launch work
 * @param[in] query device pointer to query samples in row-major order
 * @param[in] train device pointer to training samples in row-major order
 * @param[in] weights optional device pointer to sample weights of length
 * `n_train`, or nullptr for uniform weights
 * @param[out] output device pointer to log-density scores of length `n_query`
 * @param[in] n_query number of query samples
 * @param[in] n_train number of training samples
 * @param[in] n_features number of features per sample
 * @param[in] bandwidth positive KDE bandwidth
 * @param[in] sum_weights sum of `weights`, or `n_train` when weights is null
 * @param[in] kernel density kernel to evaluate
 * @param[in] metric distance metric used between query and training samples
 * @param[in] metric_arg metric-specific argument, such as p for Minkowski
 */
template <typename T>
void score_samples(raft::resources const& handle,
                   const T* query,
                   const T* train,
                   const T* weights,
                   T* output,
                   std::int64_t n_query,
                   std::int64_t n_train,
                   std::int64_t n_features,
                   T bandwidth,
                   T sum_weights,
                   DensityKernelType kernel,
                   ML::distance::DistanceType metric,
                   T metric_arg);

extern template void score_samples<float>(raft::resources const&,
                                          const float*,
                                          const float*,
                                          const float*,
                                          float*,
                                          std::int64_t,
                                          std::int64_t,
                                          std::int64_t,
                                          float,
                                          float,
                                          DensityKernelType,
                                          ML::distance::DistanceType,
                                          float);

extern template void score_samples<double>(raft::resources const&,
                                           const double*,
                                           const double*,
                                           const double*,
                                           double*,
                                           std::int64_t,
                                           std::int64_t,
                                           std::int64_t,
                                           double,
                                           double,
                                           DensityKernelType,
                                           ML::distance::DistanceType,
                                           double);

}  // namespace KDE
}  // namespace CUML_EXPORT ML
