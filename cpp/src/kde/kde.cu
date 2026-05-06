/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/neighbors/kde.hpp>

#include <raft/core/device_mdspan.hpp>

#include <cuvs/distance/kde.hpp>

#include <optional>

namespace ML::KDE {

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
                   cuvs::distance::DistanceType metric,
                   T metric_arg)
{
  auto query_view =
    raft::make_device_matrix_view<const T, std::int64_t>(query, n_query, n_features);
  auto train_view =
    raft::make_device_matrix_view<const T, std::int64_t>(train, n_train, n_features);
  auto output_view = raft::make_device_vector_view<T, std::int64_t>(output, n_query);
  auto weights_view =
    weights
      ? std::make_optional(raft::make_device_vector_view<const T, std::int64_t>(weights, n_train))
      : std::nullopt;

  cuvs::distance::kde(handle,
                      query_view,
                      train_view,
                      weights_view,
                      output_view,
                      bandwidth,
                      sum_weights,
                      kernel,
                      metric,
                      metric_arg);
}

template void score_samples<float>(raft::resources const&,
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
                                   cuvs::distance::DistanceType,
                                   float);

template void score_samples<double>(raft::resources const&,
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
                                    cuvs::distance::DistanceType,
                                    double);

}  // namespace ML::KDE
