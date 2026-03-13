/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/neighbors/kde.hpp>

#include <cuvs/distance/kde.hpp>

namespace ML::KDE {

template <typename T>
void score_samples(raft::resources const& handle,
                   const T* query,
                   const T* train,
                   const T* weights,
                   T* output,
                   int n_query,
                   int n_train,
                   int n_features,
                   T bandwidth,
                   T sum_weights,
                   DensityKernelType kernel,
                   cuvs::distance::DistanceType metric,
                   T metric_arg)
{
  cuvs::distance::kde_score_samples(handle,
                                    query,
                                    train,
                                    weights,
                                    output,
                                    n_query,
                                    n_train,
                                    n_features,
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
                                   int,
                                   int,
                                   int,
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
                                    int,
                                    int,
                                    int,
                                    double,
                                    double,
                                    DensityKernelType,
                                    cuvs::distance::DistanceType,
                                    double);

}  // namespace ML::KDE
