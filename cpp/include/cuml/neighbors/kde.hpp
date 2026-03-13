/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/resources.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/distance/kde.hpp>

namespace ML::KDE {

using cuvs::distance::DensityKernelType;

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
                   T metric_arg);

extern template void score_samples<float>(raft::resources const&,
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

extern template void score_samples<double>(raft::resources const&,
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
