/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/distance_type.hpp>

#include <raft/random/rng_state.hpp>

#include <rapids_logger/logger.hpp>

namespace cuvs::cluster::kmeans {

struct params;

}  // end namespace cuvs::cluster::kmeans

namespace ML::kmeans {

struct KMeansParams {
  enum class InitMethod { KMeansPlusPlus, Random, Array };
  ML::distance::DistanceType metric   = ML::distance::DistanceType::L2Expanded;
  int n_clusters                      = 8;
  InitMethod init                     = InitMethod::KMeansPlusPlus;
  int max_iter                        = 300;
  double tol                          = 1e-4;
  rapids_logger::level_enum verbosity = rapids_logger::level_enum::info;
  raft::random::RngState rng_state{0};
  int n_init                 = 1;
  double oversampling_factor = 2.0;
  int batch_samples          = 1 << 15;
  int batch_centroids        = 0;
  bool inertia_check         = false;

  cuvs::cluster::kmeans::params to_cuvs() const;
};

}  // end namespace ML::kmeans
