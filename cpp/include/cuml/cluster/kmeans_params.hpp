/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
