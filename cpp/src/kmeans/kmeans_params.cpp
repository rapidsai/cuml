/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/cluster/kmeans_params.hpp>

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>

namespace ML::kmeans {

cuvs::cluster::kmeans::params KMeansParams::to_cuvs() const
{
  cuvs::cluster::kmeans::params params;

  params.metric              = static_cast<cuvs::distance::DistanceType>(this->metric);
  params.n_clusters          = this->n_clusters;
  params.init                = static_cast<cuvs::cluster::kmeans::params::InitMethod>(this->init);
  params.max_iter            = this->max_iter;
  params.tol                 = this->tol;
  params.verbosity           = this->verbosity;
  params.rng_state           = this->rng_state;
  params.n_init              = this->n_init;
  params.oversampling_factor = this->oversampling_factor;
  params.batch_samples       = this->batch_samples;
  params.batch_centroids     = this->batch_centroids;
  params.inertia_check       = this->inertia_check;

  return params;
}

}  // end namespace ML::kmeans
