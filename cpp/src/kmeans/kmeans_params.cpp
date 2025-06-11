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
