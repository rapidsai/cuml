/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "umap_api.h"
#include "umap.hpp"

void initUmapParams(ML::UMAPParams *params) {
  params->n_neighbors = 15;
  params->n_components = 2;
  params->n_epochs = 500;
  params->learning_rate = 1.0;
  params->min_dist = 0.1;
  params->spread = 1.0;
  params->set_op_mix_ratio = 1.0;
  params->local_connectivity = 1.0;
  params->repulsion_strength = 1.0;
  params->negative_sample_rate = 5;
  params->transform_queue_size = 4.0;
  params->verbose = false;
  params->init = 1;                 // spectral layout
  params->target_n_neighbors = -1;  // -1 => use the n_neighbors value
  params->target_weights = 0.5;
  params->target_metric = ML::UMAPParams::MetricType::EUCLIDEAN;
}

cumlError_t cumlSpUmapFitSupervised(cumlHandle_t handle, float *X, float *y,
                                    int num_samples, int num_dims,
                                    float *embeddings, int n_components) {
  cumlError_t status;
  ML::cumlHandle *handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      ML::UMAPParams params{};
      initUmapParams(&params);
      params.n_components = n_components;
      ML::fit(*handle_ptr, X, y, num_samples, num_dims, &params, embeddings);
    } catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}

cumlError_t cumlSpUmapFit(cumlHandle_t handle, float *X, int num_samples,
                          int num_dims, float *embeddings, int n_components) {
  cumlError_t status;
  ML::cumlHandle *handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      ML::UMAPParams params{};
      initUmapParams(&params);
      params.n_components = n_components;
      ML::fit(*handle_ptr, X, num_samples, num_dims, &params, embeddings);
    } catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}
