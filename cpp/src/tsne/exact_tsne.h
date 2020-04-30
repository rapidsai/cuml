/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <common/cudart_utils.h>
#include <cuml/common/logger.hpp>
#include "exact_kernels.h"
#include "utils.h"

namespace ML {
namespace TSNE {

/**
 * @brief Slower Dimensionality reduction via TSNE using the Exact method O(N^2).
 * @param[in] VAL: The values in the attractive forces COO matrix.
 * @param[in] COL: The column indices in the attractive forces COO matrix.
 * @param[in] ROW: The row indices in the attractive forces COO matrix.
 * @param[in] NNZ: The number of non zeros in the attractive forces COO matrix.
 * @param[in] handle: The GPU handle.
 * @param[out] Y: The final embedding. Will overwrite this internally.
 * @param[in] n: Number of rows in data X.
 * @param[in] dim: Number of output columns for the output embedding Y.
 * @param[in] early_exaggeration: How much early pressure you want the clusters in TSNE to spread out more.
 * @param[in] exaggeration_iter: How many iterations you want the early pressure to run for.
 * @param[in] min_gain: Rounds up small gradient updates.
 * @param[in] pre_learning_rate: The learning rate during the exaggeration phase.
 * @param[in] post_learning_rate: The learning rate after the exaggeration phase.
 * @param[in] max_iter: The maximum number of iterations TSNE should run for.
 * @param[in] min_grad_norm: The smallest gradient norm TSNE should terminate on.
 * @param[in] pre_momentum: The momentum used during the exaggeration phase.
 * @param[in] post_momentum: The momentum used after the exaggeration phase.
 * @param[in] random_state: Set this to -1 for pure random intializations or >= 0 for reproducible outputs.
 * @param[in] intialize_embeddings: Whether to overwrite the current Y vector with random noise.
 */
void Exact_TSNE(float *VAL, const int *COL, const int *ROW, const int NNZ,
                const cumlHandle &handle, float *Y, const int n, const int dim,
                const float early_exaggeration = 12.0f,
                const int exaggeration_iter = 250, const float min_gain = 0.01f,
                const float pre_learning_rate = 200.0f,
                const float post_learning_rate = 500.0f,
                const int max_iter = 1000, const float min_grad_norm = 1e-7,
                const float pre_momentum = 0.5, const float post_momentum = 0.8,
                const long long random_state = -1,
                const bool intialize_embeddings = true) {
  auto d_alloc = handle.getDeviceAllocator();
  cudaStream_t stream = handle.getStream();

  if (intialize_embeddings)
    random_vector(Y, -0.0001f, 0.0001f, n * dim, stream, random_state);

  // Allocate space
  //---------------------------------------------------
  CUML_LOG_DEBUG("Now allocating memory for TSNE.");
  float *norm = (float *)d_alloc->allocate(sizeof(float) * n, stream);
  float *Z_sum = (float *)d_alloc->allocate(sizeof(float) * 2 * n, stream);
  float *means = (float *)d_alloc->allocate(sizeof(float) * dim, stream);

  float *attract = (float *)d_alloc->allocate(sizeof(float) * n * dim, stream);
  float *repel = (float *)d_alloc->allocate(sizeof(float) * n * dim, stream);

  float *velocity = (float *)d_alloc->allocate(sizeof(float) * n * dim, stream);
  CUDA_CHECK(cudaMemsetAsync(velocity, 0, sizeof(float) * n * dim, stream));

  float *gains = (float *)d_alloc->allocate(sizeof(float) * n * dim, stream);
  thrust::device_ptr<float> begin = thrust::device_pointer_cast(gains);
  thrust::fill(thrust::cuda::par.on(stream), begin, begin + n * dim, 1.0f);

  float *gradient = (float *)d_alloc->allocate(sizeof(float) * n * dim, stream);
  //---------------------------------------------------

  // Calculate degrees of freedom
  //---------------------------------------------------
  const float degrees_of_freedom = fmaxf(dim - 1, 1);
  const float df_power = -(degrees_of_freedom + 1.0f) / 2.0f;
  const float recp_df = 1.0f / degrees_of_freedom;
  const float C = 2.0f * (degrees_of_freedom + 1.0f) / degrees_of_freedom;

  CUML_LOG_DEBUG("Start gradient updates!");
  float momentum = pre_momentum;
  float learning_rate = pre_learning_rate;
  bool check_convergence = false;

  for (int iter = 0; iter < max_iter; iter++) {
    check_convergence = ((iter % 10) == 0) and (iter > exaggeration_iter);

    if (iter == exaggeration_iter) {
      momentum = post_momentum;
      // Divide perplexities
      const float div = 1.0f / early_exaggeration;
      MLCommon::LinAlg::scalarMultiply(VAL, VAL, div, NNZ, stream);
      learning_rate = post_learning_rate;
    }

    // Get row norm of Y
    MLCommon::LinAlg::rowNorm(norm, Y, dim, n, MLCommon::LinAlg::L2Norm, false,
                              stream);

    // Compute attractive forces
    TSNE::attractive_forces(VAL, COL, ROW, Y, norm, attract, NNZ, n, dim,
                            df_power, recp_df, stream);
    // Compute repulsive forces
    const float Z = TSNE::repulsive_forces(Y, repel, norm, Z_sum, n, dim,
                                           df_power, recp_df, stream);

    // Apply / integrate forces
    const float gradient_norm = TSNE::apply_forces(
      Y, velocity, attract, repel, means, gains, Z, learning_rate, C, momentum,
      dim, n, min_gain, gradient, check_convergence, stream);

    if (check_convergence) {
      CUML_LOG_DEBUG("Z at iter = %d = %f and gradient norm = %f", iter, Z,
                     gradient_norm);
      if (gradient_norm < min_grad_norm) {
        CUML_LOG_DEBUG(
          "Gradient norm = %f <= min_grad_norm = %f. Early stopped at iter = "
          "%d",
          gradient_norm, min_grad_norm, iter);
        break;
      }
    } else {
      CUML_LOG_DEBUG("Z at iter = %d = %f", iter, Z);
    }
  }

  d_alloc->deallocate(norm, sizeof(float) * n, stream);
  d_alloc->deallocate(Z_sum, sizeof(float) * 2 * n, stream);
  d_alloc->deallocate(means, sizeof(float) * dim, stream);

  d_alloc->deallocate(attract, sizeof(float) * n * dim, stream);
  d_alloc->deallocate(repel, sizeof(float) * n * dim, stream);

  d_alloc->deallocate(velocity, sizeof(float) * n * dim, stream);
  d_alloc->deallocate(gains, sizeof(float) * n * dim, stream);
  d_alloc->deallocate(gradient, sizeof(float) * n * dim, stream);
}

}  // namespace TSNE
}  // namespace ML
