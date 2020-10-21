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
#include <common/device_buffer.hpp>
#include <cuml/common/logger.hpp>
#include "exact_kernels.cuh"
#include "utils.cuh"

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
 * @param[in] initialize_embeddings: Whether to overwrite the current Y vector with random noise.
 */
void Exact_TSNE(float *VAL, const int *COL, const int *ROW, const int NNZ,
                const raft::handle_t &handle, float *Y, const int n,
                const int dim, const float early_exaggeration = 12.0f,
                const int exaggeration_iter = 250, const float min_gain = 0.01f,
                const float pre_learning_rate = 200.0f,
                const float post_learning_rate = 500.0f,
                const int max_iter = 1000, const float min_grad_norm = 1e-7,
                const float pre_momentum = 0.5, const float post_momentum = 0.8,
                const long long random_state = -1,
                const bool initialize_embeddings = true) {
  auto d_alloc = handle.get_device_allocator();
  cudaStream_t stream = handle.get_stream();

  if (initialize_embeddings)
    random_vector(Y, -0.0001f, 0.0001f, n * dim, stream, random_state);

  // Allocate space
  //---------------------------------------------------
  CUML_LOG_DEBUG("Now allocating memory for TSNE.");
  MLCommon::device_buffer<float> norm(d_alloc, stream, n);
  MLCommon::device_buffer<float> Z_sum(d_alloc, stream, 2 * n);
  MLCommon::device_buffer<float> means(d_alloc, stream, dim);

  MLCommon::device_buffer<float> attract(d_alloc, stream, n * dim);
  MLCommon::device_buffer<float> repel(d_alloc, stream, n * dim);

  MLCommon::device_buffer<float> velocity(d_alloc, stream, n * dim);
  CUDA_CHECK(cudaMemsetAsync(
    velocity.data(), 0, velocity.size() * sizeof(*velocity.data()), stream));

  MLCommon::device_buffer<float> gains(d_alloc, stream, n * dim);
  thrust::device_ptr<float> begin = thrust::device_pointer_cast(gains.data());
  thrust::fill(thrust::cuda::par.on(stream), begin, begin + n * dim, 1.0f);

  MLCommon::device_buffer<float> gradient(d_alloc, stream, n * dim);
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
      raft::linalg::scalarMultiply(VAL, VAL, div, NNZ, stream);
      learning_rate = post_learning_rate;
    }

    // Get row norm of Y
    MLCommon::LinAlg::rowNorm(norm.data(), Y, dim, n, MLCommon::LinAlg::L2Norm,
                              false, stream);

    // Compute attractive forces
    TSNE::attractive_forces(VAL, COL, ROW, Y, norm.data(), attract.data(), NNZ,
                            n, dim, df_power, recp_df, stream);
    // Compute repulsive forces
    const float Z =
      TSNE::repulsive_forces(Y, repel.data(), norm.data(), Z_sum.data(), n, dim,
                             df_power, recp_df, stream);

    // Apply / integrate forces
    const float gradient_norm = TSNE::apply_forces(
      Y, velocity.data(), attract.data(), repel.data(), means.data(),
      gains.data(), Z, learning_rate, C, momentum, dim, n, min_gain,
      gradient.data(), check_convergence, stream);

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
}

}  // namespace TSNE
}  // namespace ML
