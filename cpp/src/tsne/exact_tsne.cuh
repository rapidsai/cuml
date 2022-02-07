/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include "exact_kernels.cuh"
#include "utils.cuh"
#include <cuml/common/logger.hpp>
#include <raft/cudart_utils.h>

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
 * @param[in] params: Parameters for TSNE model.
 */
template <typename value_idx, typename value_t>
value_t Exact_TSNE(value_t* VAL,
                   const value_idx* COL,
                   const value_idx* ROW,
                   const value_idx NNZ,
                   const raft::handle_t& handle,
                   value_t* Y,
                   const value_idx n,
                   const TSNEParams& params)
{
  cudaStream_t stream = handle.get_stream();
  value_t kl_div      = 0;
  const value_idx dim = params.dim;

  if (params.initialize_embeddings)
    random_vector(Y, -0.0001f, 0.0001f, n * dim, stream, params.random_state);

  // Allocate space
  //---------------------------------------------------
  CUML_LOG_DEBUG("Now allocating memory for TSNE.");
  rmm::device_uvector<value_t> norm(n, stream);
  rmm::device_uvector<value_t> Z_sum(2 * n, stream);
  rmm::device_uvector<value_t> means(dim, stream);

  rmm::device_uvector<value_t> attract(n * dim, stream);
  rmm::device_uvector<value_t> repel(n * dim, stream);

  rmm::device_uvector<value_t> velocity(n * dim, stream);
  RAFT_CUDA_TRY(
    cudaMemsetAsync(velocity.data(), 0, velocity.size() * sizeof(*velocity.data()), stream));

  rmm::device_uvector<value_t> gains(n * dim, stream);
  thrust::device_ptr<value_t> begin = thrust::device_pointer_cast(gains.data());
  thrust::fill(thrust::cuda::par.on(stream), begin, begin + n * dim, 1.0f);

  rmm::device_uvector<value_t> gradient(n * dim, stream);

  rmm::device_uvector<value_t> tmp(NNZ, stream);
  value_t* Qs      = tmp.data();
  value_t* KL_divs = tmp.data();
  //---------------------------------------------------

  // Calculate degrees of freedom
  //---------------------------------------------------
  const float degrees_of_freedom = fmaxf(dim - 1, 1);
  const float df_power           = -(degrees_of_freedom + 1.0f) / 2.0f;
  const float recp_df            = 1.0f / degrees_of_freedom;
  const float C                  = 2.0f * (degrees_of_freedom + 1.0f) / degrees_of_freedom;

  CUML_LOG_DEBUG("Start gradient updates!");
  float momentum         = params.pre_momentum;
  float learning_rate    = params.pre_learning_rate;
  auto exaggeration      = params.early_exaggeration;
  bool check_convergence = false;

  for (int iter = 0; iter < params.max_iter; iter++) {
    check_convergence = ((iter % 10) == 0) and (iter > params.exaggeration_iter);

    if (iter == params.exaggeration_iter) {
      momentum      = params.post_momentum;
      learning_rate = params.post_learning_rate;
      exaggeration  = 1.0f;
    }

    // Get row norm of Y
    raft::linalg::rowNorm(norm.data(), Y, dim, n, raft::linalg::L2Norm, false, stream);

    bool last_iter = iter == params.max_iter - 1;

    // Compute attractive forces
    TSNE::attractive_forces(VAL,
                            COL,
                            ROW,
                            Y,
                            norm.data(),
                            attract.data(),
                            last_iter ? Qs : nullptr,
                            NNZ,
                            n,
                            dim,
                            fmaxf(params.dim - 1, 1),
                            stream);

    if (last_iter) { kl_div = compute_kl_div(VAL, Qs, KL_divs, NNZ, stream); }

    // Compute repulsive forces
    const float Z = TSNE::repulsive_forces(
      Y, repel.data(), norm.data(), Z_sum.data(), n, dim, df_power, recp_df, stream);

    // Apply / integrate forces
    const float gradient_norm = TSNE::apply_forces(Y,
                                                   velocity.data(),
                                                   attract.data(),
                                                   repel.data(),
                                                   means.data(),
                                                   gains.data(),
                                                   Z,
                                                   learning_rate,
                                                   C,
                                                   exaggeration,
                                                   momentum,
                                                   dim,
                                                   n,
                                                   params.min_gain,
                                                   gradient.data(),
                                                   check_convergence,
                                                   stream);

    if (check_convergence) {
      if (iter % 100 == 0) {
        CUML_LOG_DEBUG("Z at iter = %d = %f and gradient norm = %f", iter, Z, gradient_norm);
      }
      if (gradient_norm < params.min_grad_norm) {
        CUML_LOG_DEBUG(
          "Gradient norm = %f <= min_grad_norm = %f. Early stopped at iter = "
          "%d",
          gradient_norm,
          params.min_grad_norm,
          iter);
        break;
      }
    } else {
      if (iter % 100 == 0) { CUML_LOG_DEBUG("Z at iter = %d = %f", iter, Z); }
    }
  }

  return kl_div;
}

}  // namespace TSNE
}  // namespace ML
