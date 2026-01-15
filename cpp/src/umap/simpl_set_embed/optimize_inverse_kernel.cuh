/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "optimize_batch_kernel.cuh"

#include <cuml/manifold/umapparams.h>

#include <raft/linalg/unary_op.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

namespace UMAPAlgo {
namespace SimplSetEmbed {
namespace Algo {

using namespace ML;

/**
 * Kernel for inverse transform optimization.
 *
 * Optimizes points in the original high-dimensional space to minimize
 * the cross-entropy between the graph in embedding space and the
 * reconstructed positions.
 */
template <typename T, typename nnz_t, int TPB_X>
CUML_KERNEL void optimize_inverse_kernel(T* head_embedding,
                                         const T* tail_embedding,
                                         const int* head,
                                         const int* tail,
                                         const T* weight,
                                         const T* sigmas,
                                         const T* rhos,
                                         nnz_t nnz,
                                         const T* epochs_per_sample,
                                         T* epoch_of_next_negative_sample,
                                         T* epoch_of_next_sample,
                                         T alpha,
                                         int epoch,
                                         T gamma,
                                         uint64_t seed,
                                         int n_vertices,
                                         int n_components,
                                         T nsr_inv)
{
  extern __shared__ char shared_mem[];
  T* current = reinterpret_cast<T*>(shared_mem) + threadIdx.x * n_components * 2;
  T* grad    = current + n_components;

  size_t row = blockIdx.x * TPB_X + threadIdx.x;

  while (row < nnz) {
    if (epoch_of_next_sample[row] > epoch) {
      row += gridDim.x * TPB_X;
      continue;
    }

    T eps                        = epochs_per_sample[row];
    T epochs_per_negative_sample = eps * nsr_inv;
    nnz_t j                      = head[row];
    nnz_t k                      = tail[row];
    T* current_ptr               = head_embedding + j * n_components;
    const T* other               = tail_embedding + k * n_components;

    // Load current point
    for (int i = 0; i < n_components; ++i)
      current[i] = current_ptr[i];

    // Compute distance and gradient to positive sample
    T dist_sq = T(0);
    for (int i = 0; i < n_components; ++i) {
      T diff  = current[i] - other[i];
      grad[i] = diff;
      dist_sq += diff * diff;
    }
    T dist = sqrt(dist_sq);
    if (dist > T(1e-10)) {
      for (int i = 0; i < n_components; ++i)
        grad[i] /= dist;
    }

    // Attractive force: grad_coeff = -1 / (weight * sigma + eps)
    T grad_coeff = T(-1) / (weight[row] * sigmas[k] + T(1e-6));
    for (int d = 0; d < n_components; ++d)
      current[d] += clip(grad_coeff * grad[d], T(-4), T(4)) * alpha;

    epoch_of_next_sample[row] += eps;

    // Negative sampling
    T next_neg = epoch_of_next_negative_sample[row];
    int n_neg  = int((epoch - next_neg) / epochs_per_negative_sample);

    raft::random::detail::PhiloxGenerator gen(seed, row, 0);
    for (int p = 0; p < n_neg; ++p) {
      int r;
      gen.next(r);
      nnz_t t           = abs(r) % n_vertices;
      const T* negative = tail_embedding + t * n_components;

      // Compute distance to negative sample
      dist_sq = T(0);
      for (int i = 0; i < n_components; ++i) {
        T diff  = current[i] - negative[i];
        grad[i] = diff;
        dist_sq += diff * diff;
      }
      dist = sqrt(dist_sq);
      if (dist > T(1e-10)) {
        for (int i = 0; i < n_components; ++i)
          grad[i] /= dist;
      }

      // Repulsive force
      T w_h      = exp(-max(dist - rhos[t], T(1e-6)) / (sigmas[t] + T(1e-6)));
      grad_coeff = gamma * w_h / ((T(1) - w_h) * sigmas[t] + T(1e-6));

      for (int d = 0; d < n_components; ++d)
        current[d] += clip(grad_coeff * grad[d], T(-4), T(4)) * alpha;
    }

    epoch_of_next_negative_sample[row] = next_neg + n_neg * epochs_per_negative_sample;

    // Write back
    for (int d = 0; d < n_components; ++d)
      current_ptr[d] = current[d];

    row += gridDim.x * TPB_X;
  }
}

/**
 * Optimize layout for inverse transform.
 *
 * Runs SGD optimization to refine inverse-transformed points
 * in the original high-dimensional space.
 */
template <typename T, typename nnz_t, int TPB_X>
void optimize_layout_inverse(T* head_embedding,
                             int head_n,
                             const T* tail_embedding,
                             int tail_n,
                             const int* head,
                             const int* tail,
                             const T* weight,
                             const T* sigmas,
                             const T* rhos,
                             nnz_t nnz,
                             T* epochs_per_sample,
                             float gamma,
                             UMAPParams* params,
                             int n_epochs,
                             cudaStream_t stream)
{
  int n_components = params->n_components;
  T alpha          = params->initial_alpha / T(4);
  T nsr_inv        = T(1) / params->negative_sample_rate;

  rmm::device_uvector<T> epoch_of_next_negative_sample(nnz, stream);
  raft::linalg::unaryOp<T>(
    epoch_of_next_negative_sample.data(),
    epochs_per_sample,
    nnz,
    [=] __device__(T x) { return x * nsr_inv; },
    stream);

  rmm::device_uvector<T> epoch_of_next_sample(nnz, stream);
  raft::copy(epoch_of_next_sample.data(), epochs_per_sample, nnz, stream);

  dim3 grid(raft::ceildiv(nnz, static_cast<nnz_t>(TPB_X)));
  dim3 blk(TPB_X);
  size_t shared_size = TPB_X * n_components * 2 * sizeof(T);
  uint64_t seed      = params->random_state;

  for (int epoch = 0; epoch < n_epochs; ++epoch) {
    optimize_inverse_kernel<T, nnz_t, TPB_X>
      <<<grid, blk, shared_size, stream>>>(head_embedding,
                                           tail_embedding,
                                           head,
                                           tail,
                                           weight,
                                           sigmas,
                                           rhos,
                                           nnz,
                                           epochs_per_sample,
                                           epoch_of_next_negative_sample.data(),
                                           epoch_of_next_sample.data(),
                                           alpha,
                                           epoch,
                                           gamma,
                                           seed,
                                           tail_n,
                                           n_components,
                                           nsr_inv);
    RAFT_CUDA_TRY(cudaGetLastError());

    alpha = (params->initial_alpha / T(4)) * (T(1) - T(epoch) / T(n_epochs));
    seed++;
  }
}

}  // namespace Algo
}  // namespace SimplSetEmbed
}  // namespace UMAPAlgo
