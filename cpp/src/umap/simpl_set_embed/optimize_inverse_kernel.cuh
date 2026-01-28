/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "optimize_batch_kernel.cuh"  // For clip() and other utility functions

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
 *
 * @tparam use_shared_mem If true, uses shared memory for per-thread buffers.
 *                        If false, reads/writes directly from global memory.
 *                        Should be false when n_components is large enough
 *                        that shared memory would exceed device limits.
 */
template <typename T, typename nnz_t, int TPB_X, bool use_shared_mem>
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
  // Shared memory pointers (only used when use_shared_mem == true)
  extern __shared__ char shared_mem[];
  T* current  = nullptr;
  T* original = nullptr;
  T* grad     = nullptr;

  if constexpr (use_shared_mem) {
    current  = reinterpret_cast<T*>(shared_mem) + threadIdx.x * n_components * 3;
    original = current + n_components;
    grad     = original + n_components;
  }

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

    // Compute distance and gradient to positive sample
    T dist_sq = T(0);
    if constexpr (use_shared_mem) {
      // Load current point and store original for delta computation
      for (int i = 0; i < n_components; ++i) {
        current[i]  = current_ptr[i];
        original[i] = current_ptr[i];
      }
      for (int i = 0; i < n_components; ++i) {
        T diff  = current[i] - other[i];
        grad[i] = diff;
        dist_sq += diff * diff;
      }
    } else {
      for (int i = 0; i < n_components; ++i) {
        T diff = current_ptr[i] - other[i];
        dist_sq += diff * diff;
      }
    }
    T dist = sqrt(dist_sq);

    // Attractive force: grad_coeff = -1 / (weight * sigma + eps)
    T grad_coeff = T(-1) / (weight[row] * sigmas[k] + T(1e-6));
    if constexpr (use_shared_mem) {
      if (dist > T(1e-10)) {
        for (int i = 0; i < n_components; ++i)
          grad[i] /= dist;
      }
      for (int d = 0; d < n_components; ++d)
        current[d] += clip(grad_coeff * grad[d], T(-4), T(4)) * alpha;
    } else {
      // Without shared memory, compute and apply gradient directly
      T dist_inv = (dist > T(1e-10)) ? T(1) / dist : T(0);
      for (int d = 0; d < n_components; ++d) {
        T diff   = current_ptr[d] - other[d];
        T grad_d = diff * dist_inv;
        T update = clip(grad_coeff * grad_d, T(-4), T(4)) * alpha;
        raft::myAtomicAdd(current_ptr + d, update);
      }
    }

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
      if constexpr (use_shared_mem) {
        for (int i = 0; i < n_components; ++i) {
          T diff  = current[i] - negative[i];
          grad[i] = diff;
          dist_sq += diff * diff;
        }
      } else {
        for (int i = 0; i < n_components; ++i) {
          T diff = current_ptr[i] - negative[i];
          dist_sq += diff * diff;
        }
      }
      dist = sqrt(dist_sq);

      // Repulsive force
      // Note: max(..., 1e-6) matches umap-learn's implementation for numerical stability,
      // slightly dampening w_h near rho to avoid edge cases in the gradient computation.
      T w_h      = exp(-max(dist - rhos[t], T(1e-6)) / (sigmas[t] + T(1e-6)));
      grad_coeff = gamma * w_h / ((T(1) - w_h) * sigmas[t] + T(1e-6));

      if constexpr (use_shared_mem) {
        if (dist > T(1e-10)) {
          for (int i = 0; i < n_components; ++i)
            grad[i] /= dist;
        }
        for (int d = 0; d < n_components; ++d)
          current[d] += clip(grad_coeff * grad[d], T(-4), T(4)) * alpha;
      } else {
        T dist_inv = (dist > T(1e-10)) ? T(1) / dist : T(0);
        for (int d = 0; d < n_components; ++d) {
          T diff   = current_ptr[d] - negative[d];
          T grad_d = diff * dist_inv;
          T update = clip(grad_coeff * grad_d, T(-4), T(4)) * alpha;
          raft::myAtomicAdd(current_ptr + d, update);
        }
      }
    }

    epoch_of_next_negative_sample[row] = next_neg + n_neg * epochs_per_negative_sample;

    // Write back delta atomically (only needed for shared memory path)
    if constexpr (use_shared_mem) {
      for (int d = 0; d < n_components; ++d)
        raft::myAtomicAdd(current_ptr + d, current[d] - original[d]);
    }

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
                             int n_features,
                             cudaStream_t stream)
{
  int n_components = n_features;
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
  uint64_t seed = params->random_state;

  // Check if shared memory requirements exceed device limits
  // Each thread needs 3 * n_components elements (current, original, grad)
  size_t required_shared_size = TPB_X * n_components * 3 * sizeof(T);
  bool use_shared_mem = required_shared_size < static_cast<size_t>(raft::getSharedMemPerBlock());

  for (int epoch = 0; epoch < n_epochs; ++epoch) {
    if (use_shared_mem) {
      optimize_inverse_kernel<T, nnz_t, TPB_X, true>
        <<<grid, blk, required_shared_size, stream>>>(head_embedding,
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
    } else {
      // Fallback: read/write directly from global memory when shared memory
      // is insufficient (e.g., for high-dimensional feature spaces)
      optimize_inverse_kernel<T, nnz_t, TPB_X, false>
        <<<grid, blk, 0, stream>>>(head_embedding,
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
    }
    RAFT_CUDA_TRY(cudaGetLastError());

    alpha = (params->initial_alpha / T(4)) * (T(1) - T(epoch) / T(n_epochs));
    seed++;
  }
}

}  // namespace Algo
}  // namespace SimplSetEmbed
}  // namespace UMAPAlgo
