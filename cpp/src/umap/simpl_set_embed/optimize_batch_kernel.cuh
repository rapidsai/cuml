/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <common/fast_int_div.cuh>

#include <cuml/manifold/umapparams.h>

#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <stdint.h>

#include <cstddef>

namespace UMAPAlgo {
namespace SimplSetEmbed {
namespace Algo {

using namespace ML;

/**
 * Calculate the squared distance between two vectors of size n
 * @{
 */
template <typename T>
DI T rdist(const T* X, const T* Y, int n)
{
  auto result = T(0.0);
  for (int i = 0; i < n; i++) {
    auto diff = T(X[i] - Y[i]);
    result += diff * diff;
  }
  return result;
}
template <typename T, int LEN>
DI T rdist(const T (&X)[LEN], const T (&Y)[LEN])
{
  auto result = T(0.0);
#pragma unroll
  for (int i = 0; i < LEN; ++i) {
    auto diff = T(X[i] - Y[i]);
    result += diff * diff;
  }
  return result;
}
/** @} */

/**
 * Clip a value to within a lower and upper bound
 */
template <typename T>
DI T clip(T val, T lb, T ub)
{
  return min(max(val, lb), ub);
}

/**
 * Calculate the repulsive gradient
 */
template <typename T>
DI T repulsive_grad(T dist_squared, T gamma, UMAPParams params)
{
  auto grad_coeff = T(2.0) * gamma * params.b;
  grad_coeff /= (T(0.001) + dist_squared) * (params.a * pow(dist_squared, params.b) + T(1.0));
  return grad_coeff;
}

/**
 * Calculate the attractive gradient
 */
template <typename T>
DI T attractive_grad(T dist_squared, UMAPParams params)
{
  auto grad_coeff = T(-2.0) * params.a * params.b * pow(dist_squared, params.b - T(1.0));
  grad_coeff /= params.a * pow(dist_squared, params.b) + T(1.0);
  return grad_coeff;
}

template <typename T>
DI T truncate_gradient(T const rounding_factor, T const x)
{
  return (rounding_factor + x) - rounding_factor;
}

template <typename T, typename nnz_t, int TPB_X, nnz_t n_components>
CUML_KERNEL void optimize_batch_kernel_reg(T const* head_embedding,
                                           T* head_buffer,
                                           T const* tail_embedding,
                                           T* tail_buffer,
                                           MLCommon::FastIntDiv tail_n,
                                           const int* head,
                                           const int* tail,
                                           nnz_t nnz,
                                           T const* epochs_per_sample,
                                           T* epoch_of_next_negative_sample,
                                           T* epoch_of_next_sample,
                                           T alpha,
                                           int epoch,
                                           T gamma,
                                           uint64_t seed,
                                           bool move_other,
                                           UMAPParams params,
                                           T nsr_inv,
                                           T rounding)
{
  nnz_t row = (blockIdx.x * static_cast<nnz_t>(TPB_X)) + threadIdx.x;
  if (row >= nnz) return;
  auto _epoch_of_next_sample = epoch_of_next_sample[row];
  if (_epoch_of_next_sample > epoch) return;
  auto _epochs_per_sample         = epochs_per_sample[row];
  auto epochs_per_negative_sample = _epochs_per_sample * nsr_inv;
  /**
   * Positive sample stage (attractive forces)
   */
  nnz_t j          = head[row];
  nnz_t k          = tail[row];
  T const* current = head_embedding + (j * n_components);
  T const* other   = tail_embedding + (k * n_components);

  T* cur_write = head_buffer + (j * n_components);
  T* oth_write = tail_buffer + (k * n_components);

  T current_reg[n_components], other_reg[n_components], grads[n_components];
  for (int i = 0; i < n_components; ++i) {
    current_reg[i] = current[i];
    other_reg[i]   = other[i];
  }
  auto dist_squared = rdist<T, n_components>(current_reg, other_reg);
  // Attractive force between the two vertices, since they
  // are connected by an edge in the 1-skeleton.
  auto attractive_grad_coeff = T(0.0);
  if (dist_squared > T(0.0)) { attractive_grad_coeff = attractive_grad<T>(dist_squared, params); }
  /**
   * Apply attractive force between `current` and `other`
   * by updating their 'weights' to place them relative
   * to their weight in the 1-skeleton.
   * (update `other` embedding only if we are
   * performing unsupervised training).
   */
  for (int d = 0; d < n_components; d++) {
    auto diff   = current_reg[d] - other_reg[d];
    auto grad_d = clip<T>(attractive_grad_coeff * diff, T(-4.0), T(4.0));
    current_reg[d] += grad_d * alpha;
    grads[d] = grad_d * alpha;
  }
  // storing gradients for negative samples back to global memory
  if (move_other) {
    for (int d = 0; d < n_components; d++) {
      raft::myAtomicAdd(oth_write + d, truncate_gradient(rounding, -grads[d]));
    }
  }
  epoch_of_next_sample[row] = _epoch_of_next_sample + _epochs_per_sample;
  // number of negative samples to choose
  auto _epoch_of_next_negative_sample = epoch_of_next_negative_sample[row];
  int n_neg_samples = int(T(epoch - _epoch_of_next_negative_sample) / epochs_per_negative_sample);
  /**
   * Negative sampling stage
   */
  raft::random::detail::PhiloxGenerator gen((uint64_t)seed, (nnz_t)row, 0);
  for (int p = 0; p < n_neg_samples; p++) {
    int r;
    gen.next(r);
    nnz_t t                  = r % tail_n;
    T const* negative_sample = tail_embedding + (t * n_components);
    T negative_sample_reg[n_components];
    for (int i = 0; i < n_components; ++i) {
      negative_sample_reg[i] = negative_sample[i];
    }
    dist_squared = rdist<T, n_components>(current_reg, negative_sample_reg);
    // repulsive force between two vertices
    auto repulsive_grad_coeff = T(0.0);
    if (dist_squared > T(0.0)) {
      repulsive_grad_coeff = repulsive_grad<T>(dist_squared, gamma, params);
    } else if (j == t)
      continue;
    /**
     * Apply repulsive force between `current` and `other`
     * (which has been negatively sampled) by updating
     * their 'weights' to push them farther in Euclidean space.
     */
    for (int d = 0; d < n_components; d++) {
      auto diff   = current_reg[d] - negative_sample_reg[d];
      auto grad_d = T(0.0);
      if (repulsive_grad_coeff > T(0.0))
        grad_d = clip<T>(repulsive_grad_coeff * diff, T(-4.0), T(4.0));
      else
        grad_d = T(4.0);
      current_reg[d] += grad_d * alpha;
      grads[d] += grad_d * alpha;
    }
  }
  // storing gradients for positive samples back to global memory
  for (int d = 0; d < n_components; d++) {
    raft::myAtomicAdd(cur_write + d, truncate_gradient(rounding, grads[d]));
  }
  epoch_of_next_negative_sample[row] =
    _epoch_of_next_negative_sample + n_neg_samples * epochs_per_negative_sample;
}

template <typename T, typename nnz_t, int TPB_X, bool use_shared_mem>
CUML_KERNEL void optimize_batch_kernel(T const* head_embedding,
                                       T* head_buffer,
                                       T const* tail_embedding,
                                       T* tail_buffer,
                                       MLCommon::FastIntDiv tail_n,
                                       const int* head,
                                       const int* tail,
                                       nnz_t nnz,
                                       T const* epochs_per_sample,
                                       T* epoch_of_next_negative_sample,
                                       T* epoch_of_next_sample,
                                       T alpha,
                                       int epoch,
                                       T gamma,
                                       uint64_t seed,
                                       bool move_other,
                                       UMAPParams params,
                                       T nsr_inv,
                                       T rounding)
{
  extern __shared__ T embedding_shared_mem_updates[];
  nnz_t row = (blockIdx.x * static_cast<nnz_t>(TPB_X)) + threadIdx.x;
  if (row >= nnz) return;
  auto _epoch_of_next_sample = epoch_of_next_sample[row];
  if (_epoch_of_next_sample > epoch) return;
  auto _epochs_per_sample         = epochs_per_sample[row];
  auto epochs_per_negative_sample = _epochs_per_sample * nsr_inv;
  /**
   * Positive sample stage (attractive forces)
   */
  nnz_t n_components = params.n_components;

  nnz_t j          = head[row];
  nnz_t k          = tail[row];
  T const* current = head_embedding + (j * n_components);
  T const* other   = tail_embedding + (k * n_components);

  T* cur_write = head_buffer + (j * n_components);
  T* oth_write = tail_buffer + (k * n_components);

  // for reducing access to global memory. load values from global memory, and accumulate grads onto
  // this shared memory position instead of reading from global memory every time.
  T* current_buffer{nullptr};
  // for keeping track of grads, final write to global memory
  T* grads_buffer{nullptr};
  if constexpr (use_shared_mem) {
    // n_components for thread0, then the next n_components for thread1 ...
    current_buffer = (T*)embedding_shared_mem_updates + threadIdx.x * n_components;
    // TPB_X for first component, then another TPB_X for the next component for better coalescing...
    grads_buffer = (T*)embedding_shared_mem_updates + TPB_X * n_components + threadIdx.x;
  }
  auto dist_squared = rdist<T>(current, other, n_components);
  // Attractive force between the two vertices, since they
  // are connected by an edge in the 1-skeleton.
  auto attractive_grad_coeff = T(0.0);
  if (dist_squared > T(0.0)) { attractive_grad_coeff = attractive_grad<T>(dist_squared, params); }
  /**
   * Apply attractive force between `current` and `other`
   * by updating their 'weights' to place them relative
   * to their weight in the 1-skeleton.
   * (update `other` embedding only if we are
   * performing unsupervised training).
   */
  for (int d = 0; d < n_components; d++) {
    T current_val = current[d];
    if constexpr (use_shared_mem) { current_buffer[d] = current_val; }
    auto grad_d = clip<T>(attractive_grad_coeff * (current_val - other[d]), T(-4.0), T(4.0));
    grad_d *= alpha;
    if constexpr (use_shared_mem) {
      current_buffer[d] += grad_d;
      grads_buffer[d * TPB_X] = grad_d;
    } else {
      raft::myAtomicAdd<T>((T*)cur_write + d, truncate_gradient(rounding, grad_d));
      if (move_other) {  // happens only during unsupervised training
        raft::myAtomicAdd<T>((T*)oth_write + d, truncate_gradient(rounding, -grad_d));
      }
    }
  }
  // storing gradients for negative samples back to global memory
  if (use_shared_mem && move_other) {
    __syncthreads();
    for (int d = 0; d < n_components; d++) {
      auto grad = grads_buffer[d * TPB_X];
      raft::myAtomicAdd<T>((T*)oth_write + d, truncate_gradient(rounding, -grad));
    }
  }
  epoch_of_next_sample[row] = _epoch_of_next_sample + _epochs_per_sample;
  // number of negative samples to choose
  auto _epoch_of_next_negative_sample = epoch_of_next_negative_sample[row];
  int n_neg_samples = int(T(epoch - _epoch_of_next_negative_sample) / epochs_per_negative_sample);
  /**
   * Negative sampling stage
   */
  raft::random::detail::PhiloxGenerator gen((uint64_t)seed, (nnz_t)row, 0);
  for (int p = 0; p < n_neg_samples; p++) {
    int r;
    gen.next(r);
    nnz_t t                  = r % tail_n;
    T const* negative_sample = tail_embedding + (t * n_components);
    if constexpr (use_shared_mem) {
      dist_squared = rdist<T>(current_buffer, negative_sample, n_components);
    } else {
      dist_squared = rdist<T>(current, negative_sample, n_components);
    }
    // repulsive force between two vertices
    auto repulsive_grad_coeff = T(0.0);
    if (dist_squared > T(0.0)) {
      repulsive_grad_coeff = repulsive_grad<T>(dist_squared, gamma, params);
    } else if (j == t)
      continue;
    /**
     * Apply repulsive force between `current` and `other`
     * (which has been negatively sampled) by updating
     * their 'weights' to push them farther in Euclidean space.
     */
    for (int d = 0; d < n_components; d++) {
      auto grad_d = T(0.0);
      if (repulsive_grad_coeff > T(0.0)) {
        if constexpr (use_shared_mem) {
          grad_d = clip<T>(
            repulsive_grad_coeff * (current_buffer[d] - negative_sample[d]), T(-4.0), T(4.0));
        } else {
          grad_d =
            clip<T>(repulsive_grad_coeff * (current[d] - negative_sample[d]), T(-4.0), T(4.0));
        }
      } else
        grad_d = T(4.0);
      grad_d *= alpha;
      if constexpr (use_shared_mem) {
        current_buffer[d] += grad_d;
        grads_buffer[d * TPB_X] += grad_d;
      } else {
        raft::myAtomicAdd<T>((T*)cur_write + d, truncate_gradient(rounding, grad_d));
      }
    }
  }

  // storing gradients for positive samples back to global memory
  if constexpr (use_shared_mem) {
    __syncthreads();
    for (int d = 0; d < n_components; d++) {
      raft::myAtomicAdd<T>((T*)cur_write + d, truncate_gradient(rounding, grads_buffer[d * TPB_X]));
    }
  }
  epoch_of_next_negative_sample[row] =
    _epoch_of_next_negative_sample + n_neg_samples * epochs_per_negative_sample;
}

/**
 * @param head_buffer: Buffer the gradient update to head_embedding when deterministic
 *                     result is required.  They are the same pointer if random seed is not
 *                     provided.
 * @param tail_buffer: Similar to head_buffer, but for tail_embedding.
 * @param head:        Row index in COO connectivity graph.
 * @param tail:        Column index in COO connectivity graph.
 * @param alpha:       Learning rate
 * @param n:           Current epoch
 * @param rounding:    Floating rounding factor used to truncate the gradient update for
 *                     deterministic result.
 */
template <typename T, typename nnz_t, int TPB_X>
void call_optimize_batch_kernel(T const* head_embedding,
                                T* head_buffer,
                                T const* tail_embedding,
                                T* tail_buffer,
                                int tail_n,
                                const int* head,
                                const int* tail,
                                nnz_t nnz,
                                T const* epochs_per_sample,
                                T* epoch_of_next_negative_sample,
                                T* epoch_of_next_sample,
                                T alpha,
                                T gamma,
                                uint64_t seed,
                                bool move_other,
                                UMAPParams const* params,
                                int n,
                                dim3& grid,
                                dim3& blk,
                                cudaStream_t& stream,
                                T rounding)
{
  std::size_t requiredSize = TPB_X * params->n_components * 2;
  requiredSize *= sizeof(T);
  bool use_shared_mem = requiredSize < static_cast<std::size_t>(raft::getSharedMemPerBlock());
  T nsr_inv           = T(1.0) / params->negative_sample_rate;
  if (params->n_components == 2) {
    // multicore implementation with registers
    optimize_batch_kernel_reg<T, nnz_t, TPB_X, 2>
      <<<grid, blk, 0, stream>>>(head_embedding,
                                 head_buffer,
                                 tail_embedding,
                                 tail_buffer,
                                 tail_n,
                                 head,
                                 tail,
                                 nnz,
                                 epochs_per_sample,
                                 epoch_of_next_negative_sample,
                                 epoch_of_next_sample,
                                 alpha,
                                 n,
                                 gamma,
                                 seed,
                                 move_other,
                                 *params,
                                 nsr_inv,
                                 rounding);
  } else if (use_shared_mem) {
    // multicore implementation with shared memory
    optimize_batch_kernel<T, nnz_t, TPB_X, true>
      <<<grid, blk, requiredSize, stream>>>(head_embedding,
                                            head_buffer,
                                            tail_embedding,
                                            tail_buffer,
                                            tail_n,
                                            head,
                                            tail,
                                            nnz,
                                            epochs_per_sample,
                                            epoch_of_next_negative_sample,
                                            epoch_of_next_sample,
                                            alpha,
                                            n,
                                            gamma,
                                            seed,
                                            move_other,
                                            *params,
                                            nsr_inv,
                                            rounding);
  } else {
    // multicore implementation without shared memory
    optimize_batch_kernel<T, nnz_t, TPB_X, false>
      <<<grid, blk, 0, stream>>>(head_embedding,
                                 head_buffer,
                                 tail_embedding,
                                 tail_buffer,
                                 tail_n,
                                 head,
                                 tail,
                                 nnz,
                                 epochs_per_sample,
                                 epoch_of_next_negative_sample,
                                 epoch_of_next_sample,
                                 alpha,
                                 n,
                                 gamma,
                                 seed,
                                 move_other,
                                 *params,
                                 nsr_inv,
                                 rounding);
  }
}
}  // namespace Algo
}  // namespace SimplSetEmbed
}  // namespace UMAPAlgo
