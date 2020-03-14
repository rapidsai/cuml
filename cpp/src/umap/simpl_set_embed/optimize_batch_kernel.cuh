/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cuda_utils.h>
#include <cuml/manifold/umapparams.h>
#include <common/fast_int_div.cuh>

namespace UMAPAlgo {
namespace SimplSetEmbed {
namespace Algo {

using namespace ML;

/**
 * Calculate the squared distance between two vectors of size n
 */
template <typename T, typename T2>
DI T2 rdist(const T *X, const T *Y, int n) {
  auto result = T2(0.0);
  for (int i = 0; i < n; i++) {
    auto diff = T2(X[i] - Y[i]);
    result += diff * diff;
  }
  return result;
}

/**
 * Clip a value to within a lower and upper bound
 */
template <typename T2>
DI T2 clip(T2 val, T2 lb, T2 ub) {
  return min(max(val, lb), ub);
}

/**
 * Calculate the repulsive gradient
 */
template <typename T2>
DI T2 repulsive_grad(T2 dist_squared, T2 gamma, UMAPParams params) {
  auto grad_coeff = T2(2.0) * gamma * params.b;
  grad_coeff /= (T2(0.001) + dist_squared) *
                (params.a * pow(dist_squared, params.b) + T2(1.0));
  return grad_coeff;
}

/**
 * Calculate the attractive gradient
 */
template <typename T2>
DI T2 attractive_grad(T2 dist_squared, UMAPParams params) {
  auto grad_coeff =
    T2(-2.0) * params.a * params.b * pow(dist_squared, params.b - T2(1.0));
  grad_coeff /= params.a * pow(dist_squared, params.b) + T2(1.0);
  return grad_coeff;
}

template <typename T, typename T2, int TPB_X, bool multicore_implem,
          bool use_shared_mem>
__global__ void optimize_batch_kernel(
  T *head_embedding, int head_n, T *tail_embedding,
  const MLCommon::FastIntDiv tail_n, const int *head, const int *tail, int nnz,
  T *epochs_per_sample, int n_vertices, T *epoch_of_next_negative_sample,
  T *epoch_of_next_sample, T2 alpha, int epoch, T2 gamma, uint64_t seed,
  double *embedding_updates, bool move_other, UMAPParams params, T nsr_inv) {
  extern __shared__ T embedding_shared_mem_updates[];
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row >= nnz) return;
  auto _epoch_of_next_sample = epoch_of_next_sample[row];
  if (_epoch_of_next_sample > epoch) return;
  auto _epochs_per_sample = epochs_per_sample[row];
  auto epochs_per_negative_sample = _epochs_per_sample * nsr_inv;
  /**
   * Positive sample stage (attractive forces)
   */
  int j = head[row];
  int k = tail[row];
  T *current = head_embedding + (j * params.n_components);
  T *other = tail_embedding + (k * params.n_components);
  T2 *current_buffer, *other_buffer;
  if (use_shared_mem) {
    // shared memory
    current_buffer = (T2 *)embedding_shared_mem_updates + threadIdx.x;
    // initialization of shared memory
    for (int d = 0; d < params.n_components; d++) {
      current_buffer[d * TPB_X] = 0;
    }
  } else if (!multicore_implem) {
    // no shared memory and synchronized implementation
    current_buffer = (T2 *)embedding_updates + (j * params.n_components);
    other_buffer = (T2 *)embedding_updates + (k * params.n_components);
  }
  auto dist_squared = rdist<T, T2>(current, other, params.n_components);
  // Attractive force between the two vertices, since they
  // are connected by an edge in the 1-skeleton.
  auto attractive_grad_coeff = T2(0.0);
  if (dist_squared > T2(0.0)) {
    attractive_grad_coeff = attractive_grad<T2>(dist_squared, params);
  }
  /**
   * Apply attractive force between `current` and `other`
   * by updating their 'weights' to place them relative
   * to their weight in the 1-skeleton.
   * (update `other` embedding only if we are
   * performing unsupervised training).
   */
  for (int d = 0; d < params.n_components; d++) {
    auto grad_d = clip<T2>(attractive_grad_coeff * (current[d] - other[d]),
                           T2(-4.0), T2(4.0));
    grad_d *= alpha;
    if (use_shared_mem) {
      current_buffer[d * TPB_X] += grad_d;
    } else {
      if (multicore_implem) {
        atomicAdd(current + d, grad_d);
        if (move_other) {  // happens only during unsupervised training
          atomicAdd(other + d, -grad_d);
        }
      } else {
        atomicAdd(current_buffer + d, grad_d);
        if (move_other) {  // happens only during unsupervised training
          atomicAdd(other_buffer + d, -grad_d);
        }
      }
    }
  }
  // storing gradients for negative samples back to global memory
  if (use_shared_mem && move_other) {
    __syncthreads();
    if (multicore_implem) {
      for (int d = 0; d < params.n_components; d++) {
        auto grad = current_buffer[d * TPB_X];
        atomicAdd(other + d, -grad);
      }
    } else {
      T2 *tmp2 = (T2 *)embedding_updates + (k * params.n_components);
      for (int d = 0; d < params.n_components; d++) {
        auto grad = current_buffer[d * TPB_X];
        atomicAdd(tmp2 + d, -grad);
      }
    }
  }
  epoch_of_next_sample[row] = _epoch_of_next_sample + _epochs_per_sample;
  // number of negative samples to choose
  auto _epoch_of_next_negative_sample = epoch_of_next_negative_sample[row];
  int n_neg_samples =
    int(T(epoch - _epoch_of_next_negative_sample) / epochs_per_negative_sample);
  /**
   * Negative sampling stage
   */
  MLCommon::Random::detail::PhiloxGenerator gen((uint64_t)seed, (uint64_t)row,
                                                0);
  for (int p = 0; p < n_neg_samples; p++) {
    int r;
    gen.next(r);
    int t = r % tail_n;
    T *negative_sample = tail_embedding + (t * params.n_components);
    dist_squared = rdist<T, T2>(current, negative_sample, params.n_components);
    // repulsive force between two vertices
    auto repulsive_grad_coeff = T2(0.0);
    if (dist_squared > T2(0.0)) {
      repulsive_grad_coeff = repulsive_grad<T2>(dist_squared, gamma, params);
    } else if (j == t)
      continue;
    /**
     * Apply repulsive force between `current` and `other`
     * (which has been negatively sampled) by updating
     * their 'weights' to push them farther in Euclidean space.
     */
    for (int d = 0; d < params.n_components; d++) {
      auto grad_d = T2(0.0);
      if (repulsive_grad_coeff > T2(0.0))
        grad_d =
          clip<T2>(repulsive_grad_coeff * (current[d] - negative_sample[d]),
                   T2(-4.0), T2(4.0));
      else
        grad_d = T2(4.0);
      grad_d *= alpha;
      if (use_shared_mem) {
        current_buffer[d * TPB_X] += grad_d;
      } else {
        if (multicore_implem) {
          atomicAdd(current + d, grad_d);
        } else {
          atomicAdd(current_buffer + d, grad_d);
        }
      }
    }
  }
  // storing gradients for positive samples back to global memory
  if (use_shared_mem) {
    __syncthreads();
    if (multicore_implem) {
      for (int d = 0; d < params.n_components; d++) {
        atomicAdd(current + d, current_buffer[d * TPB_X]);
      }
    } else {
      T2 *tmp1 = (T2 *)embedding_updates + (j * params.n_components);
      for (int d = 0; d < params.n_components; d++) {
        atomicAdd(tmp1 + d, current_buffer[d * TPB_X]);
      }
    }
  }
  epoch_of_next_negative_sample[row] =
    _epoch_of_next_negative_sample + n_neg_samples * epochs_per_negative_sample;
}

template <typename T, int TPB_X>
void call_optimize_batch_kernel(
  T *head_embedding, int head_n, T *tail_embedding,
  const MLCommon::FastIntDiv &tail_n, const int *head, const int *tail, int nnz,
  T *epochs_per_sample, int n_vertices, T *epoch_of_next_negative_sample,
  T *epoch_of_next_sample, T alpha, int epoch, T gamma, uint64_t seed,
  double *embedding_updates, bool move_other, bool use_shared_mem,
  UMAPParams *params, int n, dim3 &grid, dim3 &blk, size_t requiredSize,
  cudaStream_t &stream) {
  T nsr_inv = T(1.0) / params->negative_sample_rate;
  if (embedding_updates) {
    if (use_shared_mem) {
      // synchronized implementation with shared memory
      optimize_batch_kernel<T, double, TPB_X, false, true>
        <<<grid, blk, requiredSize, stream>>>(
          head_embedding, head_n, tail_embedding, tail_n, head, tail, nnz,
          epochs_per_sample, n_vertices, epoch_of_next_negative_sample,
          epoch_of_next_sample, alpha, n, gamma, seed, embedding_updates,
          move_other, *params, nsr_inv);
    } else {
      // synchronized implementation without shared memory
      optimize_batch_kernel<T, double, TPB_X, false, false>
        <<<grid, blk, 0, stream>>>(
          head_embedding, head_n, tail_embedding, tail_n, head, tail, nnz,
          epochs_per_sample, n_vertices, epoch_of_next_negative_sample,
          epoch_of_next_sample, alpha, n, gamma, seed, embedding_updates,
          move_other, *params, nsr_inv);
    }
  } else {
    if (use_shared_mem) {
      // multicore implementation with shared memory
      optimize_batch_kernel<T, T, TPB_X, true, true>
        <<<grid, blk, requiredSize, stream>>>(
          head_embedding, head_n, tail_embedding, tail_n, head, tail, nnz,
          epochs_per_sample, n_vertices, epoch_of_next_negative_sample,
          epoch_of_next_sample, alpha, n, gamma, seed, embedding_updates,
          move_other, *params, nsr_inv);
    } else {
      // multicore implementation without shared memory
      optimize_batch_kernel<T, T, TPB_X, true, false><<<grid, blk, 0, stream>>>(
        head_embedding, head_n, tail_embedding, tail_n, head, tail, nnz,
        epochs_per_sample, n_vertices, epoch_of_next_negative_sample,
        epoch_of_next_sample, alpha, n, gamma, seed, embedding_updates,
        move_other, *params, nsr_inv);
    }
  }
}

}  // namespace Algo
}  // namespace SimplSetEmbed
}  // namespace UMAPAlgo
