/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cuml/manifold/umapparams.h>
#include <raft/cudart_utils.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <common/fast_int_div.cuh>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/random/rng.cuh>

namespace UMAPAlgo {
namespace SimplSetEmbed {
namespace Algo {

using namespace ML;

/**
 * Calculate the squared distance between two vectors of size n
 * @{
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
template <typename T, typename T2, int LEN>
DI T2 rdist(const T (&X)[LEN], const T (&Y)[LEN]) {
  auto result = T2(0.0);
#pragma unroll
  for (int i = 0; i < LEN; ++i) {
    auto diff = T2(X[i] - Y[i]);
    result += diff * diff;
  }
  return result;
}
/** @} */

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
          int n_components>
__global__ void optimize_batch_kernel_reg(
  T *head_embedding, int head_n, T *tail_embedding,
  const MLCommon::FastIntDiv tail_n, const int *head, const int *tail, int nnz,
  T *epochs_per_sample, int n_vertices, T *epoch_of_next_negative_sample,
  T *epoch_of_next_sample, T2 alpha, int epoch, T2 gamma, uint64_t seed,
  double *embedding_updates, bool move_other, UMAPParams params, T nsr_inv,
  int offset = 0) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x + offset;
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
  T *current = head_embedding + (j * n_components);
  T *other = tail_embedding + (k * n_components);
  T current_reg[n_components], other_reg[n_components], grads[n_components];
  for (int i = 0; i < n_components; ++i) {
    current_reg[i] = current[i];
    other_reg[i] = other[i];
  }
  auto dist_squared = rdist<T, T2, n_components>(current_reg, other_reg);
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
  for (int d = 0; d < n_components; d++) {
    auto diff = current_reg[d] - other_reg[d];
    auto grad_d = clip<T2>(attractive_grad_coeff * diff, T2(-4.0), T2(4.0));
    grads[d] = grad_d * alpha;
  }
  // storing gradients for negative samples back to global memory
  // move_other = (head_embedding == tail_embedding)
  if (move_other) {
    if (multicore_implem) {
      for (int d = 0; d < n_components; d++) {
        raft::myAtomicAdd(other + d, -grads[d]);
      }
    } else {
      T2 *tmp2 = (T2 *)embedding_updates + (k * n_components);
      for (int d = 0; d < n_components; d++) {
        raft::myAtomicAdd<T2>((T2 *)tmp2 + d, -grads[d]);
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
  raft::random::detail::PhiloxGenerator gen((uint64_t)seed, (uint64_t)row, 0);
  for (int p = 0; p < n_neg_samples; p++) {
    int r;
    gen.next(r);
    int t = r % tail_n;
    T *negative_sample = tail_embedding + (t * n_components);
    T negative_sample_reg[n_components];
    for (int i = 0; i < n_components; ++i) {
      negative_sample_reg[i] = negative_sample[i];
    }
    dist_squared = rdist<T, T2, n_components>(current_reg, negative_sample_reg);
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
    for (int d = 0; d < n_components; d++) {
      auto diff = current_reg[d] - negative_sample_reg[d];
      auto grad_d = T2(0.0);
      if (repulsive_grad_coeff > T2(0.0))
        grad_d = clip<T2>(repulsive_grad_coeff * diff, T2(-4.0), T2(4.0));
      else
        grad_d = T2(4.0);
      grads[d] += grad_d * alpha;
    }
  }
  // storing gradients for positive samples back to global memory
  if (multicore_implem) {
    for (int d = 0; d < n_components; d++) {
      raft::myAtomicAdd(current + d, grads[d]);
    }
  } else {
    T2 *tmp1 = (T2 *)embedding_updates + (j * n_components);
    for (int d = 0; d < n_components; d++) {
      raft::myAtomicAdd<T2>((T2 *)tmp1 + d, grads[d]);
    }
  }
  epoch_of_next_negative_sample[row] =
    _epoch_of_next_negative_sample + n_neg_samples * epochs_per_negative_sample;
}

template <typename T, typename Op>
DI T warpReduce(T val, uint32_t mask, Op op) {
#pragma unroll
  for (int i = raft::warp_size() / 2; i > 0; i >>= 1) {
    T tmp = raft::shfl(val, raft::laneId() + i, raft::warp_size(), mask);
    val = op(val, tmp);
  }
  return val;
}

template <typename T, typename Op>
DI T warp_reduce(T val, uint32_t n_partitions, uint32_t mask = raft::warp_full_mask(),
                 Op op = cub::Sum{}) {
  for (int i = raft::warp_size() / n_partitions / 2; i > 0; i >>= 1) {
    T tmp = raft::shfl(val, raft::laneId() + i, raft::warp_size(), mask);
    val = op(val, tmp);
  }
  return val;
}

size_t __device__ sampleId(size_t samples_per_warp) {
  size_t partition_size = raft::warp_size() / samples_per_warp;
  auto sample_id = raft::laneId() / partition_size + raft::warpId() * samples_per_warp;
  return sample_id;
}

/**
 * Kernel for optimization
 *
 * @param head_embedding  Embedding being optimized
 * @param buffer          Temporary buffer used to store the update of gradient
 * @param tail_embedding  Equal to head_embedding if it's unsupervised training, otherwise it's the reference embedding.
 * @param indptr          Row index of CSR connectivity graph
 * @param n_samples       Number of samples
 * @param indices         Column index of CSR connectivity graph
 * @param n_indices       Lenght of column index
 * @param epochs_per_sample
 * @param epoch_of_next_sample
 * @param epoch_of_next_negative_sample
 * @param seed
 * @param nsr_inv
 * @param params
 * @param epoch
 * @param alpha
 * @param gamma
 */
template <typename T, typename T2>
__global__ void optimize_batch_kernel(
  // embedding
  T const *head_embedding, T *buffer, T const *tail_embedding,
  // CSR graph
  int const *indptr, size_t n_samples, int const *indices, size_t n_indices,
  // sampling
  T const *epochs_per_sample, T *epoch_of_next_sample,
  T *epoch_of_next_negative_sample, uint64_t seed, T nsr_inv,
  // parameters
  UMAPParams params, int epoch, T2 alpha, T2 gamma, size_t n_samples_per_warp) {
  static_assert(std::is_floating_point<T>::value, "Must be floating point");
  static_assert(std::is_floating_point<T2>::value, "Must be floating point");
  // each partition of warp handles 1 sample
  size_t partition_size = raft::warp_size() / n_samples_per_warp;
  size_t id_in_partition = raft::laneId() % partition_size;

  // handles the connectivity graph with 1 warp for each row.
  size_t sample_id = sampleId(n_samples_per_warp);
  bool valid_sample = sample_id < n_samples;

  T const *current = head_embedding + (sample_id * params.n_components);
  T *writeto = buffer + (sample_id * params.n_components);

  size_t beg = 0;
  size_t end = 0;
  if (valid_sample) {
    beg = indptr[sample_id];
    end = indptr[sample_id + 1];
  }
  size_t nnz = end - beg;  // nnz in each row of the connectivity graph.
  // Use maximum of all samples within a warp
  nnz = warpReduce(nnz, raft::warp_full_mask(), cub::Max{});

  // load column indices into warp.
  for (size_t k = 0; k < nnz; k += partition_size) {
    uint32_t edge = beg + k + id_in_partition;
    uint32_t mask = __ballot_sync(raft::warp_full_mask(), edge < end && valid_sample);
    if (!valid_sample || edge >= end) {
      continue;
    }
    assert(edge < n_indices);

    bool sampled = epoch_of_next_sample[edge] <= T(epoch);

    /**
     * Apply attractive force
     */
    size_t col = indices[edge];
    T const *other =
      tail_embedding + static_cast<ptrdiff_t>(col * params.n_components);
    auto dist_squared = rdist<T, T>(current, other, params.n_components);
    auto attractive_grad_coeff = T(0.0);
    if (dist_squared > T(0.0)) {
      attractive_grad_coeff = attractive_grad<T>(dist_squared, params);
    }
    // compute gradient for each component in embedded space
    for (size_t d = 0; d < params.n_components; ++d) {
      auto grad_d = clip<T>(attractive_grad_coeff * (current[d] - other[d]),
                            T(-4.0), T(4.0)) *
                    alpha;
      grad_d *= float(sampled);
      /**
       * The gradient summed for each compoent among all threads in the warp
       */
      // reduce to starting thread of each warp partition, this equals to reducing within
      // each sample
      grad_d = warp_reduce(grad_d, n_samples_per_warp, mask, cub::Sum{});
      // since we are only writing to 1 side of the edge and rdist is symmetric, so we
      // double the value.
      grad_d *= T(2.0);
      if (id_in_partition == 0) {
        writeto[d] += grad_d;
      }
    }

    /**
     * Apply repulsive force
     */
    auto _epochs_per_sample = epochs_per_sample[edge];
    auto epochs_per_negative_sample = _epochs_per_sample * nsr_inv;
    auto _epoch_of_next_negative_sample = epoch_of_next_negative_sample[edge];
    int n_neg_samples = int(T(epoch - _epoch_of_next_negative_sample) /
                            epochs_per_negative_sample);

    raft::random::detail::PhiloxGenerator gen((uint64_t)seed, (uint64_t)edge,
                                              0);
    // use maximum for the loop to keep warp primitives in sync
    auto max_n_neg_samples = warpReduce(n_neg_samples, mask, cub::Max{});

    for (int p = 0; p < max_n_neg_samples; p += 1) {
      int r;
      gen.next(r);
      int t = r % n_samples;
      // Get the point in embedded space
      T const *negative_sample = tail_embedding + (t * params.n_components);

      auto dist_squared =
        rdist<T, T>(current, negative_sample, params.n_components);

      auto repulsive_grad_coeff =
        dist_squared > T(0.0) ? repulsive_grad<T>(dist_squared, gamma, params)
                              : T(0.0);
      for (int d = 0; d < params.n_components; d++) {
        auto diff = (current[d] - negative_sample[d]);
        auto grad_d = T(0.0);
        if (p < n_neg_samples && current != negative_sample && sampled) {
          if (repulsive_grad_coeff > T(0.0)) {
            grad_d = clip<T>(repulsive_grad_coeff * diff, T(-4.0), T(4.0));
          } else {
            grad_d = T(4.0);
          }
        }
        grad_d *= alpha;
        grad_d = warp_reduce(grad_d, n_samples_per_warp, mask, cub::Sum{});

        if (id_in_partition == 0) {
          writeto[d] += grad_d;
        }
      }
    }

    if (sampled) {
      auto _epochs_per_sample = epochs_per_sample[edge];
      // smaller the weight, greater the epochs_per_sample
      epoch_of_next_sample[edge] += _epochs_per_sample;
      epoch_of_next_negative_sample[edge] +=
        T(n_neg_samples) * epochs_per_negative_sample;
    }
  }
}

template <typename Op>
void __global__ forRangeKernel(size_t n, Op op) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n) {
    return;
  }
  op(t);
}

template <int TPB_X, typename Op>
void forRange(size_t n, cudaStream_t const& stream, Op op) {
  auto n_blocks = raft::ceildiv(n, size_t(TPB_X));
  forRangeKernel<<<n_blocks, TPB_X, 0, stream>>>(n, op);
}

/**
 * @param embedding
 * @param buffer     Buffer for writing gradient updates
 */
template <typename T, int TPB_X>
void call_optimization_batch_kernel(
  T *embedding, T *buffer, T const *other, int const *indptr, size_t n_samples,
  int const *indices, size_t n_indices, T const *epochs_per_sample,
  T *epoch_of_next_sample, T *epoch_of_next_negative_sample,
  UMAPParams const *params, uint64_t seed, int epoch, float alpha, float gamma,
  dim3 &grid, dim3 &blk, cudaStream_t &stream, int nnz, size_t n_samples_per_warp) {
  size_t embedding_size = n_samples * params->n_components;
  T nsr_inv = T(1.0) / params->negative_sample_rate;
  optimize_batch_kernel<T><<<grid, blk, 0, stream>>>(
    // embedding
    embedding, buffer, other,
    // graph
    indptr, n_samples, indices, n_indices,
    // sampling
    epochs_per_sample, epoch_of_next_sample, epoch_of_next_negative_sample,
    seed, nsr_inv,
    // parameters
    *params, epoch, alpha, gamma, n_samples_per_warp);
  forRange<TPB_X>(embedding_size, stream, [=] __device__(uint32_t i) {
    embedding[i] += buffer[i];
    buffer[i] = 0.0f;
  });
}

/**
 * @param head_embedding Either the embedding used for fit, or the output of transform.
 */
template <typename T, typename T2, int TPB_X, bool multicore_implem,
          bool use_shared_mem>
__global__ void optimize_batch_kernel(
  T *head_embedding, int head_n, T *tail_embedding,
  const MLCommon::FastIntDiv tail_n, const int *head, const int *tail, int nnz,
  T *epochs_per_sample, int n_vertices, T *epoch_of_next_negative_sample,
  T *epoch_of_next_sample, T2 alpha, int epoch, T2 gamma, uint64_t seed,
  double *embedding_updates, bool move_other, UMAPParams params, T nsr_inv,
  int offset = 0) {
  extern __shared__ T embedding_shared_mem_updates[];
  int row = (blockIdx.x * TPB_X) + threadIdx.x + offset;
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
    current_buffer = (T2 *)embedding_shared_mem_updates + threadIdx.x;
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
      current_buffer[d * TPB_X] = grad_d;
    } else {
      if (multicore_implem) {
        raft::myAtomicAdd<T>((T *)current + d, grad_d);
        if (move_other) {  // happens only during unsupervised training
          raft::myAtomicAdd<T>((T *)other + d, -grad_d);
        }
      } else {
        raft::myAtomicAdd<T2>((T2 *)current_buffer + d, grad_d);
        if (move_other) {  // happens only during unsupervised training
          raft::myAtomicAdd<T2>((T2 *)other_buffer + d, -grad_d);
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
        raft::myAtomicAdd<T>((T *)other + d, -grad);
      }
    } else {
      T2 *tmp2 = (T2 *)embedding_updates + (k * params.n_components);
      for (int d = 0; d < params.n_components; d++) {
        auto grad = current_buffer[d * TPB_X];
        raft::myAtomicAdd<T2>((T2 *)tmp2 + d, -grad);
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
  raft::random::detail::PhiloxGenerator gen((uint64_t)seed, (uint64_t)row, 0);
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
          raft::myAtomicAdd<T>((T *)current + d, grad_d);
        } else {
          raft::myAtomicAdd<T2>((T2 *)current_buffer + d, grad_d);
        }
      }
    }
  }
  // storing gradients for positive samples back to global memory
  if (use_shared_mem) {
    cub::CTA_SYNC();
    if (multicore_implem) {
      for (int d = 0; d < params.n_components; d++) {
        raft::myAtomicAdd<T>((T *)current + d, current_buffer[d * TPB_X]);
      }
    } else {
      T2 *tmp1 = (T2 *)embedding_updates + (j * params.n_components);
      for (int d = 0; d < params.n_components; d++) {
        raft::myAtomicAdd<T2>((T2 *)tmp1 + d, current_buffer[d * TPB_X]);
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
  double *embedding_updates, bool move_other, UMAPParams *params, int n,
  dim3 &grid, dim3 &blk, cudaStream_t &stream, int offset = 0) {
  size_t requiredSize = TPB_X * params->n_components;
  if (params->multicore_implem) {
    requiredSize *= sizeof(T);
  } else {
    requiredSize *= sizeof(double);
  }
  bool use_shared_mem = requiredSize < raft::getSharedMemPerBlock();
  T nsr_inv = T(1.0) / params->negative_sample_rate;
  if (embedding_updates) {
    if (params->n_components == 2) {
      // synchronized implementation with registers
      optimize_batch_kernel_reg<T, double, TPB_X, false, 2>
        <<<grid, blk, 0, stream>>>(
          head_embedding, head_n, tail_embedding, tail_n, head, tail, nnz,
          epochs_per_sample, n_vertices, epoch_of_next_negative_sample,
          epoch_of_next_sample, alpha, n, gamma, seed, embedding_updates,
          move_other, *params, nsr_inv, offset);
    } else if (use_shared_mem) {
      // synchronized implementation with shared memory
      optimize_batch_kernel<T, double, TPB_X, false, true>
        <<<grid, blk, requiredSize, stream>>>(
          head_embedding, head_n, tail_embedding, tail_n, head, tail, nnz,
          epochs_per_sample, n_vertices, epoch_of_next_negative_sample,
          epoch_of_next_sample, alpha, n, gamma, seed, embedding_updates,
          move_other, *params, nsr_inv, offset);
    } else {
      // synchronized implementation without shared memory
      optimize_batch_kernel<T, double, TPB_X, false, false>
        <<<grid, blk, 0, stream>>>(
          head_embedding, head_n, tail_embedding, tail_n, head, tail, nnz,
          epochs_per_sample, n_vertices, epoch_of_next_negative_sample,
          epoch_of_next_sample, alpha, n, gamma, seed, embedding_updates,
          move_other, *params, nsr_inv, offset);
    }
  } else {
    if (params->n_components == 2) {
      // multicore implementation with registers
      optimize_batch_kernel_reg<T, T, TPB_X, true, 2><<<grid, blk, 0, stream>>>(
        head_embedding, head_n, tail_embedding, tail_n, head, tail, nnz,
        epochs_per_sample, n_vertices, epoch_of_next_negative_sample,
        epoch_of_next_sample, alpha, n, gamma, seed, embedding_updates,
        move_other, *params, nsr_inv);
    } else if (use_shared_mem) {
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
