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

#include <cuml/manifold/umapparams.h>

#include "random/rng_impl.h"

#include <cstdlib>

#include "sparse/coo.h"

#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/system/cuda/execution_policy.h>

#include <math.h>
#include <string>

#include <internals/internals.h>

#pragma once

namespace UMAPAlgo {

namespace SimplSetEmbed {

namespace Algo {

using namespace ML;

/**
 * Calculate the squared distance between two vectors of size n
 */
template <typename T>
__device__ __host__ double rdist(const T *X, const T *Y, int n) {
  double result = 0.0;
  for (int i = 0; i < n; i++) result += pow(X[i] - Y[i], 2);
  return result;
}

/**
 * Given a set of weights and number of epochs, generate
 * the number of epochs per sample for each weight.
 *
 * @param weights: The weights of how much we wish to sample each 1-simplex
 * @param weights_n: the size of the weights array
 * @param n_epochs: the total number of epochs we want to train for
 * @returns an array of number of epochs per sample, one for each 1-simplex
 */
template <typename T>
void make_epochs_per_sample(T *weights, int weights_n, int n_epochs, T *result,
                            cudaStream_t stream) {
  thrust::device_ptr<T> d_weights = thrust::device_pointer_cast(weights);
  T weights_max = *(thrust::max_element(thrust::cuda::par.on(stream), d_weights,
                                        d_weights + weights_n));

  //  result = -1.0 * np.ones(
  //      weights.shape[0], dtype=np.float64
  //  )
  //  n_samples = n_epochs * (weights / weights.max())
  //  result[n_samples > 0] = (
  //      float(n_epochs) / n_samples[n_samples > 0]
  //  )

  MLCommon::LinAlg::unaryOp<T>(
    result, weights, weights_n,
    [=] __device__(T input) {
      T v = n_epochs * (input / weights_max);
      if (v > 0)
        return T(n_epochs) / v;
      else
        return T(-1.0);
    },
    stream);
}

/**
 * Clip a value to within a lower and upper bound
 */
__device__ __host__ double clip(double val, double lb, double ub) {
  if (val > ub)
    return ub;
  else if (val < lb)
    return lb;
  else
    return val;
}

/**
 * Calculate the repulsive gradient
 */
__device__ __host__ double repulsive_grad(double dist_squared, double gamma,
                                          UMAPParams params) {
  double grad_coeff = 2.0 * gamma * params.b;
  grad_coeff /=
    (0.001 + dist_squared) * (params.a * pow(dist_squared, params.b) + 1);
  return grad_coeff;
}

/**
  * Calculate the attractive gradient
  */
__device__ __host__ double attractive_grad(double dist_squared,
                                           UMAPParams params) {
  double grad_coeff =
    -2.0 * params.a * params.b * pow(dist_squared, params.b - 1.0);
  grad_coeff /= params.a * pow(dist_squared, params.b) + 1.0;
  return grad_coeff;
}

/**
 * Kernel for performing 1 epoch of stochastic gradient descent
 * on each call. Vectors are sampled in proportion to their
 * weights in the 1-skeleton. Negative samples are drawn
 * randomly.
 */
template <typename T, int TPB_X, bool multicore_implem, bool use_shared_mem>
__global__ void optimize_batch_kernel(
  T *head_embedding, int head_n, T *tail_embedding, int tail_n, const int *head,
  const int *tail, int nnz, T *epochs_per_sample, int n_vertices,
  T *epochs_per_negative_sample, T *epoch_of_next_negative_sample,
  T *epoch_of_next_sample, double alpha, int epoch, double gamma, uint64_t seed,
  double *embedding_updates, bool move_other, UMAPParams params) {
  extern __shared__ T embedding_shared_mem_updates[];

  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < nnz) {
    /**
       * Positive sample stage (attractive forces)
       */

    if (epoch_of_next_sample[row] <= epoch) {
      int j = head[row];
      int k = tail[row];

      T *current = head_embedding + (j * params.n_components);
      T *other = tail_embedding + (k * params.n_components);
      T *current_shared_mem_buffer, *other_shared_mem_buffer;
      double *current_buffer, *other_buffer;

      if (
        use_shared_mem) {  // do we use shared memory? If yes, means multicore_implem == true
        if (move_other) {  // are we performing a fit (or transform)?
          current_shared_mem_buffer = embedding_shared_mem_updates +
                                      ((threadIdx.x * 2) * params.n_components);
          other_shared_mem_buffer =
            embedding_shared_mem_updates +
            (((threadIdx.x * 2) + 1) * params.n_components);
        } else {
          current_shared_mem_buffer =
            embedding_shared_mem_updates + (threadIdx.x * params.n_components);
        }
      } else if (
        !multicore_implem) {  // do we use optimized but inconsistent multicore implementation?
        current_buffer = embedding_updates + (j * params.n_components);
        other_buffer = embedding_updates + (k * params.n_components);
      }

      if (use_shared_mem) {
        // initialization of shared memory
        for (int d = 0; d < params.n_components; d++) {
          current_shared_mem_buffer[d] = 0;
        }
        if (move_other) {
          for (int d = 0; d < params.n_components; d++) {
            other_shared_mem_buffer[d] = 0;
          }
        }
      }

      double dist_squared = rdist(current, other, params.n_components);

      // Attractive force between the two vertices, since they
      // are connected by an edge in the 1-skeleton.
      double attractive_grad_coeff = 0.0;
      if (dist_squared > 0.0) {
        attractive_grad_coeff = attractive_grad(dist_squared, params);
      }

      /**
         * Apply attractive force between `current` and `other`
         * by updating their 'weights' to place them relative
         * to their weight in the 1-skeleton.
         * (update `other` embedding only if we are
         * performing unsupervised training).
         */
      for (int d = 0; d < params.n_components; d++) {
        double grad_d =
          clip(attractive_grad_coeff * (current[d] - other[d]), -4.0f, 4.0f);

        grad_d *= alpha;

        if (use_shared_mem) {
          current_shared_mem_buffer[d] += grad_d;
          if (move_other) {  // happens only during unsupervised training
            other_shared_mem_buffer[d] += -grad_d;
          }
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

      epoch_of_next_sample[row] += epochs_per_sample[row];

      // number of negative samples to choose
      int n_neg_samples = int(T(epoch - epoch_of_next_negative_sample[row]) /
                              epochs_per_negative_sample[row]);

      /**
       * Negative sampling stage
       */
      MLCommon::Random::detail::PhiloxGenerator gen((uint64_t)seed,
                                                    (uint64_t)row, 0);
      for (int p = 0; p < n_neg_samples; p++) {
        int r;
        gen.next(r);
        int t = r % tail_n;
        float *negative_sample = tail_embedding + (t * params.n_components);
        dist_squared = rdist(current, negative_sample, params.n_components);

        // repulsive force between two vertices
        double repulsive_grad_coeff = 0.0;
        if (dist_squared > 0.0) {
          repulsive_grad_coeff = repulsive_grad(dist_squared, gamma, params);
        } else if (j == t)
          continue;

        /**
         * Apply repulsive force between `current` and `other`
         * (which has been negatively sampled) by updating
         * their 'weights' to push them farther in Euclidean space.
         */
        for (int d = 0; d < params.n_components; d++) {
          double grad_d = 0.0;
          if (repulsive_grad_coeff > 0.0)
            grad_d =
              clip(repulsive_grad_coeff * (current[d] - negative_sample[d]),
                   -4.0f, 4.0f);
          else
            grad_d = 4.0;

          grad_d *= alpha;

          if (use_shared_mem) {
            current_shared_mem_buffer[d] += grad_d;
          } else {
            if (multicore_implem) {
              atomicAdd(current + d, grad_d);
            } else {
              atomicAdd(current_buffer + d, grad_d);
            }
          }
        }
      }

      if (use_shared_mem) {
        // storing everything back to global memory
        __syncthreads();
        for (int d = 0; d < params.n_components; d++) {
          atomicAdd(current + d, current_shared_mem_buffer[d]);
        }
        if (move_other) {
          for (int d = 0; d < params.n_components; d++) {
            atomicAdd(other + d, other_shared_mem_buffer[d]);
          }
        }
      }

      epoch_of_next_negative_sample[row] +=
        n_neg_samples * epochs_per_negative_sample[row];
    }
  }
}

/**
 * Kernel applying updates to embedding
 */
template <typename T, int TPB_X>
__global__ void apply_optimization_kernel(T *embedding,
                                          double *embedding_updates, int n) {
  int idx = (blockIdx.x * TPB_X) + threadIdx.x;
  if (idx < n) {
    embedding[idx] += embedding_updates[idx];
  }
}

template <typename T>
inline void optimization_iteration_finalization(UMAPParams *params,
                                                T *head_embedding, T &alpha,
                                                int n, int n_epochs,
                                                uint64_t &seed) {
  if (params->callback) params->callback->on_epoch_end(head_embedding);
  alpha = params->initial_alpha * (1.0 - (T(n) / T(n_epochs)));
  seed += 1;
}

template <typename T, int TPB_X>
inline void call_optimize_batch_kernel(
  T *head_embedding, int head_n, T *tail_embedding, int tail_n, const int *head,
  const int *tail, int nnz, T *epochs_per_sample, int n_vertices,
  T *epochs_per_negative_sample, T *epoch_of_next_negative_sample,
  T *epoch_of_next_sample, T alpha, int epoch, T gamma, uint64_t seed,
  double *embedding_updates, bool move_other, bool use_shared_mem,
  UMAPParams *params, int n, dim3 &grid, dim3 &blk, size_t requiredSize,
  cudaStream_t &stream) {
  if (use_shared_mem) {  // multicore implementation with shared memory
    optimize_batch_kernel<T, TPB_X, true, true>
      <<<grid, blk, requiredSize, stream>>>(
        head_embedding, head_n, tail_embedding, tail_n, head, tail, nnz,
        epochs_per_sample, n_vertices, epochs_per_negative_sample,
        epoch_of_next_negative_sample, epoch_of_next_sample, alpha, n, gamma,
        seed, nullptr, move_other, *params);
  } else if (
    !embedding_updates) {  // multicore implementation without shared memory
    optimize_batch_kernel<T, TPB_X, true, false><<<grid, blk, 0, stream>>>(
      head_embedding, head_n, tail_embedding, tail_n, head, tail, nnz,
      epochs_per_sample, n_vertices, epochs_per_negative_sample,
      epoch_of_next_negative_sample, epoch_of_next_sample, alpha, n, gamma,
      seed, nullptr, move_other, *params);
  } else {  // singlecore implementation with global memory buffer
    optimize_batch_kernel<T, TPB_X, false, false><<<grid, blk, 0, stream>>>(
      head_embedding, head_n, tail_embedding, tail_n, head, tail, nnz,
      epochs_per_sample, n_vertices, epochs_per_negative_sample,
      epoch_of_next_negative_sample, epoch_of_next_sample, alpha, n, gamma,
      seed, embedding_updates, move_other, *params);
  }
}

/**
 * Runs gradient descent using sampling weights defined on
 * both the attraction and repulsion vectors.
 *
 * In this GD implementation, the weights being tuned are the
 * embeddings themselves, as the objective function is attracting
 * positive weights (neighbors in the 1-skeleton) and repelling
 * negative weights (non-neighbors in the 1-skeleton).
 */
template <int TPB_X, typename T>
void optimize_layout(T *head_embedding, int head_n, T *tail_embedding,
                     int tail_n, const int *head, const int *tail, int nnz,
                     T *epochs_per_sample, int n_vertices, float gamma,
                     UMAPParams *params, int n_epochs, bool multicore_implem,
                     std::shared_ptr<deviceAllocator> d_alloc,
                     cudaStream_t stream) {
  // Are we doing a fit or a transform?
  bool move_other = head_embedding == tail_embedding;

  T alpha = params->initial_alpha;

  MLCommon::device_buffer<T> epochs_per_negative_sample(d_alloc, stream, nnz);

  int nsr = params->negative_sample_rate;
  MLCommon::LinAlg::unaryOp<T>(
    epochs_per_negative_sample.data(), epochs_per_sample, nnz,
    [=] __device__(T input) { return input / T(nsr); }, stream);

  MLCommon::device_buffer<T> epoch_of_next_negative_sample(d_alloc, stream,
                                                           nnz);
  MLCommon::copy(epoch_of_next_negative_sample.data(),
                 epochs_per_negative_sample.data(), nnz, stream);

  MLCommon::device_buffer<T> epoch_of_next_sample(d_alloc, stream, nnz);
  MLCommon::copy(epoch_of_next_sample.data(), epochs_per_sample, nnz, stream);

  dim3 grid(MLCommon::ceildiv(nnz, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);
  uint64_t seed = params->random_state;

  if (multicore_implem) {
    int requiredSize = TPB_X * params->n_components * sizeof(T);
    if (move_other) requiredSize *= 2;

    // checks if enough shared memory is available
    bool use_shared_mem = requiredSize < MLCommon::getSharedMemPerBlock();

    for (int n = 0; n < n_epochs; n++) {
      call_optimize_batch_kernel<T, TPB_X>(
        head_embedding, head_n, tail_embedding, tail_n, head, tail, nnz,
        epochs_per_sample, n_vertices, epochs_per_negative_sample.data(),
        epoch_of_next_negative_sample.data(), epoch_of_next_sample.data(),
        alpha, n, gamma, seed, nullptr, move_other, use_shared_mem, params, n,
        grid, blk, requiredSize, stream);

      CUDA_CHECK(cudaGetLastError());

      optimization_iteration_finalization(params, head_embedding, alpha, n,
                                          n_epochs, seed);
    }
  } else {
    MLCommon::device_buffer<double> embedding_updates_buf(
      d_alloc, stream, n_vertices * params->n_components);
    double *embedding_updates = embedding_updates_buf.data();

    dim3 grid2(MLCommon::ceildiv(n_vertices * params->n_components, TPB_X), 1,
               1);

    for (int n = 0; n < n_epochs; n++) {
      CUDA_CHECK(cudaMemsetAsync(embedding_updates, 0,
                                 n_vertices * params->n_components * sizeof(T),
                                 stream));

      call_optimize_batch_kernel<T, TPB_X>(
        head_embedding, head_n, tail_embedding, tail_n, head, tail, nnz,
        epochs_per_sample, n_vertices, epochs_per_negative_sample.data(),
        epoch_of_next_negative_sample.data(), epoch_of_next_sample.data(),
        alpha, n, gamma, seed, embedding_updates, move_other, false, params, n,
        grid, blk, 0, stream);

      CUDA_CHECK(cudaGetLastError());

      apply_optimization_kernel<T, TPB_X><<<grid2, blk, 0, stream>>>(
        head_embedding, embedding_updates, n_vertices * params->n_components);

      CUDA_CHECK(cudaGetLastError());

      optimization_iteration_finalization(params, head_embedding, alpha, n,
                                          n_epochs, seed);
    }
  }
}

/**
 * Perform a fuzzy simplicial set embedding by minimizing
 * the fuzzy set cross entropy between the embeddings
 * and their 1-skeletons.
 */
template <int TPB_X, typename T>
void launcher(int m, int n, MLCommon::Sparse::COO<T> *in, UMAPParams *params,
              T *embedding, std::shared_ptr<deviceAllocator> d_alloc,
              cudaStream_t stream) {
  int nnz = in->nnz;

  /**
   * Find vals.max()
   */
  thrust::device_ptr<const T> d_ptr = thrust::device_pointer_cast(in->vals());
  T max =
    *(thrust::max_element(thrust::cuda::par.on(stream), d_ptr, d_ptr + nnz));

  int n_epochs = params->n_epochs;
  if (n_epochs <= 0) {
    if (m <= 10000)
      n_epochs = 500;
    else
      n_epochs = 200;
  }

  /**
   * Go through COO values and set everything that's less than
   * vals.max() / params->n_epochs to 0.0
   */
  MLCommon::LinAlg::unaryOp<T>(
    in->vals(), in->vals(), nnz,
    [=] __device__(T input) {
      if (input < (max / float(n_epochs)))
        return 0.0f;
      else
        return input;
    },
    stream);

  MLCommon::Sparse::COO<T> out(d_alloc, stream);
  MLCommon::Sparse::coo_remove_zeros<TPB_X, T>(in, &out, d_alloc, stream);

  MLCommon::device_buffer<T> epochs_per_sample(d_alloc, stream, out.nnz);
  CUDA_CHECK(
    cudaMemsetAsync(epochs_per_sample.data(), 0, out.nnz * sizeof(T), stream));

  make_epochs_per_sample(out.vals(), out.nnz, n_epochs,
                         epochs_per_sample.data(), stream);

  if (params->verbose)
    std::cout << MLCommon::arr2Str(epochs_per_sample.data(), out.nnz,
                                   "epochs_per_sample", stream)
              << std::endl;

  optimize_layout<TPB_X, T>(embedding, m, embedding, m, out.rows(), out.cols(),
                            out.nnz, epochs_per_sample.data(), m,
                            params->repulsion_strength, params, n_epochs,
                            params->multicore_implem, d_alloc, stream);

  CUDA_CHECK(cudaPeekAtLastError());
}
}  // namespace Algo
}  // namespace SimplSetEmbed
}  // namespace UMAPAlgo