/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cuml/common/device_buffer.hpp>
#include <cuml/cuml.hpp>

#include <curand.h>
#include <math.h>
#include <raft/cudart_utils.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/system/cuda/execution_policy.h>
#include <common/fast_int_div.cuh>
#include <cstdlib>
#include <cuml/common/logger.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/mr/device/allocator.hpp>
#include <raft/random/rng_impl.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.cuh>
#include <string>
#include "optimize_batch_kernel.cuh"

#include <raft/linalg/binary_op.cuh>
#include <raft/sparse/op/filter.cuh>

#pragma once

namespace UMAPAlgo {
namespace SimplSetEmbed {
namespace Algo {

using namespace ML;

/**
 * Given a set of weights and number of epochs, generate
 * the number of epochs per sample for each weight.
 *
 * @param weights: The weights of how much we wish to sample each 1-simplex
 * @param weights_n: the size of the weights array
 * @param n_epochs: the total number of epochs we want to train for
 * @param result: an array of number of epochs per sample, one for each 1-simplex
 * @param stream cuda stream
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

  raft::linalg::unaryOp<T>(
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
 * Kernel applying updates to embedding
 * TODO: Replace this kernel with modified version of Linalg::Add
 * as described at https://github.com/rapidsai/cuml/issues/1781
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
                     UMAPParams *params, int n_epochs,
                     std::shared_ptr<raft::mr::device::allocator> d_alloc,
                     cudaStream_t stream) {
  // Are we doing a fit or a transform?
  // true if fit, false if tranform
  bool move_other = head_embedding == tail_embedding;

  if (params->optim_batch_size <= 0) {
    params->optim_batch_size = 100000 / params->n_components;
  }

  T alpha = params->initial_alpha;

  MLCommon::device_buffer<T> epoch_of_next_negative_sample(d_alloc, stream,
                                                           nnz);
  T nsr_inv = T(1.0) / params->negative_sample_rate;
  raft::linalg::unaryOp<T>(
    epoch_of_next_negative_sample.data(), epochs_per_sample, nnz,
    [=] __device__(T input) { return input * nsr_inv; }, stream);

  MLCommon::device_buffer<T> epoch_of_next_sample(d_alloc, stream, nnz);
  raft::copy(epoch_of_next_sample.data(), epochs_per_sample, nnz, stream);

  dim3 grid(raft::ceildiv(nnz, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);
  uint64_t seed = params->random_state;

  MLCommon::FastIntDiv tail_n_fast(tail_n);

  if (params->multicore_implem) {
    for (int n = 0; n < n_epochs; n++) {
      call_optimize_batch_kernel<T, TPB_X>(
        head_embedding, head_n, tail_embedding, tail_n_fast, head, tail, nnz,
        epochs_per_sample, n_vertices, epoch_of_next_negative_sample.data(),
        epoch_of_next_sample.data(), alpha, n, gamma, seed, nullptr, move_other,
        params, n, grid, blk, stream);
      CUDA_CHECK(cudaGetLastError());
      optimization_iteration_finalization(params, head_embedding, alpha, n,
                                          n_epochs, seed);
    }
  } else {
    MLCommon::device_buffer<double> embedding_updates_buf(
      d_alloc, stream, n_vertices * params->n_components);
    double *embedding_updates = embedding_updates_buf.data();
    dim3 grid2(raft::ceildiv(n_vertices * params->n_components, TPB_X));

    for (int n = 0; n < n_epochs; n++) {
      CUDA_CHECK(cudaMemsetAsync(
        embedding_updates, 0,
        n_vertices * params->n_components * sizeof(double), stream));

      int toDo = nnz;
      int offset = 0;
      while (toDo > 0) {
        int curBatchSize = std::min(toDo, params->optim_batch_size);
        call_optimize_batch_kernel<T, TPB_X>(
          head_embedding, head_n, tail_embedding, tail_n_fast, head, tail,
          offset + curBatchSize, epochs_per_sample, n_vertices,
          epoch_of_next_negative_sample.data(), epoch_of_next_sample.data(),
          alpha, n, gamma, seed, embedding_updates, move_other, params, n, grid,
          blk, stream, offset);
        CUDA_CHECK(cudaGetLastError());

        toDo -= curBatchSize;
        offset += curBatchSize;
      }

      apply_optimization_kernel<T, TPB_X><<<grid2, blk, 0, stream>>>(
        head_embedding, embedding_updates, n_vertices * params->n_components);
      CUDA_CHECK(cudaGetLastError());
      optimization_iteration_finalization(params, head_embedding, alpha, n,
                                          n_epochs, seed);
    }
  }
}

template <int TPB_X, typename T>
void optimize_layout(T *embedding, T *other, int *indptr, size_t n_samples,
                     int *indices, size_t n_indices, int nnz,
                     T *epochs_per_sample, float gamma, UMAPParams *params,
                     int n_epochs, std::shared_ptr<deviceAllocator> d_alloc,
                     cudaStream_t stream) {
  if (params->optim_batch_size <= 0) {
    params->optim_batch_size = 100000 / params->n_components;
  }

  T alpha = params->initial_alpha;

  MLCommon::device_buffer<T> epoch_of_next_negative_sample(d_alloc, stream,
                                                           nnz);
  T nsr_inv = T(1.0) / params->negative_sample_rate;
  raft::linalg::unaryOp<T>(
    epoch_of_next_negative_sample.data(), epochs_per_sample, nnz,
    [=] __device__(T input) { return input * nsr_inv; }, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  MLCommon::device_buffer<T> epoch_of_next_sample(d_alloc, stream, nnz);
  raft::copy(epoch_of_next_sample.data(), epochs_per_sample, nnz, stream);
  MLCommon::device_buffer<T> epoch_of_next_sample_buffer(d_alloc, stream, nnz);

  static_assert(size_t(TPB_X) % raft::warp_size() == 0,
                "Block size must be multiple of warp size.");
  auto warps_per_blk = size_t(TPB_X) / raft::warp_size();
  dim3 grid(raft::ceildiv(n_samples, warps_per_blk), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  uint64_t seed = params->random_state;

  MLCommon::device_buffer<T> buffer(d_alloc, stream,
                                    n_samples * params->n_components);

  for (int n = 0; n < n_epochs; n++) {
    call_optimization_batch_kernel<T, TPB_X>(
      embedding, buffer.data(), other, indptr, n_samples, indices, n_indices,
      epochs_per_sample, epoch_of_next_sample.data(),
      epoch_of_next_sample_buffer.data(), epoch_of_next_negative_sample.data(),
      params, seed, n, alpha, gamma, grid, blk, stream, nnz);
    optimization_iteration_finalization(params, embedding, alpha, n, n_epochs,
                                        seed);
    CUDA_CHECK(cudaGetLastError());
  }
}

/**
 * Perform a fuzzy simplicial set embedding by minimizing
 * the fuzzy set cross entropy between the embeddings
 * and their 1-skeletons.
 *
 * @param m   Number of samples
 * @param n   Number of dimension in ambient sapce
 * @pARAM in  COO connectitivity graph
 */
template <int TPB_X, typename T>
void launcher(raft::handle_t const &handle, int m, int n,
              raft::sparse::COO<T> *in, UMAPParams *params, T *embedding) {
  int nnz = in->nnz;

  /**
   * Find vals.max()
   */
  thrust::device_ptr<const T> d_ptr = thrust::device_pointer_cast(in->vals());
  T max = *(thrust::max_element(thrust::cuda::par.on(handle.get_stream()),
                                d_ptr, d_ptr + nnz));

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
  raft::linalg::unaryOp<T>(
    in->vals(), in->vals(), nnz,
    [=] __device__(T input) {
      if (input < (max / float(n_epochs)))
        return 0.0f;
      else
        return input;
    },
    handle.get_stream());

  // COO connectivity graph
  raft::sparse::COO<T> out(handle.get_device_allocator(), handle.get_stream());
  // This will set the size correctly.
  raft::sparse::op::coo_remove_zeros<TPB_X, T>(
    in, &out, handle.get_device_allocator(), handle.get_stream());

  MLCommon::device_buffer<T> epochs_per_sample(handle.get_device_allocator(),
                                               handle.get_stream(), out.nnz);
  CUDA_CHECK(cudaMemsetAsync(epochs_per_sample.data(), 0, out.nnz * sizeof(T),
                             handle.get_stream()));

  make_epochs_per_sample(out.vals(), out.nnz, n_epochs,
                         epochs_per_sample.data(), handle.get_stream());

  if (ML::Logger::get().shouldLogFor(CUML_LEVEL_DEBUG)) {
    std::stringstream ss;
    ss << raft::arr2Str(epochs_per_sample.data(), out.nnz, "epochs_per_sample",
                        handle.get_stream());
    CUML_LOG_DEBUG(ss.str().c_str());
  }

  // FIXME(jiaming): This is redundent and consumes memory.
  raft::mr::device::buffer<int> src_offsets(handle.get_device_allocator(),
                                            handle.get_stream(), m + 1);
  raft::mr::device::buffer<int> dst_cols(handle.get_device_allocator(),
                                         handle.get_stream(), nnz);
  raft::mr::device::buffer<T> dst_vals(handle.get_device_allocator(),
                                       handle.get_stream(), nnz);
  raft::sparse::convert::coo_to_csr(handle, out.rows(), out.cols(), out.vals(),
                                    out.nnz, out.n_rows, src_offsets.data(),
                                    dst_cols.data(), dst_vals.data());

  optimize_layout<TPB_X, T>(embedding, embedding, src_offsets.data(),
                            src_offsets.size() - 1, dst_cols.data(),
                            dst_cols.size(), out.nnz, epochs_per_sample.data(),
                            params->repulsion_strength, params, n_epochs,
                            handle.get_device_allocator(), handle.get_stream());

  // optimize_layout<TPB_X, T>(embedding, m, embedding, m, out.rows(), out.cols(),
  //                           out.nnz, epochs_per_sample.data(), m,
  //                           params->repulsion_strength, params, n_epochs,
  //                           handle.get_device_allocator(), handle.get_stream());

  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Algo
}  // namespace SimplSetEmbed
}  // namespace UMAPAlgo
