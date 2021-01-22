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

#include <cusparse_v2.h>

#include <raft/cudart_utils.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>
#include <raft/mr/device/allocator.hpp>
#include <raft/mr/device/buffer.hpp>

#include <raft/device_atomics.cuh>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include <sparse/utils.h>
#include <sparse/convert/csr.cuh>
#include <sparse/coo.cuh>

namespace raft {
namespace sparse {
namespace linalg {

// TODO: value_idx param needs to be used for this once FAISS is updated to use float32
// for indices so that the index types can be uniform
template <int TPB_X = 128, typename T, typename Lambda>
__global__ void coo_symmetrize_kernel(int *row_ind, int *rows, int *cols,
                                      T *vals, int *orows, int *ocols, T *ovals,
                                      int n, int cnnz, Lambda reduction_op) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  if (row < n) {
    int start_idx = row_ind[row];  // each thread processes one row
    int stop_idx = get_stop_idx(row, n, cnnz, row_ind);

    int row_nnz = 0;
    int out_start_idx = start_idx * 2;

    for (int idx = 0; idx < stop_idx - start_idx; idx++) {
      int cur_row = rows[idx + start_idx];
      int cur_col = cols[idx + start_idx];
      T cur_val = vals[idx + start_idx];

      int lookup_row = cur_col;
      int t_start = row_ind[lookup_row];  // Start at
      int t_stop = get_stop_idx(lookup_row, n, cnnz, row_ind);

      T transpose = 0.0;

      bool found_match = false;
      for (int t_idx = t_start; t_idx < t_stop; t_idx++) {
        // If we find a match, let's get out of the loop. We won't
        // need to modify the transposed value, since that will be
        // done in a different thread.
        if (cols[t_idx] == cur_row && rows[t_idx] == cur_col) {
          // If it exists already, set transposed value to existing value
          transpose = vals[t_idx];
          found_match = true;
          break;
        }
      }

      // Custom reduction op on value and its transpose, which enables
      // specialized weighting.
      // If only simple X+X.T is desired, this op can just sum
      // the two values.
      T res = reduction_op(cur_row, cur_col, cur_val, transpose);

      // if we didn't find an exact match, we need to add
      // the computed res into our current matrix to guarantee
      // symmetry.
      // Note that if we did find a match, we don't need to
      // compute `res` on it here because it will be computed
      // in a different thread.
      if (!found_match && vals[idx] != 0.0) {
        orows[out_start_idx + row_nnz] = cur_col;
        ocols[out_start_idx + row_nnz] = cur_row;
        ovals[out_start_idx + row_nnz] = res;
        ++row_nnz;
      }

      if (res != 0.0) {
        orows[out_start_idx + row_nnz] = cur_row;
        ocols[out_start_idx + row_nnz] = cur_col;
        ovals[out_start_idx + row_nnz] = res;
        ++row_nnz;
      }
    }
  }
}

/**
 * @brief takes a COO matrix which may not be symmetric and symmetrizes
 * it, running a custom reduction function against the each value
 * and its transposed value.
 *
 * @param in: Input COO matrix
 * @param out: Output symmetrized COO matrix
 * @param reduction_op: a custom reduction function
 * @param d_alloc device allocator for temporary buffers
 * @param stream: cuda stream to use
 */
template <int TPB_X = 128, typename T, typename Lambda>
void coo_symmetrize(COO<T> *in, COO<T> *out,
                    Lambda reduction_op,  // two-argument reducer
                    std::shared_ptr<raft::mr::device::allocator> d_alloc,
                    cudaStream_t stream) {
  dim3 grid(raft::ceildiv(in->n_rows, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  ASSERT(!out->validate_mem(), "Expecting unallocated COO for output");

  raft::mr::device::buffer<int> in_row_ind(d_alloc, stream, in->n_rows);

  convert::sorted_coo_to_csr(in, in_row_ind.data(), d_alloc, stream);

  out->allocate(in->nnz * 2, in->n_rows, in->n_cols, true, stream);

  coo_symmetrize_kernel<TPB_X, T><<<grid, blk, 0, stream>>>(
    in_row_ind.data(), in->rows(), in->cols(), in->vals(), out->rows(),
    out->cols(), out->vals(), in->n_rows, in->nnz, reduction_op);
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Find how much space needed in each row.
 * We look through all datapoints and increment the count for each row.
 *
 * @param data: Input knn distances(n, k)
 * @param indices: Input knn indices(n, k)
 * @param n: Number of rows
 * @param k: Number of n_neighbors
 * @param row_sizes: Input empty row sum 1 array(n)
 * @param row_sizes2: Input empty row sum 2 array(n) for faster reduction
 */
template <typename value_idx = int64_t, typename value_t = float>
__global__ static void symmetric_find_size(const value_t *restrict data,
                                           const value_idx *restrict indices,
                                           const value_idx n, const int k,
                                           value_idx *restrict row_sizes,
                                           value_idx *restrict row_sizes2) {
  const auto row = blockIdx.x * blockDim.x + threadIdx.x;  // for every row
  const auto j =
    blockIdx.y * blockDim.y + threadIdx.y;  // for every item in row
  if (row >= n || j >= k) return;

  const auto col = indices[row * k + j];
  if (j % 2)
    atomicAdd(&row_sizes[col], value_idx(1));
  else
    atomicAdd(&row_sizes2[col], value_idx(1));
}

/**
 * @brief Reduce sum(row_sizes) + k
 * Reduction for symmetric_find_size kernel. Allows algo to be faster.
 *
 * @param n: Number of rows
 * @param k: Number of n_neighbors
 * @param row_sizes: Input row sum 1 array(n)
 * @param row_sizes2: Input row sum 2 array(n) for faster reduction
 */
template <typename value_idx>
__global__ static void reduce_find_size(const value_idx n, const int k,
                                        value_idx *restrict row_sizes,
                                        const value_idx *restrict row_sizes2) {
  const auto i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= n) return;
  row_sizes[i] += (row_sizes2[i] + k);
}

/**
 * @brief Perform data + data.T operation.
 * Can only run once row_sizes from the CSR matrix of data + data.T has been
 * determined.
 *
 * @param edges: Input row sum array(n) after reduction
 * @param data: Input knn distances(n, k)
 * @param indices: Input knn indices(n, k)
 * @param VAL: Output values for data + data.T
 * @param COL: Output column indices for data + data.T
 * @param ROW: Output row indices for data + data.T
 * @param n: Number of rows
 * @param k: Number of n_neighbors
 */
template <typename value_idx = int64_t, typename value_t = float>
__global__ static void symmetric_sum(value_idx *restrict edges,
                                     const value_t *restrict data,
                                     const value_idx *restrict indices,
                                     value_t *restrict VAL,
                                     value_idx *restrict COL,
                                     value_idx *restrict ROW, const value_idx n,
                                     const int k) {
  const auto row = blockIdx.x * blockDim.x + threadIdx.x;  // for every row
  const auto j =
    blockIdx.y * blockDim.y + threadIdx.y;  // for every item in row
  if (row >= n || j >= k) return;

  const auto col = indices[row * k + j];
  const auto original = atomicAdd(&edges[row], value_idx(1));
  const auto transpose = atomicAdd(&edges[col], value_idx(1));

  VAL[transpose] = VAL[original] = data[row * k + j];
  // Notice swapped ROW, COL since transpose
  ROW[original] = row;
  COL[original] = col;

  ROW[transpose] = col;
  COL[transpose] = row;
}

/**
 * @brief Perform data + data.T on raw KNN data.
 * The following steps are invoked:
 * (1) Find how much space needed in each row
 * (2) Compute final space needed (n*k + sum(row_sizes)) == 2*n*k
 * (3) Allocate new space
 * (4) Prepare edges for each new row
 * (5) Perform final data + data.T operation
 * (6) Return summed up VAL, COL, ROW
 *
 * @param knn_indices: Input knn distances(n, k)
 * @param knn_dists: Input knn indices(n, k)
 * @param n: Number of rows
 * @param k: Number of n_neighbors
 * @param out: Output COO Matrix class
 * @param stream: Input cuda stream
 * @param d_alloc device allocator for temporary buffers
 */
template <typename value_idx = int64_t, typename value_t = float,
          int TPB_X = 32, int TPB_Y = 32>
void from_knn_symmetrize_matrix(
  const value_idx *restrict knn_indices, const value_t *restrict knn_dists,
  const value_idx n, const int k, COO<value_t, value_idx> *out,
  cudaStream_t stream, std::shared_ptr<raft::mr::device::allocator> d_alloc) {
  // (1) Find how much space needed in each row
  // We look through all datapoints and increment the count for each row.
  const dim3 threadsPerBlock(TPB_X, TPB_Y);
  const dim3 numBlocks(raft::ceildiv(n, (value_idx)TPB_X),
                       raft::ceildiv(k, TPB_Y));

  // Notice n+1 since we can reuse these arrays for transpose_edges, original_edges in step (4)
  raft::mr::device::buffer<value_idx> row_sizes(d_alloc, stream, n);
  CUDA_CHECK(
    cudaMemsetAsync(row_sizes.data(), 0, sizeof(value_idx) * n, stream));

  raft::mr::device::buffer<value_idx> row_sizes2(d_alloc, stream, n);
  CUDA_CHECK(
    cudaMemsetAsync(row_sizes2.data(), 0, sizeof(value_idx) * n, stream));

  symmetric_find_size<<<numBlocks, threadsPerBlock, 0, stream>>>(
    knn_dists, knn_indices, n, k, row_sizes.data(), row_sizes2.data());
  CUDA_CHECK(cudaPeekAtLastError());

  reduce_find_size<<<raft::ceildiv(n, (value_idx)1024), 1024, 0, stream>>>(
    n, k, row_sizes.data(), row_sizes2.data());
  CUDA_CHECK(cudaPeekAtLastError());

  // (2) Compute final space needed (n*k + sum(row_sizes)) == 2*n*k
  // Notice we don't do any merging and leave the result as 2*NNZ
  const auto NNZ = 2 * n * k;

  // (3) Allocate new space
  out->allocate(NNZ, n, n, true, stream);

  // (4) Prepare edges for each new row
  // This mirrors CSR matrix's row Pointer, were maximum bounds for each row
  // are calculated as the cumulative rolling sum of the previous rows.
  // Notice reusing old row_sizes2 memory
  value_idx *edges = row_sizes2.data();
  thrust::device_ptr<value_idx> __edges = thrust::device_pointer_cast(edges);
  thrust::device_ptr<value_idx> __row_sizes =
    thrust::device_pointer_cast(row_sizes.data());

  // Rolling cumulative sum
  thrust::exclusive_scan(thrust::cuda::par.on(stream), __row_sizes,
                         __row_sizes + n, __edges);

  // (5) Perform final data + data.T operation in tandem with memcpying
  symmetric_sum<<<numBlocks, threadsPerBlock, 0, stream>>>(
    edges, knn_dists, knn_indices, out->vals(), out->cols(), out->rows(), n, k);
  CUDA_CHECK(cudaPeekAtLastError());
}

};  // end NAMESPACE linalg
};  // end NAMESPACE sparse
};  // end NAMESPACE raft