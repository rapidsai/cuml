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

#include <raft/cudart_utils.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>
#include <raft/mr/device/allocator.hpp>
#include <raft/mr/device/buffer.hpp>

#include <common/allocatorAdapter.hpp>

#include <sparse/distance/common.h>
#include <sparse/utils.h>
#include <sparse/csr.cuh>
#include <sparse/distance/operators.cuh>

#include <limits.h>

#include <nvfunctional>

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_store.cuh>

namespace raft {
namespace sparse {
namespace distance {

/**
 * Semiring which schedules each row of B in a different thread.
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @tparam buffer_size
 * @tparam rows_per_block
 */
template <typename value_idx, typename value_t, int tpb, typename product_f,
          typename accum_f>
struct BlockSemiring {
  __device__ inline BlockSemiring(value_idx n_, value_idx *shared_cols_,
                                  value_t *shared_vals_, value_idx *offsets_a_)
    : n(n_),
      a_cols(shared_cols_),
      a_vals(shared_vals_),
      offsets_a(offsets_a_),
      done(false),
      a_idx(0),
      b_row_count(0),
      cur_sum(0.0) {}

  /**
   * Load columns for a single row of A into shared memory
   * @param row
   * @param indptrA
   * @param indicesA
   * @param dataA
   */
  __device__ inline void load_a_shared(value_idx row, value_idx *indptrA,
                                       value_idx *indicesA, value_t *dataA) {
    if (threadIdx.x == 0) {
      offsets_a[0] = indptrA[row];
      offsets_a[1] = indptrA[row + 1];
    }
    __syncthreads();

    value_idx start_offset_a = offsets_a[0];
    value_idx stop_offset_a = offsets_a[1];

    a_size = stop_offset_a - start_offset_a;

    // Coalesce reads of row from matrix A into shared memory
    for (int i = threadIdx.x; i < a_size; i += blockDim.x) {
      a_cols[i] = indicesA[start_offset_a + i];
      a_vals[i] = dataA[start_offset_a + i];
    }

    __syncthreads();

    row_a = row;
  }

  /**
   * Sets the head for A's pointers so they can be
   * iterated in each thread. This is used for the
   * case when the maximum degree of any row in A
   * is too large to fit into shared memory, so we
   * default to increasing the size of the L1 cache
   * and suffering the uncoalesced memory accesses
   * for both A and B.
   * @param row
   * @param indptrA
   * @param indicesA
   * @param dataA
   */
  __device__ inline void load_a(value_idx row, value_idx *indptrA,
                                value_idx *indicesA, value_t *dataA) {
    offsets_a[0] = indptrA[row];
    offsets_a[1] = indptrA[row + 1];

    value_idx start_offset_a = offsets_a[0];
    value_idx stop_offset_a = offsets_a[1];

    a_size = stop_offset_a - start_offset_a;

    a_cols = indicesA + start_offset_a;
    a_vals = dataA + start_offset_a;

    row_a = row;
  }

  /**
   * Prepare index & offsets for looping through rows of B
   * @param start_row
   * @param indptrB
   */
  __device__ inline void load_b(value_idx start_row, value_idx *indptrB) {
    done = false;
    a_idx = 0;
    cur_sum = 0.0;

    value_idx start_row_b = start_row;
    value_idx stop_row_b = min(start_row_b + tpb, n);

    n_rows_b = stop_row_b - start_row_b;

    if (threadIdx.x < n_rows_b) {
      row_b = start_row_b + threadIdx.x;
      value_idx start_offset_b = indptrB[row_b];
      b_row_count = indptrB[row_b + 1] - start_offset_b;
      b_idx = start_offset_b;
      b_idx_stop = start_offset_b + b_row_count;
    }
  }

  /**
   * Perform single single column intersection/union for A & B
   * based on the row of A mapped to shared memory and the row
   * of B mapped to current thread.
   * @param product_func
   * @param accum_func
   */
  __device__ inline void step(value_idx *b_cols, value_t *b_vals,
                              product_f product_func, accum_f accum_func) {
    if (threadIdx.x < n_rows_b) {
      bool local_idx_in_bounds = b_idx < b_idx_stop && b_row_count > 0;

      value_idx b = local_idx_in_bounds ? b_cols[b_idx] : -1;
      value_t bv = local_idx_in_bounds ? b_vals[b_idx] : 0.0;

      bool a_idx_in_bounds = a_idx < a_size;

      value_idx a = a_idx_in_bounds ? a_cols[a_idx] : -1;
      value_t av = a_idx_in_bounds ? a_vals[a_idx] : 0.0;

      bool run_b = ((b <= a && b != -1) || (b != -1 && a == -1));
      b_idx += 1 * run_b;
      value_t b_side = bv * run_b;

      bool run_a = ((a <= b && a != -1) || (b == -1 && a != -1));
      a_idx += 1 * run_a;
      value_t a_side = av * run_a;

      // Apply semiring "sum" & "product" functions locally
      cur_sum = accum_func(cur_sum, product_func(b_side, a_side));

      // finished when all items in chunk have been
      // processed
      done = b == -1 && a == -1;

    } else {
      done = true;
    }
  }

  __device__ inline bool isdone() { return done; }

  __device__ inline void write(value_t *out) {
    if (threadIdx.x < n_rows_b) {
      out[(size_t)row_a * n + row_b] = cur_sum;
    }
  }

 private:
  bool done;

  int a_size;

  value_idx n_rows_b;

  value_idx b_idx;
  value_idx b_idx_stop;
  value_idx a_idx;

  value_t cur_sum;

  value_idx n;

  value_idx row_a;
  value_idx row_b;

  value_idx *offsets_a;

  // shared memory
  value_idx b_row_count;
  value_idx *a_cols;
  value_t *a_vals;
};

/**
 * Optimized for large numbers of rows but small enough numbers of columns
 * that each thread can process their rows in parallel.
 * @tparam value_idx index type
 * @tparam value_t value type
 * @tparam tpb block size
 * @tparam product_f semiring product() function
 * @tparam accum_f semiring sum() function
 * @param[in] indptrA csr column index pointer array for A
 * @param[in] indicesA csr column indices array for A
 * @param[in] dataA csr data array for A
 * @param[in] indptrB csr column index pointer array for B
 * @param[in] indicesB csr column indices array for B
 * @param[in] dataB csr data array for B
 * @param[in] m number of rows in A
 * @param[in] n number of rows in B
 * @param[out] out dense output array of size m * n in row-major layout
 * @param[in] n_blocks_per_row number of blocks of B scheduled per row of A
 * @param[in] n_rows_per_block number of rows of A scheduled per block of B
 * @param[in] buffer_size number of nonzeros to store in smem
 * @param[in] product_func semiring product() function
 * @param[in] accum_func semiring sum() function
 */
template <typename value_idx, typename value_t, int tpb, typename product_f,
          typename accum_f>
__global__ void classic_csr_semiring_spmv_smem_kernel(
  value_idx *indptrA, value_idx *indicesA, value_t *dataA, value_idx *indptrB,
  value_idx *indicesB, value_t *dataB, value_idx m, value_idx n, value_t *out,
  int n_blocks_per_row, int n_rows_per_block, int buffer_size,
  product_f product_func, accum_f accum_func) {
  value_idx out_row = blockIdx.x / n_blocks_per_row;
  value_idx out_col_start = blockIdx.x % n_blocks_per_row;

  value_idx row_b_start = out_col_start * n_rows_per_block;

  extern __shared__ char smem[];

  value_idx *offsets_a = (value_idx *)smem;
  value_idx *a_cols = offsets_a + 2;
  value_t *a_vals = (value_t *)(a_cols + buffer_size);

  BlockSemiring<value_idx, value_t, tpb, product_f, accum_f> semiring(
    n, a_cols, a_vals, offsets_a);

  semiring.load_a_shared(out_row, indptrA, indicesA, dataA);

  if (out_row > m || row_b_start > n) return;

  // for each batch, parallelize the resulting rows across threads
  for (int i = 0; i < n_rows_per_block; i += blockDim.x) {
    semiring.load_b(row_b_start + i, indptrB);
    do {
      semiring.step(indicesB, dataB, product_func, accum_func);
    } while (!semiring.isdone());

    semiring.write(out);
  }
}

template <typename value_idx, typename value_t, int tpb, typename product_f,
          typename accum_f>
__global__ void classic_csr_semiring_spmv_kernel(
  value_idx *indptrA, value_idx *indicesA, value_t *dataA, value_idx *indptrB,
  value_idx *indicesB, value_t *dataB, value_idx m, value_idx n, value_t *out,
  int n_blocks_per_row, int n_rows_per_block, product_f product_func,
  accum_f accum_func) {
  value_idx out_row = blockIdx.x / n_blocks_per_row;
  value_idx out_col_start = blockIdx.x % n_blocks_per_row;

  value_idx row_b_start = out_col_start * n_rows_per_block;

  value_idx offsets_a[2];

  BlockSemiring<value_idx, value_t, tpb, product_f, accum_f> semiring(
    n, indicesA, dataA, offsets_a);

  semiring.load_a(out_row, indptrA, indicesA, dataA);

  if (out_row > m || row_b_start > n) return;

  // for each batch, parallel the resulting rows across threads
  for (int i = 0; i < n_rows_per_block; i += blockDim.x) {
    semiring.load_b(row_b_start + i, indptrB);
    do {
      semiring.step(indicesB, dataB, product_func, accum_func);
    } while (!semiring.isdone());

    semiring.write(out);
  }
}

/**
 * Compute the maximum number of nonzeros that can be stored in shared
 * memory per block with the given index and value precision
 * @return max nnz that can be stored in smem per block
 */
template <typename value_idx, typename value_t>
inline value_idx max_nnz_per_block() {
  // max nnz = total smem - offsets for A
  // (division because we need to store cols & vals separately)
  return (raft::getSharedMemPerBlock() - (2 * sizeof(value_idx))) /
         (sizeof(value_t) + sizeof(value_idx));
}

/**
 * @tparam value_idx
 * @param out
 * @param in
 * @param n
 */
template <typename value_idx>
__global__ void max_kernel(value_idx *out, value_idx *in, value_idx n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  typedef cub::BlockReduce<value_idx, 256> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  value_idx v = tid < n ? in[tid] - in[tid - 1] : 0;
  value_idx agg = BlockReduce(temp_storage).Reduce(v, cub::Max());

  if (threadIdx.x == 0) atomicMax(out, agg);
}

template <typename value_idx>
inline value_idx max_degree(
  value_idx *indptr, value_idx n_rows,
  std::shared_ptr<raft::mr::device::allocator> allocator, cudaStream_t stream) {
  raft::mr::device::buffer<value_idx> max_d(allocator, stream, 1);
  CUDA_CHECK(cudaMemsetAsync(max_d.data(), 0, sizeof(value_idx), stream));

  /**
   * A custom max reduction is performed until https://github.com/rapidsai/cuml/issues/3431
   * is fixed.
   */
  max_kernel<<<raft::ceildiv(n_rows, 256), 256, 0, stream>>>(
    max_d.data(), indptr + 1, n_rows);

  value_idx max_h;
  raft::update_host(&max_h, max_d.data(), 1, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUML_LOG_DEBUG("max nnz: %d", max_h);

  return max_h;
}

template <typename value_idx = int, typename value_t = float,
          int threads_per_block = 64, typename product_f, typename accum_f>
void _generalized_csr_pairwise_semiring(
  value_t *out_dists, const distances_config_t<value_idx, value_t> &config_,
  product_f product_func, accum_f accum_func) {
  int n_chunks = 1;
  int n_rows_per_block = min(n_chunks * threads_per_block, config_.b_nrows);
  int n_blocks_per_row = raft::ceildiv(config_.b_nrows, n_rows_per_block);
  int n_blocks = config_.a_nrows * n_blocks_per_row;

  CUML_LOG_DEBUG("n_blocks: %d", n_blocks);
  CUML_LOG_DEBUG("n_blocks_per_row: %d", n_blocks_per_row);

  CUDA_CHECK(cudaFuncSetCacheConfig(
    classic_csr_semiring_spmv_kernel<value_idx, value_t, threads_per_block,
                                     product_f, accum_f>,
    cudaFuncCachePreferL1));

  classic_csr_semiring_spmv_kernel<value_idx, value_t, threads_per_block,
                                   product_f, accum_f>
    <<<n_blocks, threads_per_block, 0, config_.stream>>>(
      config_.a_indptr, config_.a_indices, config_.a_data, config_.b_indptr,
      config_.b_indices, config_.b_data, config_.a_nrows, config_.b_nrows,
      out_dists, n_blocks_per_row, n_rows_per_block, product_func, accum_func);
};

template <typename value_idx = int, typename value_t = float,
          int threads_per_block = 32, typename product_f, typename accum_f>
void _generalized_csr_pairwise_smem_semiring(
  value_t *out_dists, const distances_config_t<value_idx, value_t> &config_,
  product_f product_func, accum_f accum_func, value_idx max_nnz) {
  int n_chunks = 10000;
  int n_rows_per_block = min(n_chunks * threads_per_block, config_.b_nrows);
  int n_blocks_per_row = raft::ceildiv(config_.b_nrows, n_rows_per_block);
  int n_blocks = config_.a_nrows * n_blocks_per_row;

  // TODO: Figure out why performance is worse with smaller smem sizes
  int smem_size = raft::getSharedMemPerBlock();

  CUML_LOG_DEBUG("n_blocks: %d", n_blocks);
  CUML_LOG_DEBUG("n_blocks_per_row: %d", n_blocks_per_row);
  CUML_LOG_DEBUG("smem_size: %d", smem_size);

  CUDA_CHECK(cudaFuncSetCacheConfig(
    classic_csr_semiring_spmv_smem_kernel<value_idx, value_t, threads_per_block,
                                          product_f, accum_f>,
    cudaFuncCachePreferShared));

  classic_csr_semiring_spmv_smem_kernel<value_idx, value_t, threads_per_block,
                                        product_f, accum_f>
    <<<n_blocks, threads_per_block, smem_size, config_.stream>>>(
      config_.a_indptr, config_.a_indices, config_.a_data, config_.b_indptr,
      config_.b_indices, config_.b_data, config_.a_nrows, config_.b_nrows,
      out_dists, n_blocks_per_row, n_rows_per_block, max_nnz, product_func,
      accum_func);
}

/**
 * Perform generalized sparse-matrix-sparse-vector multiply in
 * a semiring algebra by allowing the product and sum operations
 * to be defined. This approach saves the most memory as it can
 * work directly on a CSR w/o the need for conversion to another
 * sparse format, does not require any transposition, nor loading
 * any vectors in dense form. The major drawback to this kernel
 * is that the non-uniform memory access pattern dominates performance.
 * When the shared memory option is used, bank conflicts also dominate
 * performance, making it slower than other options but guaranteeing
 * that the product() operation will be executed across every column
 * in A and B.
 *
 * This is primarily useful when in cases where the product() operation
 * is non-anniliating (e.g. product(x, 0) = x.
 *
 * There are two potential code paths for this primitive- if the largest
 * degree of any row is small enough to fit in shared memory then shared
 * memory is used to coalesce the reads from the vectors of A, otherwise
 * no shared memory is used and all loads from A and B happen independently
 * in separate threads.
 *
 * Iterators are maintained for the vectors from both A and B and each
 * thread iterates to a maximum of |a|+|b| (which will happen only when
 * the set of columns for vectors a and b are completely disjoint.
 *
 * TODO: Some potential things to try for future optimizations:
 *  - Always iterating for n_cols so that each warp is iterating
 *    a uniform number of times.
 *  - Computing an argsort() of B based on the number of columns
 *    in each row to attempt to load balance the warps naturally
 *  - Finding a way to coalesce the reads
 *
 *  Ref: https://github.com/rapidsai/cuml/issues/3371
 *
 * @tparam value_idx index type
 * @tparam value_t value type
 * @tparam product_f semiring product() function
 * @tparam accum_f semiring sum() function
 * @param[out] out_dists dense array of output distances size m * n in row-major layout
 * @param[in] config_ distance config object
 * @param[in] product_func semiring product() function
 * @param[in] accum_func semiring sum() function
 */
template <typename value_idx = int, typename value_t = float,
          typename product_f, typename accum_f>
void generalized_csr_pairwise_semiring(
  value_t *out_dists, const distances_config_t<value_idx, value_t> &config_,
  product_f product_func, accum_f accum_func) {
  int nnz_upper_bound = max_nnz_per_block<value_idx, value_t>();

  CUML_LOG_DEBUG("Classic block reduce");
  CUML_LOG_DEBUG("nnz_upper_bound: %d", nnz_upper_bound);

  // max_nnz set from max(diff(indptrA))
  value_idx max_nnz = max_degree<value_idx>(config_.a_indptr, config_.a_nrows,
                                            config_.allocator, config_.stream) +
                      1;

  if (max_nnz <= nnz_upper_bound)
    // use smem
    _generalized_csr_pairwise_smem_semiring<value_idx, value_t>(
      out_dists, config_, product_func, accum_func, max_nnz);

  else
    // load each row of A separately
    _generalized_csr_pairwise_semiring<value_idx, value_t>(
      out_dists, config_, product_func, accum_func);
};

}  // namespace distance
}  // namespace sparse
};  // namespace raft
