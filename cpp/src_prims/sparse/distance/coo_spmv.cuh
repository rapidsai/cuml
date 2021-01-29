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

#include <sparse/distance/common.h>
#include <sparse/utils.h>
#include <sparse/csr.cuh>

#include <limits.h>

#include <nvfunctional>

#include <cusparse_v2.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_store.cuh>

namespace raft {
namespace sparse {
namespace distance {

/**
 * Load-balanced sparse-matrix-sparse-matrix multiplication (SPMM) kernel with
 * sparse-matrix-sparse-vector multiplication layout (SPMV).
 * This is intended to be scheduled n_chunks_b times for each row of a.
 * The steps are as follows:
 *
 * 1. Load row from A into dense vector in shared memory.
 *    This can be further chunked in the future if necessary to support larger
 *    column sizes.
 * 2. Threads of block all step through chunks of B in parallel.
 *    When a new row is encountered in row_indices_b, a segmented
 *    reduction is performed across the warps and then across the
 *    block and the final value written out to host memory.
 *
 * Reference: https://www.icl.utk.edu/files/publications/2020/icl-utk-1421-2020.pdf
 *
 * @tparam value_idx index type
 * @tparam value_t value type
 * @tparam tpb threads per block configured on launch
 * @tparam rev if this is true, the reduce/accumulate functions are only
 *         executed when A[col] == 0.0. when executed before/after !rev
 *         and A & B are reversed, this allows the full symmetric difference
 *         and intersection to be computed.
 * @tparam kv_t data type stored in shared mem cache
 * @tparam product_f reduce function type (semiring product() function).
 *                  accepts two arguments of value_t and returns a value_t
 * @tparam accum_f accumulation function type (semiring sum() function).
 *                 accepts two arguments of value_t and returns a value_t
 * @tparam write_f function to write value out. this should be mathematically
 *                 equivalent to the accumulate function but implemented as
 *                 an atomic operation on global memory. Accepts two arguments
 *                 of value_t* and value_t and updates the value given by the
 *                 pointer.
 * @param[in] indptrA column pointer array for A
 * @param[in] indicesA column indices array for A
 * @param[in] dataA data array for A
 * @param[in] rowsB coo row array for B
 * @param[in] indicesB column indices array for B
 * @param[in] dataB data array for B
 * @param[in] m number of rows in A
 * @param[in] n number of rows in B
 * @param[in] dim number of features
 * @param[in] nnz_b number of nonzeros in B
 * @param[out] out array of size m*n
 * @param[in] n_blocks_per_row number of blocks of B per row of A
 * @param[in] chunk_size number of nnz for B to use for each row of A
 * @param[in] buffer_size amount of smem to use for each row of A
 * @param[in] product_func semiring product() function
 * @param[in] accum_func semiring sum() function
 * @param[in] write_func atomic semiring sum() function
 */
template <typename value_idx, typename value_t, int tpb, bool rev,
          typename kv_t, typename product_f, typename accum_f, typename write_f>
__global__ void balanced_coo_generalized_spmv_kernel(
  value_idx *indptrA, value_idx *indicesA, value_t *dataA, value_idx *rowsB,
  value_idx *indicesB, value_t *dataB, value_idx m, value_idx n, value_idx dim,
  value_idx nnz_b, value_t *out, int n_blocks_per_row, int chunk_size,
  product_f product_func, accum_f accum_func, write_f write_func) {
  typedef cub::WarpReduce<value_t> warp_reduce;

  value_idx cur_row_a = blockIdx.x / n_blocks_per_row;
  value_idx cur_chunk_offset = blockIdx.x % n_blocks_per_row;

  // chunk starting offset
  value_idx ind_offset = cur_chunk_offset * chunk_size * tpb;
  // how many total cols will be processed by this block (should be <= chunk_size * n_threads)
  value_idx active_chunk_size = min(chunk_size * tpb, nnz_b - ind_offset);

  int tid = threadIdx.x;
  int warp_id = tid / raft::warp_size();

  // compute id relative to current warp
  unsigned int lane_id = tid & (raft::warp_size() - 1);
  value_idx ind = ind_offset + threadIdx.x;

  extern __shared__ char smem[];

  value_idx *offsets_a = (value_idx *)smem;
  kv_t *A = (kv_t *)(offsets_a + 2);
  typename warp_reduce::TempStorage *temp_storage =
    (typename warp_reduce::TempStorage *)(A + dim);

  // Create dense vector A and populate with 0s
  for (int k = tid; k < dim; k += blockDim.x) A[k] = 0;

  if (tid == 0) {
    offsets_a[0] = indptrA[cur_row_a];
    offsets_a[1] = indptrA[cur_row_a + 1];
  }

  __syncthreads();

  value_idx start_offset_a = offsets_a[0];
  value_idx stop_offset_a = offsets_a[1];

  // Convert current row vector in A to dense
  for (int i = tid; i < (stop_offset_a - start_offset_a); i += blockDim.x) {
    A[indicesA[start_offset_a + i]] = dataA[start_offset_a + i];
  }

  __syncthreads();

  if (cur_row_a > m || cur_chunk_offset > n_blocks_per_row) return;
  if (ind >= nnz_b) return;

  value_idx cur_row_b = -1;
  value_t c = 0.0;

  auto warp_red = warp_reduce(*(temp_storage + warp_id));

  // coalesced reads from B
  if (tid < active_chunk_size) {
    cur_row_b = rowsB[ind];
    value_t a_col = A[indicesB[ind]];
    if (!rev || a_col == 0.0) c = product_func(a_col, dataB[ind]);
  }

  // loop through chunks in parallel, reducing when a new row is
  // encountered by each thread
  for (int i = tid; i < active_chunk_size; i += blockDim.x) {
    value_idx ind_next = ind + blockDim.x;
    value_idx next_row_b = -1;

    if (i + blockDim.x < active_chunk_size) next_row_b = rowsB[ind_next];

    bool diff_rows = next_row_b != cur_row_b;

    if (__any_sync(0xffffffff, diff_rows)) {
      // grab the threads currently participating in loops.
      // because any other threads should have returned already.
      unsigned int peer_group = __match_any_sync(0xffffffff, cur_row_b);
      bool is_leader = get_lowest_peer(peer_group) == lane_id;
      value_t v = warp_red.HeadSegmentedReduce(c, is_leader, accum_func);

      // thread with lowest lane id among peers writes out
      if (is_leader && v != 0.0) {
        // this conditional should be uniform, since rev is constant
        size_t idx = !rev ? (size_t)cur_row_a * n + cur_row_b
                          : (size_t)cur_row_b * m + cur_row_a;
        write_func(out + idx, v);
      }

      c = 0.0;
    }

    if (next_row_b != -1) {
      ind = ind_next;
      value_t a_col = A[indicesB[ind]];
      if (!rev || a_col == 0.0)
        c = accum_func(c, product_func(a_col, dataB[ind]));
      cur_row_b = next_row_b;
    }
  }
}

/**
 * Computes the maximum number of columns that can be stored
 * in shared memory in dense form with the given block size
 * and precision.
 * @return the maximum number of columns that can be stored in smem
 */
template <typename value_idx, typename value_t, int tpb = 1024>
inline int max_cols_per_block() {
  // max cols = (total smem available - offsets for A - cub reduction smem)
  return (raft::getSharedMemPerBlock() - (2 * sizeof(value_idx)) -
          ((tpb / raft::warp_size()) * sizeof(value_t))) /
         sizeof(value_t);
}

template <typename value_idx, typename value_t, int tpb = 1024>
inline int smem_per_block(int n_cols) {
  int max_cols = max_cols_per_block<value_idx, value_t, tpb>();
  ASSERT(n_cols <= max_cols, "COO SPMV Requires max dimensionality of %d",
         max_cols);
  return (n_cols * sizeof(value_t)) + (2 * sizeof(value_idx)) +
         ((tpb / raft::warp_size()) * sizeof(value_t));
}

/**
 * Performs generalized sparse-matrix-sparse-matrix multiplication via a
 * sparse-matrix-sparse-vector layout `out=A*B` where generalized product()
 * and sum() operations can be used in place of the standard sum and product:
 *
 * out_ij = sum_k(product(A_ik, B_ik)) The sum goes through values of
 * k=0..n_cols-1 where B_kj is nonzero.
 *
 * The product and sum operations shall form a semiring algebra with the
 * following properties:
 * 1. {+, 0} is a commutative sum reduction monoid with identity element 0
 * 2. {*, 1} is a product monoid with identity element 1
 * 3. Multiplication by 0 annihilates x. e.g. product(x, 0) = 0
 *
 * Each vector of A is loaded into shared memory in dense form and the
 * non-zeros of B load balanced across the threads of each block.
 * @tparam value_idx index type
 * @tparam value_t value type
 * @tparam threads_per_block block size
 * @tparam chunk_size number of nonzeros of B to process for each row of A
 *         this value was found through profiling and represents a reasonable
 *         setting for both large and small densities
 * @tparam product_f semiring product() function
 * @tparam accum_f semiring sum() function
 * @tparam write_f atomic semiring sum() function
 * @param[out] out_dists dense array of out distances of size m * n in row-major
 *             format.
 * @param[in] config_ distance config object
 * @param[in] coo_rows_b coo row array for B
 * @param[in] product_func semiring product() function
 * @param[in] accum_func semiring sum() function
 * @param[in] write_func atomic semiring sum() function
 */
template <typename value_idx, typename value_t, int threads_per_block = 1024,
          int chunk_size = 500000, typename product_f, typename accum_f,
          typename write_f>
inline void balanced_coo_pairwise_generalized_spmv(
  value_t *out_dists, const distances_config_t<value_idx, value_t> &config_,
  value_idx *coo_rows_b, product_f product_func, accum_f accum_func,
  write_f write_func) {
  CUDA_CHECK(cudaMemsetAsync(
    out_dists, 0, sizeof(value_t) * config_.a_nrows * config_.b_nrows,
    config_.stream));
  int n_blocks_per_row =
    raft::ceildiv(config_.b_nnz, chunk_size * threads_per_block);
  int n_blocks = config_.a_nrows * n_blocks_per_row;

  int smem =
    smem_per_block<value_idx, value_t, threads_per_block>(config_.a_ncols);

  CUML_LOG_DEBUG("n_blocks: %d", n_blocks);
  CUML_LOG_DEBUG("n_warps_per_row: %d", n_blocks_per_row);
  CUML_LOG_DEBUG("smem_per_block: %d", smem);

  CUDA_CHECK(cudaFuncSetCacheConfig(
    balanced_coo_generalized_spmv_kernel<value_idx, value_t, threads_per_block,
                                         false, value_t, product_f, accum_f,
                                         write_f>,
    cudaFuncCachePreferShared));

  balanced_coo_generalized_spmv_kernel<value_idx, value_t, threads_per_block,
                                       false, value_t>
    <<<n_blocks, threads_per_block, smem, config_.stream>>>(
      config_.a_indptr, config_.a_indices, config_.a_data, coo_rows_b,
      config_.b_indices, config_.b_data, config_.a_nrows, config_.b_nrows,
      config_.b_ncols, config_.b_nnz, out_dists, n_blocks_per_row, chunk_size,
      product_func, accum_func, write_func);
};

/**
 * Used for computing distances where the reduction (e.g. product()) function
 * requires an implicit union (product(x, 0) = x) to capture the difference A-B.
 * This is necessary in some applications because the standard semiring algebra
 * endowed with the default multiplication product monoid will only
 * compute the intersection & B-A.
 *
 * This particular function is meant to accompany the function
 * `balanced_coo_pairwise_generalized_spmv` and executes the product operation
 * on only those columns that exist in B and not A.
 *
 * The product and sum operations shall enable the computation of a
 * non-annihilating semiring algebra with the following properties:
 * 1. {+, 0} is a commutative sum reduction monoid with identity element 0
 * 2. {*, 0} is a product monoid with identity element 0
 * 3. Multiplication by 0 does not annihilate x. e.g. product(x, 0) = x
 *
 * Manattan distance sum(abs(x_k-y_k)) is a great example of when this type of
 * execution pattern is necessary.
 *
 * @tparam value_idx index type
 * @tparam value_t value type
 * @tparam threads_per_block block size
 * @tparam chunk_size number of nonzeros of B to process for each row of A
 *         this value was found through profiling and represents a reasonable
 *         setting for both large and small densities
 * @tparam product_f semiring product() function
 * @tparam accum_f semiring sum() function
 * @tparam write_f atomic semiring sum() function
 * @param[out] out_dists dense array of out distances of size m * n
 * @param[in] config_ distance config object
 * @param[in] coo_rows_a coo row array for A
 * @param[in] product_func semiring product() function
 * @param[in] accum_func semiring sum() function
 * @param[in] write_func atomic semiring sum() function
 */
template <typename value_idx, typename value_t, int threads_per_block = 1024,
          int chunk_size = 500000, typename product_f, typename accum_f,
          typename write_f>
inline void balanced_coo_pairwise_generalized_spmv_rev(
  value_t *out_dists, const distances_config_t<value_idx, value_t> &config_,
  value_idx *coo_rows_a, product_f product_func, accum_f accum_func,
  write_f write_func) {
  int n_blocks_per_row =
    raft::ceildiv(config_.a_nnz, chunk_size * threads_per_block);
  int n_blocks = config_.b_nrows * n_blocks_per_row;

  int smem =
    smem_per_block<value_idx, value_t, threads_per_block>(config_.a_ncols);

  CUML_LOG_DEBUG("n_blocks: %d", n_blocks);
  CUML_LOG_DEBUG("n_warps_per_row: %d", n_blocks_per_row);
  CUML_LOG_DEBUG("smem_per_block: %d", smem);

  CUDA_CHECK(cudaFuncSetCacheConfig(
    balanced_coo_generalized_spmv_kernel<value_idx, value_t, threads_per_block,
                                         true, value_t, product_f, accum_f,
                                         write_f>,
    cudaFuncCachePreferShared));

  balanced_coo_generalized_spmv_kernel<value_idx, value_t, threads_per_block,
                                       true, value_t>
    <<<n_blocks, threads_per_block, smem, config_.stream>>>(
      config_.b_indptr, config_.b_indices, config_.b_data, coo_rows_a,
      config_.a_indices, config_.a_data, config_.b_nrows, config_.a_nrows,
      config_.a_ncols, config_.a_nnz, out_dists, n_blocks_per_row, chunk_size,
      product_func, accum_func, write_func);
};
}  // namespace distance
}  // namespace sparse
};  // namespace raft
