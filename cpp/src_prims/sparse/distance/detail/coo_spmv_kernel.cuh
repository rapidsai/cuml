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

#include <sparse/utils.h>

#include <cub/cub.cuh>

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
template <typename strategy_t, typename indptr_it, typename value_idx,
          typename value_t, int tpb, bool rev, typename product_f,
          typename accum_f, typename write_f>
__global__ void balanced_coo_generalized_spmv_kernel(
  strategy_t strategy, indptr_it indptrA, value_idx *indicesA, value_t *dataA,
  value_idx nnz_a, value_idx *rowsB, value_idx *indicesB, value_t *dataB,
  value_idx m, value_idx n, value_idx dim, value_idx nnz_b, value_t *out,
  int n_blocks_per_row, int chunk_size, product_f product_func,
  accum_f accum_func, write_f write_func) {
  typedef cub::WarpReduce<value_t> warp_reduce;

  value_idx cur_row_a = indptrA.get_row_idx(n_blocks_per_row);
  value_idx cur_chunk_offset = blockIdx.x % n_blocks_per_row;

  // chunk starting offset
  value_idx ind_offset = cur_chunk_offset * chunk_size * tpb;
  // how many total cols will be processed by this block (should be <= chunk_size * n_threads)
  value_idx active_chunk_size = min(chunk_size * tpb, nnz_b - ind_offset);

  int tid = threadIdx.x;
  int warp_id = tid / raft::warp_size();

  // compute id relative to current warp
  unsigned int lane_id = tid & (raft::warp_size() - 1);
  // value_idx ind = ind_offset + threadIdx.x;
  constexpr int n_warps = tpb / raft::warp_size();
  value_idx chunk_per_warp = active_chunk_size < n_warps ? active_chunk_size : active_chunk_size / n_warps;
  // chunk_per_warp = chunk_per_warp == 0 ? active_chunk_size : chunk_per_warp;
  value_idx ind = ind_offset + chunk_per_warp * warp_id + lane_id;

  extern __shared__ char smem[];

  typename strategy_t::smem_type A = (typename strategy_t::smem_type)(smem);
  typename warp_reduce::TempStorage *temp_storage =
    (typename warp_reduce::TempStorage *)(A + dim);

  auto inserter = strategy.init_insert(A, dim);

  __syncthreads();

  value_idx start_offset_a, stop_offset_a, start_index_a, stop_index_a;
  indptrA.get_row_offsets(cur_row_a, start_offset_a, stop_offset_a,
                          n_blocks_per_row);

  // Convert current row vector in A to dense
  for (int i = tid; i < (stop_offset_a - start_offset_a); i += blockDim.x) {
    strategy.insert(inserter, indicesA[start_offset_a + i],
                    dataA[start_offset_a + i]);
  }

  __syncthreads();

  auto finder = strategy.init_find(A);

  if (cur_row_a > m || cur_chunk_offset > n_blocks_per_row) return;
  if (ind >= nnz_b) return;
  // if (tid < active_chunk_size) printf("cur_row_a: %d, m: %d, cur_chunk_offset: %d, n_blocks_per_row: %d, ind: %d, nnz_b: %d\n", cur_row_a, m, cur_chunk_offset, n_blocks_per_row, ind, nnz_b);

  indptrA.get_indices_boundary(indicesA, cur_row_a, start_offset_a, stop_offset_a,
    start_index_a, stop_index_a);
  // if (threadIdx.x == 0) {
  // printf(
  // "blockIdx.x: %d, cur_row_a: %d, start_offset: %d, end_offset: %d, "
  // "start_index_a: %d, stop_index_a: %d, nnz_a: %d, active_chunk_size: %d\n",
  // blockIdx.x, cur_row_a, start_offset_a, stop_offset_a, start_index_a,
  // stop_index_a, nnz_a, active_chunk_size);
  // }

  value_idx cur_row_b = -1;
  value_t c = 0.0;

  auto warp_red = warp_reduce(*(temp_storage + warp_id));

  // coalesced reads from B
  // if (tid < active_chunk_size) printf("HI\n");
  // printf("warp_id: %d, lane_id: %d, chunk_per_warp: %d, ind: %d, active_chunk_size: %d, n_warps: %d\n", warp_id, lane_id, chunk_per_warp, ind, active_chunk_size, n_warps);
  if (lane_id < chunk_per_warp) {
    cur_row_b = rowsB[ind];

    auto index_b = indicesB[ind];
    // auto in_bounds = true;
    // if (cur_row_a == cur_row_b)
    auto in_bounds =
      indptrA.check_indices_bounds(start_index_a, stop_index_a, index_b);

    if (in_bounds) {
      value_t a_col = strategy.find(finder, index_b, indicesA, dataA, start_offset_a, stop_offset_a);
      // if (tid < active_chunk_size)
      // printf("blockIdx.x: %d, cur_row_a: %d, index_b: %d, a_col: %f\n", blockIdx.x, cur_row_a,
      //        index_b, a_col);

      if (!rev || a_col == 0.0) {
        c = product_func(a_col, dataB[ind]);
      }
    }
  }

  // loop through chunks in parallel, reducing when a new row is
  // encountered by each thread
  for (int i = lane_id; i < chunk_per_warp; i += raft::warp_size()) {
    value_idx ind_next = ind + raft::warp_size();
    value_idx next_row_b = -1;

    if (i + raft::warp_size() < chunk_per_warp) next_row_b = rowsB[ind_next];

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

      auto index_b = indicesB[ind];
      // auto in_bounds = true;
      // if (cur_row_a == cur_row_b)
      auto in_bounds =
        indptrA.check_indices_bounds(start_index_a, stop_index_a, index_b);
      if (in_bounds) {
        value_t a_col = strategy.find(finder, index_b, indicesA, dataA, start_offset_a, stop_offset_a);

        if (!rev || a_col == 0.0) {
          c = accum_func(c, product_func(a_col, dataB[ind]));
        }
      }

      cur_row_b = next_row_b;
    }
  }
}

}  // namespace distance
}  // namespace sparse
}  // namespace raft
