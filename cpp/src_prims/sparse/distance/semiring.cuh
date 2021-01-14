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

#include <raft/cudart_utils.h>
#include <sparse/utils.h>
#include <common/device_buffer.hpp>
#include <cuml/common/cuml_allocator.hpp>
#include <raft/cuda_utils.cuh>
#include <sparse/csr.cuh>

#include <sparse/selection.cuh>

#include <limits.h>

#include <nvfunctional>

#include <cusparse_v2.h>
#include <raft/sparse/cusparse_wrappers.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_store.cuh>

#include <sparse/distance/common.h>

namespace MLCommon {
namespace Sparse {
namespace Distance {

/**
 * Semiring which schedules each row of B in a different thread.
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @tparam buffer_size
 * @tparam rows_per_block
 */
template <typename value_idx, typename value_t, int tpb, typename reduce_f,
          typename accum_f>
struct BlockSemiring {
  __device__ inline BlockSemiring(int tid_, value_idx m_, value_idx n_,
                                  value_idx *shared_cols_,
                                  value_t *shared_vals_, value_idx *chunk_cols_,
                                  value_t *chunk_vals_, value_idx *offsets_a_)
    : tid(tid_),
      p(p),
      m(m_),
      n(n_),
      shared_cols(shared_cols_),
      shared_vals(shared_vals_),
      chunk_cols(chunk_cols_),
      chunk_vals(chunk_vals_),
      offsets_a(offsets_a_),
      done(false),
      shared_idx(0),
      row_count(0),
      cur_sum(0.0) {}

  /**
   * Load columns for a single row of A into shared memory
   * @param row
   * @param indptrA
   * @param indicesA
   * @param dataA
   */
  __device__ inline void load_a(value_idx row, value_idx *indptrA,
                                value_idx *indicesA, value_t *dataA) {
    if (tid == 0) {
      offsets_a[0] = indptrA[row];
      offsets_a[1] = indptrA[row + 1];
    }
    __syncthreads();

    start_offset_a = offsets_a[0];
    stop_offset_a = offsets_a[1];

    shared_size = stop_offset_a - start_offset_a;

    // Coalesce reads of row from matrix A into shared memory
    for (int i = tid; i < shared_size; i += blockDim.x) {
      shared_cols[i] = indicesA[start_offset_a + i];
      shared_vals[i] = dataA[start_offset_a + i];
    }

    __syncthreads();

    row_a = row;
  }

  /**
   * Prepare index & offsets for looping through rows of B
   * @param start_row
   * @param indptrB
   */
  __device__ inline void load_b(value_idx start_row, value_idx *indptrB) {
    done = false;
    shared_idx = 0;
    cur_sum = 0.0;

    start_row_b = start_row;
    stop_row_b = min(start_row_b + tpb, start_row_b + (n - start_row_b));

    n_rows = (stop_row_b - start_row_b);

    for (int i = tid; i < n_rows; i += tpb) {
      row_b = start_row_b + tid;
      start_offset_b = indptrB[start_row_b + i];
      row_count = indptrB[start_row_b + i + 1] - start_offset_b;
      local_idx = start_offset_b;
      local_idx_stop = start_offset_b + row_count;
    }
  }

  /**
   * Perform single single column intersection/union for A & B
   * based on the row of A mapped to shared memory and the row
   * of B mapped to current thread.
   * @param reduce_func
   * @param accum_func
   */
  __device__ inline void step(reduce_f reduce_func, accum_f accum_func) {
    if (tid < n_rows) {
      bool local_idx_in_bounds = local_idx < local_idx_stop && row_count > 0;

      value_idx l = local_idx_in_bounds ? chunk_cols[local_idx] : -1;
      value_t lv = local_idx_in_bounds ? chunk_vals[local_idx] : 0.0;

      bool shared_idx_in_bounds = shared_idx < shared_size;

      value_idx r = shared_idx_in_bounds ? shared_cols[shared_idx] : -1;
      value_t rv = shared_idx_in_bounds ? shared_vals[shared_idx] : 0.0;

      bool run_l = ((l <= r && l != -1) || (l != -1 && r == -1));
      local_idx += 1 * run_l;
      value_t left_side = lv * run_l;

      bool run_r = ((r <= l && r != -1) || (l == -1 && r != -1));
      shared_idx += 1 * run_r;
      value_t right_side = rv * run_r;

      // Apply semiring "sum" & "product" functions locally
      cur_sum = accum_func(cur_sum, reduce_func(left_side, right_side));

      // finished when all items in chunk have been
      // processed
      done = l == -1 && r == -1;

    } else {
      done = true;
    }
  }

  __device__ inline bool isdone() { return done; }

  __device__ inline void write(value_t *out) {
    for (int i = tid; i < n_rows; i += blockDim.x) {
      out[row_a * n + row_b] = cur_sum;
    }
  }

 private:
  int tid;

  bool done;

  int shared_size;

  value_idx n_rows;

  value_t p;

  value_idx local_idx;
  value_idx local_idx_stop;
  value_idx shared_idx;

  value_t cur_sum;

  value_idx n_entries;

  value_idx m;
  value_idx n;
  value_idx start_offset_a;
  value_idx stop_offset_a;

  value_idx row_a;
  value_idx row_b;

  value_idx start_offset_b;

  value_idx start_row_b;
  value_idx stop_row_b;

  value_idx *offsets_a;

  // shared memory
  value_idx row_count;
  value_idx *shared_cols;
  value_t *shared_vals;
  value_idx *chunk_cols;
  value_t *chunk_vals;
};

/**
 * Optimized for large numbers of rows but small enough numbers of columns
 * that each thread can process their rows in parallel.
 */
template <typename value_idx, typename value_t, int tpb, typename reduce_f,
          typename accum_f>
__global__ void classic_csr_semiring_spmv_kernel(
  value_idx *indptrA, value_idx *indicesA, value_t *dataA, value_idx *indptrB,
  value_idx *indicesB, value_t *dataB, value_idx m, value_idx n, value_t *out,
  int n_blocks_per_row, int n_rows_per_block, int buffer_size,
  reduce_f reduce_func, accum_f accum_func) {
  value_idx out_row = blockIdx.x / n_blocks_per_row;
  value_idx out_col_start = blockIdx.x % n_blocks_per_row;

  value_idx row_b_start = out_col_start * n_rows_per_block;
  value_idx tid = threadIdx.x;

  if (out_row > m || row_b_start > n) return;

  extern __shared__ char smem[];

  value_idx *offsets_a = (value_idx *)smem;
  value_idx *shared_cols = offsets_a + 2;
  value_t *shared_vals = (value_t *)(shared_cols + buffer_size);

  BlockSemiring<value_idx, value_t, tpb, reduce_f, accum_f> semiring(
    tid, m, n, shared_cols, shared_vals, indicesB, dataB, offsets_a);

  semiring.load_a(out_row, indptrA, indicesA, dataA);

  // for each batch, parallel the resulting rows across threads
  for (int i = 0; i < n_rows_per_block; i += blockDim.x) {
    semiring.load_b(row_b_start + i, indptrB);
    do {
      semiring.step(reduce_func, accum_func);
    } while (!semiring.isdone());

    semiring.write(out);
  }
}

/**
 * Compute the maximum number of nonzeros that can be stored in shared
 * memory per block with the given index and value precision
 * @tparam value_idx
 * @tparam value_t
 * @return
 */
template <typename value_idx, typename value_t>
inline value_idx max_nnz_per_block() {
  // compute max shared mem to use
  return (raft::getSharedMemPerBlock() - (2 * sizeof(value_idx))) /
         (sizeof(value_t) + sizeof(value_idx));
}

/**
 * Perform generalized sparse-matrix-sparse-vector multiply in
 * semiring algebra by allowing the reduction (product) and
 * accumulation (sum) functions to be swapped out for custom
 * functions. This approach saves the most memory as it can
 * work directly on a CSR w/o the need for conversion to another
 * sparse format, does not require any transposition, nor loading
 * any vectors in dense form. The major drawback to this kernel
 * is that the non-uniform memory access pattern dominates performance,
 * making it very slow.
 *
 * Each vector of A is loaded into shared memory and each row of B
 * parallelized over threads. While vector A can be coalesced into
 * shared memory, the rows from vector B cannot, and thus this
 * kernel remains almost entirely global memory bound.
 *
 * TODO: Some potential things to try for future optimizations:
 *  - Always iterating for n_cols so that each warp is iterating
 *    a uniform number of times.
 *  - Computing an argsort() of B based on the number of columns
 *    in each row to attempt to load balance the warps naturally
 *  - Finding a way to coalesce the reads
 *
 * @tparam value_idx
 * @tparam value_t
 * @tparam max_buffer_size
 * @tparam threads_per_block
 * @tparam reduce_f
 * @tparam accum_f
 * @param out_dists
 * @param config_
 * @param reduce_func
 * @param accum_func
 */
template <typename value_idx = int, typename value_t = float,
          int threads_per_block = 1024, typename reduce_f, typename accum_f>
void generalized_csr_pairwise_semiring(
  value_t *out_dists, const distances_config_t<value_idx, value_t> &config_,
  reduce_f reduce_func, accum_f accum_func) {
  int n_chunks = 1;
  int n_rows_per_block = min(n_chunks * threads_per_block, config_.b_nrows);
  int n_blocks_per_row = raft::ceildiv(config_.b_nrows, n_rows_per_block);
  int n_blocks = config_.a_nrows * n_blocks_per_row;

  int smem_buffer_size = max_nnz_per_block<value_idx, value_t>();

  CUML_LOG_DEBUG("Classic block reduce");

  CUML_LOG_DEBUG("n_blocks: %d", n_blocks);
  CUML_LOG_DEBUG("n_blocks_per_row: %d", n_blocks_per_row);
  CUML_LOG_DEBUG("smem_buffer_size: %d", smem_buffer_size);

  classic_csr_semiring_spmv_kernel<value_idx, value_t, threads_per_block,
                                   reduce_f, accum_f>
    <<<n_blocks, threads_per_block, raft::getSharedMemPerBlock(),
       config_.stream>>>(config_.a_indptr, config_.a_indices, config_.a_data,
                         config_.b_indptr, config_.b_indices, config_.b_data,
                         config_.a_nrows, config_.b_nrows, out_dists,
                         n_blocks_per_row, n_rows_per_block, smem_buffer_size,
                         reduce_func, accum_func);
};

}  // namespace Distance
}  // namespace Sparse
};  // namespace MLCommon
