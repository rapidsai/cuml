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

#include <cuml/neighbors/knn.hpp>

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

template <typename value_idx>
__device__ value_idx binarySearch(const value_idx* array, value_idx target,
                                  value_idx begin, value_idx end) {
  while (begin < end) {
    int mid = begin + (end - begin) / 2;
    int item = array[mid];
    if (item == target) return mid;
    bool larger = (item > target);
    if (larger)
      end = mid;
    else
      begin = mid + 1;
  }
  return -1;
}

template <typename value_idx, typename value_t,
          typename reduce_f = auto(value_t, value_t)->value_t,
          typename accum_f = auto(value_t, value_t)->value_t,
          typename write_f = auto(value_t*, value_t)->void>
__global__ void csr_spgemm_kernel(value_t* C, const value_idx* A_csrRowPtr,
                                  const value_idx* A_csrColInd,
                                  const value_t* A_csrVal,
                                  const value_idx* B_cscColPtr,
                                  const value_idx* B_cscRowInd,
                                  const value_t* B_cscVal, value_idx A_nrows,
                                  value_idx B_nrows, reduce_f mul_op,
                                  accum_f add_op, write_f agg_op, bool rev) {
  value_idx thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  value_idx warp_id = thread_id / 32;  // A_row
  value_idx lane_id = thread_id & (32 - 1);
  if (warp_id < A_nrows) {
    value_idx row_start = A_csrRowPtr[warp_id];
    value_idx row_end = A_csrRowPtr[warp_id + 1];

    value_t accumulator = 0.0;
    // Entire warp works together on each nonzero
    for (value_idx edge = row_start; edge < row_end; ++edge) {
      // Load B bounds on which we must do binary search
      value_idx B_ind = B_cscRowInd[edge];
      value_idx B_col_start = B_cscColPtr[B_ind];
      value_idx B_col_end = B_cscColPtr[B_ind + 1];

      // Each thread iterates along row
      // Does binary search on B_row to try to find A_col
      // Adds result to accumulator if found
      value_idx ind = row_start + lane_id;
      for (value_idx ind_start = row_start; ind_start < row_end;
           ind_start += 32) {
        if (ind < row_end) {
          value_idx A_col = A_csrColInd[ind];
          value_idx B_row =
            binarySearch(B_cscRowInd, A_col, B_col_start, B_col_end);
          value_t A_t = A_csrVal[ind];
          value_t B_t = B_row != -1 ? B_cscVal[B_row] : 0.0;

          bool should_store = (!rev || A_t == 0.0);
          value_t C_t = should_store * mul_op(A_t, B_t);
          accumulator = add_op(C_t, accumulator);
        }
        ind += 32;
      }

      // Warp reduce for each edge
      for (int i = 1; i < 32; i *= 2)
        accumulator = add_op(__shfl_xor_sync(-1, accumulator, i), accumulator);

      // Write to output
      if (lane_id == 0 && accumulator != 0.0) {
        value_idx idx =
          !rev ? warp_id * B_nrows + edge : edge * A_nrows + warp_id;
        agg_op(C + idx, accumulator);
      }
    }
  }
}

/**
 * Perform generalized sparse-matrix-sparse-matrix-multiply
 * in semiring algebra by allowing the reduction (product)
 * and accumulation (sum) functions to be swapped out for
 * custom functions. This approach utilizes the most memory
 * as it requires B to be transposed, however, it is able
 * to work directly on the CSR format and outputs directly
 * in dense form.
 *
 * Each warp processes a single row of A in parallel,
 * using a binary search to look up the columns of B.
 *
 * @tparam value_idx
 * @tparam value_t
 * @tparam threads_per_block
 * @tparam reduce_f
 * @tparam accum_f
 * @tparam write_f
 * @param out_dists
 * @param config_
 * @param reduce_func
 * @param accum_func
 * @param write_func
 */
template <typename value_idx, typename value_t, int threads_per_block = 1024,
          typename reduce_f = auto(value_t, value_t)->value_t,
          typename accum_f = auto(value_t, value_t)->value_t,
          typename write_f = auto(value_t*, value_t)->void>
void csr_pairwise_spgemm(value_t* out_dists,
                         distances_config_t<value_idx, value_t>& config_,
                         reduce_f reduce_func, accum_f accum_func,
                         write_f write_func) {
  int n_blocks = raft::ceildiv(config_.a_nrows * 32, threads_per_block);

  CUDA_CHECK(cudaMemsetAsync(
    out_dists, 0, config_.a_nrows * config_.b_nrows * sizeof(value_t),
    config_.stream));

  csr_spgemm_kernel<value_idx, value_t, threads_per_block>
    <<<n_blocks, threads_per_block, 0, config_.stream>>>(
      out_dists, config_.a_indptr, config_.a_indices, config_.a_data,
      config_.b_indptr, config_.b_indices, config_.b_data, config_.a_nrows,
      config_.b_nrows, reduce_func, accum_func, write_func, false);
}

template <typename value_idx, typename value_t, int threads_per_block = 1024,
          typename reduce_f = auto(value_t, value_t)->value_t,
          typename accum_f = auto(value_t, value_t)->value_t,
          typename write_f = auto(value_t*, value_t)->void>
void csr_pairwise_spgemm_rev(value_t* out_dists,
                             distances_config_t<value_idx, value_t>& config_,
                             reduce_f reduce_func, accum_f accum_func,
                             write_f write_func) {
  int n_blocks = raft::ceildiv(config_.b_nrows * 32, threads_per_block);

  csr_spgemm_kernel<value_idx, value_t, threads_per_block>
    <<<n_blocks, threads_per_block, 0, config_.stream>>>(
      out_dists, config_.b_indptr, config_.b_indices, config_.b_data,
      config_.a_indptr, config_.a_indices, config_.a_data, config_.b_nrows,
      config_.a_nrows, reduce_func, accum_func, write_func, true);
}

};  // namespace Distance
};  // namespace Sparse
};  // namespace MLCommon