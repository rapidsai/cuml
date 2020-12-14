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

#include <raft/cudart_utils.h>
#include <sparse/utils.h>
#include <common/device_buffer.hpp>
#include <cuml/common/cuml_allocator.hpp>
#include <raft/cuda_utils.cuh>
#include <sparse/csr.cuh>

#include <sparse/selection.cuh>

#include <limits.h>

#include <cuml/distance/distance_type.h>
#include <cuml/neighbors/knn.hpp>

#include <nvfunctional>

#include <cusparse_v2.h>
#include <raft/sparse/cusparse_wrappers.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_store.cuh>

#include <sparse/distance_api.h>

#pragma once

namespace faiss { namespace gpu {

template <int NumThreads, typename K, typename V, int NumWarpQ,
  bool Dir, typename Comp>
struct FinalBlockMerge<32, NumThreads, K, V, NumWarpQ, Dir, Comp> {
  static inline __device__ void merge(K* sharedK, V* sharedV) {
    blockMerge<NumThreads, K, V, NumThreads / (kWarpSize * 2),
      NumWarpQ, !Dir, Comp>(sharedK, sharedV);
    blockMerge<NumThreads, K, V, NumThreads / (kWarpSize * 4),
      NumWarpQ * 2, !Dir, Comp>(sharedK, sharedV);
    blockMerge<NumThreads, K, V, NumThreads / (kWarpSize * 8),
      NumWarpQ * 4, !Dir, Comp>(sharedK, sharedV);
    blockMerge<NumThreads, K, V, NumThreads / (kWarpSize * 16),
      NumWarpQ * 8, !Dir, Comp>(sharedK, sharedV);
    // Final merge doesn't need to fully merge the second list
    blockMerge<NumThreads, K, V, NumThreads / (kWarpSize * 32),
      NumWarpQ * 16, !Dir, Comp, false>(sharedK, sharedV);
  }
};

}}

namespace MLCommon {
namespace Sparse {
namespace Distance {

const int MAX_INT = std::numeric_limits<int>::max();

// TODO: Change these to structures w/ operator() function

/**
 * Semiring which schedules each row of B in a different thread.
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @tparam buffer_size
 * @tparam rows_per_block
 */
template <
  typename value_idx,
  typename value_t,
  int tpb,
  int buffer_size,
  typename reduce_f = auto(value_t, value_t)->value_t,
  typename accum_f = auto(value_t, value_t)->value_t>
struct BlockSemiring {
  __device__ inline BlockSemiring(int tid_, value_idx m_, value_idx n_,
                                  value_idx *shared_cols_,
                                  value_t *shared_vals_, value_idx *chunk_cols_,
                                  value_t *chunk_vals_,
                                  value_idx *offsets_a_,
                                  bool verbose_)
    : tid(tid_),
      m(m_),
      n(n_),
      shared_cols(shared_cols_),
      shared_vals(shared_vals_),
      chunk_cols(chunk_cols_),
      chunk_vals(chunk_vals_),
      offsets_a(offsets_a_),
      done(false),
      shared_idx(0),
      verbose(verbose_),
      row_count(0),
      cur_sum(0.0) {}

  __device__ inline void load_a(value_idx row, value_idx *indptrA,
                                value_idx *indicesA, value_t *dataA) {

    if(tid == 0) {
      offsets_a[0] = indptrA[row];
      offsets_a[1] = indptrA[row + 1];
    }
    __syncthreads();

    start_offset_a = offsets_a[0];
    stop_offset_a = offsets_a[1];

    // Coalesce reads of row from matrix A into shared memory
    for (int i = tid; i < stop_offset_a - start_offset_a; i += blockDim.x) {
      shared_cols[i] = indicesA[start_offset_a + i];
      shared_vals[i] = dataA[start_offset_a + i];
    }

    __syncthreads();

    shared_size = stop_offset_a - start_offset_a;
    row_a = row;
  }

  __device__ inline void load_b(value_idx start_row, value_idx *indptrB) {

    done = false;
    shared_idx = 0;
    cur_sum = 0.0;

    start_row_b = start_row;
    stop_row_b = min(start_row_b + tpb,
                     n - start_row_b);

    n_rows = (stop_row_b - start_row_b);

    for (int i = tid; i < n_rows; i += tpb) {
      row_b = start_row_b + tid;
      start_offset_b = indptrB[start_row_b + i];
      row_count = indptrB[start_row_b + i + 1] - start_offset_b;
      local_idx = start_offset_b;
      local_idx_stop = start_offset_b + row_count;
    }
  }

  __device__ inline void step(reduce_f reduce_func, accum_f accum_func) {

    if (tid < n_rows) {
      bool local_idx_in_bounds = local_idx < local_idx_stop && row_count > 0;

      value_idx l = (local_idx_in_bounds) * chunk_cols[local_idx] + (!local_idx_in_bounds)*(-1);
      value_t lv = (local_idx_in_bounds) * chunk_vals[local_idx] + (!local_idx_in_bounds)*(0.0);

      bool shared_idx_in_bounds = shared_idx < shared_size;

      value_idx r = (shared_idx_in_bounds) * shared_cols[shared_idx] + (!shared_idx_in_bounds)*(-1);
      value_t rv = (shared_idx_in_bounds) * shared_vals[shared_idx] + (!shared_idx_in_bounds)*(0.0);
//

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

  __device__ inline value_t get_sum() {
    return cur_sum;
  }

  __device__ inline void write(value_t *out) {
    for (int i = tid; i < n_rows; i += blockDim.x) {
      out[row_a * n + row_b] = cur_sum;
    }
  }


  __device__ inline value_idx get_n_rows() { return n_rows; }

  __device__ inline void print() {
    printf("BlockSemiring<local_idx=%d, local_idx_stop=%d, cur_sum=%f\n",
           local_idx, local_idx_stop, cur_sum);
  }

 private:
  int tid;

  bool done;

  int shared_size;

  value_idx n_rows;

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

  bool verbose;
};

/**
 * Optimized for large numbers of rows but small enough numbers of columns
 * that each thread can process their rows in parallel.
 */

template <
  typename value_idx,
  typename value_t,
  int tpb,
  int buffer_size,
  typename reduce_f = auto(value_t, value_t)->value_t,
  typename accum_f = auto(value_t, value_t)->value_t>
__global__ void classic_csr_semiring_spmv_kernel(
    value_idx *indptrA,
    value_idx *indicesA,
    value_t *dataA,
    value_idx *indptrB,
    value_idx *indicesB,
    value_t *dataB,
    value_idx m, value_idx n, value_t *out,
    int n_blocks_per_row,
    int n_rows_per_block,
    reduce_f reduce_func,
    accum_f accum_func) {

  value_idx out_row = blockIdx.x / n_blocks_per_row;
  value_idx out_col_start = blockIdx.x % n_blocks_per_row;
  value_idx tid = threadIdx.x;

  if (out_row > m || out_col_start > n_blocks_per_row) return;

  __shared__ value_idx shared_cols[buffer_size];
  __shared__ value_t shared_vals[buffer_size];

  __shared__ value_idx offsets_a[2];

  bool verbose = tid <= 3 && out_row < 3;

  BlockSemiring<value_idx, value_t, tpb, buffer_size> semiring(
    tid, m, n, shared_cols, shared_vals, indicesB, dataB,
    offsets_a, verbose);

  semiring.load_a(out_row, indptrA, indicesA, dataA);

  // for each batch, parallel the resulting rows across threads
  for(int i = 0; i < n_rows_per_block; i+= blockDim.x) {
    semiring.load_b(out_col_start + i, indptrB);

    do {
      semiring.step(reduce_func, accum_func);
    } while(!semiring.isdone());

    semiring.write(out);
  }

}


/**
 * Perform generalized SPMV. Each vector of A is loaded into
 * shared memory and each row of B parallelized over threads.
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
template <typename value_idx = int,
          typename value_t = float,
          int max_buffer_size = 5000,
          int threads_per_block = 1024,
          typename reduce_f = auto(value_t, value_t)->value_t,
          typename accum_f = auto(value_t, value_t)->value_t>
void generalized_csr_pairwise_semiring(value_t *out_dists,
                          distances_config_t<value_idx, value_t> config_,
                          reduce_f reduce_func, accum_f accum_func) {
  int n_chunks = 1;
  int n_rows_per_block = min(n_chunks * threads_per_block, config_.b_nrows);
  int n_warps_per_row = raft::ceildiv(config_.b_nrows, n_rows_per_block);
  int n_blocks = config_.a_nrows * n_warps_per_row;

  CUML_LOG_DEBUG("Classic block reduce");

  CUML_LOG_DEBUG("n_blocks: %d", n_blocks);
  CUML_LOG_DEBUG("n_warps_per_row: %d", n_warps_per_row);

  classic_csr_semiring_spmv_kernel<value_idx, value_t, threads_per_block,
                              max_buffer_size, reduce_f, accum_f>
    <<<n_blocks, threads_per_block, 0, config_.stream>>>(
      config_.a_indptr, config_.a_indices, config_.a_data, config_.b_indptr,
      config_.b_indices, config_.b_data, config_.a_nrows, config_.b_nrows,
      out_dists, n_warps_per_row, n_rows_per_block,
      reduce_func, accum_func);
};

/**
 * This implementation follows the load-balanced implementation. This is intended
 * to be scheduled n_chunks_b times for each row of a.
 *
 * The steps are as follows:
 *
 * 1. Load row from A into dense vector in shared memory. This can be chunked if necessary.
 * 2. Threads of block all step through chunks of B in parallel. When a new row is encountered in row_indices_b,
 *    a segmented reduction is performed across the warps and then across the block and the final value written out
 *    to host memory.
 *
 * Reference: https://www.icl.utk.edu/files/publications/2020/icl-utk-1421-2020.pdf
 *
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @tparam buffer_size
 * @tparam chunk_size
 * @param indptrA
 * @param indicesA
 * @param dataA
 * @param rowsB
 * @param indicesB
 * @param dataB
 * @param m
 * @param n
 * @param out
 */
template<
  typename value_idx,
  typename value_t,
  int tpb,
  int buffer_size>
__global__ void balanced_coo_spmv_kernel(value_idx *indptrA,
                                         value_idx *indicesA,
                                         value_t *dataA,
                                         value_idx *rowsB,
                                         value_idx *indicesB,
                                         value_t *dataB,
                                         value_idx m,
                                         value_idx n,
                                         value_idx dim,
                                         value_idx nnz_b,
                                         value_t *out,
                                         int n_blocks_per_row,
                                         int chunk_size) {

  typedef cub::WarpReduce<value_t> warp_reduce;

  value_idx cur_row_a = blockIdx.x / n_blocks_per_row;
  value_idx cur_chunk_offset = blockIdx.x % n_blocks_per_row;

  // chunk starting offset
  value_idx ind_offset = cur_chunk_offset * chunk_size * tpb;

  // how many total cols will be processed by this block (should be <= chunk_size * n_threads)
  value_idx active_chunk_size = min(chunk_size*tpb, nnz_b - ind_offset);

  int tid = threadIdx.x;

  // compute id relative to current warp
  unsigned int lane_id = tid&31;
  value_idx ind = ind_offset + threadIdx.x;

  if (cur_row_a > m || cur_chunk_offset > n_blocks_per_row) return;
  if (ind >= nnz_b) return;

  __shared__ value_t A[buffer_size];
  __shared__ value_idx offsets_a[2];
  __shared__ typename warp_reduce::TempStorage temp_storage;

  if(tid == 0) {
    offsets_a[0] = indptrA[cur_row_a];
    offsets_a[1] = indptrA[cur_row_a +1];
  }

  __syncthreads();

  value_idx start_offset_a = offsets_a[0];
  value_idx stop_offset_a = offsets_a[1];

  // Create dense vector A and populate with 0s
  for(int i = threadIdx.x; i < dim; i += blockDim.x)
    A[i] = 0.0;

  // Convert current row vector in A to dense
  for(int i = tid; i < (stop_offset_a - start_offset_a)+1; i += blockDim.x) {
    value_idx ind_a = indicesA[start_offset_a+i];
    value_t val_a = dataA[start_offset_a+i];
    A[ind_a] = val_a;
  }

  __syncthreads();

  value_idx cur_row_b = -1;
  value_t c = 0.0;

  if(tid < active_chunk_size) {
    cur_row_b = rowsB[ind];
    value_idx col = indicesB[ind];
    c = A[col] * dataB[ind];
  }

  // loop through chunks in parallel, reducing when a new row is
  // encountered by each thread
  for(int i = tid; i < active_chunk_size; i+=blockDim.x) {

    value_idx ind_next = ind + blockDim.x;
    value_idx next_row_b = -1;

    if(i+blockDim.x < active_chunk_size)
      next_row_b = rowsB[ind_next];

    if(next_row_b != cur_row_b) {

      unsigned int peer_group = get_peer_group(cur_row_b);
      bool is_leader = get_lowest_peer(peer_group) == lane_id;

      value_t v = warp_reduce(temp_storage).HeadSegmentedSum(c, is_leader);

      // thread with lowest lane id among peers writes out
      if(is_leader && v != 0.0) {
        atomicAdd(out + (cur_row_a * n + cur_row_b), v);
      }
      c = 0.0;
    }

    if(next_row_b != -1) {
      ind = ind_next;
      value_idx col = indicesB[ind];
      c += A[col] * dataB[ind];
      cur_row_b = next_row_b;
    }
  }
}


/**
 * Performs generalized SPMV. Each vector of A is loaded
 * into shared memory in dense form and the columns of B
 * load balanced over threads.
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
  int max_buffer_size = 11000,
  int threads_per_block = 1024>
void balanced_coo_pairwise_spmv(value_t *out_dists,
                       distances_config_t<value_idx, value_t> config_) {

  int chunk_size = 500000;

  int n_warps_per_row =
    raft::ceildiv(config_.b_nnz, chunk_size * threads_per_block);
  int n_blocks = config_.a_nrows * n_warps_per_row;

  CUML_LOG_DEBUG("n_blocks: %d", n_blocks);
  CUML_LOG_DEBUG("n_warps_per_row: %d", n_warps_per_row);

  device_buffer<value_idx> rows_b(config_.allocator, config_.stream,
                                  config_.b_nnz);
  MLCommon::Sparse::csr_to_coo(config_.b_indptr, config_.b_nrows, rows_b.data(),
                               config_.b_nnz, config_.stream);

  balanced_coo_spmv_kernel<value_idx, value_t, threads_per_block,
                               max_buffer_size>
    <<<n_blocks, threads_per_block, 0, config_.stream>>>(
      config_.a_indptr, config_.a_indices, config_.a_data, rows_b.data(),
      config_.b_indices, config_.b_data, config_.a_nrows, config_.b_nrows,
      config_.b_ncols, config_.b_nnz, out_dists, n_warps_per_row, chunk_size);
};

template <typename value_idx = int, typename value_t = float>
class l1_distances_t : public distances_t<value_t> {
 public:
  l1_distances_t(distances_config_t<value_idx, value_t> config)
    : config_(config) {}

  void compute(value_t *out_dists) {
    CUML_LOG_DEBUG("Running l1 dists");
    generalized_csr_pairwise_semiring<value_idx, value_t>(
      out_dists, config_,
      [] __host__ __device__  (value_t a, value_t b) { return fabsf(a-b); },
      [] __host__ __device__  (value_t a, value_t b) { return a+b; });

    // TODO: Remove
    CUDA_CHECK(cudaStreamSynchronize(config_.stream));

    std::cout << "Done. printing" << std::endl;

        std::cout << raft::arr2Str(out_dists, 16,
                                   "out_dists", config_.stream)
                  << std::endl;
  }

 private:
  distances_config_t<value_idx, value_t> config_;
};

template <typename value_idx = int, typename value_t = float>
class l2_unexpanded_distances_t : public distances_t<value_t> {
 public:
  l2_unexpanded_distances_t(distances_config_t<value_idx, value_t> config)
    : config_(config) {}

  void compute(value_t *out_dists) {
    generalized_csr_pairwise_semiring<value_idx, value_t>(
      out_dists, config_,
        [] __host__ __device__  (value_t a, value_t b) { return (a - b) * (a - b); },
        [] __host__ __device__  (value_t a, value_t b) { return a+b; });
  }

 private:
  distances_config_t<value_idx, value_t> config_;
};

template <typename value_idx = int, typename value_t = float>
class chebychev_distances_t : public distances_t<value_t> {
 public:
  explicit chebychev_distances_t(distances_config_t<value_idx, value_t> config)
    : config_(config) {}

  void compute(value_t *out_dists) {
    generalized_csr_pairwise_semiring<value_idx, value_t>(
      out_dists, config_,
      [] __host__ __device__  (value_t a, value_t b) { return fabsf(a - b); },
      [] __host__ __device__  (value_t a, value_t b) { return fmaxf(a, b); });
  }

 private:
  distances_config_t<value_idx, value_t> config_;
};

template <typename value_idx = int, typename value_t = float>
class canberra_distances_t : public distances_t<value_t> {
 public:
  explicit canberra_distances_t(distances_config_t<value_idx, value_t> config)
    : config_(config) {}

  void compute(value_t *out_dists) {
    generalized_csr_pairwise_semiring<value_idx, value_t>(
      out_dists, config_,
      [] __device__(value_t a, value_t b) {
        return fabsf(a - b) / (fabsf(a) + fabsf(b));
      }, [] __host__ __device__  (value_t a, value_t b) { return a+b; });
  }

 private:
  distances_config_t<value_idx, value_t> config_;
};

template <typename value_idx = int, typename value_t = float>
class minkowski_distances_t : public distances_t<value_t> {
 public:
  explicit minkowski_distances_t(distances_config_t<value_idx, value_t> config,
                                 value_t p_)
    : config_(config), p(p_) {}

  void compute(value_t *out_dists) {
    generalized_csr_pairwise_semiring<value_idx, value_t>(
      out_dists, config_,
      [=] __device__(value_t a, value_t b) {
        return fpowf(a-b, p);
      }, [] __host__ __device__  (value_t a, value_t b) {
        return a+b; });
  }

 private:
  distances_config_t<value_idx, value_t> config_;
  value_t p;
};



}  // namespace Distance
}  // namespace Sparse
};  // namespace MLCommon
