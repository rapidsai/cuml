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


namespace MLCommon {
namespace Sparse {
namespace Distance {

#define NNZ_PER_WG 64u ///< Should be power of two

const int MAX_INT = std::numeric_limits<int>::max();

/**
 * Semiring which schedules each row of B in a different thread.
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @tparam buffer_size
 * @tparam rows_per_block
 */
template <typename value_idx, typename value_t, int tpb, int buffer_size,
          int rows_per_block>
struct BlockSemiring {
  __device__ inline BlockSemiring(int tid_, value_idx m_, value_idx n_,
                                  value_idx *shared_cols_,
                                  value_t *shared_vals_, value_idx *chunk_cols_,
                                  value_t *chunk_vals_,
                                  value_idx *offsets_a_,
                                  value_idx *offsets_b_,
                                  bool verbose_)
    : tid(tid_),
      m(m_),
      n(n_),
      shared_cols(shared_cols_),
      shared_vals(shared_vals_),
      chunk_cols(chunk_cols_),
      chunk_vals(chunk_vals_),
      offsets_a(offsets_a_),
      offsets_b(offsets_b_),
      done(false),
      shared_idx(0),
      verbose(verbose_),
      row_count(0),
      cur_sum(0.0) {}

  __device__ inline void load_a(value_idx row, value_idx *indptrA,
                                value_idx *indicesA, value_t *dataA) {

    // TODO: Hide these latencies further w/ shared memory

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

  __device__ inline void load_b(value_idx start_row, value_idx *indptrB,
                                value_idx *indicesB, value_t *dataB) {
    start_row_b = start_row * rows_per_block;
    stop_row_b = min(start_row_b + rows_per_block - 1,
                     start_row_b + (n - start_row_b) - 1);

    row_b = start_row_b + tid;
    n_rows = (stop_row_b - start_row_b) + 1;

    for (int i = tid; i < n_rows; i += blockDim.x)
      row_count = indptrB[start_row_b + i + 1] - indptrB[start_row_b + i];


    if(tid == 0) {
      offsets_b[0] = indptrB[start_row_b];
      offsets_b[1] = indptrB[stop_row_b + 1] - 1;
    }

    __syncthreads();

    start_offset_b = offsets_b[0];
    stop_offset_b = offsets_b[1];

    for (int i = tid; i < n_rows; i += blockDim.x) {
      // set starting and ending idx of local thread
      local_idx = indptrB[start_row_b + i]; //- start_offset_b;
      local_idx_stop = min(local_idx + row_count, stop_offset_b);
    }
  }

  __device__ inline void step() {
    if (tid < n_rows) {
      bool local_idx_in_bounds = local_idx < local_idx_stop && row_count > 0;

      value_idx l = local_idx_in_bounds ? chunk_cols[local_idx] : -1;
      value_t lv = local_idx_in_bounds ? chunk_vals[local_idx] : 0.0;

      bool shared_idx_in_bounds = shared_idx < shared_size;

      value_idx r = shared_idx_in_bounds ? shared_cols[shared_idx] : -1;
      value_t rv = shared_idx_in_bounds ? shared_vals[shared_idx] : 0.0;

      value_t left_side = 0.0;
      value_t right_side = 0.0;

      if ((l <= r && l != -1) || (l != -1 && r == -1)) {
        local_idx++;
        left_side = lv;
      }

      if ((r <= l && r != -1) || (l == -1 && r != -1)) {
        shared_idx++;
        right_side = rv;
      }

      // Apply semiring "sum" & "product" functions locally
      cur_sum += fabsf(left_side - right_side);

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
  value_idx stop_offset_b;

  value_idx start_row_b;
  value_idx stop_row_b;

  value_idx *offsets_a;
  value_idx *offsets_b;

  // shared memory
  value_idx row_count;
  value_idx *shared_cols;
  value_t *shared_vals;
  value_idx *chunk_cols;
  value_t *chunk_vals;

  bool verbose;
};

template<typename value_t>
__device__ value_t prev_power_of_2 (value_t n) {
  while (n & n - 1)
    n = n & n - 1;
  return n;
}

template <typename data_type>
__global__ void adaptive_csr_spmv_semiring_kernel (
  const unsigned int n_rows,
  const unsigned int *indices_b,
  const unsigned int *row_ind_b,
  const unsigned int *row_blocks,
  const data_type *data_b,
  const data_type *x,
  data_type *out) {
  const unsigned int block_row_begin = row_blocks[blockIdx.x];
  const unsigned int block_row_end = row_blocks[blockIdx.x + 1];
  const unsigned int nnz =
    row_ind_b[block_row_end] - row_ind_b[block_row_begin];

  __shared__ data_type cache[NNZ_PER_WG];

  if (block_row_end - block_row_begin > 1)
  {
    /// CSR-Stream case
    const unsigned int i = threadIdx.x;
    const unsigned int block_data_begin = row_ind_b[block_row_begin];
    const unsigned int thread_data_begin = block_data_begin + i;

    if (i < nnz)
      cache[i] = data_b[thread_data_begin] * x[indices_b[thread_data_begin]];
    __syncthreads ();

    const unsigned int threads_for_reduction =
      prev_power_of_2 (blockDim.x / (block_row_end - block_row_begin));

    if (threads_for_reduction > 1)
    {
      /// Reduce all non zeroes of row by multiple thread
      const unsigned int thread_in_block = i % threads_for_reduction;
      const unsigned int local_row = block_row_begin + i / threads_for_reduction;

      data_type dot = 0.0;

      if (local_row < block_row_end)
      {
        const unsigned int local_first_element =
          row_ind_b[local_row] - row_ind_b[block_row_begin];
        const unsigned int local_last_element =
          row_ind_b[local_row + 1] - row_ind_b[block_row_begin];

        for (unsigned int local_element = local_first_element + thread_in_block;
             local_element < local_last_element;
             local_element += threads_for_reduction)
        {
          dot += cache[local_element];
        }
      }
      __syncthreads ();
      cache[i] = dot;

      /// Now each row has threads_for_reduction values in cache
      for (int j = threads_for_reduction / 2; j > 0; j /= 2)
      {
        /// Reduce for each row
        __syncthreads ();

        const bool use_result = thread_in_block < j && i + j < NNZ_PER_WG;

        if (use_result)
          dot += cache[i + j];
        __syncthreads ();

        if (use_result)
          cache[i] = dot;
      }

      if (thread_in_block == 0 && local_row < block_row_end)
        out[local_row] = dot;
    }
    else
    {
      /// Reduce all non zeroes of row by single thread
      unsigned int local_row = block_row_begin + i;
      while (local_row < block_row_end)
      {
        data_type dot = 0.0;

        for (unsigned int j = row_ind_b[local_row] - block_data_begin;
             j < row_ind_b[local_row + 1] - block_data_begin;
             j++)
        {
          dot += cache[j];
        }

        out[local_row] = dot;
        local_row += NNZ_PER_WG;
      }
    }
  }
  else
  {
    const unsigned int row = block_row_begin;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane = threadIdx.x % 32;

    data_type dot = 0;

    if (nnz <= 64 || NNZ_PER_WG <= 32)
    {
      /// CSR-Vector case
      if (row < n_rows)
      {
        const unsigned int row_start = row_ind_b[row];
        const unsigned int row_end = row_ind_b[row + 1];

        for (unsigned int element = row_start + lane; element < row_end; element += 32)
          dot += data_b[element] * x[indices_b[element]];
      }

      dot = warp_reduce (dot);

      if (lane == 0 && warp_id == 0 && row < n_rows)
      {
        out[row] = dot;
      }
    }
    else
    {
      /// CSR-VectorL case
      if (row < n_rows)
      {
        const unsigned int row_start = row_ind_b[row];
        const unsigned int row_end = row_ind_b[row + 1];

        for (unsigned int element = row_start + threadIdx.x; element < row_end; element += blockDim.x)
          dot += data_b[element] * x[indices_b[element]];
      }

      dot = warp_reduce (dot);

      if (lane == 0)
        cache[warp_id] = dot;
      __syncthreads ();

      if (warp_id == 0)
      {
        dot = 0.0;

        for (unsigned int element = lane; element < blockDim.x / 32; element += 32)
          dot += cache[element];

        dot = warp_reduce (dot);

        if (lane == 0 && row < n_rows)
        {
          out[row] = dot;
        }
      }
    }
  }
}

/**
 * Optimized for large numbers of rows but small enough numbers of columns
 * that each thread can process their rows in parallel.
 */
template <typename value_idx, typename value_t, int tpb, int buffer_size,
          int max_chunk_size, int rows_per_block>
__global__ void classic_csr_semiring_spmv_kernel(
    value_idx *indptrA,
    value_idx *indicesA,
    value_t *dataA,
    value_idx *indptrB,
    value_idx *indicesB,
    value_t *dataB,
    value_idx m, value_idx n, value_t *out,
    int n_blocks_per_row
  ) {
  value_idx out_row = blockIdx.x / n_blocks_per_row;
  value_idx out_col_start = blockIdx.x % n_blocks_per_row;
  value_idx tid = threadIdx.x;

  if (out_row > m || out_col_start > n_blocks_per_row) return;

  __shared__ value_idx shared_cols[buffer_size];
  __shared__ value_t shared_vals[buffer_size];

  __shared__ value_idx offsets_a[2];
  __shared__ value_idx offsets_b[2];

  bool verbose = tid <= 3 && out_row < 3;

  BlockSemiring<value_idx, value_t, tpb, buffer_size, rows_per_block> semiring(
    tid, m, n, shared_cols, shared_vals, indicesB, dataB,
    offsets_a, offsets_b, verbose);

  semiring.load_a(out_row, indptrA, indicesA, dataA);
  semiring.load_b(out_col_start, indptrB, indicesB, dataB);

  while (!semiring.isdone()) semiring.step();
  semiring.write(out);
}


/**
 * This implementation follows the load-balanced implementation
 * from
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
template<typename value_idx, typename value_t, int tpb, int buffer_size, int chunk_size>
__global__ void balanced_coo_semiring_kernel(value_idx *indptrA, value_idx *indicesA,
                                             value_t *dataA, value_idx *rowsB,
                                             value_idx *indicesB, value_t *dataB, value_idx m,
                                             value_idx n, value_idx dim, value_t *out,
                                             int n_blocks_per_row) {

  value_idx cur_row_a = blockIdx.x / n_blocks_per_row;
  value_idx cur_chunk_offset = blockIdx.x % n_blocks_per_row;
  value_idx tid = threadIdx.x;

  if (cur_row_a > m || cur_chunk_offset > n_blocks_per_row) return;

  __shared__ value_t A[buffer_size];

  value_idx start_offset_a = indptrA[cur_row_a];
  value_idx stop_offset_a = indptrA[cur_row_a +1];

  // Create dense vector A and populate with 0s
  for(int i = threadIdx.x; i < dim; i += blockDim.x)
    A[i] = 0;

  // Convert current row vector in A to dense
  for(int i = threadIdx.x; i < (start_offset_a - stop_offset_a)+1; i += blockDim.x) {
    value_idx ind_a = indicesA[start_offset_a+i];
    value_t val_a = dataA[start_offset_a+i];
    A[ind_a] = val_a;
  }

  value_idx ind = cur_chunk_offset / threadIdx.x;
  value_idx cur_row_b = rowsB[ind];

  value_t c = A[ind] * dataB[indicesB[ind]];

  for(int i = 1; i < chunk_size; i+=blockDim.x) {
    value_idx ind_next = ind + blockDim.x;
    value_idx next_row_b = rowsB[ind_next];
    if(next_row_b != cur_row_b) {
      unsigned mask = __ballot_sync(0xffffffff, cur_row_b);
      c = __reduce_add_sync(mask, c);
      atomicAdd(out + (cur_row_a * n + cur_row_b), c);
      c = 0;
    }

    ind = ind_next;
    c += A[ind] * dataB[indicesB[ind]];
    cur_row_b= next_row_b;
  }
}


/**
 * A version of the block semiring that chunks offsets and rows
 * over threads in the warps to keep them busy.
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @tparam buffer_size
 * @tparam rows_per_block
 */
template <typename value_idx, typename value_t, int tpb, int buffer_size,
          int rows_per_block>
struct ChunkedBlockSemiring {
  __device__ inline ChunkedBlockSemiring(
    int tid_, value_idx m_, value_idx n_, value_idx *start_rows_,
    value_idx *offsets_, value_idx *shared_cols_, value_t *shared_vals_,
    value_idx *chunk_cols_, value_t *chunk_vals_, value_t *sums_,
    value_idx *row_counts_, value_idx *cum_row_counts_, bool verbose_)
    : tid(tid_),
      m(m_),
      n(n_),
      start_rows(start_rows_),
      offsets(offsets_),
      shared_cols(shared_cols_),
      shared_vals(shared_vals_),
      chunk_cols(chunk_cols_),
      chunk_vals(chunk_vals_),
      sums(sums_),
      done(false),
      shared_idx(0),
      cur_sum(0),
      row_counts(row_counts_),
      cum_row_counts(cum_row_counts_),
      verbose(verbose_) {}

  __device__ inline void load_a(value_idx row, value_idx *indptrA,
                                value_idx *indicesA, value_t *dataA) {
    start_offset_a = indptrA[row];
    stop_offset_a = indptrA[row + 1];

    // Coalesce reads of row from matrix A into shared memory
    for (int i = tid; i < stop_offset_a - start_offset_a; i += blockDim.x) {
      shared_cols[i] = indicesA[start_offset_a + i];
      shared_vals[i] = dataA[start_offset_a + i];
    }

    __syncthreads();

    shared_size = stop_offset_a - start_offset_a;

    if (verbose && tid == 0) {
      printf("row_a=%d, shared_cols=[", row_a);
      for (int i = 0; i < (stop_offset_a - start_offset_a); i++) {
        printf("%d, ", shared_cols[i]);
      }
      printf("]\n");
    }

    done = false;

    row_a = row;
  }

  __device__ inline void load_b(value_idx start_row, value_idx *indptrB,
                                value_idx *indicesB, value_t *dataB) {
    if (verbose)
      printf("tid=%d, start_col=%d, load_b called\n", tid, start_row);

    start_row_b = start_row * rows_per_block;
    stop_row_b =
      min(start_row_b + rows_per_block, start_row_b + (n - start_row_b) - 1);

    // initialize start_rows
    for (int i = tid; i < (stop_row_b - start_row_b) + 2; i += blockDim.x) {
      start_rows[i] = MAX_INT;
      value_idx diff = indptrB[start_row_b + i + 1] - indptrB[start_row_b + i];
      cum_row_counts[i] = max(1, diff);
      row_counts[i] = diff;
    }
    __syncthreads();

    // Row counts need to be cumulative. Last entry will total number of
    // rows that need to be processed.
    if (tid == 0) {
      for (int i = 1; i < (stop_row_b - start_row_b) + 2; i += 1)
        cum_row_counts[i] += cum_row_counts[i - 1];
    }
    __syncthreads();

    if (verbose && tid == 0) {
      printf("row_a=%d, row_counts=[", row_a);
      for (int i = 0; i < (stop_row_b - start_row_b) + 2; i++) {
        printf("%d, ", cum_row_counts[i]);
      }
      printf("]\n");
    }

    // divide the work evenly across threads in the warp
    value_idx n_offsets = cum_row_counts[(stop_row_b - start_row_b) + 2];

    if (verbose) printf("n_offsets=%d\n", n_offsets);

    // TOOO: Don't use integer division
    working_chunk_size = n_offsets / blockDim.x;

    // TODO: Don't use modulo
    working_chunk_size += n_offsets % blockDim.x <= blockDim.x ? 1 : 0;

    if (verbose)
      printf("tid=%d, start_col=%d, start_row_b=%d, stop_row_b=%d\n", tid,
             start_row, start_row_b, stop_row_b);

    // load start & end offsets to compute chunk size
    start_offset_b = indptrB[start_row_b];
    stop_offset_b = indptrB[stop_row_b + 1];

    if (verbose)
      printf(
        "tid=%d, start_offset_b=%d, stop_offset_b=%d, working_chunk_size=%d\n",
        tid, start_offset_b, stop_offset_b, working_chunk_size);

    if (verbose) printf("Initialized start_rows to -1\n");

    // get starting offsets of each row being processed by the current block
    for (int i = tid; i < (stop_row_b - start_row_b) + 1; i += blockDim.x) {
      value_idx offset = indptrB[start_row_b + i];

      // make offsets start at 0
      value_idx adj_offset = offset - start_offset_b;

      if (verbose)
        printf("building offsets: tid=%d, row_a=%d, offset=%d, adj_offset=%d\n",
               tid, row_a, offset, adj_offset);

      //      offsets[i] = adj_offset;
      //
      atomicMin(start_rows + int(ceilf(float(start_row_b + cum_row_counts[i]) /
                                       float(working_chunk_size))),
                int(ceilf(float(start_row_b + cum_row_counts[i]) /
                          float(working_chunk_size))));
    }
    __syncthreads();

    // iterate start_rows and repeat any missing values
    // TODO: Faster way to perform repeat?
    if (tid == 0) {
      start_rows[0] = start_row_b;
      offsets[0] = 0;
      for (int i = 1; i < tpb; i++) {
        if (start_rows[i] == MAX_INT) start_rows[i] = start_rows[i - 1];

        value_idx prev_offset = offsets[i - 1];
        if (row_counts[i - 1] > 0) prev_offset += working_chunk_size;
        offsets[i] = prev_offset;
      }
    }

    if (verbose && tid == 0) {
      printf("row_a=%d, offsets=[", row_a);
      for (int i = 0; i < tpb; i++) {
        printf("%d, ", offsets[i]);
      }
      printf("]\n");
    }

    if (verbose) printf("Computed starting row offsets for each block.\n");

    __syncthreads();

    if (verbose && tid == 0) {
      printf(
        "Performed a 'repeat' to fill in rows that span blocks. row_a=%d, "
        "start_rows=[",
        row_a);
      for (int i = 0; i < tpb; i++) {
        printf("%d, ", start_rows[i]);
      }
      printf("]\n");
    }

    row_b = start_rows[tid];

    // coalesce reads of B rows into shared memory
    for (int i = tid; i < (stop_offset_b - start_offset_b) + 1;
         i += blockDim.x) {
      chunk_cols[i] = indicesB[start_offset_b + i];
      chunk_vals[i] = dataB[start_offset_b + i];
    }

    __syncthreads();

    if (verbose) printf("Read B rows into shared memory\n");

    // set starting and ending idx of local thread
    local_idx = offsets[tid];  //threadIdx.x * working_chunk_size;
    local_idx_stop = min(local_idx + working_chunk_size, stop_offset_b);

    n_entries = min(tpb, (stop_offset_b - start_offset_b) + 1);

    if (verbose)
      printf(
        "row_a=%d, tid=%d, local_idx=%d, local_idx_stop=%d, n_entries=%d\n",
        row_a, tid, local_idx, local_idx_stop, n_entries);

    /**
     * Need to account for rows of b that are either being continued from tid-1 or
     * need to stop mid-row because tid+1 is continuing
     *
     * Case 1: first row in thread is continuing from tid-1
     *  - look at the last column in the tid-1 and make sure rv > that col
     *
     * Case 2: last row in thread is continued in tid+1
     *  - look at the first column in tid+1 and make sure rv < that col
     */
    case1 = (tid > 0 && row_b == start_rows[tid - 1])
              ? chunk_cols[local_idx - 1]
              : -1;
    case2 = (tid < blockDim.x && row_b == start_rows[tid + 1])
              ? chunk_cols[local_idx_stop]
              : MAX_INT;

    if (verbose)
      printf(
        "tid=%d, row_a=%d, row_b=%d, Computed overlapping cases: case1: %d, "
        "case2: %d\n",
        tid, row_a, row_b, case1, case2);
  }

  __device__ inline void step() {
    bool local_idx_in_bounds =
      local_idx < local_idx_stop && row_counts[row_b] > 0;

    if (verbose)
      printf("About to load chunk_cols/chunk_vals. local_idx_in_bounds=%d\n",
             local_idx_in_bounds);

    value_idx l = local_idx_in_bounds ? chunk_cols[local_idx] : -1;
    value_t lv = local_idx_in_bounds ? chunk_vals[local_idx] : 0.0;

    bool shared_idx_in_bounds = shared_idx < shared_size;

    if (verbose)
      printf(
        "tid=%d, row_a=%d, row_b=%d, About to load shared_cols/shared_vals. "
        "shared_idx_in_bounds=%d\n",
        tid, row_a, row_b, shared_idx_in_bounds);

    value_idx r = shared_idx_in_bounds ? shared_cols[shared_idx] : -1;
    value_t rv = shared_idx_in_bounds ? shared_vals[shared_idx] : 0.0;

    if (verbose)
      printf(
        "Loaded chunk_cols/chunk_vals. row_a=%d, row_b=%d, tid=%d, l=%d, "
        "lv=%f, r=%d, rv=%f\n",
        row_a, row_b, tid, l, lv, r, rv);

    r = r > case1 && r < case2 ? r : -1;

    value_t left_side = 0.0;
    value_t right_side = 0.0;

    if (l <= r && l != -1 || (l != -1 && r == -1)) {
      local_idx++;
      left_side = lv;
    }

    if (r <= l && r != -1 || (l == -1 && r != -1)) {
      shared_idx++;
      right_side = rv;
    }

    // Apply semiring "sum" & "product" functions locally
    cur_sum += fabsf(left_side - right_side);

    if (verbose)
      printf(
        "Middle of step(). row_a=%d, row_b=%d, tid=%d, l=%d, r=%d, "
        "left_side=%f, right_side=%f, done=%d, cur_sum=%f\n",
        row_a, row_b, tid, l, r, left_side, right_side, done, cur_sum);

    // finished when all items in chunk have been
    // processed
    done = l == -1 && r == -1;

    // adjust state when a new row is encountered
    if (tid < n_entries && (local_idx > local_idx_stop || done)) {
      // apply "sum" function globally
      atomicAdd(sums + row_b, cur_sum);

      if (verbose)
        printf("Processing new row. tid=%d, row_a=%d, row_b=%d, new_row_b=%d\n",
               tid, row_a, row_b, row_b + 1);
      row_b++;
      cur_sum = 0.0;
    }

    __syncthreads();

    if (verbose)
      printf(
        "End of step(). row_a=%d, row_b=%d, tid=%d, l=%d, r=%d, left_side=%f, "
        "right_side=%f, done=%d, cur_sum=%f, offsets[row_b]=%d, local_idx=%d\n",
        row_a, row_b, tid, l, r, left_side, right_side, done, cur_sum,
        offsets[row_b], local_idx);
  }

  __device__ inline bool isdone() { return done; }

  __device__ inline void write(value_t *out) {
    for (int i = tid; i < (stop_row_b - start_row_b) + 1; i += blockDim.x) {
      out[row_a * n + i] = sums[i];
    }
  }

  __device__ inline void print() {
    printf(
      "BlockSemiring<local_idx=%d, local_idx_stop=%d, row_b=%d, cur_sum=%f, "
      "working_chunk_size=%d\n",
      local_idx, local_idx_stop, row_b, cur_sum, working_chunk_size);
  }

 private:
  int tid;

  bool done;

  value_idx working_chunk_size;

  int shared_size;

  value_idx case1;
  value_idx case2;

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
  value_idx stop_offset_b;

  value_idx start_row_b;
  value_idx stop_row_b;

  // shared memory
  value_idx *offsets;
  value_idx *start_rows;
  value_idx *row_counts;
  value_idx *cum_row_counts;
  value_idx *shared_cols;
  value_t *shared_vals;
  value_idx *chunk_cols;
  value_t *chunk_vals;

  value_t *sums;

  bool verbose;
};

template <typename value_idx, typename value_t, int tpb, int buffer_size,
          int max_chunk_size, int rows_per_block>
__global__ void chunked_block_semiring(value_idx *indptrA, value_idx *indicesA,
                                       value_t *dataA, value_idx *indptrB,
                                       value_idx *indicesB, value_t *dataB,
                                       value_idx m, value_idx n, value_t *out,
                                       int n_blocks_per_row) {
  value_idx out_row = blockIdx.x / n_blocks_per_row;
  value_idx out_col_start = blockIdx.x % n_blocks_per_row;
  value_idx tid = threadIdx.x;

  if (out_row > m || out_col_start > n_blocks_per_row) return;

  __shared__ value_idx offsets[tpb];
  __shared__ value_idx start_rows[tpb];

  // TODO: This should really be computed once and passed in
  __shared__ value_idx row_counts[rows_per_block + 1];
  __shared__ value_idx cum_row_counts[rows_per_block + 1];

  __shared__ value_idx shared_cols[buffer_size];
  __shared__ value_t shared_vals[buffer_size];

  __shared__ value_idx chunk_cols[buffer_size];
  __shared__ value_t chunk_vals[buffer_size];

  __shared__ value_t sums[rows_per_block + 1];

  // TODO: Can chunk extremely large rows further by executing the semiring multiple times

  bool verbose = tid <= 10 && out_row < 1;

  ChunkedBlockSemiring<value_idx, value_t, tpb, buffer_size, rows_per_block>
    semiring(tid, m, n, offsets, start_rows, shared_cols, shared_vals,
             chunk_cols, chunk_vals, sums, row_counts, cum_row_counts, verbose);

  semiring.load_a(out_row, indptrA, indicesA, dataA);

  semiring.load_b(out_col_start, indptrB, indicesB, dataB);

  int iter = 0;
  while (!semiring.isdone()) {
    semiring.step();
    ++iter;
  }
  semiring.write(out);
}

//unsigned int
//fill_row_blocks (
//  bool fill,
//  unsigned int rows_count,
//  const unsigned int *row_ptr,
//  unsigned int *row_blocks
//)
//{
//  if (fill)
//    row_blocks[0] = 0;
//
//  int last_i = 0;
//  int current_wg = 1;
//  unsigned int nnz_sum = 0;
//  for (int i = 1; i <= rows_count; i++)
//  {
//    nnz_sum += row_ptr[i] - row_ptr[i - 1];
//
//    if (nnz_sum == NNZ_PER_WG)
//    {
//      last_i = i;
//
//      if (fill)
//        row_blocks[current_wg] = i;
//      current_wg++;
//      nnz_sum = 0;
//    }
//    else if (nnz_sum > NNZ_PER_WG)
//    {
//      if (i - last_i > 1)
//      {
//        if (fill)
//          row_blocks[current_wg] = i - 1;
//        current_wg++;
//        i--;
//      }
//      else
//      {
//        if (fill)
//          row_blocks[current_wg] = i;
//        current_wg++;
//      }
//
//      last_i = i;
//      nnz_sum = 0;
//    }
//    else if (i - last_i > NNZ_PER_WG)
//    {
//      last_i = i;
//      if (fill)
//        row_blocks[current_wg] = i;
//      current_wg++;
//      nnz_sum = 0;
//    }
//  }
//
//  if (fill)
//    row_blocks[current_wg] = rows_count;
//
//  return current_wg;
//}
//
//template <typename value_idx, typename value_t>
//void gpu_csr_adaptive_spmv (
//  const value_idx *indptr_a,
//  const value_idx *indices_a,
//  const value_t *data_a,
//  value_idx n_rows_a,
//  value_idx nnz_a,
//  const value_idx *indptr_b,
//  const value_idx *indices_b,
//  const value_t *data_b,
//  value_idx n_rows_b,
//  value_idx nnz_b,
//  value_t *out)
//{
//  // fill delimiters
//  const unsigned int blocks_count = fill_row_blocks (false, n_rows_a, indptr_a, nullptr);
//  std::unique_ptr<unsigned int[]> row_blocks(new unsigned int[blocks_count + 1]);
//  fill_row_blocks (true, n_rows_a, indptr_a, row_blocks.get ());
//
//  // TODO: Do this on device
//  unsigned int *d_row_blocks {};
//  cudaMalloc (&d_row_blocks, (blocks_count + 1) * sizeof (unsigned int));
//  cudaMemcpy (d_row_blocks, row_blocks.get (), sizeof (unsigned int) * (blocks_count + 1), cudaMemcpyHostToDevice);
//
//  dim3 block_size = dim3 (NNZ_PER_WG);
//  dim3 grid_size {};
//
//  grid_size.x = blocks_count;
//
//  // TODO: Need to schedule exhaustively for all rows of b.
//  adaptive_csr_spmv_semiring_kernel<<<grid_size, block_size>>> (
//    n_rows_a, indices_b, indptr_a, d_row_blocks, data_b, out);
//}


template <typename value_idx = int, typename value_t = float,
          int max_buffer_size = 1000,  //
          int threads_per_block = 1024,
          typename reduce_f = auto(value_t, value_t)->value_t,
          typename accum_f = auto(value_t, value_t)->value_t>
void distance_block_reduce(value_t *out_dists,
                           distances_config_t<value_idx, value_t> config_,
                           reduce_f reduce_func, accum_f accum_func) {
  int n_warps_per_row = raft::ceildiv(config_.b_nrows, threads_per_block);
  int n_blocks = config_.a_nrows * n_warps_per_row;

  CUML_LOG_DEBUG("n_blocks: %d", n_blocks);
  CUML_LOG_DEBUG("n_warps_per_row: %d", n_warps_per_row);

  classic_csr_semiring_spmv_kernel<value_idx, value_t, threads_per_block,
                              max_buffer_size, 256, threads_per_block>
    <<<n_blocks, threads_per_block, 0, config_.stream>>>(
      config_.a_indptr, config_.a_indices, config_.a_data, config_.b_indptr,
      config_.b_indices, config_.b_data, config_.a_nrows, config_.b_nrows,
      out_dists, n_warps_per_row);
};

template <typename value_idx = int, typename value_t = float>
class l1_distances_t : public distances_t<value_t> {
 public:
  l1_distances_t(distances_config_t<value_idx, value_t> config)
    : config_(config) {}

  void compute(value_t *out_dists) {
    CUML_LOG_DEBUG("Running l1 dists");
    distance_block_reduce<value_idx, value_t>(
      out_dists, config_,
      [] __device__(value_t a, value_t b) { return fabsf(a - b); },
      [] __device__(value_t out, value_t b) { return out +  b; });

    CUDA_CHECK(cudaStreamSynchronize(config_.stream));

    //    std::cout << raft::arr2Str(out_dists, config_.a_nrows * config_.b_nrows,
    //                               "out_dists", config_.stream)
    //              << std::endl;
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
    distance_block_reduce<value_idx, value_t>(
      out_dists, config_,
      [] __device__(value_t a, value_t b) { return (a - b) * (a - b); },
      [] __device__(value_t a, value_t b) { return a + b; });
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
    distance_block_reduce<value_idx, value_t>(
      out_dists, config_,
      [] __device__(value_t a, value_t b) { return fabsf(a - b); },
      [] __device__(value_t a, value_t b) { return fmaxf(a, b); });
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
    distance_block_reduce<value_idx, value_t>(
      out_dists, config_,
      [] __device__(value_t a, value_t b) {
        return fabsf(a - b) / (fabsf(a) + fabsf(b));
      },
      [] __device__(value_t a, value_t b) { return a + b; });
  }

 private:
  distances_config_t<value_idx, value_t> config_;
};

}  // namespace Distance
}  // namespace Sparse
};  // namespace MLCommon
