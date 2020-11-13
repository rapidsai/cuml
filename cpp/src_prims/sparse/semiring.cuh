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
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>

#include <sparse/distance.cuh>

namespace MLCommon {
namespace Sparse {
namespace Distance {

const int MAX_INT = std::numeric_limits<int>::max();

template <typename value_idx, typename value_t, int tpb, int buffer_size,
          int rows_per_block>
struct BlockSemiring {
  __device__ inline BlockSemiring(int tid_, value_idx m_, value_idx n_,
                                  value_idx *start_rows_, value_idx *offsets_,
                                  value_idx *shared_cols_,
                                  value_t *shared_vals_, value_idx *chunk_cols_,
                                  value_t *chunk_vals_, value_t *sums_, value_idx *row_counts_,
                                  bool verbose_)
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

    shared_size = stop_offset_a - start_offset_a;
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

    if (verbose)
      printf("tid=%d, start_col=%d, start_row_b=%d, stop_row_b=%d\n", tid,
             start_row, start_row_b, stop_row_b);

    // load start & end offsets to compute chunk size
    start_offset_b = indptrB[start_row_b];
    stop_offset_b = indptrB[stop_row_b+1]-1;

    // divide the work evenly across threads in the warp
    value_idx n_offsets = stop_offset_b - start_offset_b + 1;

    // TOOO: Don't use integer division
    working_chunk_size = n_offsets / blockDim.x;

    // TODO: Don't use modulo
    working_chunk_size += n_offsets % blockDim.x < blockDim.x ? 1 : 0;

    if (verbose)
      printf(
        "tid=%d, start_offset_b=%d, stop_offset_b=%d, working_chunk_size=%d\n",
        tid, start_offset_b, stop_offset_b, working_chunk_size);

    // initialize start_rows to -1
    for (int i = tid; i < (stop_row_b - start_row_b) + 1; i += blockDim.x) {
      start_rows[i] = MAX_INT;
      row_counts[i] = indptrB[start_row_b+i+1] - indptrB[start_row_b+i];
    }
    __syncthreads();

    if (verbose) printf("Initialized start_rows to -1\n");

    // get starting offsets of each row being processed by the current block
    for (int i = tid; i < (stop_row_b - start_row_b) + 1; i += blockDim.x) {
      value_idx offset = indptrB[start_row_b + i];

      // make offsets start at 0
      value_idx adj_offset = offset - start_offset_b;

      if (verbose)
        printf("building offsets: tid=%d, row_a=%d, offset=%d, adj_offset=%d\n",
               tid, row_a, offset, adj_offset);

      offsets[i] = adj_offset;

      atomicMin(
        start_rows + int(ceilf(float(i) / float(working_chunk_size))),
        ceilf(float(i) / float(working_chunk_size)));
    }
    __syncthreads();

    if (verbose && tid == 0) {
      printf("row_a=%d, offsets=[", row_a);
      for (int i = 0; i < (stop_row_b - start_row_b) + 1; i++) {
        printf("%d, ", offsets[i]);
      }
      printf("]\n");
    }

    if (verbose) printf("Computed starting row offsets for each block.\n");

    // iterate start_rows and repeat any missing values
    // TODO: Faster way to perform repeat?
    if (tid == 0) {
      start_rows[0] = start_row_b;
      for (int i = 1; i < (stop_row_b - start_row_b) + 1; i++) {
        if (start_rows[i] == MAX_INT) start_rows[i] = start_rows[i - 1];
      }
    }
    __syncthreads();

    if (verbose && tid == 0) {
      printf(
        "Performed a 'repeat' to fill in rows that span blocks. row_a=%d, start_rows=[",
        row_a);
      for (int i = 0; i < (stop_row_b - start_row_b) + 1; i++) {
        printf("%d, ", start_rows[i]);
      }
      printf("]\n");
    }

    if (verbose && tid == 0) {
      printf(
        "row_a=%d, row_counts=[",
        row_a);
      for (int i = 0; i < (stop_row_b - start_row_b) + 1; i++) {
        printf("%d, ", row_counts[i]);
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
    local_idx = offsets[start_rows[tid]];//threadIdx.x * working_chunk_size;
    local_idx_stop = min(local_idx + working_chunk_size, stop_offset_b);

    if (verbose)
      printf("row_a=%d, tid=%d, local_idx=%d, local_idx_stop=%d\n", row_a, tid, local_idx,
             local_idx_stop);

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
              ? shared_cols[local_idx - 1]
              : -1;
    case2 = (tid < blockDim.x && row_b == start_rows[tid + 1])
              ? shared_cols[local_idx_stop]
              : MAX_INT;

    if (verbose)
      printf("Computed overlapping cases: case1: %d, case2: %d\n", case1,
             case2);
  }

  __device__ inline void step() {
    bool local_idx_in_bounds = local_idx < local_idx_stop && row_counts[row_b] > 0;

    if (verbose)
      printf("About to load chunk_cols/chunk_vals. local_idx_in_bounds=%d\n",
             local_idx_in_bounds);

    value_idx l = local_idx_in_bounds ? chunk_cols[local_idx] : -1;
    value_t lv = local_idx_in_bounds ? chunk_vals[local_idx] : 0.0;

    bool shared_idx_in_bounds = shared_idx < shared_size;

    value_idx r = shared_idx_in_bounds ? shared_cols[shared_idx] : -1;
    value_t rv = shared_idx_in_bounds ? shared_vals[shared_idx] : 0.0;

    if (verbose)
      printf("Loaded chunk_cols/chunk_vals. row_a=%d, row_b=%d, tid=%d, l=%d, lv=%f, r=%d, rv=%f\n",
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
        "Middle of step(). row_a=%d, row_b=%d, tid=%d, l=%d, r=%d, left_side=%f, right_side=%f, done=%d, cur_sum=%f\n",
        row_a, row_b, tid, l, r, left_side, right_side, done, cur_sum);

    // adjust state when a new row is encountered
    if (local_idx > offsets[row_b]) {
      // apply "sum" function globally
      sums[row_b] += cur_sum;

      if (verbose)
        printf("Processing new row. tid=%d, row_a=%d, row_b=%d, new_row_b=%d\n",
               tid, row_a, row_b, row_b + 1);
      row_b++;
      cur_sum = 0.0;
    }

    // finished when all items in chunk have been
    // processed
    done = l == -1 && r == -1;

    if (verbose)
      printf(
        "End of step(). row_a=%d, row_b=%d, tid=%d, l=%d, r=%d, left_side=%f, right_side=%f, done=%d, cur_sum=%f, offsets[row_b]=%d, local_idx=%d\n",
        row_a, row_b, tid, l, r, left_side, right_side, done, cur_sum,
        offsets[row_b], local_idx);

    __syncthreads();
  }

  __device__ inline bool isdone() { return done; }

  __device__ inline void write(value_t *out) {
    for (int i = tid; i < (stop_row_b - start_row_b) + 1; i += blockDim.x) {
      printf("Writing: row_a=%d, row_b=%d, tid=%d, cur_sum=%f, idx=%d\n", row_a,
             row_b, tid, cur_sum, row_a * n + i);

      // Pick up any straggling threads that didn't get to write to shared memory
      if (cur_sum != 0.0) {
        sums[i] += cur_sum;
        cur_sum = 0.0;
      }

      out[row_a * n + i] = sums[i];
    }
  }

  __device__ inline void print() {
    printf(
      "BlockSemiring<local_idx=%d, local_idx_stop=%d, row_b=%d, cur_sum=%f, working_chunk_size=%d\n",
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
  value_idx *shared_cols;
  value_t *shared_vals;
  value_idx *chunk_cols;
  value_t *chunk_vals;

  value_t *sums;

  bool verbose;
};

template <typename value_idx, typename value_t, int tpb, int buffer_size,
          int max_chunk_size, int rows_per_block>
__global__ void semiring_kernel_load_balanced_matvec_layout(
  value_idx *indptrA, value_idx *indicesA, value_t *dataA, value_idx *indptrB,
  value_idx *indicesB, value_t *dataB, value_idx m, value_idx n, value_t *out,
  int n_blocks_per_row) {
  value_idx out_row = blockIdx.x / n_blocks_per_row;
  value_idx out_col_start = blockIdx.x % n_blocks_per_row;
  value_idx tid = threadIdx.x;

  if (out_row > m || out_col_start > n_blocks_per_row) return;

  // num_warps = n_rows_a * (n_rows_b / rows_per_warp)

  __shared__ value_idx offsets[rows_per_block + 1];
  __shared__ value_idx start_rows[tpb];
  __shared__ value_idx row_counts[rows_per_block+1];

  __shared__ value_idx shared_cols[buffer_size];
  __shared__ value_t shared_vals[buffer_size];

  __shared__ value_idx chunk_cols[buffer_size];
  __shared__ value_t chunk_vals[buffer_size];

  __shared__ value_t sums[rows_per_block + 1];

  // TODO: Can chunk extremely large rows further by executing the semiring multiple times

  bool verbose = tid <= 4 && out_row < 1;

  if (verbose) printf("Building block semiring\n");

  BlockSemiring<value_idx, value_t, tpb, buffer_size, rows_per_block> semiring(
    tid, m, n, offsets, start_rows, shared_cols, shared_vals, chunk_cols,
    chunk_vals, sums, row_counts, verbose);

  if (verbose) printf("Calling load_a\n");

  semiring.load_a(out_row, indptrA, indicesA, dataA);

  if (verbose) printf("Calling load_b\n");

  semiring.load_b(out_col_start, indptrB, indicesB, dataB);

  int iter = 0;
  while (!semiring.isdone()) {
    if (verbose) printf("Iteration %d\n", iter);
    semiring.step();

    ++iter;
  }

  semiring.write(out);
}

template <typename value_idx = int, typename value_t = float,
          int max_buffer_size = 1000,
          int threads_per_block =
            32,  // TODO: These should be conditional based on the data
          typename reduce_f = auto(value_t, value_t)->value_t,
          typename accum_f = auto(value_t, value_t)->value_t>
void distance_block_reduce(value_t *out_dists,
                           distances_config_t<value_idx, value_t> config_,
                           reduce_f reduce_func, accum_f accum_func) {
  //  naive_semiring_kernel<value_idx, value_t, threads_per_block, buffer_size>
  //    <<<config_.a_nrows * config_.b_nrows, threads_per_block, 0,
  //       config_.stream>>>(
  //      out_dists, config_.a_indptr, config_.a_indices, config_.a_data,
  //      config_.a_nnz, config_.b_indptr, config_.b_indices, config_.b_data,
  //      config_.b_nnz, config_.a_nrows, config_.b_nrows, reduce_func, accum_func);

  // number of rows processed within each warp/block. This is split across the threads of the block

  // TODO: Might be able to load balance even further if one side is a COO
  constexpr int rows_per_block = 64;

  int n_warps_per_row = raft::ceildiv(config_.b_nrows, rows_per_block);
  int n_blocks = config_.a_nrows * n_warps_per_row;

  CUML_LOG_DEBUG("n_blocks: %d", n_blocks);
  CUML_LOG_DEBUG("n_warps_per_row: %d", n_warps_per_row);

  semiring_kernel_load_balanced_matvec_layout<
    value_idx, value_t, threads_per_block, max_buffer_size, 256, rows_per_block>
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
      [] __device__(value_t a, value_t b) { return a + b; });

    CUDA_CHECK(cudaStreamSynchronize(config_.stream));

    std::cout << raft::arr2Str(out_dists, config_.a_nrows * config_.b_nrows,
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

}  // END namespace distance
}  // END namespace sparse
}; // END namespace mlcommon
