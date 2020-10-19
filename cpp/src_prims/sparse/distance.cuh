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

#include <common/cudart_utils.h>
#include <sparse/utils.h>
#include <common/device_buffer.hpp>
#include <cuda_utils.cuh>
#include <cuml/common/cuml_allocator.hpp>
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

namespace MLCommon {
namespace Sparse {
namespace Distance {

template <typename value_idx, typename value_t>
struct distances_config_t {
  // left side
  value_idx a_nrows;
  value_idx a_ncols;
  value_idx a_nnz;
  value_idx *a_indptr;
  value_idx *a_indices;
  value_t *a_data;

  // right side
  value_idx b_nrows;
  value_idx b_ncols;
  value_idx b_nnz;
  value_idx *b_indptr;
  value_idx *b_indices;
  value_t *b_data;

  cusparseHandle_t handle;

  std::shared_ptr<deviceAllocator> allocator;
  cudaStream_t stream;
};

template <typename value_t>
class distances_t {
 public:
  virtual void compute(value_t *out) { CUML_LOG_DEBUG("INside base"); }
  virtual ~distances_t() = default;
};

/**
 * Simple inner product distance with sparse matrix multiply
 */
template <typename value_idx = int, typename value_t = float>
class ip_distances_t : public distances_t<value_t> {
 public:
  /**
   * Computes simple sparse inner product distances as sum(x_y * y_k)
   * @param[in] config specifies inputs, outputs, and sizes
   */
  explicit ip_distances_t(distances_config_t<value_idx, value_t> config)
    : config_(config),
      workspace(config.allocator, config.stream, 0),
      csc_indptr(config.allocator, config.stream, 0),
      csc_indices(config.allocator, config.stream, 0),
      csc_data(config.allocator, config.stream, 0),
      alpha(1.0) {
    init_mat_descriptor(matA);
    init_mat_descriptor(matB);
    init_mat_descriptor(matC);
    init_mat_descriptor(matD);

    CUSPARSE_CHECK(cusparseCreateCsrgemm2Info(&info));

    CUSPARSE_CHECK(
      cusparseSetPointerMode(config.handle, CUSPARSE_POINTER_MODE_HOST));
  }

  /**
   * Performs pairwise distance computation and computes output distances
   * @param out_distances dense output matrix (size a_nrows * b_nrows)
   */
  void compute(value_t *out_distances) {
    /**
	   * Compute pairwise distances and return dense matrix in column-major format
	   */

    CUML_LOG_DEBUG("Compute() inside inner-product d");
    device_buffer<value_idx> out_batch_indptr(config_.allocator, config_.stream,
                                              config_.a_nrows + 1);
    device_buffer<value_idx> out_batch_indices(config_.allocator,
                                               config_.stream, 0);
    device_buffer<value_t> out_batch_data(config_.allocator, config_.stream, 0);

    value_idx out_batch_nnz = get_nnz(out_batch_indptr.data());

    out_batch_indices.resize(out_batch_nnz, config_.stream);
    out_batch_data.resize(out_batch_nnz, config_.stream);

    compute_gemm(out_batch_indptr.data(), out_batch_indices.data(),
                 out_batch_data.data());

    /**
     * Convert output to dense
     * TODO: This is wasteful of memory and adds additional latency.
     * It would be nice if there was a gemm that could do
     * (sparse, sparse)->dense natively.
     */
    csr_to_dense(config_.handle, config_.a_nrows, config_.b_nrows,
                 out_batch_indptr.data(), out_batch_indices.data(),
                 out_batch_data.data(), config_.a_nrows, out_distances,
                 config_.stream, true);
  }

  value_idx *trans_indptr() { return csc_indptr.data(); }

  value_idx *trans_indices() { return csc_indices.data(); }

  value_t *trans_data() { return csc_data.data(); }

  ~ip_distances_t() {
    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(matA));
    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(matB));
    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(matC));
    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(matD));
  }

 private:
  void init_mat_descriptor(cusparseMatDescr_t &mat) {
    CUSPARSE_CHECK(cusparseCreateMatDescr(&mat));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(mat, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CHECK(cusparseSetMatType(mat, CUSPARSE_MATRIX_TYPE_GENERAL));
  }

  value_idx get_nnz(value_idx *csr_out_indptr) {
    value_idx m = config_.a_nrows, n = config_.b_nrows, k = config_.a_ncols;

    transpose_b();

    size_t workspace_size;

    CUSPARSE_CHECK(raft::sparse::cusparsecsrgemm2_buffersizeext<value_t>(
      config_.handle, m, n, k, &alpha, NULL, matA, config_.a_nnz,
      config_.a_indptr, config_.a_indices, matB, config_.b_nnz,
      csc_indptr.data(), csc_indices.data(), matD, 0, NULL, NULL, info,
      &workspace_size, config_.stream));

    workspace.resize(workspace_size, config_.stream);

    value_idx out_nnz = 0;

    CUSPARSE_CHECK(raft::sparse::cusparsecsrgemm2nnz(
      config_.handle, m, n, k, matA, config_.a_nnz, config_.a_indptr,
      config_.a_indices, matB, config_.b_nnz, csc_indptr.data(),
      csc_indices.data(), matD, 0, NULL, NULL, matC, csr_out_indptr, &out_nnz,
      info, workspace.data(), config_.stream));

    return out_nnz;
  }

  void compute_gemm(const value_idx *csr_out_indptr, value_idx *csr_out_indices,
                    value_t *csr_out_data) {
    value_idx m = config_.a_nrows, n = config_.b_nrows, k = config_.a_ncols;

    CUSPARSE_CHECK(raft::sparse::cusparsecsrgemm2<value_t>(
      config_.handle, m, n, k, &alpha, matA, config_.a_nnz, config_.a_data,
      config_.a_indptr, config_.a_indices, matB, config_.b_nnz, csc_data.data(),
      csc_indptr.data(), csc_indices.data(), NULL, matD, 0, NULL, NULL, NULL,
      matC, csr_out_data, csr_out_indptr, csr_out_indices, info,
      workspace.data(), config_.stream));
  }

  void transpose_b() {
    /**
     * Transpose index array into csc
     */
    CUML_LOG_DEBUG("Transposing index CSR. rows=%d, cols=%d, nnz=%d",
                   config_.b_nrows, config_.b_ncols, config_.b_nnz);

    csc_indptr.resize(config_.b_ncols + 1, config_.stream);
    csc_indices.resize(config_.b_nnz, config_.stream);
    csc_data.resize(config_.b_nnz, config_.stream);

    csr_transpose(config_.handle, config_.b_indptr, config_.b_indices,
                  config_.b_data, csc_indptr.data(), csc_indices.data(),
                  csc_data.data(), config_.b_nrows, config_.b_ncols,
                  config_.b_nnz, config_.allocator, config_.stream);
  }

  value_t alpha;
  csrgemm2Info_t info;
  cusparseMatDescr_t matA;
  cusparseMatDescr_t matB;
  cusparseMatDescr_t matC;
  cusparseMatDescr_t matD;
  device_buffer<char> workspace;
  device_buffer<value_idx> csc_indptr;
  device_buffer<value_idx> csc_indices;
  device_buffer<value_t> csc_data;
  distances_config_t<value_idx, value_t> config_;
};

template <typename value_idx, typename value_t>
__global__ void compute_sq_norm_kernel(value_t *out, const value_idx *coo_rows,
                                       const value_t *data, value_idx nnz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nnz) {
    atomicAdd(&out[coo_rows[i]], data[i] * data[i]);
  }
}

template <typename value_idx, typename value_t>
__global__ void compute_euclidean_warp_kernel(value_t *C,
                                              const value_t *Q_sq_norms,
                                              const value_t *R_sq_norms,
                                              value_idx n_cols) {
  value_idx i = blockIdx.x;
  value_idx tid = threadIdx.x;

  __shared__ value_t q_norm;

  if (tid == 0) {
    q_norm = Q_sq_norms[i];
  }

  __syncthreads();

  for (int j = tid; j < n_cols; j += blockDim.x) {
    value_t r_norm = R_sq_norms[j];
    value_t dot = C[i * n_cols + j];

    value_t val = q_norm + r_norm - 2.0 * dot;
    if (fabsf(val) < 0.0001) val = 0.0;

    C[i * n_cols + j] = val;
  }
}

template <typename value_idx, typename value_t>
void compute_euclidean(value_t *C, const value_t *Q_sq_norms,
                       const value_t *R_sq_norms, value_idx n_rows,
                       value_idx n_cols, cudaStream_t stream) {
  int blockdim = block_dim(n_cols);

  compute_euclidean_warp_kernel<<<n_rows, blockdim, 0, stream>>>(
    C, Q_sq_norms, R_sq_norms, n_cols);
}

template <typename value_idx, typename value_t, int tpb = 256>
void compute_l2(value_t *out, const value_idx *Q_coo_rows,
                const value_t *Q_data, value_idx Q_nnz,
                const value_idx *R_coo_rows, const value_t *R_data,
                value_idx R_nnz, value_idx m, value_idx n,
                cusparseHandle_t handle, std::shared_ptr<deviceAllocator> alloc,
                cudaStream_t stream) {
  device_buffer<value_t> Q_sq_norms(alloc, stream, m);
  device_buffer<value_t> R_sq_norms(alloc, stream, n);
  CUDA_CHECK(
    cudaMemsetAsync(Q_sq_norms.data(), 0, Q_sq_norms.size() * sizeof(value_t)));
  CUDA_CHECK(
    cudaMemsetAsync(R_sq_norms.data(), 0, R_sq_norms.size() * sizeof(value_t)));

  compute_sq_norm_kernel<<<raft::ceildiv(Q_nnz, tpb), tpb, 0, stream>>>(
    Q_sq_norms.data(), Q_coo_rows, Q_data, Q_nnz);
  compute_sq_norm_kernel<<<raft::ceildiv(R_nnz, tpb), tpb, 0, stream>>>(
    R_sq_norms.data(), R_coo_rows, R_data, R_nnz);

  compute_euclidean(out, Q_sq_norms.data(), R_sq_norms.data(), m, n, stream);
}

/**
 * L2 distance using the expanded form: sum(x_k)^2 + sum(y_k)^2 - 2 * sum(x_k * y_k)
 * The expanded form is more efficient for sparse data.
 */
template <typename value_idx = int, typename value_t = float>
class l2_distances_t : public distances_t<value_t> {
 public:
  explicit l2_distances_t(distances_config_t<value_idx, value_t> config)
    : config_(config),
      workspace(config.allocator, config.stream, 0),
      ip_dists(config) {}

  void compute(value_t *out_dists) {
    CUML_LOG_DEBUG("Computing inner products");
    ip_dists.compute(out_dists);

    value_idx *b_indices = ip_dists.trans_indices();
    value_t *b_data = ip_dists.trans_data();

    CUML_LOG_DEBUG("Computing COO row index array");
    device_buffer<value_idx> search_coo_rows(config_.allocator, config_.stream,
                                             config_.a_nnz);
    csr_to_coo(config_.a_indptr, config_.a_nrows, search_coo_rows.data(),
               config_.a_nnz, config_.stream);

    CUML_LOG_DEBUG("Done.");

    CUML_LOG_DEBUG("Computing L2");
    compute_l2(out_dists, search_coo_rows.data(), config_.a_data, config_.a_nnz,
               b_indices, b_data, config_.b_nnz, config_.a_nrows,
               config_.b_nrows, config_.handle, config_.allocator,
               config_.stream);
    CUML_LOG_DEBUG("Done.");
  }

  ~l2_distances_t() = default;

 private:
  distances_config_t<value_idx, value_t> config_;
  device_buffer<char> workspace;
  ip_distances_t<value_idx, value_t> ip_dists;
};

template <typename value_t>
__device__ void swap(value_t *shared_mem, int idx1, int idx2) {
  if(idx1 != idx2) {
    value_t temp = shared_mem[idx1];
    shared_mem[idx1] = shared_mem[idx2];
    shared_mem[idx2] = temp;
  }
}

template <typename value_idx, typename value_t>
__device__ void bitonic_sort_by_key(value_t *keys, value_idx *vals,
                                    int n_elements, int tid, int row, int col, bool verbose) {
  for (unsigned int k = 2; k <= n_elements; k <<= 1) {
    for (unsigned int j = k >> 1; j > 0; j >>= 1) {
      unsigned int x = tid ^ j;
      if (x > tid) {
        if ((tid & k) == 0) {
          if (vals[tid] > vals[x]) {
            swap(vals, tid, x);
            swap(keys, tid, x);
          }
        } else {
          if (vals[tid] < vals[x]) {
            swap(vals, tid, x);
            swap(keys, tid, x);
          }
        }
      }

      __syncthreads();
    }
  }
}

template<typename value_idx, typename value_t>
__device__ void bitonic_sort_by_key_2(value_t *keys, value_idx *vals,
                                      int n_elements, int tid, int row, int col, bool verbose) {
  for(int size = 2; size <= n_elements; size <<=1) {
    for(int stride = size >> 1; stride > 0; stride >>= 1) {
      if(tid < n_elements) {

        if(verbose)
          printf("tid=%d, size=%d, stride=%d, l=%d, r=%d\n", tid, size, stride, tid, tid+stride);

        if ((tid & size) == 0) {
          if (vals[tid] > vals[tid+stride]) {
            swap(vals, tid, tid+stride);
            swap(keys, tid, tid+stride);
          }
        } else {
          if (vals[tid] < vals[tid+stride]) {
            swap(vals, tid, tid+stride);
            swap(keys, tid, tid+stride);
          }
        }
      }

      __syncthreads();
    }
  }
}


template<typename value_idx, typename value_t>
__device__ void bitonic_sort_by_key_3(value_t *keys, value_idx *vals,
                                      int n_elements, int tid, int row, int col, bool verbose) {
  for(int split = 2; split < n_elements; split <<=1) {
    for(int away = split; away >= 1; away >>= 1) {
      for(int thread_id = tid; thread_id < n_elements; thread_id += blockDim.x) {
        int i_mask = ((1<<30) - 1) - away;
        int j_mask = away;
        int is_inc_mask = split << 1;

        int i = tid & i_mask;
        int j = tid | j_mask;

        int is_inc = (tid & is_inc_mask) == 0;

        bool need_swap = false;

        need_swap |= (is_inc && (vals[i] > vals[j]));
        need_swap |= (!is_inc && (vals[i] < vals[j]));

        if(need_swap && i == tid) {


//          if(verbose)
//            printf("row=%d, col=%d, tid=%d, split=%d, away=%d, i=%d, j=%d, vals[i]=%d, vals[j]=%d\n", row, col, tid, split, away, i, j, vals[i], vals[j]);

          swap(vals, i, j);
          swap(keys, i, j);
        }
      }
    }

    __syncthreads();
  }
}

template<typename value_idx, typename value_t>
__device__ void even_odd_merge_sort(value_t *keys, value_idx *vals,
                                      int n_elements, int tid, int row,
                                    int col, bool verbose) {
  for(int i = 0; i < n_elements; i++) {

    bool is_even = i % 2 == 0; // TODO: use fastmath here
    for(int j = tid; j < n_elements >> 1; j += blockDim.x) {
      int idx = is_even ? j << 1 : (j<<1)+1;
      if(vals[idx+1] < vals[idx] && idx+1 < n_elements) {
        swap(keys, idx, idx+1);
        swap(vals, idx, idx+1);
      }
    }

    __syncthreads();
  }
}

template<typename value_idx, typename value_t, int tpb, int items_per_thread>
__device__ void cub_block_radix_sort(
  value_idx (&cols)[items_per_thread], value_t (&vals)[items_per_thread],
                                      value_idx *cols_inA, value_idx *cols_inB,
                                     value_t *vals_inA, value_t *vals_inB,
                                     int start_offsetA, int start_offsetB,
                                     int stop_offsetA, int stop_offsetB,
                                     int nnz_a, int nnz_b,
                                     value_idx *cols_shared, value_t *vals_shared) {

  typedef cub::BlockRadixSort<value_idx, tpb, items_per_thread, value_t> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  BlockRadixSort(temp_storage).Sort(cols, vals);

  __syncthreads();
//
//  cub::BlockStore<value_idx, tpb, items_per_thread>().Store(cols_shared, cols);
//  cub::BlockStore<value_t, tpb, items_per_thread>().Store(vals_shared, vals);
}


const int MAX_INT = std::numeric_limits<int>::max();
const float MAX_FLOAT = std::numeric_limits<float>::max();


template<typename value_idx, typename value_t>
__device__ void thread_step(int idx, int tid, int local_idx, int shared_idx,
                            value_idx *local_cols,
                            value_idx *shared_cols, value_t *local_vals,
                            value_t *shared_vals, int shared_size,
                                 value_t sum,
                            int chunk_size,
                            value_idx *offsets,
                            value_t *out) {

  value_idx l = local_idx < chunk_size ? 1 : 0;//local_cols[local_idx] : 0;
  value_idx r = shared_idx < shared_size ? 1 : 0;//shared_cols[shared_idx] : 0;

  value_t lv = local_idx < chunk_size ? 1 : 0;//local_vals[local_idx] : 0;
  value_t rv = shared_idx < shared_size ? 1 : 0;//shared_vals[shared_idx] : 0;

  value_t left_side = 0;
  value_t right_side = 0;

  if(l < r) {
    local_idx++;
    left_side = lv;
  }

  if(r > l) {
    shared_idx++;
    right_side = rv;
  }

//  int col_offset_idx = tid * chunk_size + local_idx;
//  int col_offset_idx_lookup = 0;
//
//  // should be 0 if we've hit the end of the offset
//  bool new_row = col_offset_idx - offsets[col_offset_idx_lookup+1] == 0;
//
//  value_t val = fabsf(l - r);
//
//  if(new_row)
//    atomicAdd(out, sum + val);
//
//  sum *= new_row;
//  sum += val;
}
template <typename value_idx, typename value_t, int tpb, int buffer_size,
  int max_chunk_size, int rows_per_warp>
struct Semiring {

  __device__ inline Semiring(int tid_, value_idx m_, value_idx n_
  ): tid(tid_), m(m_), n(n_){}

  __device__ inline void load_a(value_idx row, value_idx *indptrA, value_idx *indicesA, value_t *dataA) {
    start_offset_a = indptrA[row];
    stop_offset_a = indptrA[row + 1];

    for(int i = tid; i < stop_offset_a - start_offset_a; i += blockDim.x) {
      shared_cols[i] = indicesA[start_offset_a+i];
      shared_vals[i] = dataA[start_offset_a+i];
    }
  }

  __device__ inline void load_b(value_idx start_row, value_idx *indptrB, value_idx *indicesB, value_t *dataB) {
    start_row_b = start_row * rows_per_warp;
    stop_row_b = min(start_row_b + rows_per_warp,
                               start_row_b + (n-start_row_b)-1);

    for(int i = tid; i < (stop_row_b - start_row_b)+1; i+= blockDim.x) {
      offsets[i] = indptrB[start_row_b + i];
    }
    __syncthreads();

    start_offset_b = offsets[0];
    stop_offset_b = offsets[stop_row_b-start_row_b];

    for(int i = tid; i < stop_offset_b - start_offset_b; i += blockDim.x) {
      chunk_cols[i] = indicesB[start_offset_b+i];
      chunk_vals[i] = dataB[start_offset_b+i];
    }
  }

  __device__ inline bool step() {

    value_idx working_chunk_size = (stop_offset_b - start_offset_b) / blockDim.x;

    value_idx local_idx = 0;
    value_idx shared_idx = 0;
    int idx = working_chunk_size * tid;

    value_t sum = 0;

    int shared_size = stop_offset_a - start_offset_a;

    for(int i = 0; i < max(working_chunk_size, shared_size); i++) {
      value_idx l = local_idx < working_chunk_size ? chunk_cols[local_idx] : 0;
      value_idx r = shared_idx < shared_size ? shared_cols[shared_idx] : 0;

      value_t lv = local_idx < working_chunk_size ? chunk_vals[local_idx] : 0;
      value_t rv = shared_idx < shared_size ? shared_vals[shared_idx] : 0;

      value_t left_side = 0;
      value_t right_side = 0;

      if(l < r) {
        local_idx++;
        left_side = lv;
      }

      if(r > l) {
        shared_idx++;
        right_side = rv;
      }



      __syncthreads();
    }

  }

  __device__ inline void write(value_t *out) {}

 private:

  int tid;

  value_idx m;
  value_idx n;
  value_idx start_offset_a;
  value_idx stop_offset_a;

  value_idx start_offset_b;
  value_idx stop_offset_b;

  value_idx start_row_b;
  value_idx stop_row_b;

};



template <typename value_idx, typename value_t, int tpb, int buffer_size,
          int max_chunk_size, int rows_per_warp>
__global__ void semiring_kernel_thread_per_col(
  value_idx *indptrA, value_idx *indicesA, value_t *dataA,
  value_idx *indptrB, value_idx *indicesB, value_t *dataB,
  value_idx m, value_idx n, value_t *out,
  int n_warps_per_row

) {
  value_idx out_row = blockIdx.x / n_warps_per_row;
  value_idx out_col = blockIdx.x % n_warps_per_row;
  value_idx tid = threadIdx.x;

//
//  if(tid == 0 && out_row < 5) {
//    printf("out_row=%d, out_col=%d, tid=%d\n", out_row, out_col, tid);
//  }
//
  if(out_row > m || out_col > n_warps_per_row) return;
  // num_warps = n_rows_a * (n_rows_b / rows_per_warp)

  __shared__ value_idx offsets[rows_per_warp+1];
  __shared__ value_idx shared_cols[buffer_size];
  __shared__ value_t shared_vals[buffer_size];
  __shared__ value_idx chunk_cols[buffer_size];
  __shared__ value_t chunk_vals[buffer_size];


  Semiring<value_idx, value_t, tpb, buffer_size, max_chunk_size, rows_per_warp> semiring(tid, m, n, offsets, shared_cols, shared_vals, chunk_cols, chunk_vals);

  semiring.load_a(out_row, indptrA, indicesA, dataA);
  semiring.load_b(out_col, indptrB, indicesB, dataB);

  semiring.step();



//  if(stop_row_b - start_row_b > rows_per_warp || stop_row_b >= n) {
//    printf("WE HAVE A PROBLEM!: %d", stop_row_b - start_row_b);
//  }

  //
//  if(tid == 0 && out_row == 0) {
//    printf("start_row_a=%d, start_row_b=%d, stop_row_b=%d\n", out_row, start_row_b, stop_row_b);
//  }

//

//    if(tid == 0 && out_row == 0) {
//    printf("start_offset_b=%d, stop_offset_b=%d\n", start_offset_b, stop_offset_b);
//  }

//
////
  __syncthreads();
//
}



/*
 * Algorithm #2:
 *   each thread maintains a queue buffer for some row/col pair which they load
 *
 */

/**
 * Assuming the nnz cols of A and B (About 5k each) can both fit into shared memory for now
 * just to test the algorithm.
 *
 * This has several advantages over using the cusparse csrgemm:
 * - The output is dense by default
 * - The input B doesn't need to be transposed
 * - This enables several minkowski-class metrics, as well as a matrix multiply from this
 * - The k-select could be done right in this kernel so that intermediate results never need to be written to global memory.
 */
template <typename value_idx, typename value_t, int tpb, int buffer_size,
          typename accumulator_f, typename reduction_f>
__global__ void semiring_kernel(
  value_t *out, value_idx *indptrA, value_idx *indicesA, value_t *dataA,
  size_t nnzA, value_idx *indptrB, value_idx *indicesB, value_t *dataB,
  size_t nnzB, size_t m, size_t n, reduction_f func_reducer,
  accumulator_f func_accumulator) {


  constexpr int items_per_thread = buffer_size / tpb;

  value_idx out_row = blockIdx.x / n;
  value_idx out_col = blockIdx.x % n;
  value_idx tid = threadIdx.x;

  if (out_row > m || out_col > n) return;

  __shared__ value_idx cols_buffer[buffer_size];
  __shared__ value_t vals_buffer[buffer_size];

  value_idx cols[items_per_thread];
  value_t vals[items_per_thread];

  value_idx start_offset_a;
  value_idx stop_offset_a;
  value_idx start_offset_b;
  value_idx stop_offset_b;

  value_idx col_nnz_a;
  value_idx col_nnz_b;

  //  __shared__ value_idx n_loaded_a;
  //  __shared__ value_idx n_loaded_b;

  start_offset_a = indptrA[out_row];
  stop_offset_a = indptrA[out_row + 1];
  start_offset_b = indptrB[out_col];
  stop_offset_b = indptrB[out_col + 1];

  col_nnz_a = stop_offset_a - start_offset_a;
  col_nnz_b = stop_offset_b - start_offset_b;

#pragma unroll
for(int i = 0; i < items_per_thread; i++) {
  cols[i] = 50000;
}

//  // Load A into buffer
#pragma unroll
  for(int i = 0; i < items_per_thread; i++) {
    int index = (tid + (i * tpb));
    if(index < col_nnz_a) {
      cols[i] = indicesA[start_offset_a + index];
      vals[i] = dataA[start_offset_a + index];
    }
  }

  // Load B into buffer
#pragma unroll
  for(int i = 0; i < items_per_thread; i++) {
    int index = (tid + (i * tpb));
    if(index < col_nnz_b) {
      cols[i+col_nnz_a] = indicesB[start_offset_b + index];
      vals[i+col_nnz_a] = dataB[start_offset_b + index];
    }
  }
//
//  __syncthreads();
//
//  if (tid==1 && out_row == 0 && out_col == 1) {
//    printf("unsorted: tid=%d, row=%d, col=%d, [\n", tid, out_row, out_col);
//    for (int i = 0; i < items_per_thread; i++) {
//      printf("%d, ", cols[i]);
//    }
//    printf("]\n");
//  }


  cub_block_radix_sort<value_idx, value_t, tpb, items_per_thread>(
    cols, vals, indicesA, indicesB, dataA, dataB, start_offset_a, start_offset_b,
    stop_offset_a, stop_offset_b, col_nnz_a, col_nnz_b, cols_buffer, vals_buffer
  );

  __syncthreads();

//  if (tid==1 && out_row == 0 && out_col == 1) {
//    printf("unsorted: tid=%d, row=%d, col=%d, [\n", tid, out_row, out_col);
//    for (int i = 0; i < items_per_thread; i++) {
//      printf("%d, ", cols[i]);
//    }
//    printf("]\n");
//  }
//
//
//  if (tid == 0 && out_row == 0 && out_col == 1) {
//    printf("sorted: row=%d, col=%d, [\n", out_row, out_col);
//    for (int i = 0; i < col_nnz_b + col_nnz_a; i++) {
//      printf("%d, ", cols_buffer[i]);
//    }
//    printf("]\n");
//  }
//  __syncthreads();

  // Sort
//  bitonic_sort_by_key_3(vals_buffer, cols_buffer, col_nnz_a + col_nnz_b, tid,
//                      out_row, out_col, out_row == 0 && out_col == 1 && tid == 0);
//
//  __syncthreads();
//
//  if (tid == 0 && out_row == 0 && out_col == 1) {
//    printf("sorted bufsize=%d [", col_nnz_a + col_nnz_b);
//    for (int i = 0; i < col_nnz_b + col_nnz_a; i++) {
//      printf("%d, ", cols_buffer[i]);
//    }
//    printf("]\n");
//  }

  // Reduce duplicates
  for (int i = tid; i < (col_nnz_a + col_nnz_b) - 2; i += blockDim.x) {

    value_idx one = cols_buffer[i];
    value_idx two = cols_buffer[i+1];
    value_idx three = cols_buffer[i+2];

    if (two == three) {
      vals_buffer[i+1] = fabsf(vals_buffer[i+1] - vals_buffer[i+2]);
      vals_buffer[i+2] = 0.0;
    } else if(one != two && two != three) {
      vals_buffer[i+1] = fabsf(vals_buffer[i+1] - 0.0);
    }
  }
  __syncthreads();

  if(tid == 0) {
    if(cols_buffer[0] == cols_buffer[1]) {
      vals_buffer[0] = fabsf(vals_buffer[0] - vals_buffer[1]);
      vals_buffer[1] = 0.0;
    } else if(cols_buffer[0] != cols_buffer[1]) {
      vals_buffer[0] = fabsf(vals_buffer[0] - 0.0);
    }
  }

  __syncthreads();

//  if (tid == 0 && out_row == 0 && out_col == 1) {
//    printf("vals_buffer after reduce dupes[");
//    for (int i = 0; i < col_nnz_b + col_nnz_a; i++) {
//      printf("%f, ", vals_buffer[i]);
//    }
//    printf("]\n");
//  }
//
//  if (tid == 0 && out_row == 0 && out_col == 1) {
//    printf("cols_buffer after reduce dupes [");
//    for (int i = 0; i < col_nnz_b + col_nnz_a; i++) {
//      printf("%d, ", cols_buffer[i]);
//    }
//    printf("]\n");
//  }



//  // Tree-reduce
//  for (unsigned int i = blockDim.x >> 1; i > 0; i >>= 1) {
//      vals_buffer[tid] = func_accumulator(vals_buffer[tid], vals_buffer[tid + i]);
//    }
//    __syncthreads();
//  }
//////
//  if (tid == 0 && out_row == 0 && out_col == 1) {
//    printf("tree reduction row=%d, col=%d[", out_row, out_col);
//    for (int i = 0; i < col_nnz_b + col_nnz_a; i++) {
//      printf("%f, ", vals_buffer[i]);
//    }
//    printf("]\n");
//  }

  if (tid == 0) {
    value_t sum = 0.0;
//    for(int i = 0; i < (col_nnz_b + col_nnz_a); i++) {
//      sum += vals_buffer[i];
//    }
    out[out_row * n + out_col] = sum;
  }

//  if(tid < (col_nnz_a + col_nnz_b))
//    atomicAdd(out + (our_row * n + out_col), vals_buffer[tid]);
}

template <typename value_idx = int, typename value_t = float,
          int max_buffer_size = 10000, int threads_per_block = 32,   // TODO: These should be conditional based on the data
          typename reduce_f = auto(value_t, value_t)->value_t,
          typename accum_f = auto(value_t, value_t)->value_t>
void distance_block_reduce(value_t *out_dists,
                           distances_config_t<value_idx, value_t> config_,
                           reduce_f reduce_func, accum_f accum_func) {

//  semiring_kernel<value_idx, value_t, threads_per_block, buffer_size>
//    <<<config_.a_nrows * config_.b_nrows, threads_per_block, 0,
//       config_.stream>>>(
//      out_dists, config_.a_indptr, config_.a_indices, config_.a_data,
//      config_.a_nnz, config_.b_indptr, config_.b_indices, config_.b_data,
//      config_.b_nnz, config_.a_nrows, config_.b_nrows, reduce_func, accum_func);

  int n_warps_per_row = ceildiv(config_.b_nrows, 128);
  semiring_kernel_thread_per_col<value_idx, value_t, threads_per_block, max_buffer_size, 256, 128>
    <<<config_.a_nrows * (config_.b_nrows/128), threads_per_block, 0,
       config_.stream>>>(
      config_.a_indptr, config_.a_indices, config_.a_data,
      config_.b_indptr, config_.b_indices, config_.b_data,
      config_.a_nrows, config_.b_nrows, out_dists, n_warps_per_row);
};

template <typename value_idx = int, typename value_t = float>
class l1_distances_t : public distances_t<value_t> {
 public:
  l1_distances_t(distances_config_t<value_idx, value_t> config)
    : config_(config) {}

  void compute(value_t *out_dists) {
    distance_block_reduce<value_idx, value_t>(
      out_dists, config_,
      [] __device__(value_t a, value_t b) { return fabsf(a - b); },
      [] __device__(value_t a, value_t b) { return a + b; });
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

/**
 * Compute pairwise distances between A and B, using the provided
 * input configuration and distance function.
 *
 * @tparam value_idx index type
 * @tparam value_t value type
 * @param[out] out dense output array (size A.nrows * B.nrows)
 * @param[in] input_config input argument configuration
 * @param[in] metric distance metric to use
 */
template class ip_distances_t<int, float>;
template class l2_distances_t<int, float>;
template class distances_config_t<int, float>;

template <typename value_idx = int, typename value_t = float>
void pairwiseDistance(value_t *out,
                      distances_config_t<value_idx, value_t> input_config,
                      ML::Distance::DistanceType metric) {
  CUML_LOG_DEBUG("Running sparse pairwise distances with metric=%d", metric);

  switch (metric) {
    case ML::Distance::DistanceType::EucExpandedL2:
      // EucExpandedL2
      l2_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case ML::Distance::DistanceType::InnerProduct:
      // InnerProduct
      ip_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case ML::Distance::DistanceType::EucUnexpandedL1:
      l1_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case ML::Distance::DistanceType::ChebyChev:
      chebychev_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case ML::Distance::DistanceType::Canberra:
      canberra_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    default:
      THROW("Unsupported metric: %d", metric);
  }
}

};  // END namespace Distance
};  // END namespace Sparse
};  // END namespace MLCommon
