/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "cuda_utils.h"

#include "label/classlabels.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>

namespace MLCommon {
namespace Sparse {

static const float MIN_FLOAT = std::numeric_limits<float>::min();

/** @brief A Container object for CSR format. There are two motivations
 * behind using a container for CSR arrays.
 *
 * The first motivation is that it simplifies code, rather than always having
 * to pass three arrays as function arguments.
 *
 * The second is more subtle, but much more important. The size
 * of the resulting COO from a sparse operation is often not known ahead of time,
 * since it depends on the contents of the underlying graph. The COO object can
 * allocate the underlying arrays lazily so that the object can be created by the
 * user and passed as an output argument in a sparse primitive. The sparse primitive
 * would have the responsibility for allocating and populating the output arrays,
 * while the original caller still maintains ownership of the underlying memory.
 *
 * @tparam T: the type of the value array.
 * @tparam Index_Type: The type of the index arrays
 *
 */

template <typename T, typename Index_Type = int>
class CSR {
 protected:
  device_buffer<Index_Type> row_ind_arr;
  device_buffer<Index_Type> row_ind_ptr_arr;
  device_buffer<T> vals_arr;

 public:
  Index_Type nnz;
  Index_Type n_rows;
  Index_Type n_cols;

  /**
    * @brief default constructor
    * @param alloc device allocator
    * @param stream cuda stream
    */
  CSR(std::shared_ptr<deviceAllocator> d_alloc, cudaStream_t stream)
    : row_ind_arr(d_alloc, stream, 0),
      row_ind_ptr_arr(d_alloc, stream, 0),
      vals_arr(d_alloc, stream, 0),
      nnz(0),
      n_rows(0),
      n_cols(0) {}

  /**
    * @brief construct a CSR object with pre-allocated device buffers
    * @param row_ind_: csr row index array
    * @param row_ind_ptr_: csr row index pointer array
    * @param vals: csr vals array
    * @param nnz: size of the rows/cols/vals arrays
    * @param n_rows: number of rows in the dense matrix
    * @param n_cols: number of cols in the dense matrix
    */
  CSR(device_buffer<Index_Type> &row_ind_,
      device_buffer<Index_Type> &row_ind_ptr_, device_buffer<T> &vals,
      Index_Type nnz, Index_Type n_rows = 0, Index_Type n_cols = 0)
    : row_ind_arr(row_ind_),
      row_ind_ptr_arr(row_ind_ptr_),
      vals_arr(vals),
      nnz(nnz),
      n_rows(n_rows),
      n_cols(n_cols) {}

  void init_arrays(cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(this->row_ind_arr.data(), 0,
                               this->n_rows + 1 * sizeof(Index_Type), stream));
    CUDA_CHECK(cudaMemsetAsync(this->row_ind_ptr_arr.data(), 0,
                               this->nnz * sizeof(Index_Type), stream));
    CUDA_CHECK(
      cudaMemsetAsync(this->vals_arr.data(), 0, this->nnz * sizeof(T), stream));
  }

  /**
     * @brief Allocate a CSR given its size
    * @param alloc: device allocator for temporary buffers
    * @param stream: CUDA stream to use
    * @param nnz: size of the rows/cols/vals arrays
    * @param n_rows: number of rows in the dense matrix
    * @param n_cols: number of cols in the dense matrix
    * @param init: initialize arrays with zeros
    */
  CSR(std::shared_ptr<deviceAllocator> d_alloc, cudaStream_t stream,
      Index_Type nnz, Index_Type n_rows = 0, Index_Type n_cols = 0,
      bool init = true)
    : row_ind_arr(d_alloc, stream, nnz),
      row_ind_ptr_arr(d_alloc, stream, nnz),
      vals_arr(d_alloc, stream, nnz),
      nnz(nnz),
      n_rows(n_rows),
      n_cols(n_cols) {
    if (init) init_arrays(stream);
  }

  ~CSR() {}

  /**
    * @brief Size should be > 0, with the number of rows
    * and cols in the dense matrix being > 0.
    */
  bool validate_size() const {
    if (this->nnz < 0 || n_rows < 0 || n_cols < 0) return false;
    return true;
  }

  /**
    * @brief If the underlying arrays have not been set,
    * return false. Otherwise true.
    */
  bool validate_mem() const {
    if (this->row_ind_arr.size() == 0 || this->row_ind_ptr_arr.size() == 0 ||
        this->vals_arr.size() == 0) {
      return false;
    }

    return true;
  }

  /**
   * @brief Returns the row index array
   */
  Index_Type *row_ind() { return this->row_ind_arr.data(); }

  /**
   * @brief Returns the row index pointer array
   */
  Index_Type *row_ind_ptr() { return this->row_ind_ptr_arr.data(); }

  /**
   * Returns the vals_arr array
   */
  T *vals() { return this->vals_arr.data(); }

  /**
    * @brief Send human-readable state information to output stream
    */
  friend std::ostream &operator<<(std::ostream &out, const CSR<T> &c) {
    if (c.validate_size() && c.validate_mem()) {
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreate(&stream));

      out << arr2Str(c.row_ind_arr.data(), c.n_rows + 1, "row_ind", stream)
          << std::endl;
      out << arr2Str(c.row_ind_ptr_arr.data(), c.nnz, "row_ind_ptr_arr", stream)
          << std::endl;
      out << arr2Str(c.vals_arr.data(), c.nnz, "vals_arr", stream) << std::endl;
      out << "nnz=" << c.nnz << std::endl;
      out << "n_rows=" << c.n_rows << std::endl;
      out << "n_cols=" << c.n_cols << std::endl;

      CUDA_CHECK(cudaStreamDestroy(stream));
    } else {
      out << "Cannot print COO object: Uninitialized or invalid." << std::endl;
    }

    return out;
  }

  /**
    * @brief Set the number of rows and cols
    * @param n_rows: number of rows in the dense matrix
    * @param n_cols: number of columns in the dense matrix
    */
  void setSize(int n_rows, int n_cols) {
    this->n_rows = n_rows;
    this->n_cols = n_cols;
  }

  /**
    * @brief Set the number of rows and cols for a square dense matrix
    * @param n: number of rows and cols
    */
  void setSize(int n) {
    this->n_rows = n;
    this->n_cols = n;
  }

  /**
    * @brief Allocate the underlying arrays
    * @param nnz: size of underlying row/col/val arrays
    * @param init: should values be initialized to 0?
    * @param stream: CUDA stream to use
    */
  void allocate(int nnz, bool init, cudaStream_t stream) {
    this->allocate(nnz, 0, init, stream);
  }

  /**
    * @brief Allocate the underlying arrays
    * @param nnz: size of the underlying row/col/val arrays
    * @param size: the number of rows/cols in a square dense matrix
    * @param init: should values be initialized to 0?
    * @param stream: CUDA stream to use
    */
  void allocate(int nnz, int size, bool init, cudaStream_t stream) {
    this->allocate(nnz, size, size, init, stream);
  }

  /**
    * @brief Allocate the underlying arrays
    * @param nnz: size of the underlying row/col/val arrays
    * @param n_rows: number of rows in the dense matrix
    * @param n_cols: number of columns in the dense matrix
    * @param init: should values be initialized to 0?
    * @param stream: stream to use for init
    */
  void allocate(int nnz, int n_rows, int n_cols, bool init,
                cudaStream_t stream) {
    this->n_rows = n_rows;
    this->n_cols = n_cols;
    this->nnz = nnz;

    this->row_ind_arr.resize(this->n_rows + 1, stream);
    this->row_ind_ptr_arr.resize(this->nnz, stream);
    this->vals_arr.resize(this->nnz, stream);

    if (init) init_arrays(stream);
  }
};

template <int TPB_X, typename T>
__global__ void csr_row_normalize_l1_kernel(
  const int *ia,           // csr row ex_scan (sorted by row)
  const T *vals, int nnz,  // array of values and number of non-zeros
  int m,                   // num rows in csr
  T *result) {             // output array

  // row-based matrix 1 thread per row
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  // sum all vals_arr for row and divide each val by sum
  if (row < m) {
    int start_idx = ia[row];
    int stop_idx = 0;
    if (row < m - 1) {
      stop_idx = ia[row + 1];
    } else
      stop_idx = nnz;

    T sum = T(0.0);
    for (int j = start_idx; j < stop_idx; j++) {
      sum = sum + fabs(vals[j]);
    }

    for (int j = start_idx; j < stop_idx; j++) {
      if (sum != 0.0) {
        T val = vals[j];
        result[j] = val / sum;
      } else {
        result[j] = 0.0;
      }
    }
  }
}

/**
 * @brief Perform L1 normalization on the rows of a given CSR-formatted sparse matrix
 *
 * @param ia: row_ind array
 * @param vals: data array
 * @param nnz: size of data array
 * @param m: size of row_ind array
 * @param result: l1 normalized data array
 * @param stream: cuda stream to use
 */
template <int TPB_X = 32, typename T>
void csr_row_normalize_l1(const int *ia,  // csr row ex_scan (sorted by row)
                          const T *vals,
                          int nnz,  // array of values and number of non-zeros
                          int m,    // num rows in csr
                          T *result,
                          cudaStream_t stream) {  // output array

  dim3 grid(MLCommon::ceildiv(m, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  csr_row_normalize_l1_kernel<TPB_X, T>
    <<<grid, blk, 0, stream>>>(ia, vals, nnz, m, result);
  CUDA_CHECK(cudaGetLastError());
}

template <int TPB_X = 32, typename T>
__global__ void csr_row_normalize_max_kernel(
  const int *ia,           // csr row ind array (sorted by row)
  const T *vals, int nnz,  // array of values and number of non-zeros
  int m,                   // num total rows in csr
  T *result) {             // output array

  // row-based matrix 1 thread per row
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  // find max across columns and divide
  if (row < m) {
    int start_idx = ia[row];
    int stop_idx = 0;
    if (row < m - 1) {
      stop_idx = ia[row + 1];
    } else
      stop_idx = nnz;

    T max = MIN_FLOAT;
    for (int j = start_idx; j < stop_idx; j++) {
      if (vals[j] > max) max = vals[j];
    }

    // divide nonzeros in current row by max
    for (int j = start_idx; j < stop_idx; j++) {
      if (max != 0.0 && max > MIN_FLOAT) {
        T val = vals[j];
        result[j] = val / max;
      } else {
        result[j] = 0.0;
      }
    }
  }
}

/**
 * @brief Perform L_inf normalization on a given CSR-formatted sparse matrix
 *
 * @param ia: row_ind array
 * @param vals: data array
 * @param nnz: size of data array
 * @param m: size of row_ind array
 * @param result: l1 normalized data array
 * @param stream: cuda stream to use
 */

template <int TPB_X = 32, typename T>
void csr_row_normalize_max(const int *ia,  // csr row ind array (sorted by row)
                           const T *vals,
                           int nnz,  // array of values and number of non-zeros
                           int m,    // num total rows in csr
                           T *result, cudaStream_t stream) {
  dim3 grid(MLCommon::ceildiv(m, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  csr_row_normalize_max_kernel<TPB_X, T>
    <<<grid, blk, 0, stream>>>(ia, vals, nnz, m, result);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__device__ int get_stop_idx(T row, T m, T nnz, const T *ind) {
  int stop_idx = 0;
  if (row < (m - 1))
    stop_idx = ind[row + 1];
  else
    stop_idx = nnz;

  return stop_idx;
}

template <int TPB_X = 32>
__global__ void csr_to_coo_kernel(const int *row_ind, int m, int *coo_rows,
                                  int nnz) {
  // row-based matrix 1 thread per row
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < m) {
    int start_idx = row_ind[row];
    int stop_idx = get_stop_idx(row, m, nnz, row_ind);
    for (int i = start_idx; i < stop_idx; i++) coo_rows[i] = row;
  }
}

/**
 * @brief Convert a CSR row_ind array to a COO rows array
 * @param row_ind: Input CSR row_ind array
 * @param m: size of row_ind array
 * @param coo_rows: Output COO row array
 * @param nnz: size of output COO row array
 * @param stream: cuda stream to use
 */
template <int TPB_X>
void csr_to_coo(const int *row_ind, int m, int *coo_rows, int nnz,
                cudaStream_t stream) {
  dim3 grid(MLCommon::ceildiv(m, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  csr_to_coo_kernel<TPB_X><<<grid, blk, 0, stream>>>(row_ind, m, coo_rows, nnz);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T, int TPB_X = 32>
__global__ void csr_add_calc_row_counts_kernel(
  const int *a_ind, const int *a_indptr, const T *a_val, int nnz1,
  const int *b_ind, const int *b_indptr, const T *b_val, int nnz2, int m,
  int *out_rowcounts) {
  // loop through columns in each set of rows and
  // calculate number of unique cols across both rows
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  if (row < m) {
    int a_start_idx = a_ind[row];
    int a_stop_idx = get_stop_idx(row, m, nnz1, a_ind);

    int b_start_idx = b_ind[row];
    int b_stop_idx = get_stop_idx(row, m, nnz2, b_ind);

    /**
         * Union of columns within each row of A and B so that we can scan through
         * them, adding their values together.
         */
    int max_size = (a_stop_idx - a_start_idx) + (b_stop_idx - b_start_idx);

    int *arr = new int[max_size];
    int cur_arr_idx = 0;
    for (int j = a_start_idx; j < a_stop_idx; j++) {
      arr[cur_arr_idx] = a_indptr[j];
      cur_arr_idx++;
    }

    int arr_size = cur_arr_idx;
    int final_size = arr_size;

    for (int j = b_start_idx; j < b_stop_idx; j++) {
      int cur_col = b_indptr[j];
      bool found = false;
      for (int k = 0; k < arr_size; k++) {
        if (arr[k] == cur_col) {
          found = true;
          break;
        }
      }

      if (!found) {
        final_size++;
      }
    }

    out_rowcounts[row] = final_size;
    atomicAdd(out_rowcounts + m, final_size);

    delete arr;
  }
}

template <typename T, int TPB_X = 32>
__global__ void csr_add_kernel(const int *a_ind, const int *a_indptr,
                               const T *a_val, int nnz1, const int *b_ind,
                               const int *b_indptr, const T *b_val, int nnz2,
                               int m, int *out_ind, int *out_indptr,
                               T *out_val) {
  // 1 thread per row
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  if (row < m) {
    int a_start_idx = a_ind[row];
    int a_stop_idx = get_stop_idx(row, m, nnz1, a_ind);

    int b_start_idx = b_ind[row];
    int b_stop_idx = get_stop_idx(row, m, nnz2, b_ind);

    int o_idx = out_ind[row];

    int cur_o_idx = o_idx;
    for (int j = a_start_idx; j < a_stop_idx; j++) {
      out_indptr[cur_o_idx] = a_indptr[j];
      out_val[cur_o_idx] = a_val[j];
      cur_o_idx++;
    }

    int arr_size = cur_o_idx - o_idx;
    for (int j = b_start_idx; j < b_stop_idx; j++) {
      int cur_col = b_indptr[j];
      bool found = false;
      for (int k = o_idx; k < o_idx + arr_size; k++) {
        // If we found a match, sum the two values
        if (out_indptr[k] == cur_col) {
          out_val[k] += b_val[j];
          found = true;
          break;
        }
      }

      // if we didn't find a match, add the value for b
      if (!found) {
        out_indptr[o_idx + arr_size] = cur_col;
        out_val[o_idx + arr_size] = b_val[j];
        arr_size++;
      }
    }
  }
}

/**
 * @brief Calculate the CSR row_ind array that would result
 * from summing together two CSR matrices
 * @param a_ind: left hand row_ind array
 * @param a_indptr: left hand index_ptr array
 * @param a_val: left hand data array
 * @param nnz1: size of left hand index_ptr and val arrays
 * @param b_ind: right hand row_ind array
 * @param b_indptr: right hand index_ptr array
 * @param b_val: right hand data array
 * @param nnz2: size of right hand index_ptr and val arrays
 * @param m: size of output array (number of rows in final matrix)
 * @param out_ind: output row_ind array
 * @param alloc: deviceAllocator to use for temp memory
 * @param stream: cuda stream to use
 */
template <typename T, int TPB_X = 32>
size_t csr_add_calc_inds(const int *a_ind, const int *a_indptr, const T *a_val,
                         int nnz1, const int *b_ind, const int *b_indptr,
                         const T *b_val, int nnz2, int m, int *out_ind,
                         std::shared_ptr<deviceAllocator> d_alloc,
                         cudaStream_t stream) {
  dim3 grid(ceildiv(m, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  device_buffer<int> row_counts(d_alloc, stream, m + 1);
  CUDA_CHECK(
    cudaMemsetAsync(row_counts.data(), 0, (m + 1) * sizeof(int), stream));

  csr_add_calc_row_counts_kernel<T, TPB_X>
    <<<grid, blk, 0, stream>>>(a_ind, a_indptr, a_val, nnz1, b_ind, b_indptr,
                               b_val, nnz2, m, row_counts.data());
  CUDA_CHECK(cudaPeekAtLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));

  int cnnz = 0;
  MLCommon::updateHost(&cnnz, row_counts.data() + m, 1, stream);

  // create csr compressed row index from row counts
  thrust::device_ptr<int> row_counts_d =
    thrust::device_pointer_cast(row_counts.data());
  thrust::device_ptr<int> c_ind_d = thrust::device_pointer_cast(out_ind);
  exclusive_scan(thrust::cuda::par.on(stream), row_counts_d, row_counts_d + m,
                 c_ind_d);

  return cnnz;
}

/**
 * @brief Calculate the CSR row_ind array that would result
 * from summing together two CSR matrices
 * @param a_ind: left hand row_ind array
 * @param a_indptr: left hand index_ptr array
 * @param a_val: left hand data array
 * @param nnz1: size of left hand index_ptr and val arrays
 * @param b_ind: right hand row_ind array
 * @param b_indptr: right hand index_ptr array
 * @param b_val: right hand data array
 * @param nnz2: size of right hand index_ptr and val arrays
 * @param m: size of output array (number of rows in final matrix)
 * @param c_ind: output row_ind array
 * @param c_indptr: output ind_ptr array
 * @param c_val: output data array
 * @param stream: cuda stream to use
 */
template <typename T, int TPB_X = 32>
void csr_add_finalize(const int *a_ind, const int *a_indptr, const T *a_val,
                      int nnz1, const int *b_ind, const int *b_indptr,
                      const T *b_val, int nnz2, int m, int *c_ind,
                      int *c_indptr, T *c_val, cudaStream_t stream) {
  dim3 grid(MLCommon::ceildiv(m, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  csr_add_kernel<T, TPB_X>
    <<<grid, blk, 0, stream>>>(a_ind, a_indptr, a_val, nnz1, b_ind, b_indptr,
                               b_val, nnz2, m, c_ind, c_indptr, c_val);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T, int TPB_X = 32, typename Lambda = auto(T, T, T)->void>
__global__ void csr_row_op_kernel(const T *row_ind, T n_rows, T nnz,
                                  Lambda op) {
  T row = blockIdx.x * TPB_X + threadIdx.x;
  if (row < n_rows) {
    T start_idx = row_ind[row];
    T stop_idx = row < n_rows - 1 ? row_ind[row + 1] : nnz;
    op(row, start_idx, stop_idx);
  }
}

/**
 * @brief Perform a custom row operation on a CSR matrix in batches.
 * @tparam T numerical type of row_ind array
 * @tparam TPB_X number of threads per block to use for underlying kernel
 * @tparam Lambda type of custom operation function
 * @param row_ind the CSR row_ind array to perform parallel operations over
 * @param total_rows total number vertices in graph
 * @param batchSize size of row_ind
 * @param op custom row operation functor accepting the row and beginning index.
 * @param stream cuda stream 121to use
 */
template <typename Index_, int TPB_X = 32,
          typename Lambda = auto(Index_, Index_, Index_)->void>
void csr_row_op(const Index_ *row_ind, Index_ n_rows, Index_ nnz, Lambda op,
                cudaStream_t stream) {
  dim3 grid(MLCommon::ceildiv(n_rows, Index_(TPB_X)), 1, 1);
  dim3 blk(TPB_X, 1, 1);
  csr_row_op_kernel<Index_, TPB_X>
    <<<grid, blk, 0, stream>>>(row_ind, n_rows, nnz, op);

  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Constructs an adjacency graph CSR row_ind_ptr array from
 * a row_ind array and adjacency array.
 * @tparam T the numeric type of the index arrays
 * @tparam TPB_X the number of threads to use per block for kernels
 * @tparam Lambda function for fused operation in the adj_graph construction
 * @param row_ind the input CSR row_ind array
 * @param total_rows number of vertices in graph
 * @param batchSize number of vertices in current batch
 * @param adj an adjacency array (size batchSize x total_rows)
 * @param row_ind_ptr output CSR row_ind_ptr for adjacency graph
 * @param stream cuda stream to use
 * @param fused_op: the fused operation
 */
template <typename Index_, int TPB_X = 32,
          typename Lambda = auto(Index_, Index_, Index_)->void>
void csr_adj_graph_batched(const Index_ *row_ind, Index_ total_rows, Index_ nnz,
                           Index_ batchSize, const bool *adj,
                           Index_ *row_ind_ptr, cudaStream_t stream,
                           Lambda fused_op) {
  csr_row_op<Index_, TPB_X>(
    row_ind, batchSize, nnz,
    [fused_op, adj, total_rows, row_ind_ptr, batchSize, nnz] __device__(
      Index_ row, Index_ start_idx, Index_ stop_idx) {
      fused_op(row, start_idx, stop_idx);
      Index_ k = 0;
      for (Index_ i = 0; i < total_rows; i++) {
        // @todo: uncoalesced mem accesses!
        if (adj[batchSize * i + row]) {
          row_ind_ptr[start_idx + k] = i;
          k += 1;
        }
      }
    },
    stream);
}

template <typename Index_, int TPB_X = 32,
          typename Lambda = auto(Index_, Index_, Index_)->void>
void csr_adj_graph_batched(const Index_ *row_ind, Index_ total_rows, Index_ nnz,
                           Index_ batchSize, const bool *adj,
                           Index_ *row_ind_ptr, cudaStream_t stream) {
  csr_adj_graph_batched(
    row_ind, total_rows, nnz, batchSize, adj, row_ind_ptr, stream,
    [] __device__(Index_ row, Index_ start_idx, Index_ stop_idx) {});
}

/**
 * @brief Constructs an adjacency graph CSR row_ind_ptr array from a
 * a row_ind array and adjacency array.
 * @tparam T the numeric type of the index arrays
 * @tparam TPB_X the number of threads to use per block for kernels
 * @param row_ind the input CSR row_ind array
 * @param n_rows number of total vertices in graph
 * @param adj an adjacency array
 * @param row_ind_ptr output CSR row_ind_ptr for adjacency graph
 * @param stream cuda stream to use
 * @param fused_op the fused operation
 */
template <typename Index_, int TPB_X = 32,
          typename Lambda = auto(Index_, Index_, Index_)->void>
void csr_adj_graph(const Index_ *row_ind, Index_ total_rows, Index_ nnz,
                   const bool *adj, Index_ *row_ind_ptr, cudaStream_t stream,
                   Lambda fused_op) {
  csr_adj_graph_batched<Index_, TPB_X>(row_ind, total_rows, nnz, total_rows,
                                       adj, row_ind_ptr, stream, fused_op);
}

struct WeakCCState {
 public:
  bool *xa;
  bool *fa;
  bool *m;

  WeakCCState(bool *xa, bool *fa, bool *m) : xa(xa), fa(fa), m(m) {}
};

template <typename Index_, int TPB_X = 32>
__global__ void weak_cc_label_device(Index_ *labels, const Index_ *row_ind,
                                     const Index_ *row_ind_ptr, Index_ nnz,
                                     bool *fa, bool *xa, bool *m,
                                     Index_ startVertexId, Index_ batchSize) {
  Index_ tid = threadIdx.x + blockIdx.x * TPB_X;
  if (tid < batchSize) {
    if (fa[tid + startVertexId]) {
      fa[tid + startVertexId] = false;
      Index_ row_ind_val = row_ind[tid];

      Index_ start = row_ind_val;
      Index_ ci, cj;
      bool ci_mod = false;
      ci = labels[tid + startVertexId];

      Index_ degree = get_stop_idx(tid, batchSize, nnz, row_ind) - row_ind_val;
      for (Index_ j = 0; j < degree;
           j++) {  // TODO: Can't this be calculated from the ex_scan?
        Index_ j_ind = row_ind_ptr[start + j];
        cj = labels[j_ind];
        if (ci < cj) {
          if (sizeof(Index_) == 4)
            atomicMin((int *)(labels + j_ind), ci);
          else if (sizeof(Index_) == 8)
            atomicMin((long long int *)(labels + j_ind), ci);
          xa[j_ind] = true;
          m[0] = true;
        } else if (ci > cj) {
          ci = cj;
          ci_mod = true;
        }
      }
      if (ci_mod) {
        if (sizeof(Index_) == 4)
          atomicMin((int *)(labels + startVertexId + tid), ci);
        else if (sizeof(Index_) == 8)
          atomicMin((long long int *)(labels + startVertexId + tid), ci);
        xa[startVertexId + tid] = true;
        m[0] = true;
      }
    }
  }
}

template <typename Index_, int TPB_X = 32, typename Lambda>
__global__ void weak_cc_init_label_kernel(Index_ *labels, Index_ startVertexId,
                                          Index_ batchSize, Index_ MAX_LABEL,
                                          Lambda filter_op) {
  /** F1 and F2 in the paper correspond to fa and xa */
  /** Cd in paper corresponds to db_cluster */
  Index_ tid = threadIdx.x + blockIdx.x * TPB_X;
  if (tid < batchSize) {
    if (filter_op(tid) && labels[tid + startVertexId] == MAX_LABEL)
      labels[startVertexId + tid] = startVertexId + tid + 1;
  }
}

template <typename Index_, int TPB_X = 32>
__global__ void weak_cc_init_all_kernel(Index_ *labels, bool *fa, bool *xa,
                                        Index_ N, Index_ MAX_LABEL) {
  Index_ tid = threadIdx.x + blockIdx.x * TPB_X;
  if (tid < N) {
    labels[tid] = MAX_LABEL;
    fa[tid] = true;
    xa[tid] = false;
  }
}

template <typename Index_, int TPB_X = 32, typename Lambda>
void weak_cc_label_batched(Index_ *labels, const Index_ *row_ind,
                           const Index_ *row_ind_ptr, Index_ nnz, Index_ N,
                           WeakCCState *state, Index_ startVertexId,
                           Index_ batchSize, cudaStream_t stream,
                           Lambda filter_op) {
  ASSERT(sizeof(Index_) == 4 || sizeof(Index_) == 8,
         "Index_ should be 4 or 8 bytes");

  bool host_m;
  bool *host_fa = (bool *)malloc(sizeof(bool) * N);
  bool *host_xa = (bool *)malloc(sizeof(bool) * N);

  dim3 blocks(ceildiv(batchSize, Index_(TPB_X)));
  dim3 threads(TPB_X);
  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();

  weak_cc_init_label_kernel<Index_, TPB_X><<<blocks, threads, 0, stream>>>(
    labels, startVertexId, batchSize, MAX_LABEL, filter_op);
  CUDA_CHECK(cudaPeekAtLastError());

  int n_iters = 0;
  do {
    CUDA_CHECK(cudaMemsetAsync(state->m, false, sizeof(bool), stream));

    weak_cc_label_device<Index_, TPB_X><<<blocks, threads, 0, stream>>>(
      labels, row_ind, row_ind_ptr, nnz, state->fa, state->xa, state->m,
      startVertexId, batchSize);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    //** swapping F1 and F2
    MLCommon::updateHost(host_fa, state->fa, N, stream);
    MLCommon::updateHost(host_xa, state->xa, N, stream);
    MLCommon::updateDevice(state->fa, host_xa, N, stream);
    MLCommon::updateDevice(state->xa, host_fa, N, stream);

    //** Updating m *
    MLCommon::updateHost(&host_m, state->m, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    n_iters++;
  } while (host_m);

  free(host_fa);
  free(host_xa);
}

/**
 * @brief Compute weakly connected components. Note that the resulting labels
 * may not be taken from a monotonically increasing set (eg. numbers may be
 * skipped). The MLCommon::Label package contains a primitive `make_monotonic`,
 * which will make a monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam Type the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param startVertexId the starting vertex index for the current batch
 * @param batchSize number of vertices for current batch
 * @param state instance of inter-batch state management
 * @param stream the cuda stream to use
 * @param filter_op an optional filtering function to determine which points
 * should get considered for labeling.
 */
template <typename Index_, int TPB_X = 32, typename Lambda = auto(Index_)->bool>
void weak_cc_batched(Index_ *labels, const Index_ *row_ind,
                     const Index_ *row_ind_ptr, Index_ nnz, Index_ N,
                     Index_ startVertexId, Index_ batchSize, WeakCCState *state,
                     cudaStream_t stream, Lambda filter_op) {
  dim3 blocks(ceildiv(N, Index_(TPB_X)));
  dim3 threads(TPB_X);

  Index_ MAX_LABEL = std::numeric_limits<Index_>::max();

  if (startVertexId == 0) {
    weak_cc_init_all_kernel<Index_, TPB_X><<<blocks, threads, 0, stream>>>(
      labels, state->fa, state->xa, N, MAX_LABEL);
    CUDA_CHECK(cudaPeekAtLastError());
  }

  weak_cc_label_batched<Index_, TPB_X>(labels, row_ind, row_ind_ptr, nnz, N,
                                       state, startVertexId, batchSize, stream,
                                       filter_op);
}

/**
 * @brief Compute weakly connected components. Note that the resulting labels
 * may not be taken from a monotonically increasing set (eg. numbers may be
 * skipped). The MLCommon::Label package contains a primitive `make_monotonic`,
 * which will make a monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam Type the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param startVertexId the starting vertex index for the current batch
 * @param batchSize number of vertices for current batch
 * @param state instance of inter-batch state management
 * @param stream the cuda stream to use
 */
template <typename Index_, int TPB_X = 32>
void weak_cc_batched(Index_ *labels, const Index_ *row_ind,
                     const Index_ *row_ind_ptr, Index_ nnz, Index_ N,
                     Index_ startVertexId, Index_ batchSize, WeakCCState *state,
                     cudaStream_t stream) {
  weak_cc_batched(labels, row_ind, row_ind_ptr, nnz, N, startVertexId,
                  batchSize, state, stream,
                  [] __device__(Index_ tid) { return true; });
}

/**
 * @brief Compute weakly connected components. Note that the resulting labels
 * may not be taken from a monotonically increasing set (eg. numbers may be
 * skipped). The MLCommon::Label package contains a primitive `make_monotonic`,
 * which will make a monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam Type the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param alloc: deviceAllocator to use for temp memory
 * @param stream the cuda stream to use
 * @param filter_op an optional filtering function to determine which points
 * should get considered for labeling.
 */
template <typename Index_ = int, int TPB_X = 32,
          typename Lambda = auto(Index_)->bool>
void weak_cc(Index_ *labels, const Index_ *row_ind, const Index_ *row_ind_ptr,
             Index_ nnz, Index_ N, std::shared_ptr<deviceAllocator> d_alloc,
             cudaStream_t stream, Lambda filter_op) {
  device_buffer<bool> xa(d_alloc, stream, N);
  device_buffer<bool> fa(d_alloc, stream, N);
  device_buffer<bool> m(d_alloc, stream, 1);

  WeakCCState state(xa.data(), fa.data(), m.data());
  weak_cc_batched<Index_, TPB_X>(labels, row_ind, row_ind_ptr, nnz, N, 0, N,
                                 stream, filter_op);
}

/**
 * @brief Compute weakly connected components. Note that the resulting labels
 * may not be taken from a monotonically increasing set (eg. numbers may be
 * skipped). The MLCommon::Label package contains a primitive `make_monotonic`,
 * which will make a monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam Type the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param alloc: deviceAllocator to use for temp memory
 * @param stream the cuda stream to use
 */
template <typename Index_, int TPB_X = 32>
void weak_cc(Index_ *labels, const Index_ *row_ind, const Index_ *row_ind_ptr,
             Index_ nnz, Index_ N, std::shared_ptr<deviceAllocator> d_alloc,
             cudaStream_t stream) {
  device_buffer<bool> xa(d_alloc, stream, N);
  device_buffer<bool> fa(d_alloc, stream, N);
  device_buffer<bool> m(d_alloc, stream, 1);
  WeakCCState state(xa.data(), fa.data(), m.data());
  weak_cc_batched<Index_, TPB_X>(labels, row_ind, row_ind_ptr, nnz, N, 0, N,
                                 stream, [](Index_) { return true; });
}

};  // namespace Sparse
};  // namespace MLCommon
