/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cuml/common/cuml_allocator.hpp>
#include "csr.cuh"

#include <raft/sparse/cusparse_wrappers.h>

#include <common/device_buffer.hpp>

#include <cusparse_v2.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>

#include <iostream>
#define restrict __restrict__

#pragma once

namespace MLCommon {
namespace Sparse {

/** @brief A Container object for sparse coordinate. There are two motivations
 * behind using a container for COO arrays.
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
 * @tparam Index_Type: the type of index array
 *
 */
template <typename T, typename Index_Type = int>
class COO {
 protected:
  device_buffer<Index_Type> rows_arr;
  device_buffer<Index_Type> cols_arr;
  device_buffer<T> vals_arr;

 public:
  Index_Type nnz;
  Index_Type n_rows;
  Index_Type n_cols;

  /**
    * @param d_alloc: the device allocator to use for the underlying buffers
    * @param stream: CUDA stream to use
    */
  COO(std::shared_ptr<deviceAllocator> d_alloc, cudaStream_t stream)
    : rows_arr(d_alloc, stream, 0),
      cols_arr(d_alloc, stream, 0),
      vals_arr(d_alloc, stream, 0),
      nnz(0),
      n_rows(0),
      n_cols(0) {}

  /**
    * @param rows: coo rows array
    * @param cols: coo cols array
    * @param vals: coo vals array
    * @param nnz: size of the rows/cols/vals arrays
    * @param n_rows: number of rows in the dense matrix
    * @param n_cols: number of cols in the dense matrix
    */
  COO(device_buffer<Index_Type> &rows, device_buffer<Index_Type> &cols,
      device_buffer<T> &vals, Index_Type nnz, Index_Type n_rows = 0,
      Index_Type n_cols = 0)
    : rows_arr(rows),
      cols_arr(cols),
      vals_arr(vals),
      nnz(nnz),
      n_rows(n_rows),
      n_cols(n_cols) {}

  /**
    * @param d_alloc: the device allocator use
    * @param stream: CUDA stream to use
    * @param nnz: size of the rows/cols/vals arrays
    * @param n_rows: number of rows in the dense matrix
    * @param n_cols: number of cols in the dense matrix
    * @param init: initialize arrays with zeros
    */
  COO(std::shared_ptr<deviceAllocator> d_alloc, cudaStream_t stream,
      Index_Type nnz, Index_Type n_rows = 0, Index_Type n_cols = 0,
      bool init = true)
    : rows_arr(d_alloc, stream, nnz),
      cols_arr(d_alloc, stream, nnz),
      vals_arr(d_alloc, stream, nnz),
      nnz(nnz),
      n_rows(n_rows),
      n_cols(n_cols) {
    if (init) init_arrays(stream);
  }

  void init_arrays(cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(this->rows_arr.data(), 0,
                               this->nnz * sizeof(Index_Type), stream));
    CUDA_CHECK(cudaMemsetAsync(this->cols_arr.data(), 0,
                               this->nnz * sizeof(Index_Type), stream));
    CUDA_CHECK(
      cudaMemsetAsync(this->vals_arr.data(), 0, this->nnz * sizeof(T), stream));
  }

  ~COO() {}

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
    if (this->rows_arr.size() == 0 || this->cols_arr.size() == 0 ||
        this->vals_arr.size() == 0) {
      return false;
    }

    return true;
  }

  /*
   * @brief Returns the rows array
   */
  Index_Type *rows() { return this->rows_arr.data(); }

  /**
   * @brief Returns the cols array
   */
  Index_Type *cols() { return this->cols_arr.data(); }

  /**
   * @brief Returns the vals array
   */
  T *vals() { return this->vals_arr.data(); }

  /**
    * @brief Send human-readable state information to output stream
    */
  friend std::ostream &operator<<(std::ostream &out, const COO<T> &c) {
    if (c.validate_size() && c.validate_mem()) {
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

      out << raft::arr2Str(c.rows_arr.data(), c.nnz, "rows", stream)
          << std::endl;
      out << raft::arr2Str(c.cols_arr.data(), c.nnz, "cols", stream)
          << std::endl;
      out << raft::arr2Str(c.vals_arr.data(), c.nnz, "vals", stream)
          << std::endl;
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

    this->rows_arr.resize(this->nnz, stream);
    this->cols_arr.resize(this->nnz, stream);
    this->vals_arr.resize(this->nnz, stream);

    if (init) init_arrays(stream);
  }
};

/**
 * @brief Sorts the arrays that comprise the coo matrix
 * by row.
 *
 * @param m number of rows in coo matrix
 * @param n number of cols in coo matrix
 * @param nnz number of non-zeros
 * @param rows rows array from coo matrix
 * @param cols cols array from coo matrix
 * @param vals vals array from coo matrix
 * @param d_alloc device allocator for temporary buffers
 * @param stream: cuda stream to use
 */
template <typename T>
void coo_sort(int m, int n, int nnz, int *rows, int *cols, T *vals,
              std::shared_ptr<deviceAllocator> d_alloc, cudaStream_t stream) {
  cusparseHandle_t handle = NULL;

  size_t pBufferSizeInBytes = 0;

  CUSPARSE_CHECK(cusparseCreate(&handle));
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  CUSPARSE_CHECK(cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, rows, cols,
                                                &pBufferSizeInBytes));

  device_buffer<int> d_P(d_alloc, stream, nnz);
  device_buffer<char> pBuffer(d_alloc, stream, pBufferSizeInBytes);

  CUSPARSE_CHECK(cusparseCreateIdentityPermutation(handle, nnz, d_P.data()));

  CUSPARSE_CHECK(cusparseXcoosortByRow(handle, m, n, nnz, rows, cols,
                                       d_P.data(), pBuffer.data()));

  device_buffer<T> vals_sorted(d_alloc, stream, nnz);

  CUSPARSE_CHECK(raft::sparse::cusparsegthr<T>(
    handle, nnz, vals, vals_sorted.data(), d_P.data(), stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  raft::copy(vals, vals_sorted.data(), nnz, stream);

  CUSPARSE_CHECK(cusparseDestroy(handle));
}

/**
 * @brief Sort the underlying COO arrays by row
 * @tparam T: the type name of the underlying value array
 * @param in: COO to sort by row
 * @param d_alloc device allocator for temporary buffers
 * @param stream: the cuda stream to use
 */
template <typename T>
void coo_sort(COO<T> *const in, std::shared_ptr<deviceAllocator> d_alloc,
              cudaStream_t stream) {
  coo_sort<T>(in->n_rows, in->n_cols, in->nnz, in->rows(), in->cols(),
              in->vals(), d_alloc, stream);
}

template <int TPB_X, typename T>
__global__ void coo_remove_zeros_kernel(const int *rows, const int *cols,
                                        const T *vals, int nnz, int *crows,
                                        int *ccols, T *cvals, int *ex_scan,
                                        int *cur_ex_scan, int m) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  if (row < m) {
    int start = cur_ex_scan[row];
    int stop = MLCommon::Sparse::get_stop_idx(row, m, nnz, cur_ex_scan);
    int cur_out_idx = ex_scan[row];

    for (int idx = start; idx < stop; idx++) {
      if (vals[idx] != 0.0) {
        crows[cur_out_idx] = rows[idx];
        ccols[cur_out_idx] = cols[idx];
        cvals[cur_out_idx] = vals[idx];
        ++cur_out_idx;
      }
    }
  }
}

template <int TPB_X, typename T>
__global__ void coo_remove_scalar_kernel(const int *rows, const int *cols,
                                         const T *vals, int nnz, int *crows,
                                         int *ccols, T *cvals, int *ex_scan,
                                         int *cur_ex_scan, int m, T scalar) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  if (row < m) {
    int start = cur_ex_scan[row];
    int stop = MLCommon::Sparse::get_stop_idx(row, m, nnz, cur_ex_scan);
    int cur_out_idx = ex_scan[row];

    for (int idx = start; idx < stop; idx++) {
      if (vals[idx] != scalar) {
        crows[cur_out_idx] = rows[idx];
        ccols[cur_out_idx] = cols[idx];
        cvals[cur_out_idx] = vals[idx];
        ++cur_out_idx;
      }
    }
  }
}

/**
 * @brief Count all the rows in the coo row array and place them in the
 * results matrix, indexed by row.
 *
 * @tparam TPB_X: number of threads to use per block
 * @param rows the rows array of the coo matrix
 * @param nnz the size of the rows array
 * @param results array to place results
 */
template <int TPB_X>
__global__ void coo_row_count_kernel(const int *rows, int nnz, int *results) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < nnz) {
    raft::myAtomicAdd(results + rows[row], 1);
  }
}

/**
 * @brief Count the number of values for each row
 * @tparam TPB_X: number of threads to use per block
 * @param rows: rows array of the COO matrix
 * @param nnz: size of the rows array
 * @param results: output result array
 * @param stream: cuda stream to use
 */
template <int TPB_X>
void coo_row_count(const int *rows, int nnz, int *results,
                   cudaStream_t stream) {
  dim3 grid_rc(raft::ceildiv(nnz, TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);

  coo_row_count_kernel<TPB_X>
    <<<grid_rc, blk_rc, 0, stream>>>(rows, nnz, results);
  CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Count the number of values for each row
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: type name of underlying values array
 * @param in: input COO object for counting rows
 * @param results: output array with row counts (size=in->n_rows)
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T>
void coo_row_count(COO<T> *in, int *results, cudaStream_t stream) {
  dim3 grid_rc(raft::ceildiv(in->nnz, TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);

  coo_row_count_kernel<TPB_X>
    <<<grid_rc, blk_rc, 0, stream>>>(in->rows(), in->nnz, results);
  CUDA_CHECK(cudaGetLastError());
}

template <int TPB_X, typename T>
__global__ void coo_row_count_nz_kernel(const int *rows, const T *vals, int nnz,
                                        int *results) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < nnz && vals[row] != 0.0) {
    raft::myAtomicAdd(results + rows[row], 1);
  }
}

template <int TPB_X, typename T>
__global__ void coo_row_count_scalar_kernel(const int *rows, const T *vals,
                                            int nnz, T scalar, int *results) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < nnz && vals[row] != scalar) {
    raft::myAtomicAdd(results + rows[row], 1);
  }
}

/**
 * @brief Count the number of values for each row matching a particular scalar
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: the type name of the underlying value arrays
 * @param in: Input COO array
 * @param scalar: scalar to match for counting rows
 * @param results: output row counts
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T>
void coo_row_count_scalar(COO<T> *in, T scalar, int *results,
                          cudaStream_t stream) {
  dim3 grid_rc(raft::ceildiv(in->nnz, TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);

  coo_row_count_scalar_kernel<TPB_X, T><<<grid_rc, blk_rc, 0, stream>>>(
    in->rows(), in->vals(), in->nnz, scalar, results);
  CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Count the number of values for each row matching a particular scalar
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: the type name of the underlying value arrays
 * @param rows: Input COO row array
 * @param vals: Input COO val arrays
 * @param nnz: size of input COO arrays
 * @param scalar: scalar to match for counting rows
 * @param results: output row counts
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T>
void coo_row_count_scalar(const int *rows, const T *vals, int nnz, T scalar,
                          int *results, cudaStream_t stream = 0) {
  dim3 grid_rc(raft::ceildiv(nnz, TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);

  coo_row_count_scalar_kernel<TPB_X, T>
    <<<grid_rc, blk_rc, 0, stream>>>(rows, vals, nnz, scalar, results);
  CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Count the number of nonzeros for each row
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: the type name of the underlying value arrays
 * @param rows: Input COO row array
 * @param vals: Input COO val arrays
 * @param nnz: size of input COO arrays
 * @param results: output row counts
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T>
void coo_row_count_nz(const int *rows, const T *vals, int nnz, int *results,
                      cudaStream_t stream) {
  dim3 grid_rc(raft::ceildiv(nnz, TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);

  coo_row_count_nz_kernel<TPB_X, T>
    <<<grid_rc, blk_rc, 0, stream>>>(rows, vals, nnz, results);
  CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Count the number of nonzero values for each row
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: the type name of the underlying value arrays
 * @param in: Input COO array
 * @param results: output row counts
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T>
void coo_row_count_nz(COO<T> *in, int *results, cudaStream_t stream) {
  dim3 grid_rc(raft::ceildiv(in->nnz, TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);

  coo_row_count_nz_kernel<TPB_X, T>
    <<<grid_rc, blk_rc, 0, stream>>>(in->rows(), in->vals(), in->nnz, results);
  CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Removes the values matching a particular scalar from a COO formatted sparse matrix.
 *
 * @param rows: input array of rows (size n)
 * @param cols: input array of cols (size n)
 * @param vals: input array of vals (size n)
 * @param nnz: size of current rows/cols/vals arrays
 * @param crows: compressed array of rows
 * @param ccols: compressed array of cols
 * @param cvals: compressed array of vals
 * @param cnnz: array of non-zero counts per row
 * @param cur_cnnz array of counts per row
 * @param scalar: scalar to remove from arrays
 * @param n: number of rows in dense matrix
 * @param d_alloc device allocator for temporary buffers
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T>
void coo_remove_scalar(const int *rows, const int *cols, const T *vals, int nnz,
                       int *crows, int *ccols, T *cvals, int *cnnz,
                       int *cur_cnnz, T scalar, int n,
                       std::shared_ptr<deviceAllocator> d_alloc,
                       cudaStream_t stream) {
  device_buffer<int> ex_scan(d_alloc, stream, n);
  device_buffer<int> cur_ex_scan(d_alloc, stream, n);

  CUDA_CHECK(cudaMemsetAsync(ex_scan.data(), 0, n * sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(cur_ex_scan.data(), 0, n * sizeof(int), stream));

  thrust::device_ptr<int> dev_cnnz = thrust::device_pointer_cast(cnnz);
  thrust::device_ptr<int> dev_ex_scan =
    thrust::device_pointer_cast(ex_scan.data());
  thrust::exclusive_scan(thrust::cuda::par.on(stream), dev_cnnz, dev_cnnz + n,
                         dev_ex_scan);
  CUDA_CHECK(cudaPeekAtLastError());

  thrust::device_ptr<int> dev_cur_cnnz = thrust::device_pointer_cast(cur_cnnz);
  thrust::device_ptr<int> dev_cur_ex_scan =
    thrust::device_pointer_cast(cur_ex_scan.data());
  thrust::exclusive_scan(thrust::cuda::par.on(stream), dev_cur_cnnz,
                         dev_cur_cnnz + n, dev_cur_ex_scan);
  CUDA_CHECK(cudaPeekAtLastError());

  dim3 grid(raft::ceildiv(n, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  coo_remove_scalar_kernel<TPB_X><<<grid, blk, 0, stream>>>(
    rows, cols, vals, nnz, crows, ccols, cvals, dev_ex_scan.get(),
    dev_cur_ex_scan.get(), n, scalar);
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Removes the values matching a particular scalar from a COO formatted sparse matrix.
 *
 * @param in: input COO matrix
 * @param out: output COO matrix
 * @param scalar: scalar to remove from arrays
 * @param d_alloc device allocator for temporary buffers
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T>
void coo_remove_scalar(COO<T> *in, COO<T> *out, T scalar,
                       std::shared_ptr<deviceAllocator> d_alloc,
                       cudaStream_t stream) {
  device_buffer<int> row_count_nz(d_alloc, stream, in->n_rows);
  device_buffer<int> row_count(d_alloc, stream, in->n_rows);

  CUDA_CHECK(
    cudaMemsetAsync(row_count_nz.data(), 0, in->n_rows * sizeof(int), stream));
  CUDA_CHECK(
    cudaMemsetAsync(row_count.data(), 0, in->n_rows * sizeof(int), stream));

  MLCommon::Sparse::coo_row_count<TPB_X>(in->rows(), in->nnz, row_count.data(),
                                         stream);
  CUDA_CHECK(cudaPeekAtLastError());

  MLCommon::Sparse::coo_row_count_scalar<TPB_X>(
    in->rows(), in->vals(), in->nnz, scalar, row_count_nz.data(), stream);
  CUDA_CHECK(cudaPeekAtLastError());

  thrust::device_ptr<int> d_row_count_nz =
    thrust::device_pointer_cast(row_count_nz.data());
  int out_nnz = thrust::reduce(thrust::cuda::par.on(stream), d_row_count_nz,
                               d_row_count_nz + in->n_rows);

  out->allocate(out_nnz, in->n_rows, in->n_cols, false, stream);

  coo_remove_scalar<TPB_X, T>(in->rows(), in->cols(), in->vals(), in->nnz,
                              out->rows(), out->cols(), out->vals(),
                              row_count_nz.data(), row_count.data(), scalar,
                              in->n_rows, d_alloc, stream);
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Removes zeros from a COO formatted sparse matrix.
 *
 * @param in: input COO matrix
 * @param out: output COO matrix
 * @param d_alloc device allocator for temporary buffers
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T>
void coo_remove_zeros(COO<T> *in, COO<T> *out,
                      std::shared_ptr<deviceAllocator> d_alloc,
                      cudaStream_t stream) {
  coo_remove_scalar<TPB_X, T>(in, out, T(0.0), d_alloc, stream);
}

template <int TPB_X, typename T>
__global__ void from_knn_graph_kernel(const long *knn_indices,
                                      const T *knn_dists, int m, int k,
                                      int *rows, int *cols, T *vals) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < m) {
    for (int i = 0; i < k; i++) {
      rows[row * k + i] = row;
      cols[row * k + i] = knn_indices[row * k + i];
      vals[row * k + i] = knn_dists[row * k + i];
    }
  }
}

/**
 * @brief Converts a knn graph, defined by index and distance matrices,
 * into COO format.
 *
 * @param knn_indices: knn index array
 * @param knn_dists: knn distance array
 * @param m: number of vertices in graph
 * @param k: number of nearest neighbors
 * @param rows: output COO row array
 * @param cols: output COO col array
 * @param vals: output COO val array
 */
template <typename T>
void from_knn(const long *knn_indices, const T *knn_dists, int m, int k,
              int *rows, int *cols, T *vals) {
  dim3 grid(raft::ceildiv(m, 32), 1, 1);
  dim3 blk(32, 1, 1);
  from_knn_graph_kernel<32, T>
    <<<grid, blk>>>(knn_indices, knn_dists, m, k, rows, cols, vals);
  CUDA_CHECK(cudaGetLastError());
}

/**
 * Converts a knn graph, defined by index and distance matrices,
 * into COO format.
 * @param knn_indices: KNN index array (size m * k)
 * @param knn_dists: KNN dist array (size m * k)
 * @param m: number of vertices in graph
 * @param k: number of nearest neighbors
 * @param out: The output COO graph from the KNN matrices
 * @param stream: CUDA stream to use
 */
template <typename T>
void from_knn(const long *knn_indices, const T *knn_dists, int m, int k,
              COO<T> *out, cudaStream_t stream) {
  out->allocate(m * k, m, m, true, stream);

  from_knn(knn_indices, knn_dists, m, k, out->rows(), out->cols(), out->vals());
}

/**
 * @brief Generate the row indices array for a sorted COO matrix
 *
 * @param rows: COO rows array
 * @param nnz: size of COO rows array
 * @param row_ind: output row indices array
 * @param m: number of rows in dense matrix
 * @param d_alloc device allocator for temporary buffers
 * @param stream: cuda stream to use
 */
template <typename T>
void sorted_coo_to_csr(const T *rows, int nnz, T *row_ind, int m,
                       std::shared_ptr<deviceAllocator> d_alloc,
                       cudaStream_t stream) {
  device_buffer<T> row_counts(d_alloc, stream, m);

  CUDA_CHECK(cudaMemsetAsync(row_counts.data(), 0, m * sizeof(T), stream));

  coo_row_count<32>(rows, nnz, row_counts.data(), stream);

  // create csr compressed row index from row counts
  thrust::device_ptr<T> row_counts_d =
    thrust::device_pointer_cast(row_counts.data());
  thrust::device_ptr<T> c_ind_d = thrust::device_pointer_cast(row_ind);
  exclusive_scan(thrust::cuda::par.on(stream), row_counts_d, row_counts_d + m,
                 c_ind_d);
}

/**
 * @brief Generate the row indices array for a sorted COO matrix
 *
 * @param coo: Input COO matrix
 * @param row_ind: output row indices array
 * @param d_alloc device allocator for temporary buffers
 * @param stream: cuda stream to use
 */
template <typename T>
void sorted_coo_to_csr(COO<T> *coo, int *row_ind,
                       std::shared_ptr<deviceAllocator> d_alloc,
                       cudaStream_t stream) {
  sorted_coo_to_csr(coo->rows(), coo->nnz, row_ind, coo->n_rows, d_alloc,
                    stream);
}

template <int TPB_X, typename T, typename Lambda>
__global__ void coo_symmetrize_kernel(int *row_ind, int *rows, int *cols,
                                      T *vals, int *orows, int *ocols, T *ovals,
                                      int n, int cnnz, Lambda reduction_op) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  if (row < n) {
    int start_idx = row_ind[row];  // each thread processes one row
    int stop_idx = MLCommon::Sparse::get_stop_idx(row, n, cnnz, row_ind);

    int row_nnz = 0;
    int out_start_idx = start_idx * 2;

    for (int idx = 0; idx < stop_idx - start_idx; idx++) {
      int cur_row = rows[idx + start_idx];
      int cur_col = cols[idx + start_idx];
      T cur_val = vals[idx + start_idx];

      int lookup_row = cur_col;
      int t_start = row_ind[lookup_row];  // Start at
      int t_stop = MLCommon::Sparse::get_stop_idx(lookup_row, n, cnnz, row_ind);

      T transpose = 0.0;

      bool found_match = false;
      for (int t_idx = t_start; t_idx < t_stop; t_idx++) {
        // If we find a match, let's get out of the loop. We won't
        // need to modify the transposed value, since that will be
        // done in a different thread.
        if (cols[t_idx] == cur_row && rows[t_idx] == cur_col) {
          // If it exists already, set transposed value to existing value
          transpose = vals[t_idx];
          found_match = true;
          break;
        }
      }

      // Custom reduction op on value and its transpose, which enables
      // specialized weighting.
      // If only simple X+X.T is desired, this op can just sum
      // the two values.
      T res = reduction_op(cur_row, cur_col, cur_val, transpose);

      // if we didn't find an exact match, we need to add
      // the computed res into our current matrix to guarantee
      // symmetry.
      // Note that if we did find a match, we don't need to
      // compute `res` on it here because it will be computed
      // in a different thread.
      if (!found_match && vals[idx] != 0.0) {
        orows[out_start_idx + row_nnz] = cur_col;
        ocols[out_start_idx + row_nnz] = cur_row;
        ovals[out_start_idx + row_nnz] = res;
        ++row_nnz;
      }

      if (res != 0.0) {
        orows[out_start_idx + row_nnz] = cur_row;
        ocols[out_start_idx + row_nnz] = cur_col;
        ovals[out_start_idx + row_nnz] = res;
        ++row_nnz;
      }
    }
  }
}

/**
 * @brief takes a COO matrix which may not be symmetric and symmetrizes
 * it, running a custom reduction function against the each value
 * and its transposed value.
 *
 * @param in: Input COO matrix
 * @param out: Output symmetrized COO matrix
 * @param reduction_op: a custom reduction function
 * @param d_alloc device allocator for temporary buffers
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T, typename Lambda>
void coo_symmetrize(COO<T> *in, COO<T> *out,
                    Lambda reduction_op,  // two-argument reducer
                    std::shared_ptr<deviceAllocator> d_alloc,
                    cudaStream_t stream) {
  dim3 grid(raft::ceildiv(in->n_rows, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  ASSERT(!out->validate_mem(), "Expecting unallocated COO for output");

  device_buffer<int> in_row_ind(d_alloc, stream, in->n_rows);

  sorted_coo_to_csr(in, in_row_ind.data(), d_alloc, stream);

  out->allocate(in->nnz * 2, in->n_rows, in->n_cols, true, stream);

  coo_symmetrize_kernel<TPB_X, T><<<grid, blk, 0, stream>>>(
    in_row_ind.data(), in->rows(), in->cols(), in->vals(), out->rows(),
    out->cols(), out->vals(), in->n_rows, in->nnz, reduction_op);
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Find how much space needed in each row.
 * We look through all datapoints and increment the count for each row.
 *
 * @param data: Input knn distances(n, k)
 * @param indices: Input knn indices(n, k)
 * @param n: Number of rows
 * @param k: Number of n_neighbors
 * @param row_sizes: Input empty row sum 1 array(n)
 * @param row_sizes2: Input empty row sum 2 array(n) for faster reduction
 */
template <typename math_t>
__global__ static void symmetric_find_size(const math_t *restrict data,
                                           const long *restrict indices,
                                           const int n, const int k,
                                           int *restrict row_sizes,
                                           int *restrict row_sizes2) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;  // for every row
  const int j = blockIdx.y * blockDim.y + threadIdx.y;  // for every item in row
  if (row >= n || j >= k) return;

  const int col = indices[row * k + j];
  if (j % 2)
    raft::myAtomicAdd(&row_sizes[col], 1);
  else
    raft::myAtomicAdd(&row_sizes2[col], 1);
}

/**
 * @brief Reduce sum(row_sizes) + k
 * Reduction for symmetric_find_size kernel. Allows algo to be faster.
 *
 * @param n: Number of rows
 * @param k: Number of n_neighbors
 * @param row_sizes: Input row sum 1 array(n)
 * @param row_sizes2: Input row sum 2 array(n) for faster reduction
 */
__global__ static void reduce_find_size(const int n, const int k,
                                        int *restrict row_sizes,
                                        const int *restrict row_sizes2) {
  const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= n) return;
  row_sizes[i] += (row_sizes2[i] + k);
}

/**
 * @brief Perform data + data.T operation.
 * Can only run once row_sizes from the CSR matrix of data + data.T has been
 * determined.
 *
 * @param edges: Input row sum array(n) after reduction
 * @param data: Input knn distances(n, k)
 * @param indices: Input knn indices(n, k)
 * @param VAL: Output values for data + data.T
 * @param COL: Output column indices for data + data.T
 * @param ROW: Output row indices for data + data.T
 * @param n: Number of rows
 * @param k: Number of n_neighbors
 */
template <typename math_t>
__global__ static void symmetric_sum(int *restrict edges,
                                     const math_t *restrict data,
                                     const long *restrict indices,
                                     math_t *restrict VAL, int *restrict COL,
                                     int *restrict ROW, const int n,
                                     const int k) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;  // for every row
  const int j = blockIdx.y * blockDim.y + threadIdx.y;  // for every item in row
  if (row >= n || j >= k) return;

  const int col = indices[row * k + j];
  const int original = atomicAdd(&edges[row], 1);
  const int transpose = atomicAdd(&edges[col], 1);

  VAL[transpose] = VAL[original] = data[row * k + j];
  // Notice swapped ROW, COL since transpose
  ROW[original] = row;
  COL[original] = col;

  ROW[transpose] = col;
  COL[transpose] = row;
}

/**
 * @brief Perform data + data.T on raw KNN data.
 * The following steps are invoked:
 * (1) Find how much space needed in each row
 * (2) Compute final space needed (n*k + sum(row_sizes)) == 2*n*k
 * (3) Allocate new space
 * (4) Prepare edges for each new row
 * (5) Perform final data + data.T operation
 * (6) Return summed up VAL, COL, ROW
 *
 * @param knn_indices: Input knn distances(n, k)
 * @param knn_dists: Input knn indices(n, k)
 * @param n: Number of rows
 * @param k: Number of n_neighbors
 * @param out: Output COO Matrix class
 * @param stream: Input cuda stream
 * @param d_alloc device allocator for temporary buffers
 */
template <typename math_t, int TPB_X = 32, int TPB_Y = 32>
void from_knn_symmetrize_matrix(const long *restrict knn_indices,
                                const math_t *restrict knn_dists, const int n,
                                const int k, COO<math_t> *out,
                                cudaStream_t stream,
                                std::shared_ptr<deviceAllocator> d_alloc) {
  // (1) Find how much space needed in each row
  // We look through all datapoints and increment the count for each row.
  const dim3 threadsPerBlock(TPB_X, TPB_Y);
  const dim3 numBlocks(raft::ceildiv(n, TPB_X), raft::ceildiv(k, TPB_Y));

  // Notice n+1 since we can reuse these arrays for transpose_edges, original_edges in step (4)
  device_buffer<int> row_sizes(d_alloc, stream, n);
  CUDA_CHECK(cudaMemsetAsync(row_sizes.data(), 0, sizeof(int) * n, stream));

  device_buffer<int> row_sizes2(d_alloc, stream, n);
  CUDA_CHECK(cudaMemsetAsync(row_sizes2.data(), 0, sizeof(int) * n, stream));

  symmetric_find_size<<<numBlocks, threadsPerBlock, 0, stream>>>(
    knn_dists, knn_indices, n, k, row_sizes.data(), row_sizes2.data());
  CUDA_CHECK(cudaPeekAtLastError());

  reduce_find_size<<<raft::ceildiv(n, 1024), 1024, 0, stream>>>(
    n, k, row_sizes.data(), row_sizes2.data());
  CUDA_CHECK(cudaPeekAtLastError());

  // (2) Compute final space needed (n*k + sum(row_sizes)) == 2*n*k
  // Notice we don't do any merging and leave the result as 2*NNZ
  const int NNZ = 2 * n * k;

  // (3) Allocate new space
  out->allocate(NNZ, n, n, true, stream);

  // (4) Prepare edges for each new row
  // This mirrors CSR matrix's row Pointer, were maximum bounds for each row
  // are calculated as the cumulative rolling sum of the previous rows.
  // Notice reusing old row_sizes2 memory
  int *edges = row_sizes2.data();
  thrust::device_ptr<int> __edges = thrust::device_pointer_cast(edges);
  thrust::device_ptr<int> __row_sizes =
    thrust::device_pointer_cast(row_sizes.data());

  // Rolling cumulative sum
  thrust::exclusive_scan(thrust::cuda::par.on(stream), __row_sizes,
                         __row_sizes + n, __edges);

  // (5) Perform final data + data.T operation in tandem with memcpying
  symmetric_sum<<<numBlocks, threadsPerBlock, 0, stream>>>(
    edges, knn_dists, knn_indices, out->vals(), out->cols(), out->rows(), n, k);
  CUDA_CHECK(cudaPeekAtLastError());
}

};  // namespace Sparse
};  // namespace MLCommon
