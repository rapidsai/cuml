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

#include "csr.h"

#include "cusparse_wrappers.h"

#include <cusparse_v2.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include "cuda_utils.h"

#include <iostream>

#pragma once

namespace MLCommon {
namespace Sparse {

/** @brief A Container object for sparse coordinate
 * format.
 *
 * @tparam T: the type of the value array.
 *
 */
template <typename T>
class COO {
 protected:
  bool owner;

 public:
  int *rows;
  int *cols;
  T *vals;
  int nnz;
  int n_rows;
  int n_cols;
  bool device;

  /**
        * @param device: are the underlying arrays going to be on device?
        */
  COO(bool device = true)
    : rows(nullptr),
      cols(nullptr),
      vals(nullptr),
      nnz(-1),
      n_rows(-1),
      n_cols(-1),
      device(device),
      owner(true) {}

  /**
        * @param rows: coo rows array
        * @param cols: coo cols array
        * @param vals: coo vals array
        * @param nnz: size of the rows/cols/vals arrays
        * @param n_rows: number of rows in the dense matrix
        * @param n_cols: number of cols in the dense matrix
        * @param device: are the underlying arrays on device?
        */
  COO(int *rows, int *cols, T *vals, int nnz, int n_rows = -1, int n_cols = -1,
      bool device = true)
    : rows(rows),
      cols(cols),
      vals(vals),
      nnz(nnz),
      n_rows(n_rows),
      n_cols(n_cols),
      device(device),
      owner(false) {}

  /**
        * @param nnz: size of the rows/cols/vals arrays
        * @param n_rows: number of rows in the dense matrix
        * @param n_cols: number of cols in the dense matrix
        * @param device: are the underlying arrays on device?
        */
  COO(int nnz, int n_rows = -1, int n_cols = -1, bool device = true,
      bool init = true)
    : rows(nullptr),
      cols(nullptr),
      vals(nullptr),
      nnz(nnz),
      n_rows(n_rows),
      n_cols(n_cols),
      device(device),
      owner(true) {
    this->allocate(nnz, n_rows, n_cols, device, init);
  }

  ~COO() { this->destroy(); }

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
    if (this->rows == nullptr || this->cols == nullptr ||
        this->vals == nullptr) {
      return false;
    }

    return true;
  }

  /**
        * @brief Send human-readable state information to output stream
        */
  friend std::ostream &operator<<(std::ostream &out, const COO<T> &c) {
    if (c.validate_size() && c.validate_mem()) {
      cudaStream_t stream;
      cudaStreamCreate(&stream);

      out << arr2Str(c.rows, c.nnz, "rows", stream) << std::endl;
      out << arr2Str(c.cols, c.nnz, "cols", stream) << std::endl;
      out << arr2Str(c.vals, c.nnz, "vals", stream) << std::endl;
      out << "nnz=" << c.nnz << std::endl;
      out << "n_rows=" << c.n_rows << std::endl;
      out << "n_cols=" << c.n_cols << std::endl;
      out << "owner=" << c.owner << std::endl;

      cudaStreamDestroy(stream);
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
        * @param device: allocate on device or host?
        * @param init: should values be initialized to 0?
        */
  void allocate(int nnz, bool device = true, bool init = true) {
    this->allocate(nnz, -1, device, init);
  }

  /**
        * @brief Allocate the underlying arrays
        * @param nnz: size of the underlying row/col/val arrays
        * @param size: the number of rows/cols in a square dense matrix
        * @param device: allocate on device or host?
        * @param init: should values be initialized to 0?
        */
  void allocate(int nnz, int size, bool device = true, bool init = true) {
    this->allocate(nnz, size, size, device, init);
  }

  /**
        * @brief Allocate the underlying arrays
        * @param nnz: size of the underlying row/col/val arrays
        * @param n_rows: number of rows in the dense matrix
        * @param n_cols: number of columns in the dense matrix
        * @param device: allocate on device or host?
        * @param init: should values be initialized to 0?
        */
  void allocate(int nnz, int n_rows, int n_cols, bool device = true,
                bool init = true) {
    this->n_rows = n_rows;
    this->n_cols = n_cols;
    this->nnz = nnz;
    this->owner = true;

    if (device) {
      MLCommon::allocate(this->rows, this->nnz, init);
      MLCommon::allocate(this->cols, this->nnz, init);
      MLCommon::allocate(this->vals, this->nnz, init);
    } else {
      this->rows = (int *)malloc(this->nnz * sizeof(int));
      this->cols = (int *)malloc(this->nnz * sizeof(int));
      this->vals = (T *)malloc(this->nnz * sizeof(T));
    }
  }

  /**
        * @brief Deallocate the underlying arrays if this object
        * owns the underlying memory
        */
  void destroy() {
    if (this->owner) {
      try {
        if (rows != nullptr) {
          if (this->device)
            CUDA_CHECK(cudaFree(rows));
          else
            free(rows);
        }

        if (cols != nullptr) {
          if (this->device)
            CUDA_CHECK(cudaFree(cols));
          else
            free(cols);
        }

        if (vals != nullptr) {
          if (this->device)
            CUDA_CHECK(cudaFree(vals));
          else
            free(vals);
        }

        rows = nullptr;
        cols = nullptr;
        vals = nullptr;

      } catch (Exception &e) {
        std::cout << "An exception occurred freeing COO memory" << std::endl;
      }
    }
  }
};

template <typename T>
cusparseStatus_t cusparse_gthr(cusparseHandle_t handle, int nnz, float *vals,
                               float *vals_sorted, int *d_P) {
  return cusparseSgthr(handle, nnz, vals, vals_sorted, d_P,
                       CUSPARSE_INDEX_BASE_ZERO);
}

template <typename T>
cusparseStatus_t cusparse_gthr(cusparseHandle_t handle, int nnz, double *vals,
                               double *vals_sorted, int *d_P) {
  return cusparseDgthr(handle, nnz, vals, vals_sorted, d_P,
                       CUSPARSE_INDEX_BASE_ZERO);
}

/**
 * @brief Sorts the arrays that comprise the coo matrix
 * by row.
 *
 * @param m number of rows in coo matrix
 * @param n number of cols in coo matrix
 * @param rows rows array from coo matrix
 * @param cols cols array from coo matrix
 * @param vals vals array from coo matrix
 * @param stream: cuda stream to use
 */
template <typename T>
void coo_sort(int m, int n, int nnz, int *rows, int *cols, T *vals,
              cudaStream_t stream = 0) {
  cusparseHandle_t handle = NULL;

  size_t pBufferSizeInBytes = 0;
  void *pBuffer = NULL;
  int *d_P = NULL;

  CUSPARSE_CHECK(cusparseCreate(&handle));

  CUSPARSE_CHECK(cusparseSetStream(handle, stream));

  CUSPARSE_CHECK(cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, rows, cols,
                                                &pBufferSizeInBytes));

  allocate(d_P, nnz);
  cudaMalloc(&pBuffer, pBufferSizeInBytes * sizeof(char));

  CUSPARSE_CHECK(cusparseCreateIdentityPermutation(handle, nnz, d_P));

  CUSPARSE_CHECK(
    cusparseXcoosortByRow(handle, m, n, nnz, rows, cols, d_P, pBuffer));

  T *vals_sorted;
  allocate(vals_sorted, nnz);

  CUSPARSE_CHECK(cusparse_gthr<T>(handle, nnz, vals, vals_sorted, d_P));

  cudaDeviceSynchronize();

  copy(vals, vals_sorted, nnz, stream);

  cudaFree(d_P);
  cudaFree(vals_sorted);
  cudaFree(pBuffer);
  CUSPARSE_CHECK(cusparseDestroy(handle));
}

/**
 * @brief Sort the underlying COO arrays by row
 * @tparam T: the type name of the underlying value array
 * @param in: COO to sort by row
 * @param stream: the cuda stream to use
 */
template <typename T>
void coo_sort(COO<T> *const in, cudaStream_t stream = 0) {
  coo_sort<T>(in->n_rows, in->n_cols, in->nnz, in->rows, in->cols, in->vals,
              stream);
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
__global__ void coo_row_count_kernel(int *const rows, int nnz, int *results) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < nnz) {
    atomicAdd(results + rows[row], 1);
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
void coo_row_count(int *const rows, int nnz, int *results,
                   cudaStream_t stream) {
  dim3 grid_rc(MLCommon::ceildiv(nnz, TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);

  coo_row_count_kernel<TPB_X>
    <<<grid_rc, blk_rc, 0, stream>>>(rows, nnz, results);
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
void coo_row_count(COO<T> *const in, int *results, cudaStream_t stream = 0) {
  dim3 grid_rc(MLCommon::ceildiv(in->nnz, TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);

  coo_row_count_kernel<TPB_X>
    <<<grid_rc, blk_rc, 0, stream>>>(in->rows, in->nnz, results);
}

template <int TPB_X, typename T>
__global__ void coo_row_count_nz_kernel(int *const rows, T *const vals, int nnz,
                                        int *results) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < nnz && vals[row] != 0.0) {
    atomicAdd(results + rows[row], 1);
  }
}

template <int TPB_X, typename T>
__global__ void coo_row_count_scalar_kernel(int *const rows, T *const vals,
                                            int nnz, T scalar, int *results) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < nnz && vals[row] != scalar) {
    atomicAdd(results + rows[row], 1);
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
void coo_row_count_scalar(COO<T> *const in, T scalar, int *results,
                          cudaStream_t stream = 0) {
  dim3 grid_rc(MLCommon::ceildiv(in->nnz, TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);

  coo_row_count_scalar_kernel<TPB_X, T><<<grid_rc, blk_rc, 0, stream>>>(
    in->rows, in->vals, in->nnz, scalar, results);
}

/**
 * @brief Count the number of values for each row matching a particular scalar
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: the type name of the underlying value arrays
 * @param rows: Input COO row array
 * @param cols: Input COO col array
 * @param vals: Input COO val arrays
 * @param nnz: size of input COO arrays
 * @param scalar: scalar to match for counting rows
 * @param results: output row counts
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T>
void coo_row_count_scalar(int *const rows, T *const vals, int nnz, T scalar,
                          int *results, cudaStream_t stream = 0) {
  dim3 grid_rc(MLCommon::ceildiv(nnz, TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);

  coo_row_count_scalar_kernel<TPB_X, T>
    <<<grid_rc, blk_rc, 0, stream>>>(rows, vals, nnz, scalar, results);
}

/**
 * @brief Count the number of nonzeros for each row
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: the type name of the underlying value arrays
 * @param rows: Input COO row array
 * @param cols: Input COO col array
 * @param vals: Input COO val arrays
 * @param nnz: size of input COO arrays
 * @param results: output row counts
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T>
void coo_row_count_nz(int *const rows, T *const vals, int nnz, int *results,
                      cudaStream_t stream = 0) {
  dim3 grid_rc(MLCommon::ceildiv(nnz, TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);

  coo_row_count_nz_kernel<TPB_X, T>
    <<<grid_rc, blk_rc, 0, stream>>>(rows, vals, nnz, results);
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
void coo_row_count_nz(COO<T> *const in, int *results, cudaStream_t stream = 0) {
  dim3 grid_rc(MLCommon::ceildiv(in->nnz, TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);

  coo_row_count_nz_kernel<TPB_X, T>
    <<<grid_rc, blk_rc, 0, stream>>>(in->rows, in->vals, in->nnz, results);
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
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T>
void coo_remove_scalar(const int *rows, const int *cols, const T *vals, int nnz,
                       int *crows, int *ccols, T *cvals, int *cnnz,
                       int *cur_cnnz, T scalar, int n, cudaStream_t stream) {
  int *ex_scan, *cur_ex_scan;
  MLCommon::allocate(ex_scan, n, true);
  MLCommon::allocate(cur_ex_scan, n, true);

  thrust::device_ptr<int> dev_cnnz = thrust::device_pointer_cast(cnnz);
  thrust::device_ptr<int> dev_ex_scan = thrust::device_pointer_cast(ex_scan);
  thrust::exclusive_scan(thrust::cuda::par.on(stream), dev_cnnz, dev_cnnz + n,
                         dev_ex_scan);
  CUDA_CHECK(cudaPeekAtLastError());

  thrust::device_ptr<int> dev_cur_cnnz = thrust::device_pointer_cast(cur_cnnz);
  thrust::device_ptr<int> dev_cur_ex_scan =
    thrust::device_pointer_cast(cur_ex_scan);
  thrust::exclusive_scan(thrust::cuda::par.on(stream), dev_cur_cnnz,
                         dev_cur_cnnz + n, dev_cur_ex_scan);
  CUDA_CHECK(cudaPeekAtLastError());

  dim3 grid(ceildiv(n, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  coo_remove_scalar_kernel<TPB_X><<<grid, blk, 0, stream>>>(
    rows, cols, vals, nnz, crows, ccols, cvals, dev_ex_scan.get(),
    dev_cur_ex_scan.get(), n, scalar);
  CUDA_CHECK(cudaPeekAtLastError());

  CUDA_CHECK(cudaFree(ex_scan));
  CUDA_CHECK(cudaFree(cur_ex_scan));
}

/**
 * @brief Removes the values matching a particular scalar from a COO formatted sparse matrix.
 *
 * @param in: input COO matrix
 * @param out: output COO matrix
 * @param scalar: scalar to remove from arrays
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T>
void coo_remove_scalar(COO<T> *const in, COO<T> *out, T scalar,
                       cudaStream_t stream) {
  int *row_count_nz, *row_count;

  MLCommon::allocate(row_count, in->n_rows, true);
  MLCommon::allocate(row_count_nz, in->n_rows, true);

  MLCommon::Sparse::coo_row_count<TPB_X>(in->rows, in->nnz, row_count, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  MLCommon::Sparse::coo_row_count_scalar<TPB_X>(in->rows, in->vals, in->nnz,
                                                scalar, row_count_nz, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));

  thrust::device_ptr<int> d_row_count_nz =
    thrust::device_pointer_cast(row_count_nz);
  int out_nnz = thrust::reduce(thrust::cuda::par.on(stream), d_row_count_nz,
                               d_row_count_nz + in->n_rows);

  out->allocate(out_nnz, in->n_rows, in->n_cols);

  coo_remove_scalar<TPB_X, T>(in->rows, in->cols, in->vals, in->nnz, out->rows,
                              out->cols, out->vals, row_count_nz, row_count,
                              scalar, in->n_rows, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  CUDA_CHECK(cudaFree(row_count));
  CUDA_CHECK(cudaFree(row_count_nz));
}

/**
 * @brief Removes zeros from a COO formatted sparse matrix.
 *
 * @param in: input COO matrix
 * @param out: output COO matrix
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T>
void coo_remove_zeros(COO<T> *const in, COO<T> *out, cudaStream_t stream) {
  coo_remove_scalar<TPB_X, T>(in, out, T(0.0), stream);
}

template <int TPB_X, typename T>
__global__ void from_knn_graph_kernel(long *const knn_indices,
                                      T *const knn_dists, int m, int k,
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
void from_knn(long *const knn_indices, T *const knn_dists, int m, int k,
              int *rows, int *cols, T *vals) {
  dim3 grid(ceildiv(m, 32), 1, 1);
  dim3 blk(32, 1, 1);
  from_knn_graph_kernel<32, T>
    <<<grid, blk>>>(knn_indices, knn_dists, m, k, rows, cols, vals);
}

/**
 * Converts a knn graph, defined by index and distance matrices,
 * into COO format.
 */
template <typename T>
void from_knn(long *const knn_indices, T *const knn_dists, int m, int k,
              COO<T> *out) {
  out->allocate(m * k, m, m);

  from_knn(knn_indices, knn_dists, m, k, out->rows, out->cols, out->vals);
}

/**
 * @brief Generate the row indices array for a sorted COO matrix
 *
 * @param rows: COO rows array
 * @param nnz: size of COO rows array
 * @param row_ind: output row indices array
 * @param m: number of rows in dense matrix
 * @param stream: cuda stream to use
 */
template <typename T>
void sorted_coo_to_csr(T *const rows, int nnz, T *row_ind, int m,
                       cudaStream_t stream = 0) {
  T *row_counts;
  MLCommon::allocate(row_counts, m, true);

  dim3 grid(ceildiv(m, 32), 1, 1);
  dim3 blk(32, 1, 1);

  coo_row_count<32>(rows, nnz, row_counts, stream);

  // create csr compressed row index from row counts
  thrust::device_ptr<T> row_counts_d = thrust::device_pointer_cast(row_counts);
  thrust::device_ptr<T> c_ind_d = thrust::device_pointer_cast(row_ind);
  exclusive_scan(thrust::cuda::par.on(stream), row_counts_d, row_counts_d + m,
                 c_ind_d);

  CUDA_CHECK(cudaFree(row_counts));
}

/**
 * @brief Generate the row indices array for a sorted COO matrix
 *
 * @param coo: Input COO matrix
 * @param row_ind: output row indices array
 * @param stream: cuda stream to use
 */
template <typename T>
void sorted_coo_to_csr(COO<T> *const coo, int *row_ind,
                       cudaStream_t stream = 0) {
  sorted_coo_to_csr(coo->rows, coo->nnz, row_ind, coo->n_rows, stream);
}

template <int TPB_X, typename T, typename Lambda>
__global__ void coo_symmetrize_kernel(int *row_ind, int *rows, int *cols,
                                      T *vals, int *orows, int *ocols, T *ovals,
                                      int n, int cnnz, Lambda reduction_op) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  if (row < n) {
    int start_idx = row_ind[row];  // each thread processes one row
    int stop_idx = MLCommon::Sparse::get_stop_idx(row, n, cnnz, row_ind);

    int nnz = 0;
    for (int idx = 0; idx < stop_idx - start_idx; idx++) {
      int out_idx = start_idx * 2 + nnz;
      int row_lookup = cols[idx + start_idx];
      int t_start = row_ind[row_lookup];  // Start at
      int t_stop = MLCommon::Sparse::get_stop_idx(row_lookup, n, cnnz, row_ind);

      T transpose = 0.0;
      bool found_match = false;
      for (int t_idx = t_start; t_idx < t_stop; t_idx++) {
        // If we find a match, let's get out of the loop
        if (cols[t_idx] == rows[idx + start_idx] &&
            rows[t_idx] == cols[idx + start_idx] && vals[t_idx] != 0.0) {
          transpose = vals[t_idx];
          found_match = true;
          break;
        }
      }

      // if we didn't find an exact match, we need to add
      // the transposed value into our current matrix.
      if (!found_match && vals[idx] != 0.0) {
        orows[out_idx + nnz] = cols[idx + start_idx];
        ocols[out_idx + nnz] = rows[idx + start_idx];
        ovals[out_idx + nnz] = vals[idx + start_idx];
        ++nnz;
      }

      T val = vals[idx + start_idx];

      // Custom reduction op on value and its transpose
      T res = reduction_op(rows[idx + start_idx], cols[idx + start_idx], val,
                           transpose);

      if (res != 0.0) {
        orows[out_idx + nnz] = rows[idx + start_idx];
        ocols[out_idx + nnz] = cols[idx + start_idx];
        ovals[out_idx + nnz] = T(res);
        ++nnz;
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
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T, typename Lambda>
void coo_symmetrize(COO<T> *const in, COO<T> *out,
                    Lambda reduction_op,  // two-argument reducer
                    cudaStream_t stream) {
  dim3 grid(ceildiv(in->n_rows, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  ASSERT(!out->validate_mem(), "Expecting unallocated COO for output");

  int *in_row_ind;
  MLCommon::allocate(in_row_ind, in->n_rows);

  sorted_coo_to_csr(in, in_row_ind, stream);

  out->allocate(in->nnz * 2, in->n_rows, in->n_cols);

  coo_symmetrize_kernel<TPB_X, T><<<grid, blk, 0, stream>>>(
    in_row_ind, in->rows, in->cols, in->vals, out->rows, out->cols, out->vals,
    in->n_rows, in->nnz, reduction_op);
  CUDA_CHECK(cudaPeekAtLastError());
}

};  // namespace Sparse
};  // namespace MLCommon
