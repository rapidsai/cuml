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

#include <cuml/common/cuml_allocator.hpp>
#include "csr.h"
#include "linalg/unary_op.h"

#include "cusparse_wrappers.h"

#include <cusparse_v2.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include "cuda_utils.h"

#include <iostream>
#define restrict __restrict

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
        std::cout << "An exception occurred freeing COO memory: " << e.what()
                  << std::endl;
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
__global__ void coo_row_count_kernel(const int *rows, int nnz, int *results) {
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
void coo_row_count(const int *rows, int nnz, int *results,
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
void coo_row_count(const COO<T> *in, int *results, cudaStream_t stream = 0) {
  dim3 grid_rc(MLCommon::ceildiv(in->nnz, TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);

  coo_row_count_kernel<TPB_X>
    <<<grid_rc, blk_rc, 0, stream>>>(in->rows, in->nnz, results);
}

template <int TPB_X, typename T>
__global__ void coo_row_count_nz_kernel(const int *rows, const T *vals, int nnz,
                                        int *results) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < nnz && vals[row] != 0.0) {
    atomicAdd(results + rows[row], 1);
  }
}

template <int TPB_X, typename T>
__global__ void coo_row_count_scalar_kernel(const int *rows, const T *vals,
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
void coo_row_count_scalar(const COO<T> *in, T scalar, int *results,
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
void coo_row_count_scalar(const int *rows, const T *vals, int nnz, T scalar,
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
void coo_row_count_nz(const int *rows, const T *vals, int nnz, int *results,
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
void coo_row_count_nz(const COO<T> *in, int *results, cudaStream_t stream = 0) {
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
void coo_remove_scalar(const COO<T> *in, COO<T> *out, T scalar,
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
void coo_remove_zeros(const COO<T> *in, COO<T> *out, cudaStream_t stream) {
  coo_remove_scalar<TPB_X, T>(in, out, T(0.0), stream);
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
void from_knn(const long *knn_indices, const T *knn_dists, int m, int k,
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
void sorted_coo_to_csr(const T *rows, int nnz, T *row_ind, int m,
                       cudaStream_t stream = 0) {
  T *row_counts;
  MLCommon::allocate(row_counts, m, true);

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
void sorted_coo_to_csr(const COO<T> *coo, int *row_ind,
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
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T, typename Lambda>
void coo_symmetrize(const COO<T> *in, COO<T> *out,
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
  const int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in row
  const int row = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every row
  if (row >= n || j >= k) return;

  const int col = indices[row * k + j];
  if (j % 2)
    atomicAdd(&row_sizes[col], 1);
  else
    atomicAdd(&row_sizes2[col], 1);
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
                                        const int *restrict row_sizes2)
{
  const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= n) return;
  row_sizes[i] += (row_sizes2[i]);
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
                                     math_t *restrict VAL,
                                     int *restrict COL,
                                     int *restrict ROW, const int n,
                                     const int k)
{
  const int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in row
  const int row = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every row
  if (row >= n || j >= k) return;

  const int index = row * k + j;
  const int col = indices[index];
  // const int original = atomicAdd(&edges[row], 1);

  // Notice swapped ROW, COL since transpose
  // ROW[original] = row;
  // COL[original] = col;
  ROW[index] = row;
  COL[index] = col;

  const int transpose = atomicAdd(&edges[col], 1);
  VAL[transpose] /*= VAL[original]*/ = data[index];
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
 */
template <typename math_t, int TPB_X = 32, int TPB_Y = 32>
void from_knn_symmetrize_matrix(const long *restrict knn_indices,
                                const math_t *restrict knn_dists,
                                const int n,
                                const int k,
                                /* math_t *restrict out, */
                                math_t *restrict VAL,
                                int *restrict COL,
                                int *restrict ROW,
                                int *restrict row_sizes,
                                cudaStream_t stream,
                                std::shared_ptr<deviceAllocator> d_alloc)
{
  // (1) Find how much space needed in each row
  // We look through all datapoints and increment the count for each row.
  const dim3 threadsPerBlock(TPB_X, TPB_Y);
  const dim3 numBlocks(MLCommon::ceildiv(k, TPB_X),
                       MLCommon::ceildiv(n, TPB_Y));

  // Notice n+1 since we can reuse these arrays for transpose_edges, original_edges in step (4)
  int *row_sizes1, *row_sizes2;
  if (row_sizes == NULL) {
    row_sizes1 = (int*)d_alloc->allocate(sizeof(int)*n*2, stream);
    row_sizes2 = row_sizes1 + n;
  }
  else {
    row_sizes1 = row_sizes;
    row_sizes2 = row_sizes1 + n;
  }
  CUDA_CHECK(cudaMemsetAsync(row_sizes1, 0, sizeof(int)*n*2, stream));

  symmetric_find_size<<<numBlocks, threadsPerBlock, 0, stream>>>(
    knn_dists, knn_indices, n, k, row_sizes1, row_sizes2);
  CUDA_CHECK(cudaPeekAtLastError());

  reduce_find_size<<<MLCommon::ceildiv(n, 1024), 1024, 0, stream>>>(
    n, k, row_sizes1, row_sizes2);
  CUDA_CHECK(cudaPeekAtLastError());

  // (2) Compute final space needed (n*k + sum(row_sizes1)) == 2*n*k
  // Notice we don't do any merging and leave the result as 2*NNZ
  // const int NNZ = 2 * n * k;

  // // (3) Allocate new space
  // out->allocate(NNZ, n, n);

  // (4) Prepare edges for each new row
  // This mirrors CSR matrix's row Pointer, were maximum bounds for each row
  // are calculated as the cumulative rolling sum of the previous rows.
  // Notice reusing old row_sizes2 memory
  int *edges = row_sizes2;
  thrust::device_ptr<int> __edges = thrust::device_pointer_cast(edges);
  thrust::device_ptr<int> __row_sizes = thrust::device_pointer_cast(row_sizes1);

  // Rolling cumulative sum
  thrust::exclusive_scan(thrust::cuda::par.on(stream), __row_sizes,
                         __row_sizes + n, __edges);

  const int nk = n*k;
  LinAlg::unaryOp(edges, edges, n, [nk] __device__(int x) { return x + nk; }, stream);

  // Set last to NNZ only if CSR needed
  // CUDA_CHECK(cudaMemcpy(edges + n, &NNZ, sizeof(int), cudaMemcpyHostToDevice));

  // (5) Perform final data + data.T operation in tandem with memcpying
  symmetric_sum<<<numBlocks, threadsPerBlock, 0, stream>>>(
    edges, knn_dists, knn_indices, VAL, COL, ROW, n, k);
  CUDA_CHECK(cudaPeekAtLastError());

  if (row_sizes == NULL)
    d_alloc->deallocate(row_sizes1, sizeof(int)*n*2, stream);
}

};  // namespace Sparse
};  // namespace MLCommon

#undef restrict
