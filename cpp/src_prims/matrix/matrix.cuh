/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <raft/linalg/cublas_wrappers.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <cstddef>
#include <cuda_utils.cuh>

namespace MLCommon {
namespace Matrix {

using namespace std;

/**
 * @brief Copy selected rows of the input matrix into contiguous space.
 *
 * On exit out[i + k*n_rows] = in[indices[i] + k*n_rows],
 * where i = 0..n_rows_indices-1, and k = 0..n_cols-1.
 *
 * @param in input matrix
 * @param n_rows number of rows of output matrix
 * @param n_cols number of columns of output matrix
 * @param out output matrix
 * @param indices of the rows to be copied
 * @param n_rows_indices number of rows to copy
 * @param stream cuda stream
 * @param rowMajor whether the matrix has row major layout
 */
template <typename m_t>
void copyRows(const m_t *in, int n_rows, int n_cols, m_t *out,
              const int *indices, int n_rows_indices, cudaStream_t stream,
              bool rowMajor = false) {
  if (rowMajor) {
    ASSERT(false, "matrix.h: row major is not supported yet!");
  }

  auto size = n_rows_indices * n_cols;
  auto counting = thrust::make_counting_iterator<int>(0);

  thrust::for_each(thrust::cuda::par.on(stream), counting, counting + size,
                   [=] __device__(int idx) {
                     int row = idx % n_rows_indices;
                     int col = idx / n_rows_indices;

                     out[col * n_rows_indices + row] =
                       in[col * n_rows + indices[row]];
                   });
}

/**
 * @brief copy matrix operation for column major matrices.
 * @param in: input matrix
 * @param out: output matrix
 * @param n_rows: number of rows of output matrix
 * @param n_cols: number of columns of output matrix
 * @param stream: cuda stream
 */
template <typename m_t>
void copy(const m_t *in, m_t *out, int n_rows, int n_cols,
          cudaStream_t stream) {
  copyAsync(out, in, n_rows * n_cols, stream);
}

/**
 * @brief copy matrix operation for column major matrices. First n_rows and
 * n_cols of input matrix "in" is copied to "out" matrix.
 * @param in: input matrix
 * @param in_n_rows: number of rows of input matrix
 * @param out: output matrix
 * @param out_n_rows: number of rows of output matrix
 * @param out_n_cols: number of columns of output matrix
 * @param stream: cuda stream
 */
template <typename m_t>
void truncZeroOrigin(m_t *in, int in_n_rows, m_t *out, int out_n_rows,
                     int out_n_cols, cudaStream_t stream) {
  auto m = out_n_rows;
  auto k = in_n_rows;
  auto size = out_n_rows * out_n_cols;
  auto d_q = in;
  auto d_q_trunc = out;
  auto counting = thrust::make_counting_iterator<int>(0);

  thrust::for_each(thrust::cuda::par.on(stream), counting, counting + size,
                   [=] __device__(int idx) {
                     int row = idx % m;
                     int col = idx / m;
                     d_q_trunc[col * m + row] = d_q[col * k + row];
                   });
}

/**
 * @brief Columns of a column major matrix is reversed (i.e. first column and
 * last column are swapped)
 * @param inout: input and output matrix
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param stream: cuda stream
 */
template <typename m_t>
void colReverse(m_t *inout, int n_rows, int n_cols, cudaStream_t stream) {
  auto n = n_cols;
  auto m = n_rows;
  auto size = n_rows * n_cols;
  auto d_q = inout;
  auto d_q_reversed = inout;
  auto counting = thrust::make_counting_iterator<int>(0);

  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + (size / 2), [=] __device__(int idx) {
                     int dest_row = idx % m;
                     int dest_col = idx / m;
                     int src_row = dest_row;
                     int src_col = (n - dest_col) - 1;
                     m_t temp = (m_t)d_q_reversed[idx];
                     d_q_reversed[idx] = d_q[src_col * m + src_row];
                     d_q[src_col * m + src_row] = temp;
                   });
}

/**
 * @brief Rows of a column major matrix is reversed (i.e. first row and last
 * row are swapped)
 * @param inout: input and output matrix
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param stream: cuda stream
 */
template <typename m_t>
void rowReverse(m_t *inout, int n_rows, int n_cols, cudaStream_t stream) {
  auto m = n_rows;
  auto size = n_rows * n_cols;
  auto d_q = inout;
  auto d_q_reversed = inout;
  auto counting = thrust::make_counting_iterator<int>(0);

  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + (size / 2), [=] __device__(int idx) {
                     int dest_row = idx % m;
                     int dest_col = idx / m;
                     int src_row = (m - dest_row) - 1;
                     ;
                     int src_col = dest_col;

                     m_t temp = (m_t)d_q_reversed[idx];
                     d_q_reversed[idx] = d_q[src_col * m + src_row];
                     d_q[src_col * m + src_row] = temp;
                   });
}

/**
 * @brief Prints the data stored in GPU memory
 * @param in: input matrix
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param h_separator: horizontal separator character
 * @param v_separator: vertical separator character
 */
template <typename m_t>
void print(const m_t *in, int n_rows, int n_cols, char h_separator = ' ',
           char v_separator = '\n') {
  std::vector<m_t> h_matrix = std::vector<m_t>(n_cols * n_rows);
  CUDA_CHECK(cudaMemcpy(h_matrix.data(), in, n_cols * n_rows * sizeof(m_t),
                        cudaMemcpyDeviceToHost));

  for (auto i = 0; i < n_rows; i++) {
    for (auto j = 0; j < n_cols; j++) {
      printf("%1.4f%c", h_matrix[j * n_rows + i],
             j < n_cols - 1 ? h_separator : v_separator);
    }
  }
}

/**
 * @brief Prints the data stored in CPU memory
 * @param in: input matrix
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 */
template <typename m_t>
void printHost(const m_t *in, int n_rows, int n_cols) {
  for (auto i = 0; i < n_rows; i++) {
    for (auto j = 0; j < n_cols; j++) {
      printf("%1.4f ", in[j * n_rows + i]);
    }
    printf("\n");
  }
}

/**
 * @brief Kernel for copying a slice of a big matrix to a small matrix with a
 * size matches that slice
 * @param src_d: input matrix
 * @param m: number of rows of input matrix
 * @param n: number of columns of input matrix
 * @param dst_d: output matrix
 * @param x1, y1: coordinate of the top-left point of the wanted area (0-based)
 * @param x2, y2: coordinate of the bottom-right point of the wanted area
 * (1-based)
 */
template <typename m_t>
__global__ void slice(m_t *src_d, int m, int n, m_t *dst_d, int x1, int y1,
                      int x2, int y2) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int dm = x2 - x1, dn = y2 - y1;
  if (idx < dm * dn) {
    int i = idx % dm, j = idx / dm;
    int is = i + x1, js = j + y1;
    dst_d[idx] = src_d[is + js * m];
  }
}

/**
 * @brief Slice a matrix (in-place)
 * @param in: input matrix
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param out: output matrix
 * @param x1, y1: coordinate of the top-left point of the wanted area (0-based)
 * @param x2, y2: coordinate of the bottom-right point of the wanted area
 * (1-based)
 * example: Slice the 2nd and 3rd columns of a 4x3 matrix: slice_matrix(M_d, 4,
 * 3, 0, 1, 4, 3);
 * @param stream: cuda stream
 */
template <typename m_t>
void sliceMatrix(m_t *in, int n_rows, int n_cols, m_t *out, int x1, int y1,
                 int x2, int y2, cudaStream_t stream) {
  // Slicing
  dim3 block(64);
  dim3 grid(((x2 - x1) * (y2 - y1) + block.x - 1) / block.x);
  slice<<<grid, block, 0, stream>>>(in, n_rows, n_cols, out, x1, y1, x2, y2);
}

/**
 * @brief Kernel for copying the upper triangular part of a matrix to another
 * @param src: input matrix with a size of mxn
 * @param dst: output matrix with a size of kxk
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param k: min(n_rows, n_cols)
 */
template <typename m_t>
__global__ void getUpperTriangular(m_t *src, m_t *dst, int n_rows, int n_cols,
                                   int k) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int m = n_rows, n = n_cols;
  if (idx < m * n) {
    int i = idx % m, j = idx / m;
    if (i < k && j < k && j >= i) {
      dst[i + j * k] = src[idx];
    }
  }
}

/**
 * @brief Copy the upper triangular part of a matrix to another
 * @param src: input matrix with a size of n_rows x n_cols
 * @param dst: output matrix with a size of kxk, k = min(n_rows, n_cols)
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param stream: cuda stream
 */
template <typename m_t>
void copyUpperTriangular(m_t *src, m_t *dst, int n_rows, int n_cols,
                         cudaStream_t stream) {
  int m = n_rows, n = n_cols;
  int k = min(m, n);
  dim3 block(64);
  dim3 grid((m * n + block.x - 1) / block.x);
  getUpperTriangular<<<grid, block, 0, stream>>>(src, dst, m, n, k);
}

/**
 * @brief Copy a vector to the diagonal of a matrix
 * @param vec: vector of length k = min(n_rows, n_cols)
 * @param matrix: matrix of size n_rows x n_cols
 * @param m: number of rows of the matrix
 * @param n: number of columns of the matrix
 * @param k: dimensionality
 */
template <typename m_t>
__global__ void copyVectorToMatrixDiagonal(m_t *vec, m_t *matrix, int m, int n,
                                           int k) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < k) {
    matrix[idx + idx * m] = vec[idx];
  }
}

/**
 * @brief Initialize a diagonal matrix with a vector
 * @param vec: vector of length k = min(n_rows, n_cols)
 * @param matrix: matrix of size n_rows x n_cols
 * @param n_rows: number of rows of the matrix
 * @param n_cols: number of columns of the matrix
 * @param stream: cuda stream
 */
template <typename m_t>
void initializeDiagonalMatrix(m_t *vec, m_t *matrix, int n_rows, int n_cols,
                              cudaStream_t stream) {
  int k = min(n_rows, n_cols);
  dim3 block(64);
  dim3 grid((k + block.x - 1) / block.x);
  copyVectorToMatrixDiagonal<<<grid, block, 0, stream>>>(vec, matrix, n_rows,
                                                         n_cols, k);
}

/**
 * @brief Calculate the inverse of the diagonal of a square matrix
 * element-wise and in place
 * @param in: square input matrix with size len x len
 * @param len: size of one side of the matrix
 */
template <typename m_t>
__global__ void matrixDiagonalInverse(m_t *in, int len) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < len) {
    in[idx + idx * len] = 1.0 / in[idx + idx * len];
  }
}

/**
 * @brief Get a square matrix with elements on diagonal reversed (in-place)
 * @param in: square input matrix with size len x len
 * @param len: size of one side of the matrix
 * @param stream: cuda stream
 */
template <typename m_t>
void getDiagonalInverseMatrix(m_t *in, int len, cudaStream_t stream) {
  dim3 block(64);
  dim3 grid((len + block.x - 1) / block.x);
  matrixDiagonalInverse<m_t><<<grid, block, 0, stream>>>(in, len);
}

/**
 * @brief Get the L2/F-norm of a matrix/vector
 * @param in: input matrix/vector with totally size elements
 * @param size: size of the matrix/vector
 * @param cublasH cublas handle
 * @param stream: cuda stream
 */
template <typename m_t>
m_t getL2Norm(m_t *in, int size, cublasHandle_t cublasH, cudaStream_t stream) {
  m_t normval = 0;
  CUBLAS_CHECK(
    raft::linalg::cublasnrm2(cublasH, size, in, 1, &normval, stream));
  return normval;
}

};  // end namespace Matrix
};  // end namespace MLCommon
