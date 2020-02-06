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

#include <algorithm>
#include <memory>
#include <vector>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuml/common/utils.hpp>
#include <cuml/cuml.hpp>

#include "linalg/batched/matrix.h"
#include "linalg/cusolver_wrappers.h"
#include "matrix/matrix.h"

namespace MLCommon {
namespace Sparse {
namespace Batched {

/**
 * Kernel to construct batched CSR sparse matrices from batched dense matrices
 * 
 * @note The construction could coalesce writes to the values array if we
 *       stored a mix of COO and CSR, but the performance gain is not
 *       significant enough to justify complexifying the class.
 * 
 * @param[in]  dense      Batched dense matrices. Size: m * n * batch_size
 * @param[in]  col_index  CSR column index.       Size: nnz
 * @param[in]  row_index  CSR row index.          Size: m + 1
 * @param[out] values     CSR values array.       Size: nnz * batch_size
 * @param[in]  batch_size Number of matrices in the batch
 * @param[in]  m          Number of rows per matrix
 * @param[in]  n          Number of columns per matrix
 * @param[in]  nnz        Number of non-zero elements in each matrix
 */
template <typename T>
static __global__ void dense_to_csr_kernel(const T* dense, const int* col_index,
                                           const int* row_index, T* values,
                                           int batch_size, int m, int n,
                                           int nnz) {
  int bid = blockIdx.x * blockDim.x + threadIdx.x;

  if (bid < batch_size) {
    int stride = m * n;
    for (int i = 0; i < m; i++) {
      for (int idx = row_index[i]; idx < row_index[i + 1]; idx++) {
        int j = col_index[idx];
        values[bid * nnz + idx] = dense[bid * stride + j * m + i];
      }
    }
  }
}

/**
 * Kernel to construct batched dense matrices from batched CSR sparse matrices
 * 
 * @param[out] dense      Batched dense matrices. Size: m * n * batch_size
 * @param[in]  col_index  CSR column index.       Size: nnz
 * @param[in]  row_index  CSR row index.          Size: m + 1
 * @param[in]  values     CSR values array.       Size: nnz * batch_size
 * @param[in]  batch_size Number of matrices in the batch
 * @param[in]  m          Number of rows per matrix
 * @param[in]  n          Number of columns per matrix
 * @param[in]  nnz        Number of non-zero elements in each matrix
 */
template <typename T>
static __global__ void csr_to_dense_kernel(T* dense, const int* col_index,
                                           const int* row_index,
                                           const T* values, int batch_size,
                                           int m, int n, int nnz) {
  int bid = blockIdx.x * blockDim.x + threadIdx.x;

  if (bid < batch_size) {
    int stride = m * n;
    for (int i = 0; i < m; i++) {
      for (int idx = row_index[i]; idx < row_index[i + 1]; idx++) {
        int j = col_index[idx];
        dense[bid * stride + j * m + i] = values[bid * nnz + idx];
      }
    }
  }
}

/**
 * @brief The Batched::CSR class provides storage and a few operations for
 *        a batch of matrices in Compressed Sparse Row representation, that
 *        share a common structure (index arrays) but different values.
 */
template <typename T>
class CSR {
 public:
  /**
   * @brief Constructor that leaves the matrix uninitialized
   * 
   * @param[in] m                Number of rows per matrix
   * @param[in] n                Number of columns per matrix
   * @param[in] nnz              Number of non-zero elements per matrix
   * @param[in] batch_size       Number of matrices in the batch
   * @param[in] cublasHandle     cuBLAS handle
   * @param[in] cusolverSpHandle cuSOLVER sparse handle
   * @param[in] allocator        Device memory allocator
   * @param[in] stream           CUDA stream
   */
  CSR(int m, int n, int nnz, int batch_size, cublasHandle_t cublasHandle,
      cusolverSpHandle_t cusolverSpHandle,
      std::shared_ptr<ML::deviceAllocator> allocator, cudaStream_t stream)
    : m_batch_size(batch_size),
      m_allocator(allocator),
      m_cublasHandle(cublasHandle),
      m_cusolverSpHandle(cusolverSpHandle),
      m_stream(stream),
      m_shape(std::pair<int, int>(m, n)),
      m_nnz(nnz) {
    // Allocate the values
    T* values =
      (T*)m_allocator->allocate(sizeof(T) * m_nnz * m_batch_size, m_stream);
    // Allocate the index arrays
    int* col_index = (int*)m_allocator->allocate(sizeof(int) * m_nnz, m_stream);
    int* row_index =
      (int*)m_allocator->allocate(sizeof(int) * (m_shape.first + 1), m_stream);

    /* Take these references to extract them from member-storage for the
     * lambda below. There are better C++14 ways to do this, but I'll keep
     * it C++11 for now. */
    auto& shape = m_shape;

    /* Note: we create these "free" functions with explicit copies to ensure
     * that the deallocate function gets called with the correct values. */
    auto deallocate_values = [allocator, batch_size, nnz, stream](T* A) {
      allocator->deallocate(A, batch_size * nnz * sizeof(T), stream);
    };
    auto deallocate_col = [allocator, nnz, stream](int* A) {
      allocator->deallocate(A, sizeof(int) * nnz, stream);
    };
    auto deallocate_row = [allocator, shape, stream](int* A) {
      allocator->deallocate(A, sizeof(int) * (shape.first + 1), stream);
    };

    // When the shared pointers go to 0, the memory is deallocated
    m_values = std::shared_ptr<T>(values, deallocate_values);
    m_col_index = std::shared_ptr<int>(col_index, deallocate_col);
    m_row_index = std::shared_ptr<int>(row_index, deallocate_row);
  }

  /**
   * @brief Construct from a dense batched matrix and its mask
   * 
   * @param[in]  dense  Dense batched matrix
   * @param[in]  mask   Col-major host device matrix containing a mask of the
   *                    non-zero values common to all matrices in the batch.
   *                    Note: the point of using a mask is that some values
   *                    might be zero in a few matrices but not generally in
   *                    the batch so we shouldn't rely on a single matrix to
   *                    get the mask
   * @return Batched CSR matrix
   */
  static CSR<T> from_dense(const LinAlg::Batched::Matrix<T>& dense,
                           const std::vector<bool>& mask,
                           cusolverSpHandle_t cusolverSpHandle) {
    std::pair<int, int> shape = dense.shape();

    // Create the index arrays from the mask
    std::vector<int> h_col_index;
    std::vector<int> h_row_index = std::vector<int>(shape.first + 1);
    int nnz = 0;
    for (int i = 0; i < shape.first; i++) {
      h_row_index[i] = nnz;
      for (int j = 0; j < shape.second; j++) {
        if (mask[j * shape.first + i]) {
          h_col_index.push_back(j);
          nnz++;
        }
      }
    }
    h_row_index[shape.first] = nnz;

    CSR<T> out = CSR<T>(shape.first, shape.second, nnz, dense.batches(),
                        dense.cublasHandle(), cusolverSpHandle,
                        dense.allocator(), dense.stream());

    // Copy the host index arrays to the device
    MLCommon::copy(out.get_col_index(), h_col_index.data(), nnz, out.stream());
    MLCommon::copy(out.get_row_index(), h_row_index.data(), shape.first + 1,
                   out.stream());

    // Copy the data from the dense matrix to its sparse representation
    constexpr int TPB = 256;
    dense_to_csr_kernel<<<MLCommon::ceildiv<int>(out.batches(), TPB), TPB, 0,
                          out.stream()>>>(
      dense.raw_data(), out.get_col_index(), out.get_row_index(),
      out.get_values(), out.batches(), shape.first, shape.second, nnz);
    CUDA_CHECK(cudaPeekAtLastError());

    return out;
  }

  /**
   * @brief Construct a dense batched matrix
   * 
   * @return Batched::Matrix representing the same data as this object
   */
  LinAlg::Batched::Matrix<T> to_dense() {
    LinAlg::Batched::Matrix<T> dense(m_shape.first, m_shape.second,
                                     m_batch_size, m_cublasHandle, m_allocator,
                                     m_stream, true);

    // Copy the data from the sparse to the dense representation
    constexpr int TPB = 256;
    csr_to_dense_kernel<<<MLCommon::ceildiv<int>(m_batch_size, TPB), TPB, 0,
                          m_stream>>>(
      dense.raw_data(), m_col_index.get(), m_row_index.get(), m_values.get(),
      m_batch_size, m_shape.first, m_shape.second, m_nnz);
    CUDA_CHECK(cudaPeekAtLastError());

    return dense;
  }

  //! Return batch size
  size_t batches() const { return m_batch_size; }

  //! Return number of non-zero elements
  size_t nnz() const { return m_nnz; }

  //! Return cublas handle
  cublasHandle_t cublasHandle() const { return m_cublasHandle; }

  //! Return cusolver sparse handle
  cusolverSpHandle_t cusolverSpHandle() const { return m_cusolverSpHandle; }

  //! Return allocator
  std::shared_ptr<deviceAllocator> allocator() const { return m_allocator; }

  //! Return stream
  cudaStream_t stream() const { return m_stream; }

  //! Return shape
  const std::pair<int, int>& shape() const { return m_shape; }

  //! Return values array
  T* get_values() { return m_values.get(); }
  const T* get_values() const { return m_values.get(); }

  //! Return columns index array
  int* get_col_index() { return m_col_index.get(); }
  const int* get_col_index() const { return m_col_index.get(); }

  //! Return rows index array
  int* get_row_index() { return m_row_index.get(); }
  const int* get_row_index() const { return m_row_index.get(); }

 protected:
  //! Shape (rows, cols) of matrices.
  std::pair<int, int> m_shape;

  //! Number of non-zero values per matrix
  int m_nnz;

  //! Array(pointer) to the values in all the batched matrices.
  std::shared_ptr<T> m_values;

  //! Array(pointer) to the column index of the CSR.
  std::shared_ptr<int> m_col_index;

  //! Array(pointer) to the row index of the CSR.
  std::shared_ptr<int> m_row_index;

  //! Number of matrices in batch
  size_t m_batch_size;

  std::shared_ptr<ML::deviceAllocator> m_allocator;
  cublasHandle_t m_cublasHandle;
  cusolverSpHandle_t m_cusolverSpHandle;
  cudaStream_t m_stream;
};

/**
 * Kernel to compute a batched SpMV: alpha*A*x + beta*y
 * (where A is a sparse matrix, x and y dense vectors)
 * 
 * @note One thread per batch (this is intended for very large batches)
 *       Rows don't have the same number of non-zero elements, so an approach
 *       to parallelize on the rows would lead to divergence
 * 
 * @param[in]     alpha        Scalar alpha
 * @param[in]     A_col_index  CSR column index of batched matrix A
 * @param[in]     A_row_index  CSR row index of batched matrix A
 * @param[in]     A_values     Values of the non-zero elements of A
 * @param[in]     x            Dense vector x
 * @param[in]     beta         Scalar beta
 * @param[in,out] y            Dense vector y
 * @param[in]     m            Number of rows of A
 * @param[in]     n            Number of columns of A
 * @param[in]     batch_size   Number of individual matrices in the batch
 */
template <typename T>
__global__ void batched_spmv_kernel(T alpha, const int* A_col_index,
                                    const int* A_row_index, const T* A_values,
                                    const T* x, T beta, T* y, int m, int n,
                                    int batch_size) {
  int bid = blockIdx.x * blockDim.x + threadIdx.x;

  if (bid < batch_size) {
    int nnz = A_row_index[m];
    for (int i = 0; i < m; i++) {
      T acc = 0.0;
      for (int idx = A_row_index[i]; idx < A_row_index[i + 1]; idx++) {
        int j = A_col_index[idx];
        acc += A_values[bid * nnz + idx] * x[bid * n + j];
      }
      y[bid * m + i] =
        alpha * acc + (beta == 0.0 ? 0.0 : beta * y[bid * m + i]);
    }
  }
}

/**
 * Compute a batched SpMV: alpha*A*x + beta*y
 * (where A is a sparse matrix, x and y dense vectors)
 * 
 * @note Not supporting transpose yet for simplicity as it isn't needed
 *       Also currently the strides between batched vectors are assumed to
 *       be exactly the dimensions of the problem
 * 
 * @param[in]     alpha  Scalar alpha
 * @param[in]     A      Batched sparse matrix (CSR)
 * @param[in]     x      Batched dense vector x
 * @param[in]     beta   Scalar beta
 * @param[in,out] y      Batched dense vector y
 */
template <typename T>
void b_spmv(T alpha, const CSR<T>& A, const LinAlg::Batched::Matrix<T>& x,
            T beta, LinAlg::Batched::Matrix<T>& y) {
  int m = A.shape().first;
  int n = A.shape().second;
  // A few checks
  ASSERT(std::min(x.shape().first, x.shape().second) == 1 &&
           std::max(x.shape().first, x.shape().second) == n,
         "SpMV: Dimension mismatch: x");
  ASSERT(std::min(y.shape().first, y.shape().second) == 1 &&
           std::max(y.shape().first, y.shape().second) == m,
         "SpMV: Dimension mismatch: y");
  ASSERT(A.batches() == x.batches(),
         "SpMV: A and x must have the same batch size");
  ASSERT(A.batches() == y.batches(),
         "SpMV: A and y must have the same batch size");

  // Execute the kernel
  constexpr int TPB = 256;
  batched_spmv_kernel<<<MLCommon::ceildiv<int>(A.batches(), TPB), TPB, 0,
                        A.stream()>>>(
    alpha, A.get_col_index(), A.get_row_index(), A.get_values(), x.raw_data(),
    beta, y.raw_data(), m, n, A.batches());
}

/**
 * Kernel to compute a batched SpMM: alpha*A*B + beta*C
 * (where A is a sparse matrix, B and C dense matrices)
 * 
 * @note Parallelized over the batch and the columns of individual matrices
 * 
 * @param[in]     alpha           Scalar alpha
 * @param[in]     A_col_index     CSR column index of batched matrix A
 * @param[in]     A_row_index     CSR row index of batched matrix A
 * @param[in]     A_values        Values of the non-zero elements of A
 * @param[in]     B               Dense matrix B
 * @param[in]     beta            Scalar beta
 * @param[in,out] C               Dense matrix C
 * @param[in]     m               Number of rows of A and C
 * @param[in]     k               Number of columns of A, rows of B
 * @param[in]     n               Number of columns of B and C
 * @param[in]     batch_size      Number of individual matrices in the batch
 * @param[in]     threads_per_bid Number of threads per batch index
 */
template <typename T>
__global__ void batched_spmm_kernel(T alpha, const int* A_col_index,
                                    const int* A_row_index, const T* A_values,
                                    const T* B, T beta, T* C, int m, int k,
                                    int n, int batch_size,
                                    int threads_per_bid) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = thread_idx / threads_per_bid;

  if (bid < batch_size) {
    int nnz = A_row_index[m];
    const T* b_A_values = A_values + bid * nnz;
    const T* b_B = B + bid * k * n;
    for (int j = thread_idx % threads_per_bid; j < n; j += threads_per_bid) {
      for (int i = 0; i < m; i++) {
        T acc = 0.0;
        for (int idx = A_row_index[i]; idx < A_row_index[i + 1]; idx++) {
          int ik = A_col_index[idx];
          acc += b_A_values[idx] * b_B[j * k + ik];
        }
        int ci = bid * m * n + j * m + i;
        C[ci] = alpha * acc + (beta == 0.0 ? 0.0 : beta * C[ci]);
      }
    }
  }
}

/**
 * @todo: docs
 * @note: this is more performant when the matrices are large enough and
 *        assuming that almost all elements of B need to be read
 */
template <typename T>
__global__ void batched_spmm_kernel_shared_mem(T alpha, const int* A_col_index,
                                               const int* A_row_index,
                                               const T* A_values, const T* B,
                                               T beta, T* C, int m, int k,
                                               int n, int nnz) {
  int bid = blockIdx.x;
  int j = threadIdx.x;

  // Using dynamic shared memory
  extern __shared__ int8_t shared_mem[];
  int* s_A_col_index = (int*)shared_mem;
  int* s_A_row_index = (int*)(shared_mem + nnz * sizeof(int));
  T* s_A_values = (T*)(shared_mem + (nnz + m + 1) * sizeof(int));
  T* s_B = (T*)(shared_mem + (nnz + m + 1) * sizeof(int) + nnz * sizeof(T));

  // Load A in shared memory
  const T* b_A_values = A_values + bid * nnz;
  for (int i_nnz = j; i_nnz < nnz; i_nnz += blockDim.x) {
    s_A_col_index[i_nnz] = A_col_index[i_nnz];
    s_A_values[i_nnz] = b_A_values[i_nnz];
  }
  for (int i_m = j; i_m < m; i_m += blockDim.x) {
    s_A_row_index[i_m] = A_row_index[i_m];
  }
  if (j == 0) s_A_row_index[m] = nnz;

  // Load B in shared memory
  const T* b_B = B + bid * k * n;
  for (int i_kn = j; i_kn < k * n; i_kn += blockDim.x) {
    s_B[i_kn] = b_B[i_kn];
  }

  __syncthreads();

  for (int i = 0; i < m; i++) {
    T acc = 0.0;
    for (int idx = s_A_row_index[i]; idx < s_A_row_index[i + 1]; idx++) {
      int ik = s_A_col_index[idx];
      acc += s_A_values[idx] * s_B[j * k + ik];
    }
    int ci = bid * m * n + j * m + i;
    C[ci] = alpha * acc + (beta == 0.0 ? 0.0 : beta * C[ci]);
  }
}

/**
 * Compute a batched SpMM: alpha*A*B + beta*C
 * (where A is a sparse matrix, B and C dense matrices)
 * 
 * @note Not supporting transpose yet for simplicity as it isn't needed
 *       Also not supporting leading dim different than the problem dimensions
 * 
 * @param[in]     alpha  Scalar alpha
 * @param[in]     A      Batched sparse matrix (CSR)
 * @param[in]     B      Batched dense matrix B
 * @param[in]     beta   Scalar beta
 * @param[in,out] C      Batched dense matrix C
 */
template <typename T>
void b_spmm(T alpha, const CSR<T>& A, const LinAlg::Batched::Matrix<T>& B,
            T beta, LinAlg::Batched::Matrix<T>& C) {
  int m = A.shape().first;
  int n = B.shape().second;
  int k = A.shape().second;
  int nb = A.batches();
  int nnz = A.nnz();
  // Check the parameters
  ASSERT(B.batches() == nb, "SpMM: A and B must have the same batch size");
  ASSERT(C.batches() == nb, "SpMM: A and C must have the same batch size");
  ASSERT(B.shape().first == k, "SpMM: Dimension mismatch: A and B");
  ASSERT(C.shape().first == m && C.shape().second == n,
         "SpMM: Dimension mismatch: C");

  // Execute the kernel
  if (true) {  // Shared memory kernel (large matrices)
    size_t shared_mem_size =
      (nnz + m + 1) * sizeof(int) + (nnz + k * n) * sizeof(T);
    batched_spmm_kernel<<<nb, n, shared_mem_size, A.stream()>>>(
      alpha, A.get_col_index(), A.get_row_index(), A.get_values(), B.raw_data(),
      beta, C.raw_data(), m, k, n, nnz);
  } else {  // No shared memory (small matrices)
    constexpr int TPB = 256;
    int threads_per_bid =
      nb <= 1024 ? 8 : (nb <= 2048 ? 4 : (nb <= 4096 ? 2 : 1));
    batched_spmm_kernel<<<MLCommon::ceildiv<int>(nb, TPB), TPB, 0,
                          A.stream()>>>(
      alpha, A.get_col_index(), A.get_row_index(), A.get_values(), B.raw_data(),
      beta, C.raw_data(), m, k, n, nb, threads_per_bid);
  }
}

/**
 * @brief Solve discrete Lyapunov equation A*X*A' - X + Q = 0
 *
 * @param[in]  A       Batched matrix A in CSR representation
 * @param[in]  A_mask  Density mask of A (host)
 * @param[in]  Q       Batched dense matrix Q
 * @return             Batched dense matrix X solving the Lyapunov equation
 */
template <typename T>
LinAlg::Batched::Matrix<T> b_lyapunov(const CSR<T>& A,
                                      const std::vector<bool>& A_mask,
                                      const LinAlg::Batched::Matrix<T>& Q) {
  int n = A.shape().first;
  int n2 = n * n;
  int A_nnz = A.nnz();
  int batch_size = A.batches();
  auto stream = A.stream();
  auto allocator = A.allocator();
  cusolverSpHandle_t cusolverSpHandle = A.cusolverSpHandle();
  auto counting = thrust::make_counting_iterator(0);

  //
  // Construct sparse matrix I-AxA
  //

  // Note: if the substraction cancels elements, they are still considered
  // as non-zero, but it shouldn't be an issue

  // Copy the index arrays of A to the host
  std::vector<int> h_A_col_index = std::vector<int>(A_nnz);
  std::vector<int> h_A_row_index = std::vector<int>(n + 1);
  copy(h_A_col_index.data(), A.get_col_index(), A_nnz, stream);
  copy(h_A_row_index.data(), A.get_row_index(), n + 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Compute the index arrays of AxA
  std::vector<int> h_AxA_col_index;
  std::vector<int> h_AxA_row_index = std::vector<int>(n2 + 1);
  int nnz = 0;
  {
    int i_p = -1, j_p = -1;
    for (int i1 = 0; i1 < n; i1++) {
      for (int i2 = 0; i2 < n; i2++) {
        for (int idx1 = h_A_row_index[i1]; idx1 < h_A_row_index[i1 + 1];
             idx1++) {
          int j1 = h_A_col_index[idx1];
          for (int idx2 = h_A_row_index[i2]; idx2 < h_A_row_index[i2 + 1];
               idx2++) {
            int j2 = h_A_col_index[idx2];
            // Compute the next non-zero element
            int i = i1 * n + i2;
            int j = j1 * n + j2;
            // Fill all the diagonal elements in-between
            for (int k = (j_p >= i_p ? i_p + 1 : i_p);
                 k <= (j <= i ? i - 1 : i); k++) {
              h_AxA_col_index.push_back(k);
              h_AxA_row_index[k + 1] = ++nnz;
            }
            // Add the current index
            h_AxA_col_index.push_back(j);
            h_AxA_row_index[i + 1] = ++nnz;
            // Remember the current index in the next iteration
            i_p = i;
            j_p = j;
          }
        }
      }
    }
    // Fill diagonal elements after the last element of AxA
    for (int k = (j_p >= i_p ? i_p + 1 : i_p); k < n2; k++) {
      h_AxA_col_index.push_back(k);
      h_AxA_row_index[k + 1] = ++nnz;
    }
  }

  // Create the uninitialized batched CSR matrix
  CSR<T> I_m_AxA(n2, n2, nnz, batch_size, A.cublasHandle(), cusolverSpHandle,
                 allocator, stream);

  // Copy the host index arrays to the device arrays
  copy(I_m_AxA.get_col_index(), h_AxA_col_index.data(), nnz, stream);
  copy(I_m_AxA.get_row_index(), h_AxA_row_index.data(), n2 + 1, stream);

  // Compute values of I-AxA
  const double* d_A_values = A.get_values();
  const int* d_A_col_index = A.get_col_index();
  const int* d_A_row_index = A.get_row_index();
  double* d_values = I_m_AxA.get_values();
  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + batch_size, [=] __device__(int bid) {
                     const double* b_A_values = d_A_values + A_nnz * bid;
                     double* b_values = d_values + nnz * bid;
                     int i_p = -1, j_p = -1;
                     int i_nnz = 0;
                     for (int i1 = 0; i1 < n; i1++) {
                       for (int i2 = 0; i2 < n; i2++) {
                         for (int idx1 = d_A_row_index[i1];
                              idx1 < d_A_row_index[i1 + 1]; idx1++) {
                           int j1 = d_A_col_index[idx1];
                           double value1 = b_A_values[idx1];
                           for (int idx2 = d_A_row_index[i2];
                                idx2 < d_A_row_index[i2 + 1]; idx2++) {
                             int j2 = d_A_col_index[idx2];
                             double value2 = b_A_values[idx2];
                             // Compute the next non-zero element
                             int i = i1 * n + i2;
                             int j = j1 * n + j2;
                             // Fill all the diagonal elements in-between
                             for (int k = (j_p >= i_p ? i_p + 1 : i_p);
                                  k <= (j <= i ? i - 1 : i); k++) {
                               b_values[i_nnz++] = 1.0;
                             }
                             // Set the current element
                             b_values[i_nnz++] =
                               (i == j ? 1.0 : 0.0) - value1 * value2;
                             // Remember the current index in the next iteration
                             i_p = i;
                             j_p = j;
                           }
                         }
                       }
                     }
                     // Fill diagonal elements after the last element of AxA
                     for (int k = (j_p >= i_p ? i_p + 1 : i_p); k < n2; k++) {
                       b_values[i_nnz++] = 1.0;
                     }
                   });

  //
  // Solve (I - TxT) vec(X) = vec(Q)
  //

  csrqrInfo_t info = NULL;
  cusparseMatDescr_t descr = NULL;

  // Create cuSPARSE matrix descriptor
  cusparseCreateMatDescr(&descr);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);

  // Create empty info structure
  CUSOLVER_CHECK(cusolverSpCreateCsrqrInfo(&info));

  // Symbolic analysis
  CUSOLVER_CHECK(cusolverSpXcsrqrAnalysisBatched(
    cusolverSpHandle, n2, n2, nnz, descr, I_m_AxA.get_row_index(),
    I_m_AxA.get_col_index(), info));

  // Sometimes not all the batch can be computed at once. So we divide
  // it into smaller batches and perform multiple cuSOLVER calls
  int group_size = 2 * batch_size;
  size_t internalDataInBytes, workspaceInBytes;
  size_t mem_free, mem_total;
  do {
    group_size = (group_size - 1) / 2 + 1;
    CUDA_CHECK(cudaMemGetInfo(&mem_free, &mem_total));
    CUSOLVER_CHECK(LinAlg::cusolverSpcsrqrBufferInfoBatched(
      cusolverSpHandle, n2, n2, nnz, descr, I_m_AxA.get_values(),
      I_m_AxA.get_row_index(), I_m_AxA.get_col_index(), group_size, info,
      &internalDataInBytes, &workspaceInBytes));
  } while (group_size > 1 && internalDataInBytes > mem_free / 2);

  // Allocate working space
  void* pBuffer = allocator->allocate(workspaceInBytes, stream);

  // Create output matrix
  LinAlg::Batched::Matrix<T> X(n, n, batch_size, A.cublasHandle(), allocator,
                               stream, false);

  // Then loop over the groups and solve
  for (int start_id = 0; start_id < batch_size; start_id += group_size) {
    int actual_group_size = std::min(batch_size - start_id, group_size);
    CUSOLVER_CHECK(LinAlg::cusolverSpcsrqrsvBatched(
      cusolverSpHandle, n2, n2, nnz, descr,
      I_m_AxA.get_values() + nnz * start_id, I_m_AxA.get_row_index(),
      I_m_AxA.get_col_index(), Q.raw_data() + n2 * start_id,
      X.raw_data() + n2 * start_id, actual_group_size, info, pBuffer, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  allocator->deallocate(pBuffer, workspaceInBytes, stream);

  CUSOLVER_CHECK(cusolverSpDestroyCsrqrInfo(info));

  return X;
}

}  // namespace Batched
}  // namespace Sparse
}  // namespace MLCommon
