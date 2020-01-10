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

#include <memory>
#include <vector>

#include <cuml/common/utils.hpp>
#include <cuml/cuml.hpp>

#include "linalg/batched/batched_matrix.h"

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
 * @brief The BatchedCSR class provides storage and a few operations for
 *        a batch of matrices in Compressed Sparse Row representation, that
 *        share a common structure (index arrays) but different values.
 */
template <typename T>
class BatchedCSR {
 public:
  /**
   * @brief Constructor from dense batched matrix and mask
   * 
   * @param[in]  dense  Dense batched matrix
   * @param[in]  mask   Col-major host device matrix containing a mask of the
   *                    non-zero values common to all matrices in the batch.
   *                    Note: the point of using a mask is that some values
   *                    might be zero in a few matrices but not generally in
   *                    the batch so we shouldn't rely on a single matrix to
   *                    get the mask
   */
  BatchedCSR(const LinAlg::Batched::BatchedMatrix<T>& dense,
             const std::vector<bool>& mask)
    : m_batch_size(dense.batches()),
      m_allocator(dense.allocator()),
      m_stream(dense.stream()),
      m_shape(dense.shape()) {
    // Create the index arrays from the mask
    std::vector<int> h_col_index;
    std::vector<int> h_row_index = std::vector<int>(m_shape.first + 1);
    int k = 0;
    for (int i = 0; i < m_shape.first; i++) {
      h_row_index[i] = k;
      for (int j = 0; j < m_shape.second; j++) {
        if (mask[j * m_shape.first + i]) {
          h_col_index.push_back(j);
          k++;
        }
      }
    }
    h_row_index[m_shape.first] = k;
    m_nnz = k;

    // Allocate the values
    T* values =
      (T*)m_allocator->allocate(sizeof(T) * m_nnz * m_batch_size, m_stream);
    // Allocate the index arrays
    int* col_index = (int*)m_allocator->allocate(sizeof(int) * m_nnz, m_stream);
    int* row_index =
      (int*)m_allocator->allocate(sizeof(int) * (m_shape.first + 1), m_stream);

    // Copy the host index arrays to the device
    MLCommon::copy(col_index, h_col_index.data(), m_nnz, m_stream);
    MLCommon::copy(row_index, h_row_index.data(), m_shape.first + 1, m_stream);

    // Copy the data from the dense matrix to its sparse representation
    constexpr int TPB = 256;
    dense_to_csr_kernel<<<MLCommon::ceildiv<int>(m_batch_size, TPB), TPB, 0,
                          m_stream>>>(dense.raw_data(), col_index, row_index,
                                      values, m_batch_size, m_shape.first,
                                      m_shape.second, m_nnz);
    CUDA_CHECK(cudaPeekAtLastError());

    /* Take these references to extract them from member-storage for the
     * lambda below. There are better C++14 ways to do this, but I'll keep
     * it C++11 for now. */
    auto& shape = m_shape;
    auto& batch_size = m_batch_size;
    auto& nnz = m_nnz;
    auto& allocator = m_allocator;
    auto& stream = m_stream;

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

  //! Return batch size
  size_t batches() const { return m_batch_size; }

  //! Return allocator
  std::shared_ptr<deviceAllocator> allocator() const { return m_allocator; }

  //! Return stream
  cudaStream_t stream() const { return m_stream; }

  //! Return shape
  const std::pair<int, int>& shape() const { return m_shape; }

  //! Return values array
  T* get_values() const { return m_values.get(); }

  //! Return columns index array
  int* get_col_index() const { return m_col_index.get(); }

  //! Return rows index array
  int* get_row_index() const { return m_row_index.get(); }

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
void b_spmv(T alpha, const BatchedCSR<T>& A,
            const LinAlg::Batched::BatchedMatrix<T>& x, T beta,
            LinAlg::Batched::BatchedMatrix<T>& y) {
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
void b_spmm(T alpha, const BatchedCSR<T>& A,
            const LinAlg::Batched::BatchedMatrix<T>& B, T beta,
            LinAlg::Batched::BatchedMatrix<T>& C) {
  int m = A.shape().first;
  int n = B.shape().second;
  int k = A.shape().second;
  int nb = A.batches();
  // Check the parameters
  ASSERT(B.batches() == nb, "SpMM: A and B must have the same batch size");
  ASSERT(C.batches() == nb, "SpMM: A and C must have the same batch size");
  ASSERT(B.shape().first == k, "SpMM: Dimension mismatch: A and B");
  ASSERT(C.shape().first == m && C.shape().second == n,
         "SpMM: Dimension mismatch: C");

  // Execute the kernel
  constexpr int TPB = 256;
  int threads_per_bid =
    nb <= 1024 ? 8 : (nb <= 2048 ? 4 : (nb <= 4096 ? 2 : 1));
  batched_spmm_kernel<<<MLCommon::ceildiv<int>(nb, TPB), TPB, 0, A.stream()>>>(
    alpha, A.get_col_index(), A.get_row_index(), A.get_values(), B.raw_data(),
    beta, C.raw_data(), m, k, n, nb, threads_per_bid);
}

}  // namespace Batched
}  // namespace Sparse
}  // namespace MLCommon
