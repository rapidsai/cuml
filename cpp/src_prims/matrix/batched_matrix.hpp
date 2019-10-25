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

#include <functional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <linalg/binary_op.h>
#include <linalg/cublas_wrappers.h>
#include <memory>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <common/cumlHandle.hpp>
#include <cuml/cuml.hpp>

namespace MLCommon {
namespace Matrix {

/**
 * @brief An allocation function for `BatchedMatrixMemory`.
 * 
 * @note Written as a free function because I had trouble getting the
 *       __device__ lambda to compile as a member function of the
 *       `BatchedMatrixMemory` struct.
 * 
 * @param[in]  shape        Shape of each matrix (rows, columns)
 * @param[in]  num_batches  Number of matrices in the batch
 * @param[in]  allocator    Device memory allocator
 * @param[in]  stream       CUDA stream
 * 
 * @return Pair (A_dense, A_array): pointer to the raw data and pointer
 *         to the array of pointers to each matrix in the batch
 */
std::pair<double*, double**> BMM_Allocate(
  std::pair<int, int> shape, int num_batches, bool setZero,
  std::shared_ptr<ML::deviceAllocator> allocator, cudaStream_t stream) {
  int m = shape.first;
  int n = shape.second;

  // Allocate dense batched matrix and possibly set to zero
  double* A_dense =
    (double*)allocator->allocate(sizeof(double) * m * n * num_batches, stream);
  if (setZero)
    CUDA_CHECK(cudaMemsetAsync(A_dense, 0, sizeof(double) * m * n * num_batches,
                               stream));

  // Allocate and fill array of pointers to each batch matrix
  double** A_array =
    (double**)allocator->allocate(sizeof(double*) * num_batches, stream);
  // Fill array of pointers to each batch matrix.
  auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(
    thrust::cuda::par.on(stream), counting, counting + num_batches,
    [=] __device__(int bid) { A_array[bid] = &(A_dense[bid * m * n]); });
  return std::make_pair(A_dense, A_array);
}

/**
 * @brief Kernel to creates an identity matrix
 * 
 * @note The block id is the batch id, and the thread id is the starting
 *       row/column for this thread (then looping to cover all the diagonal)
 * 
 * @param[out]  I  Pointer to the raw data of the identity matrix to create
 * @param[in]   m  Number of rows/columns of matrix
 */
__global__ void identity_matrix_kernel(double* I, int m) {
  double* I_b = I + blockIdx.x * m * m;
  int stride = (m + 1);
  for (int idx = threadIdx.x; idx < m; idx += blockDim.x) {
    I_b[idx * stride] = 1;
  }
}

/**
 * @brief The BatchedMatrix class provides storage and a number of linear
 *        operations on collections (batches) of matrices of identical shape.
 */
class BatchedMatrix {
 public:
  /**
   * @brief Constructor that allocates memory using the memory pool.
   * 
   * @param[in]  m            Number of rows
   * @param[in]  n            Number of columns
   * @param[in]  num_batches  Number of matrices in the batch
   * @param[in]  pool         The memory pool
   * @param[in]  setZero      Should matrix be zeroed on allocation?
   */
  BatchedMatrix(int m, int n, int num_batches, cublasHandle_t cublasHandle,
                std::shared_ptr<ML::deviceAllocator> allocator,
                cudaStream_t stream, bool setZero = true)
    : m_num_batches(num_batches),
      m_allocator(allocator),
      m_cublasHandle(cublasHandle),
      m_stream(stream) {
    m_shape = std::make_pair(m, n);

    // Allocate memory
    auto memory =
      BMM_Allocate(m_shape, num_batches, setZero, allocator, stream);

    /* Take these references to extract them from member-storage for the
     * lambda below. There are better C++14 ways to do this, but I'll keep
     * it C++11 for now. */
    auto& shape = m_shape;

    /* Note: we create this "free" function with explicit copies to ensure that
     * the deallocate function gets called with the correct values. */
    auto f1 = [allocator, num_batches, shape, stream](double* A) {
      allocator->deallocate(
        A, num_batches * shape.first * shape.second * sizeof(double), stream);
    };

    auto f2 = [allocator, num_batches, stream](double** A) {
      allocator->deallocate(A, sizeof(double*) * num_batches, stream);
    };

    // When this shared pointer count goes to 0, `f` is called to deallocate the memory
    m_A_dense = std::shared_ptr<double>(memory.first, f1);
    m_A_batches = std::shared_ptr<double*>(memory.second, f2);
  }

  //! Return batches
  size_t batches() const { return m_num_batches; }

  //! Return cublas handle
  cublasHandle_t cublasHandle() const { return m_cublasHandle; }

  //! Return allocator
  std::shared_ptr<deviceAllocator> allocator() const { return m_allocator; }

  //! Return stream
  cudaStream_t stream() const { return m_stream; }

  //! Return shape
  const std::pair<int, int>& shape() const { return m_shape; }

  //! Return pointer array
  double** data() const { return m_A_batches.get(); }

  //! Return pointer to the underlying memory
  double* raw_data() const { return m_A_dense.get(); }

  /**
   * @brief Return pointer to the data of a specific matrix
   * 
   * @param[in]  id  id of the matrix
   * @return         A pointer to the raw data of the matrix
   */
  double* operator[](int id) const {
    return &(m_A_dense.get()[id * m_shape.first * m_shape.second]);
  }

  //! Stack the matrix by columns creating a long vector
  BatchedMatrix vec() const {
    int m = m_shape.first;
    int n = m_shape.second;
    int r = m * n;
    BatchedMatrix toVec(r, 1, m_num_batches, m_cublasHandle, m_allocator,
                        m_stream);
    cudaMemcpyAsync(toVec[0], m_A_dense.get(),
                    m_num_batches * r * sizeof(double),
                    cudaMemcpyDeviceToDevice, m_stream);
    return toVec;
  }

  /**
   * @brief Create a matrix from a long vector.
   * 
   * @param[in]  m  Number of desired rows
   * @param[in]  n  Number of desired columns
   * @return        A batched matrix
   */
  BatchedMatrix mat(int m, int n) const {
    const int r = m_shape.first * m_shape.second;
    if (r != m * n)
      throw std::runtime_error(
        "ERROR BatchedMatrix::mat(m,n): Size mismatch - Cannot reshape array "
        "into desired size");
    BatchedMatrix toMat(m, n, m_num_batches, m_cublasHandle, m_allocator,
                        m_stream);
    cudaMemcpyAsync(toMat[0], m_A_dense.get(),
                    m_num_batches * r * sizeof(double),
                    cudaMemcpyDeviceToDevice, m_stream);

    return toMat;
  }

  //! Visualize the first matrix.
  void print(std::string name) const {
    size_t len = m_shape.first * m_shape.second * m_num_batches;
    std::vector<double> A(len);
    updateHost(A.data(), m_A_dense.get(), len, m_stream);
    std::cout << name << "=\n";
    for (int i = 0; i < m_shape.first; i++) {
      for (int j = 0; j < m_shape.second; j++) {
        // column major
        std::cout << std::setprecision(10) << A[j * m_shape.first + i] << ",";
      }
      std::cout << "\n";
    }
  }

  /**
   * @brief Initialize a batched identity matrix.
   * 
   * @param[in]  m            Number of rows/columns of matrix
   * @param[in]  num_batches  Number of matrices in batch
   * @param[in]  pool         Memory pool
   * 
   * @return A batched identity matrix
   */
  static BatchedMatrix Identity(int m, int num_batches, cublasHandle_t handle,
                                std::shared_ptr<ML::deviceAllocator> allocator,
                                cudaStream_t stream) {
    BatchedMatrix I(m, m, num_batches, handle, allocator, stream, true);

    identity_matrix_kernel<<<num_batches, std::min(1024, m), 0, stream>>>(
      I.raw_data(), m);
    CUDA_CHECK(cudaGetLastError());
    return I;
  }

 private:
  //! Shape (rows, cols) of matrices. We assume all matrices in batch have same shape.
  std::pair<int, int> m_shape;

  //! Array(pointer) to each matrix.
  std::shared_ptr<double*> m_A_batches;

  //! Data pointer to first element of dense matrix data.
  std::shared_ptr<double> m_A_dense;

  //! Number of matrices in batch
  size_t m_num_batches;

  std::shared_ptr<deviceAllocator> m_allocator;
  cublasHandle_t m_cublasHandle;
  cudaStream_t m_stream;
};

/**
 * @brief Computes batched kronecker product between AkB <- A (x) B
 * 
 * @note The block x is the batch id, the thread x is the starting row
 *       in B and the thread y is the starting column in B
 * 
 * @param[in]   A    Pointer to the raw data of matrix `A`
 * @param[in]   m    Number of rows (A)
 * @param[in]   n    Number of columns (A)
 * @param[in]   B    Pointer to the raw data of matrix `B`
 * @param[in]   p    Number of rows (B)
 * @param[in]   q    Number of columns (B)
 * @param[out]  AkB  Pointer to raw data of the result kronecker product
 * @param[in]   k_m  Number of rows of the result    (m * p)
 * @param[in]   k_n  Number of columns of the result (n * q)
 */
__global__ void kronecker_product_kernel(const double* A, int m, int n,
                                         const double* B, int p, int q,
                                         double* AkB, int k_m, int k_n) {
  const double* A_b = A + blockIdx.x * m * n;
  const double* B_b = B + blockIdx.x * p * q;
  double* AkB_b = AkB + blockIdx.x * k_m * k_n;

  for (int ia = 0; ia < m; ia++) {
    for (int ja = 0; ja < n; ja++) {
      double A_ia_ja = A_b[ia + ja * m];

      for (int ib = threadIdx.x; ib < p; ib += blockDim.x) {
        for (int jb = threadIdx.y; jb < q; jb += blockDim.y) {
          int i_ab = ia * p + ib;
          int j_ab = ja * q + jb;
          AkB_b[i_ab + j_ab * k_m] = A_ia_ja * B_b[ib + jb * p];
        }
      }
    }
  }
}

/**
 * @brief Multiplies each matrix in a batch-A with it's batch-B counterpart.
 *        A = [A1,A2,A3], B=[B1,B2,B3] returns [A1*B1, A2*B2, A3*B3]
 * 
 * @param[in]  A   First matrix batch
 * @param[in]  B   Second matrix batch
 * @param[in]  aT  Is `A` transposed?
 * @param[in]  bT  is `B` transposed?
 * 
 * @return Member-wise A*B
 */
BatchedMatrix b_gemm(const BatchedMatrix& A, const BatchedMatrix& B,
                     bool aT = false, bool bT = false) {
  if (A.batches() != B.batches()) {
    throw std::runtime_error("A & B must have same number of batches");
  }

  // Logic for matrix dimensions with optional transpose
  // m = number of rows of matrix op(A) and C.
  int m = !aT ? A.shape().first : A.shape().second;

  // n = number of columns of matrix op(B) and C.
  int n = !bT ? B.shape().second : B.shape().first;

  // k = number of columns of op(A) and rows of op(B).
  int k = !aT ? A.shape().second : A.shape().first;
  int kB = !bT ? B.shape().first : B.shape().second;

  if (k != kB) {
    throw std::runtime_error("Matrix-Multiplication dimensions don't match!");
  }

  auto num_batches = A.batches();
  const auto& handle = A.cublasHandle();

  // set transpose
  cublasOperation_t opA = aT ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = bT ? CUBLAS_OP_T : CUBLAS_OP_N;

  // Create C(m,n)
  BatchedMatrix C(m, n, num_batches, A.cublasHandle(), A.allocator(),
                  A.stream());

  double alpha = 1.0;
  double beta = 0.0;

  // [C1,C2,C3] = [A1*B1, A2*B2, A3*B3]
  CUBLAS_CHECK(
    MLCommon::LinAlg::cublasgemmBatched(handle,
                                        opA,     // A.T?
                                        opB,     // B.T?
                                        m,       // rows op(A), C
                                        n,       // cols of op(B), C
                                        k,       // cols of op(A), rows of op(B)
                                        &alpha,  // alpha * A * B
                                        A.data(),
                                        A.shape().first,  // rows of A
                                        B.data(),
                                        B.shape().first,  // rows of B
                                        &beta,            // + beta * C
                                        C.data(),
                                        C.shape().first,  // rows of C
                                        num_batches, A.stream()));
  return C;
}

/**
 * @brief A utility method to implement pointwise operations between elements
 *        of two batched matrices.
 * 
 * @param[in]  A          Batched matrix A
 * @param[in]  B          Batched matrix B
 * @param[in]  binary_op  The binary operation used on elements of A and B
 * @return A batched matrix, the result of A binary_op B
 */
template <typename F>
BatchedMatrix b_aA_op_B(const BatchedMatrix& A, const BatchedMatrix& B,
                        F binary_op) {
  if (A.shape().first != B.shape().first &&
      A.shape().second != B.shape().second) {
    throw std::runtime_error(
      "Batched Matrix Addition ERROR: Matrices must be same size");
  }
  if (A.batches() != B.batches()) {
    throw std::runtime_error("A & B must have same number of batches");
  }

  auto num_batches = A.batches();
  int m = A.shape().first;
  int n = A.shape().second;

  BatchedMatrix C(m, n, num_batches, A.cublasHandle(), A.allocator(),
                  A.stream());

  LinAlg::binaryOp(C.raw_data(), A.raw_data(), B.raw_data(),
                   m * n * num_batches, binary_op, A.stream());

  return C;
}

/**
 * @brief Multiplies each matrix in a batch-A with it's batch-B counterpart.
 *        A = [A1,A2,A3], B=[B1,B2,B3] return [A1*B1, A2*B2, A3*B3]
 * 
 * @param[in]  A  Batched matrix A
 * @param[in]  B  Batched matrix B
 * @return The result of the batched matrix-matrix multiplication of A * B
 */
BatchedMatrix operator*(const BatchedMatrix& A, const BatchedMatrix& B) {
  return b_gemm(A, B);
}

/**
 * @brief Adds two batched matrices together element-wise.
 * 
 * @param[in]  A  Batched matrix A
 * @param[in]  B  Batched matrix B
 * @return A+B
 */
BatchedMatrix operator+(const BatchedMatrix& A, const BatchedMatrix& B) {
  return b_aA_op_B(A, B, [] __device__(double a, double b) { return a + b; });
}

/**
 * @brief Subtract two batched matrices together element-wise.
 * 
 * @param[in]  A  Batched matrix A
 * @param[in]  B  Batched matrix B
 * @return A-B
 */
BatchedMatrix operator-(const BatchedMatrix& A, const BatchedMatrix& B) {
  return b_aA_op_B(A, B, [] __device__(double a, double b) { return a - b; });
}

/**
 * @brief Solve Ax = b for given batched matrix A and batched vector b
 * 
 * @param[in]  A  Batched matrix A
 * @param[in]  b  Batched vector b
 * @return A\b
 */
BatchedMatrix b_solve(const BatchedMatrix& A, const BatchedMatrix& b) {
  auto num_batches = A.batches();
  const auto& handle = A.cublasHandle();

  int n = A.shape().first;
  auto allocator = A.allocator();
  int* P = (int*)allocator->allocate(sizeof(int) * n * num_batches, 0);
  int* info = (int*)allocator->allocate(sizeof(int) * num_batches, 0);

  // A copy of A is necessary as the cublas operations write in A
  BatchedMatrix Acopy(n, n, num_batches, A.cublasHandle(), A.allocator(),
                      A.stream());
  copy(Acopy.raw_data(), A.raw_data(), n * n * num_batches, A.stream());

  BatchedMatrix Ainv(n, n, num_batches, A.cublasHandle(), A.allocator(),
                     A.stream());

  CUBLAS_CHECK(MLCommon::LinAlg::cublasgetrfBatched(handle, n, Acopy.data(), n,
                                                    P, info, num_batches));
  CUBLAS_CHECK(MLCommon::LinAlg::cublasgetriBatched(
    handle, n, Acopy.data(), n, P, Ainv.data(), n, info, num_batches));

  BatchedMatrix x = Ainv * b;

  allocator->deallocate(P, sizeof(int) * n * num_batches, A.stream());
  allocator->deallocate(info, sizeof(int) * num_batches, A.stream());

  return x;
}

/**
 * @brief The batched kroneker product A (x) B for given batched matrix A
 *        and batched matrix B
 * 
 * @param[in]  A  Matrix A
 * @param[in]  B  Matrix B
 * @return A (x) B
 */
BatchedMatrix b_kron(const BatchedMatrix& A, const BatchedMatrix& B) {
  int m = A.shape().first;
  int n = A.shape().second;

  int p = B.shape().first;
  int q = B.shape().second;

  // Resulting shape
  int k_m = m * p;
  int k_n = n * q;

  BatchedMatrix AkB(k_m, k_n, A.batches(), A.cublasHandle(), A.allocator(),
                    A.stream());

  // Run kronecker
  dim3 threads(std::min(p, 32), std::min(q, 32));
  kronecker_product_kernel<<<A.batches(), threads, 0, A.stream()>>>(
    A.raw_data(), m, n, B.raw_data(), p, q, AkB.raw_data(), k_m, k_n);
  CUDA_CHECK(cudaPeekAtLastError());
  return AkB;
}

}  // namespace Matrix
}  // namespace MLCommon
