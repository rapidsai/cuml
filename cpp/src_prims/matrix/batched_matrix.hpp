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
#include <cuML.hpp>

namespace MLCommon {
namespace Matrix {

/* An allocation function for `BatchedMatrixMemory`. Written as a free function because I had trouble
 * getting the __device__ lambda to compile as a member function of the `BatchedMatrixMemory` struct.
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

  // allocate and fill array of pointers to each batch matrix
  double** A_array =
    (double**)allocator->allocate(sizeof(double*) * num_batches, stream);
  // fill array of pointers to each batch matrix.
  auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(
    thrust::cuda::par.on(stream), counting, counting + num_batches,
    [=] __device__(int bid) { A_array[bid] = &(A_dense[bid * m * n]); });
  return std::make_pair(A_dense, A_array);
}

//! The BatchedMatrix class provides storage and a number of linear operations on collections (batches) of matrices of identical shape.
class BatchedMatrix {
 public:
  /* Constructor that allocates memory using the memory pool.
  * @param m Number of rows
  * @param n Number of columns
  * @param num_batches Number of matrices in the batch
  * @param pool The memory pool
  * @param setZero Should matrix be zeroed on allocation?
  */
  BatchedMatrix(int m, int n, int num_batches, cublasHandle_t cublasHandle,
                std::shared_ptr<ML::deviceAllocator> allocator,
                cudaStream_t stream, bool setZero = true)
    : m_num_batches(num_batches),
      m_allocator(allocator),
      m_cublasHandle(cublasHandle),
      m_stream(stream) {
    m_shape = std::make_pair(m, n);

    // allocate memory
    auto memory =
      BMM_Allocate(m_shape, num_batches, setZero, allocator, stream);

    // Take these references to extract them from member-storage for the
    // lambda below. There are better C++14 ways to do this, but I'll keep it C++11 for now.
    auto& shape = m_shape;

    // note: we create this "free" function with explicit copies to ensure that the
    // deallocate function gets called with the correct values.
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

  //! return stream
  cudaStream_t stream() const { return m_stream; }

  //! Return shape
  const std::pair<int, int>& shape() const { return m_shape; }

  //! Return pointer array
  double** data() const { return m_A_batches.get(); }

  /* Return specific matrix pointer
   * @param id Return the pointer to `id` matrix.
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
    cudaMemcpyAsync(toVec[0], this->operator[](0),
                    m_num_batches * r * sizeof(double),
                    cudaMemcpyDeviceToDevice, m_stream);
    return toVec;
  }

  /* Create a matrix from a long vector.
   * @param m Number of desired rows
   * @param n Number of desired columns
   */
  BatchedMatrix mat(int m, int n) const {
    int r = m_shape.first * m_shape.second;
    BatchedMatrix toMat(m, n, m_num_batches, m_cublasHandle, m_allocator,
                        m_stream);
    cudaMemcpyAsync(toMat[0], this->operator[](0),
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

  //! Kernel to creates an Identity matrix
  __global__ void identity_matrix_kernel(double** I, int r) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    for (int b = 0; b < r; b += blockDim.x) {
      int idx = tid + b;
      if (idx < r) {
        I[bid][idx + r * idx] = 1;
      }
    }
  }

  /* Initialize a batched identity matrix.
   * @param m Number of rows/columns of matrix
   * @param num_batches Number of matrices in batch
   * @param pool Memory pool
   */
  static BatchedMatrix Identity(int m, int num_batches, cublasHandle_t handle,
                                std::shared_ptr<ML::deviceAllocator> allocator,
                                cudaStream_t stream) {
    BatchedMatrix I(m, m, num_batches, handle, allocator, stream, true);

    identity_matrix_kernel<<<num_batches, std::min(1024, m), 0, stream>>>(
      I.data(), m);

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

/* Computes batched kronecker prodcut between AkB <- A (x) B
 * @param A Array of pointers to matrices `A`
 * @param m number of rows (A)
 * @param n number of columns (A)
 * @param B Array of pointers to matrices `B`
 * @param p number of rows (B)
 * @param q number of columns (B)
 * @param AkB Result kronecker product pointer array.
 * @param k_m Result number of rows
 * @param k_n Result number of columns
 */
__global__ void kronecker_product_kernel(double** A, int m, int n, double** B,
                                         int p, int q, double** AkB, int k_m,
                                         int k_n) {
  int bid = blockIdx.x;

  for (int ia = 0; ia < n; ia++) {
    for (int ja = 0; ja < m; ja++) {
      double A_ia_ja = A[bid][ia + ja * m];

      for (int iblock = 0; iblock < p; iblock += blockDim.x) {
        int ib = threadIdx.x + iblock * blockDim.x;
        if (ib < p) {
          for (int jblock = 0; jblock < q; jblock += blockDim.y) {
            int jb = threadIdx.y + jblock * blockDim.y;
            if (jb < q) {
              int i_ab = ia * p + ib;
              int j_ab = ja * q + jb;
              AkB[bid][i_ab + j_ab * k_m] = A_ia_ja * B[bid][ib + jb * p];
            }
          }
        }
      }
    }
  }
}

/* Multiplies each matrix in a batch-A with it's batch-B counterpart.
 * A = [A1,A2,A3], B=[B1,B2,B3]
 * returns [A1*B1, A2*B2, A3*B3]
 * @param A First matrix batch
 * @param B Second matrix batch
 * @param aT Is `A` transposed?
 * @param bT is `B` transposed?
 * @return member-wise A*B
 */
BatchedMatrix b_gemm(const BatchedMatrix& A, const BatchedMatrix& B,
                     bool aT = false, bool bT = false) {
  if (A.batches() != B.batches()) {
    throw std::runtime_error("A & B must have same number of batches");
  }

  // logic for matrix dimensions with optional transpose
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
  CUBLAS_CHECK(cublasDgemmBatched(handle,
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
                                  num_batches));
  return C;
}

/* A utility method to implement pointwise operations between elements of two batched matrices.
 * @param A First matrix
 * @param B Second matrix
 * @param binary_op The binary operation used on elements of `A` and `B`
 * @return The result of `A` `binary_op` `B`
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

  LinAlg::binaryOp(C[0], A[0], B[0], m * n * num_batches, binary_op,
                   A.stream());

  return C;
}

/* Multiplies each matrix in a batch-A with it's batch-B counterpart.
 * A = [A1,A2,A3], B=[B1,B2,B3]
 * return [A1*B1, A2*B2, A3*B3]
 * @param A Matrix `A`
 * @param B Matrix `B`
 * @returns The result of the batched matrix-matrix multiplication of `A` x `B`
 */
BatchedMatrix operator*(const BatchedMatrix& A, const BatchedMatrix& B) {
  return b_gemm(A, B);
}

/* Adds two batched matrices together element-wise.
 * @param Matrix A
 * @param Matrix B
 * @return A+B
 */
BatchedMatrix operator+(const BatchedMatrix& A, const BatchedMatrix& B) {
  return b_aA_op_B(A, B, [] __device__(double a, double b) { return a + b; });
}

/* Subtract two batched matrices together element-wise.
 * @param Matrix A
 * @param Matrix B
 * @return A-B
 */
BatchedMatrix operator-(const BatchedMatrix& A, const BatchedMatrix& B) {
  return b_aA_op_B(A, B, [] __device__(double a, double b) { return a - b; });
}

/* Solve Ax = b for given batched matrix A and batched vector b
 * @param Matrix A
 * @param vector b
 * @return A\b
 */
BatchedMatrix b_solve(const BatchedMatrix& A, const BatchedMatrix& b) {
  auto num_batches = A.batches();
  const auto& handle = A.cublasHandle();

  int n = A.shape().first;
  auto allocator = A.allocator();
  int* P = (int*)allocator->allocate(sizeof(int) * n * num_batches, 0);
  int* info = (int*)allocator->allocate(sizeof(int) * num_batches, 0);

  BatchedMatrix Ainv(n, n, num_batches, A.cublasHandle(), A.allocator(),
                     A.stream());

  CUBLAS_CHECK(
    cublasDgetrfBatched(handle, n, A.data(), n, P, info, num_batches));
  CUBLAS_CHECK(cublasDgetriBatched(handle, n, A.data(), n, P, Ainv.data(), n,
                                   info, num_batches));

  BatchedMatrix x = Ainv * b;

  allocator->deallocate(P, sizeof(int) * n * num_batches, A.stream());
  allocator->deallocate(info, sizeof(int) * num_batches, A.stream());

  return x;
}

/* The batched kroneker product A (x) B for given batched matrix A and batched matrix B
 * @param Matrix A
 * @param Matrix B
 * @return A (x) B
 */
BatchedMatrix b_kron(const BatchedMatrix& A, const BatchedMatrix& B) {
  int m = A.shape().first;
  int n = A.shape().second;

  int p = A.shape().first;
  int q = B.shape().second;

  // resulting shape
  int km = m * p;
  int kn = n * q;

  BatchedMatrix AkB(km, kn, A.batches(), A.cublasHandle(), A.allocator(),
                    A.stream());

  // run kronecker...
  dim3 threads(std::min(p, 32), std::min(q, 32));
  kronecker_product_kernel<<<A.batches(), threads, 0, A.stream()>>>(
    A.data(), m, n, B.data(), p, q, AkB.data(), km, kn);

  return AkB;
}

}  // namespace Matrix
}  // namespace MLCommon
