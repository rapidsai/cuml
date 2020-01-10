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
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuml/common/utils.hpp>
#include <cuml/cuml.hpp>

#include <linalg/binary_op.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/unary_op.h>


namespace MLCommon {
namespace LinAlg {
namespace Batched {

/**
 * @brief An allocation function for `BatchedMatrix`.
 * 
 * @param[in]  shape        Shape of each matrix (rows, columns)
 * @param[in]  batch_size   Number of matrices in the batch
 * @param[in]  setZero      Whether to initialize the allocated matrix with all zeros
 * @param[in]  allocator    Device memory allocator
 * @param[in]  stream       CUDA stream
 * 
 * @return Pair (A_dense, A_array): pointer to the raw data and pointer
 *         to the array of pointers to each matrix in the batch
 */
template <typename T>
std::pair<T*, T**> BMM_Allocate(std::pair<int, int> shape, int batch_size,
                                bool setZero,
                                std::shared_ptr<ML::deviceAllocator> allocator,
                                cudaStream_t stream) {
  int m = shape.first;
  int n = shape.second;

  // Allocate dense batched matrix and possibly set to zero
  T* A_dense = (T*)allocator->allocate(sizeof(T) * m * n * batch_size, stream);
  if (setZero)
    CUDA_CHECK(
      cudaMemsetAsync(A_dense, 0, sizeof(T) * m * n * batch_size, stream));

  // Allocate and fill array of pointers to each batch matrix
  T** A_array = (T**)allocator->allocate(sizeof(T*) * batch_size, stream);
  // Fill array of pointers to each batch matrix.
  auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(
    thrust::cuda::par.on(stream), counting, counting + batch_size,
    [=] __device__(int bid) { A_array[bid] = &(A_dense[bid * m * n]); });
  return std::make_pair(A_dense, A_array);
}

/**
 * @brief Kernel to create an identity matrix
 * 
 * @note The block id is the batch id, and the thread id is the starting
 *       row/column for this thread (then looping to cover all the diagonal)
 * 
 * @param[out]  I  Pointer to the raw data of the identity matrix to create
 * @param[in]   m  Number of rows/columns of matrix
 */
template <typename T>
__global__ void identity_matrix_kernel(T* I, int m) {
  T* I_b = I + blockIdx.x * m * m;
  int stride = (m + 1);
  for (int idx = threadIdx.x; idx < m; idx += blockDim.x) {
    I_b[idx * stride] = 1;
  }
}

/**
 * @brief Kernel to compute the difference of batched vectors with a given
 *        period (1 for simple difference, s for seasonal difference)
 *
 * @note: The thread id is the starting position in each vector and the block
 *        id is the batch id.
 * 
 * @param[in]  in      Input vector
 * @param[out] out     Output vector
 * @param[in]  n_elem  Number of elements in the input vector
 * @param[in]  period  Period of the difference
 */
template <typename T>
__global__ void batched_diff_kernel(const T* in, T* out, int n_elem,
                                    int period = 1) {
  const T* batch_in = in + n_elem * blockIdx.x;
  T* batch_out = out + (n_elem - period) * blockIdx.x;

  for (int i = threadIdx.x; i < n_elem - period; i += blockDim.x) {
    batch_out[i] = batch_in[i + period] - batch_in[i];
  }
}

/**
 * @brief Kernel to compute the second difference of batched vectors with given
 *        periods (1 for simple difference, s for seasonal difference)
 *
 * @note: The thread id is the starting position in each vector and the block
 *        id is the batch id.
 * 
 * @param[in]  in       Input vector
 * @param[out] out      Output vector
 * @param[in]  n_elem   Number of elements in the input vector
 * @param[in]  period1  Period for the 1st difference
 * @param[in]  period2  Period for the 2nd difference
 */
template <typename T>
__global__ void batched_second_diff_kernel(const T* in, T* out, int n_elem,
                                           int period1 = 1, int period2 = 1) {
  const T* batch_in = in + n_elem * blockIdx.x;
  T* batch_out = out + (n_elem - period1 - period2) * blockIdx.x;

  for (int i = threadIdx.x; i < n_elem - period1 - period2; i += blockDim.x) {
    batch_out[i] = batch_in[i + period1 + period2] - batch_in[i + period1] -
                   batch_in[i + period2] + batch_in[i];
  }
}

/**
 * @brief The BatchedMatrix class provides storage and a number of linear
 *        operations on collections (batches) of matrices of identical shape.
 */
template <typename T>
class BatchedMatrix {
 public:
  /**
   * @brief Constructor that allocates memory using the memory pool.
   * 
   * @param[in]  m            Number of rows
   * @param[in]  n            Number of columns
   * @param[in]  batch_size   Number of matrices in the batch
   * @param[in]  cublasHandle cublas handle
   * @param[in]  allocator    device allocator
   * @param[in]  stream       cuda stream where to schedule work
   * @param[in]  setZero      Should matrix be zeroed on allocation?
   */
  BatchedMatrix(int m, int n, int batch_size, cublasHandle_t cublasHandle,
                std::shared_ptr<ML::deviceAllocator> allocator,
                cudaStream_t stream, bool setZero = true)
    : m_batch_size(batch_size),
      m_allocator(allocator),
      m_cublasHandle(cublasHandle),
      m_stream(stream) {
    m_shape = std::make_pair(m, n);

    // Allocate memory
    auto memory =
      BMM_Allocate<T>(m_shape, batch_size, setZero, allocator, stream);

    /* Take these references to extract them from member-storage for the
     * lambda below. There are better C++14 ways to do this, but I'll keep
     * it C++11 for now. */
    auto& shape = m_shape;

    /* Note: we create this "free" function with explicit copies to ensure that
     * the deallocate function gets called with the correct values. */
    auto f1 = [allocator, batch_size, shape, stream](T* A) {
      allocator->deallocate(
        A, batch_size * shape.first * shape.second * sizeof(T), stream);
    };

    auto f2 = [allocator, batch_size, stream](T** A) {
      allocator->deallocate(A, sizeof(T*) * batch_size, stream);
    };

    // When this shared pointer count goes to 0, `f` is called to deallocate
    // the memory
    m_A_dense = std::shared_ptr<T>(memory.first, f1);
    m_A_batches = std::shared_ptr<T*>(memory.second, f2);
  }

  //! Return batches
  size_t batches() const { return m_batch_size; }

  //! Return cublas handle
  cublasHandle_t cublasHandle() const { return m_cublasHandle; }

  //! Return allocator
  std::shared_ptr<deviceAllocator> allocator() const { return m_allocator; }

  //! Return stream
  cudaStream_t stream() const { return m_stream; }

  //! Return shape
  const std::pair<int, int>& shape() const { return m_shape; }

  //! Return pointer array
  T** data() const { return m_A_batches.get(); }

  //! Return pointer to the underlying memory
  T* raw_data() const { return m_A_dense.get(); }

  /**
   * @brief Return pointer to the data of a specific matrix
   * 
   * @param[in]  id  id of the matrix
   * @return         A pointer to the raw data of the matrix
   */
  T* operator[](int id) const {
    return &(m_A_dense.get()[id * m_shape.first * m_shape.second]);
  }

  /**
   * @brief   Deep copy of the batched matrix
   * @note    Avoiding a copy constructor at the moment (rule of 3/5/0)
   * @return  A batched matrix containing the same data
   */
  BatchedMatrix<T> deepcopy() const {
    BatchedMatrix<T> out(m_shape.first, m_shape.second, m_batch_size,
                         m_cublasHandle, m_allocator, m_stream);
    copy(out[0], m_A_dense.get(), m_batch_size * m_shape.first * m_shape.second,
         m_stream);
    return out;
  }

  //! Stack the matrix by columns creating a long vector
  BatchedMatrix<T> vec() const {
    int m = m_shape.first;
    int n = m_shape.second;
    int r = m * n;
    BatchedMatrix<T> toVec(r, 1, m_batch_size, m_cublasHandle, m_allocator,
                           m_stream);
    copy(toVec[0], m_A_dense.get(), m_batch_size * r, m_stream);
    return toVec;
  }

  /**
   * @brief Create a matrix from a long vector.
   * 
   * @param[in]  m  Number of desired rows
   * @param[in]  n  Number of desired columns
   * @return        A batched matrix
   */
  BatchedMatrix<T> mat(int m, int n) const {
    const int r = m_shape.first * m_shape.second;
    ASSERT(
      r == m * n,
      "ERROR BatchedMatrix::mat(m,n): Size mismatch - Cannot reshape array "
      "into desired size");
    BatchedMatrix<T> toMat(m, n, m_batch_size, m_cublasHandle, m_allocator,
                           m_stream);
    copy(toMat[0], m_A_dense.get(), m_batch_size * r, m_stream);

    return toMat;
  }

  //! Visualize the first matrix.
  void print(std::string name) const {
    size_t len = m_shape.first * m_shape.second * m_batch_size;
    std::vector<T> A(len);
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
   * @brief Compute the difference of the batched vector with a given period
   *        (1 for simple difference, s for seasonal)
   * 
   * @param[in]  period  Period of the difference (defaults to 1)
   *
   * @return A batched vector corresponding to the first difference. Matches
   *         the layout of the input vector (row or column vector)
   */
  BatchedMatrix<T> difference(int period = 1) const {
    ASSERT(m_shape.first == 1 || m_shape.second == 1,
           "Invalid operation: must be a vector");
    int len = m_shape.second * m_shape.first;
    ASSERT(len > period, "Length of the vector must be > period");

    // Create output batched vector
    bool row_vector = (m_shape.first == 1);
    BatchedMatrix<T> out(row_vector ? 1 : len - period,
                         row_vector ? len - period : 1, m_batch_size,
                         m_cublasHandle, m_allocator, m_stream);

    // Execute kernel
    const int TPB = (len - period) > 512 ? 256 : 128;  // quick heuristics
    batched_diff_kernel<<<m_batch_size, TPB, 0, m_stream>>>(
      raw_data(), out.raw_data(), len, period);
    CUDA_CHECK(cudaPeekAtLastError());

    return out;
  }

  /**
   * @brief Initialize a batched identity matrix.
   * 
   * @param[in]  m            Number of rows/columns of matrix
   * @param[in]  batch_size   Number of matrices in batch
   * @param[in]  handle       cuml handle
   * @param[in]  allocator    device allocator
   * @param[in]  stream       cuda stream to schedule work on
   * 
   * @return A batched identity matrix
   */
  static BatchedMatrix Identity(int m, int batch_size, cublasHandle_t handle,
                                std::shared_ptr<ML::deviceAllocator> allocator,
                                cudaStream_t stream) {
    BatchedMatrix I(m, m, batch_size, handle, allocator, stream, true);

    identity_matrix_kernel<T>
      <<<batch_size, std::min(1024, m), 0, stream>>>(I.raw_data(), m);
    CUDA_CHECK(cudaGetLastError());
    return I;
  }

 protected:
  //! Shape (rows, cols) of matrices. We assume all matrices in batch have same shape.
  std::pair<int, int> m_shape;

  //! Array(pointer) to each matrix.
  std::shared_ptr<T*> m_A_batches;

  //! Data pointer to first element of dense matrix data.
  std::shared_ptr<T> m_A_dense;

  //! Number of matrices in batch
  size_t m_batch_size;

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
template <typename T>
__global__ void kronecker_product_kernel(const T* A, int m, int n, const T* B,
                                         int p, int q, T* AkB, int k_m,
                                         int k_n) {
  const T* A_b = A + blockIdx.x * m * n;
  const T* B_b = B + blockIdx.x * p * q;
  T* AkB_b = AkB + blockIdx.x * k_m * k_n;

  for (int ia = 0; ia < m; ia++) {
    for (int ja = 0; ja < n; ja++) {
      T A_ia_ja = A_b[ia + ja * m];

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
 * @brief Batched GEMM operation (exhaustive version)
 *        [C1, C2, ...] = [alpha*A1*B1+beta*C1, alpha*A2*B2+beta*C2, ...]
 *
 * @param[in]      aT     Is `A` transposed?
 * @param[in]      bT     Is `B` transposed?
 * @param[in]      m      Number of rows of A or A.T
 * @param[in]      n      Number of columns of B or B.T
 * @param[in]      k      Common dimension of A or A.T and B or B.T
 * @param[in]      alpha  Parameter alpha
 * @param[in]      A      Batch of matrices A
 * @param[in]      B      Batch of matrices B
 * @param[in]      beta   Parameter beta
 * @param[in,out]  C      Batch of matrices C
 */
template <typename T>
void b_gemm(bool aT, bool bT, int m, int n, int k, T alpha,
            const BatchedMatrix<T>& A, const BatchedMatrix<T>& B, T beta,
            const BatchedMatrix<T>& C) {
  // Check the parameters
  {
    ASSERT(A.batches() == B.batches(),
           "A and B must have the same number of batches");
    ASSERT(A.batches() == C.batches(),
           "A and C must have the same number of batches");
    int Arows = !aT ? A.shape().first : A.shape().second;
    int Acols = !aT ? A.shape().second : A.shape().first;
    int Brows = !bT ? B.shape().first : B.shape().second;
    int Bcols = !bT ? B.shape().second : B.shape().first;
    ASSERT(m <= Arows, "m should be <= number of rows of A");
    ASSERT(k <= Acols, "k should be <= number of columns of A");
    ASSERT(k <= Brows, "k should be <= number of rows of B");
    ASSERT(n <= Bcols, "n should be <= number of columns of B");
    ASSERT(m <= C.shape().first, "m should be <= number of rows of C");
    ASSERT(n <= C.shape().second, "n should be <= number of columns of C");
  }

  // Set transpose modes
  cublasOperation_t opA = aT ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = bT ? CUBLAS_OP_T : CUBLAS_OP_N;

  // Call cuBLAS
  CUBLAS_CHECK(LinAlg::cublasgemmStridedBatched(
    A.cublasHandle(), opA, opB, m, n, k, &alpha, A.raw_data(), A.shape().first,
    A.shape().first * A.shape().second, B.raw_data(), B.shape().first,
    B.shape().first * B.shape().second, &beta, C.raw_data(), C.shape().first,
    C.shape().first * C.shape().second, A.batches(), A.stream()));
}

/**
 * @brief Multiplies each matrix in a batch-A with it's batch-B counterpart.
 *        A = [A1,A2,A3], B=[B1,B2,B3] returns [A1*B1, A2*B2, A3*B3]
 * 
 * @param[in]  A   First matrix batch
 * @param[in]  B   Second matrix batch
 * @param[in]  aT  Is `A` transposed?
 * @param[in]  bT  Is `B` transposed?
 * 
 * @return Member-wise A*B
 */
template <typename T>
BatchedMatrix<T> b_gemm(const BatchedMatrix<T>& A, const BatchedMatrix<T>& B,
                        bool aT = false, bool bT = false) {
  // m = number of rows of matrix op(A) and C.
  int m = !aT ? A.shape().first : A.shape().second;
  // n = number of columns of matrix op(B) and C.
  int n = !bT ? B.shape().second : B.shape().first;

  // k = number of columns of op(A) and rows of op(B).
  int k = !aT ? A.shape().second : A.shape().first;
  int kB = !bT ? B.shape().first : B.shape().second;

  ASSERT(k == kB, "Matrix-Multiplication dimensions don't match!");

  // Create C(m,n)
  BatchedMatrix<T> C(m, n, A.batches(), A.cublasHandle(), A.allocator(),
                     A.stream());

  b_gemm(aT, bT, m, n, k, (T)1, A, B, (T)0, C);
  return C;
}

/**
 * @brief Wrapper around cuBLAS batched gels (least-square solver of Ax=C)
 * 
 * @details: - This simple wrapper only supports non-transpose mode.
 *           - There isn't any strided version in cuBLAS yet.
 *           - cuBLAS only supports overdetermined systems.
 *           - This function copies A to avoid modifying the original one.
 * 
 * @param[in]      A  Batched matrix A (must have more rows than columns)
 * @param[in|out]  C  Batched matrix C (the number of rows must match A)
 */
template <typename T>
void b_gels(const BatchedMatrix<T>& A, BatchedMatrix<T>& C) {
  ASSERT(A.batches() == C.batches(),
         "A and C must have the same number of batches");
  int m = A.shape().first;
  ASSERT(C.shape().first == m, "Dimension mismatch: A rows, C rows");
  int n = A.shape().second;
  ASSERT(m > n, "Only overdetermined systems (m > n) are supported");
  int nrhs = C.shape().second;

  BatchedMatrix<T> Acopy = A.deepcopy();

  int info;
  CUBLAS_CHECK(MLCommon::LinAlg::cublasgelsBatched(
    A.cublasHandle(), CUBLAS_OP_N, m, n, nrhs, Acopy.data(), m, C.data(), m,
    &info, nullptr, A.batches()));
}

/**
 * @brief A utility method to implement a unary operation on a batched matrix
 * 
 * @param[in]  A          Batched matrix A
 * @param[in]  unary_op   The unary operation applied on the elements of A
 * @return A batched matrix, the result of unary_op A
 */
template <typename T, typename F>
BatchedMatrix<T> b_op_A(const BatchedMatrix<T>& A, F unary_op) {
  auto batch_size = A.batches();
  int m = A.shape().first;
  int n = A.shape().second;

  BatchedMatrix<T> C(m, n, batch_size, A.cublasHandle(), A.allocator(),
                     A.stream());

  LinAlg::unaryOp(C.raw_data(), A.raw_data(), m * n * batch_size, unary_op,
                  A.stream());

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
template <typename T, typename F>
BatchedMatrix<T> b_aA_op_B(const BatchedMatrix<T>& A, const BatchedMatrix<T>& B,
                           F binary_op) {
  ASSERT(
    A.shape().first == B.shape().first && A.shape().second == B.shape().second,
    "Batched Matrix Addition ERROR: Matrices must be same size");

  ASSERT(A.batches() == B.batches(), "A & B must have same number of batches");

  auto batch_size = A.batches();
  int m = A.shape().first;
  int n = A.shape().second;

  BatchedMatrix<T> C(m, n, batch_size, A.cublasHandle(), A.allocator(),
                     A.stream());

  LinAlg::binaryOp(C.raw_data(), A.raw_data(), B.raw_data(), m * n * batch_size,
                   binary_op, A.stream());

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
template <typename T>
BatchedMatrix<T> operator*(const BatchedMatrix<T>& A,
                           const BatchedMatrix<T>& B) {
  return b_gemm(A, B);
}

/**
 * @brief Adds two batched matrices together element-wise.
 * 
 * @param[in]  A  Batched matrix A
 * @param[in]  B  Batched matrix B
 * @return A+B
 */
template <typename T>
BatchedMatrix<T> operator+(const BatchedMatrix<T>& A,
                           const BatchedMatrix<T>& B) {
  return b_aA_op_B(A, B, [] __device__(T a, T b) { return a + b; });
}

/**
 * @brief Subtract two batched matrices together element-wise.
 * 
 * @param[in]  A  Batched matrix A
 * @param[in]  B  Batched matrix B
 * @return A-B
 */
template <typename T>
BatchedMatrix<T> operator-(const BatchedMatrix<T>& A,
                           const BatchedMatrix<T>& B) {
  return b_aA_op_B(A, B, [] __device__(T a, T b) { return a - b; });
}

/**
 * @brief Unary substraction
 * 
 * @param[in]  A  Batched matrix A
 * @return -A
 */
template <typename T>
BatchedMatrix<T> operator-(const BatchedMatrix<T>& A) {
  return b_op_A(A, [] __device__(T a) { return -a; });
}

/**
 * @brief Solve Ax = b for given batched matrix A and batched vector b
 * 
 * @param[in]  A  Batched matrix A
 * @param[in]  b  Batched vector b
 * @return A\b
 */
template <typename T>
BatchedMatrix<T> b_solve(const BatchedMatrix<T>& A, const BatchedMatrix<T>& b) {
  auto batch_size = A.batches();
  const auto& handle = A.cublasHandle();

  int n = A.shape().first;
  auto allocator = A.allocator();
  int* P = (int*)allocator->allocate(sizeof(int) * n * batch_size, A.stream());
  int* info = (int*)allocator->allocate(sizeof(int) * batch_size, A.stream());

  // A copy of A is necessary as the cublas operations write in A
  BatchedMatrix<T> Acopy = A.deepcopy();

  BatchedMatrix<T> Ainv(n, n, batch_size, A.cublasHandle(), A.allocator(),
                        A.stream());

  CUBLAS_CHECK(LinAlg::cublasgetrfBatched(handle, n, Acopy.data(), n, P, info,
                                          batch_size, A.stream()));
  CUBLAS_CHECK(LinAlg::cublasgetriBatched(handle, n, Acopy.data(), n, P,
                                          Ainv.data(), n, info, batch_size,
                                          A.stream()));

  BatchedMatrix<T> x = Ainv * b;

  allocator->deallocate(P, sizeof(int) * n * batch_size, A.stream());
  allocator->deallocate(info, sizeof(int) * batch_size, A.stream());

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
template <typename T>
BatchedMatrix<T> b_kron(const BatchedMatrix<T>& A, const BatchedMatrix<T>& B) {
  int m = A.shape().first;
  int n = A.shape().second;

  int p = B.shape().first;
  int q = B.shape().second;

  // Resulting shape
  int k_m = m * p;
  int k_n = n * q;

  BatchedMatrix<T> AkB(k_m, k_n, A.batches(), A.cublasHandle(), A.allocator(),
                       A.stream());

  // Run kronecker
  dim3 threads(std::min(p, 32), std::min(q, 32));
  kronecker_product_kernel<T><<<A.batches(), threads, 0, A.stream()>>>(
    A.raw_data(), m, n, B.raw_data(), p, q, AkB.raw_data(), k_m, k_n);
  CUDA_CHECK(cudaPeekAtLastError());
  return AkB;
}

/**
 * @brief Kernel to create a batched lagged matrix from a given batched vector
 * 
 * @note The block id is the batch id and the thread id is the starting index
 * 
 * @param[in]  vec              Input vector
 * @param[out] mat              Output lagged matrix
 * @param[in]  lags             Number of lags
 * @param[in]  lagged_height    Height of the lagged matrix
 * @param[in]  vec_offset       Offset in the input vector
 * @param[in]  ld               Length of the underlying vector
 * @param[in]  mat_offset       Offset in the lagged matrix
 * @param[in]  ls_batch_stride  Stride between batches in the output matrix
 * @param[in]  s                Seasonality of the lags
 */
template <typename T>
__global__ void lagged_mat_kernel(const T* vec, T* mat, int lags,
                                  int lagged_height, int vec_offset, int ld,
                                  int mat_offset, int ls_batch_stride,
                                  int s = 1) {
  const T* batch_in = vec + blockIdx.x * ld + vec_offset;
  T* batch_out = mat + blockIdx.x * ls_batch_stride + mat_offset;

  for (int lag = 0; lag < lags; lag++) {
    const T* b_in = batch_in + s * (lags - lag - 1);
    T* b_out = batch_out + lag * lagged_height;
    for (int i = threadIdx.x; i < lagged_height; i += blockDim.x) {
      b_out[i] = b_in[i];
    }
  }
}

/**
 * @brief Create a batched lagged matrix from a given batched vector
 * 
 * @note This overload takes both batched matrices as inputs
 * 
 * @param[in]  vec            Input vector
 * @param[out] lagged_mat     Output matrix
 * @param[in]  lags           Number of lags
 * @param[in]  lagged_height  Height of the lagged matrix
 * @param[in]  vec_offset     Offset in the input vector
 * @param[in]  mat_offset     Offset in the lagged matrix
 * @param[in]  s              Period of the lags
 */
template <typename T>
void b_lagged_mat(const BatchedMatrix<T>& vec, BatchedMatrix<T>& lagged_mat,
                  int lags, int lagged_height, int vec_offset, int mat_offset,
                  int s = 1) {
  // Verify all the dimensions ; it's better to fail loudly than hide errors
  ASSERT(vec.batches() == lagged_mat.batches(),
         "The numbers of batches of the matrix and the vector must match");
  ASSERT(vec.shape().first == 1 || vec.shape().second == 1,
         "The first argument must be a vector (either row or column)");
  int len = vec.shape().first == 1 ? vec.shape().second : vec.shape().first;
  int mat_batch_stride = lagged_mat.shape().first * lagged_mat.shape().second;
  ASSERT(lagged_height <= len - s * lags - vec_offset,
         "Lagged height can't exceed vector length - s * lags - vector offset");
  ASSERT(mat_offset <= mat_batch_stride - lagged_height * lags,
         "Matrix offset can't exceed real matrix size - lagged matrix size");

  // Execute the kernel
  const int TPB = lagged_height > 512 ? 256 : 128;  // quick heuristics
  lagged_mat_kernel<<<vec.batches(), TPB, 0, vec.stream()>>>(
    vec.raw_data(), lagged_mat.raw_data(), lags, lagged_height, vec_offset, len,
    mat_offset, mat_batch_stride, s);
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Create a batched lagged matrix from a given batched vector
 * 
 * @note This overload takes the input vector and returns the output matrix.
 *       For more control, use the other overload.
 * 
 * @param[in]  vec            Input vector
 * @param[in]  lags           Number of lags
 * 
 * @return A batched matrix corresponding to the output lagged matrix
 */
template <typename T>
BatchedMatrix<T> b_lagged_mat(const BatchedMatrix<T>& vec, int lags) {
  ASSERT(vec.shape().first == 1 || vec.shape().second == 1,
         "The first argument must be a vector (either row or column)");
  int len = vec.shape().first * vec.shape().second;
  ASSERT(lags < len, "The number of lags can't exceed the vector length");
  int lagged_height = len - lags;

  // Create output matrix
  BatchedMatrix<T> lagged_mat(lagged_height, lags, vec.batches(),
                              vec.cublasHandle(), vec.allocator(), vec.stream(),
                              false);
  // Call exhaustive version of the function
  b_lagged_mat(vec, lagged_mat, lags, lagged_height, 0, 0);

  return lagged_mat;
}

/**
 * @brief Kernel to compute a 2D copy of a window in a batched matrix.
 * 
 * @note The blocks are the batches and the threads are the matrix elements,
 *       column-wise.
 * 
 * @param[in]  in            Input matrix
 * @param[out] out           Output matrix
 * @param[in]  starting_row  First row to copy
 * @param[in]  starting_col  First column to copy
 * @param[in]  in_rows       Number of rows in the input matrix
 * @param[in]  in_cols       Number of columns in the input matrix
 * @param[in]  out_rows      Number of rows to copy
 * @param[in]  out_cols      Number of columns to copy
 */
template <typename T>
static __global__ void batched_2dcopy_kernel(const T* in, T* out,
                                             int starting_row, int starting_col,
                                             int in_rows, int in_cols,
                                             int out_rows, int out_cols) {
  const T* in_ =
    in + blockIdx.x * in_rows * in_cols + starting_col * in_rows + starting_row;
  T* out_ = out + blockIdx.x * out_rows * out_cols;

  for (int i = threadIdx.x; i < out_rows * out_cols; i += blockDim.x) {
    int i_col = i / out_rows;
    int i_row = i % out_rows;
    out_[i] = in_[i_row + in_rows * i_col];
  }
}

/**
 * @brief Compute a 2D copy of a window in a batched matrix.
 * 
 * @note This overload takes two matrices as inputs
 * 
 * @param[in]  in            Batched input matrix
 * @param[out] out           Batched output matrix
 * @param[in]  starting_row  First row to copy
 * @param[in]  starting_col  First column to copy
 * @param[in]  out_rows      Number of rows to copy
 * @param[in]  out_cols      Number of columns to copy
 */
template <typename T>
void b_2dcopy(const BatchedMatrix<T>& in, BatchedMatrix<T>& out,
              int starting_row, int starting_col, int rows, int cols) {
  ASSERT(out.shape().first == rows, "Dimension mismatch: rows");
  ASSERT(out.shape().second == cols, "Dimension mismatch: columns");

  // Execute the kernel
  const int TPB = rows * cols > 512 ? 256 : 128;  // quick heuristics
  batched_2dcopy_kernel<<<in.batches(), TPB, 0, in.stream()>>>(
    in.raw_data(), out.raw_data(), starting_row, starting_col, in.shape().first,
    in.shape().second, rows, cols);
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Compute a 2D copy of a window in a batched matrix.
 * 
 * @note This overload only takes the input matrix as input and creates and
 *       returns the output matrix
 * 
 * @param[in]  in            Batched input matrix
 * @param[in]  starting_row  First row to copy
 * @param[in]  starting_col  First column to copy
 * @param[in]  out_rows      Number of rows to copy
 * @param[in]  out_cols      Number of columns to copy
 * 
 * @return The batched output matrix
 */
template <typename T>
BatchedMatrix<T> b_2dcopy(const BatchedMatrix<T>& in, int starting_row,
                          int starting_col, int rows, int cols) {
  // Create output matrix
  BatchedMatrix<T> out(rows, cols, in.batches(), in.cublasHandle(),
                       in.allocator(), in.stream(), false);

  // Call the other overload of the function
  b_2dcopy(in, out, starting_row, starting_col, rows, cols);

  return out;
}

}  // namespace Batched
}  // namespace LinAlg
}  // namespace MLCommon
