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

#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <cuml/common/utils.hpp>
#include <cuml/cuml.hpp>

#include <linalg/binary_op.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/unary_op.h>
#include <matrix/matrix.h>
#include <common/device_buffer.hpp>

namespace MLCommon {
namespace LinAlg {
namespace Batched {

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
 * @brief Helper kernel for the allocation: computes the array of pointers to each
 *        matrix in the strided batch
 *
 * @param[in]  A_dense     Raw data array
 * @param[out] A_array     Array of strided pointers to A_dense
 * @param[in]  batch_size  Number of matrices in the batch
 * @param[in]  m           Number of rows of each matrix
 * @param[in]  n           Number of columns of each matrix
 */
template <typename T>
__global__ void fill_strided_pointers_kernel(T* A_dense, T** A_array,
                                             int batch_size, int m, int n) {
  int bid = blockIdx.x * blockDim.x + threadIdx.x;
  if (bid < batch_size) {
    A_array[bid] = A_dense + bid * m * n;
  }
}

/**
 * @brief The Batched::Matrix class provides storage and a number of linear
 *        operations on collections (batches) of matrices of identical shape.
 */
template <typename T>
class Matrix {
 protected:
  /**
   * @brief Allocation method used by the constructors
   * 
   * @param[in]  setZero  Whether to initialize the allocated matrix with all zeros
   */
  void allocate(bool setZero = false) {
    int m = m_shape.first;
    int n = m_shape.second;
    auto& allocator = m_allocator;
    auto& stream = m_stream;
    auto batch_size = m_batch_size;

    // Allocate dense batched matrix and possibly set to zero
    T* A_dense =
      (T*)allocator->allocate(sizeof(T) * m * n * m_batch_size, m_stream);
    if (setZero)
      CUDA_CHECK(cudaMemsetAsync(A_dense, 0, sizeof(T) * m * n * m_batch_size,
                                 m_stream));

    // Allocate and fill array of pointers to each batch matrix
    T** A_array = (T**)allocator->allocate(sizeof(T*) * m_batch_size, m_stream);
    // Fill array of pointers to each batch matrix.
    constexpr int TPB = 256;
    fill_strided_pointers_kernel<<<ceildiv<int>(batch_size, TPB), TPB, 0,
                                   m_stream>>>(A_dense, A_array, batch_size, m,
                                               n);
    CUDA_CHECK(cudaPeekAtLastError());

    /* Note: we create this "free" function with explicit copies to ensure that
     * the deallocate function gets called with the correct values. */
    auto f1 = [allocator, batch_size, m, n, stream](T* A) {
      allocator->deallocate(A, batch_size * m * n * sizeof(T), stream);
    };

    auto f2 = [allocator, batch_size, stream](T** A) {
      allocator->deallocate(A, sizeof(T*) * batch_size, stream);
    };

    // When this shared pointer count goes to 0, `f` is called to deallocate
    // the memory
    m_A_dense = std::shared_ptr<T>(A_dense, f1);
    m_A_batches = std::shared_ptr<T*>(A_array, f2);
  }

 public:
  /**
   * @brief Constructor that allocates memory using the memory pool.
   * 
   * @param[in]  m            Number of rows
   * @param[in]  n            Number of columns
   * @param[in]  batch_size   Number of matrices in the batch
   * @param[in]  cublasHandle cuBLAS handle
   * @param[in]  allocator    Device memory allocator
   * @param[in]  stream       CUDA stream
   * @param[in]  setZero      Should matrix be zeroed on allocation?
   */
  Matrix(int m, int n, int batch_size, cublasHandle_t cublasHandle,
         std::shared_ptr<ML::deviceAllocator> allocator, cudaStream_t stream,
         bool setZero = true)
    : m_batch_size(batch_size),
      m_allocator(allocator),
      m_cublasHandle(cublasHandle),
      m_stream(stream),
      m_shape(std::make_pair(m, n)) {
    // Allocate memory
    allocate(setZero);
  }

  //! Destructor: nothing to destroy explicitely
  ~Matrix() {}

  //! Copy constructor
  Matrix(const Matrix<T>& other)
    : m_batch_size(other.m_batch_size),
      m_allocator(other.m_allocator),
      m_cublasHandle(other.m_cublasHandle),
      m_stream(other.m_stream),
      m_shape(other.m_shape) {
    // Allocate memory
    allocate(false);

    // Copy the raw data
    copy(m_A_dense.get(), other.m_A_dense.get(),
         m_batch_size * m_shape.first * m_shape.second, m_stream);
  }

  //! Copy assignment operator
  Matrix<T>& operator=(const Matrix<T>& other) {
    m_batch_size = other.m_batch_size;
    m_shape = other.m_shape;

    // Allocate memory
    allocate(false);

    // Copy the raw data
    copy(m_A_dense.get(), other.m_A_dense.get(),
         m_batch_size * m_shape.first * m_shape.second, m_stream);

    return *this;
  }

  //! Move constructor
  Matrix<T>(Matrix<T>&& other) noexcept
    : m_batch_size(other.m_batch_size),
      m_allocator(other.m_allocator),
      m_cublasHandle(other.m_cublasHandle),
      m_stream(other.m_stream),
      m_shape(other.m_shape),
      m_A_dense(std::move(other.m_A_dense)),
      m_A_batches(std::move(other.m_A_batches)) {}

  //! Move assignment operator
  Matrix<T>& operator=(Matrix<T>&& other) noexcept {
    m_batch_size = other.m_batch_size;
    m_shape = other.m_shape;

    m_A_dense = std::move(other.m_A_dense);
    m_A_batches = std::move(other.m_A_batches);

    return *this;
  }

  //! Return batches
  int batches() const { return m_batch_size; }

  //! Return cublas handle
  cublasHandle_t cublasHandle() const { return m_cublasHandle; }

  //! Return allocator
  std::shared_ptr<deviceAllocator> allocator() const { return m_allocator; }

  //! Return stream
  cudaStream_t stream() const { return m_stream; }

  //! Return shape
  const std::pair<int, int>& shape() const { return m_shape; }

  //! Return pointer array
  T** data() { return m_A_batches.get(); }
  const T** data() const { return m_A_batches.get(); }

  //! Return pointer to the underlying memory
  T* raw_data() { return m_A_dense.get(); }
  const T* raw_data() const { return m_A_dense.get(); }

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
   * @brief Reshape the matrix (the new shape must have the same size)
   *        The column-major data is left unchanged
   * 
   * @param[in]  m  Number of desired rows
   * @param[in]  n  Number of desired columns
   */
  void reshape(int m, int n) {
    const int r = m_shape.first * m_shape.second;
    ASSERT(r == m * n,
           "ERROR: Size mismatch - Cannot reshape matrix into desired shape");
    m_shape = std::pair<int, int>(m, n);
  }

  //! Stack the matrix by columns creating a long vector
  Matrix<T> vec() const {
    int m = m_shape.first;
    int n = m_shape.second;
    int r = m * n;
    Matrix<T> toVec(r, 1, m_batch_size, m_cublasHandle, m_allocator, m_stream,
                    false);
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
  Matrix<T> mat(int m, int n) const {
    const int r = m_shape.first * m_shape.second;
    ASSERT(r == m * n,
           "ERROR: Size mismatch - Cannot reshape array into desired size");
    Matrix<T> toMat(m, n, m_batch_size, m_cublasHandle, m_allocator, m_stream,
                    false);
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
  Matrix<T> difference(int period = 1) const {
    ASSERT(m_shape.first == 1 || m_shape.second == 1,
           "Invalid operation: must be a vector");
    int len = m_shape.second * m_shape.first;
    ASSERT(len > period, "Length of the vector must be > period");

    // Create output batched vector
    bool row_vector = (m_shape.first == 1);
    Matrix<T> out(row_vector ? 1 : len - period, row_vector ? len - period : 1,
                  m_batch_size, m_cublasHandle, m_allocator, m_stream, false);

    // Execute kernel
    const int TPB = (len - period) > 512 ? 256 : 128;  // quick heuristics
    batched_diff_kernel<<<m_batch_size, TPB, 0, m_stream>>>(
      raw_data(), out.raw_data(), len, period);
    CUDA_CHECK(cudaPeekAtLastError());

    return out;
  }

  /**
  * @brief Compute the inverse of the batched matrix
  * 
  * @return Batched inverse matrix
  */
  Matrix<T> inv() const {
    int n = m_shape.first;

    int* P =
      (int*)m_allocator->allocate(sizeof(int) * n * m_batch_size, m_stream);
    int* info =
      (int*)m_allocator->allocate(sizeof(int) * m_batch_size, m_stream);

    // A copy of A is necessary as the cublas operations write in A
    Matrix<T> Acopy(*this);

    Matrix<T> Ainv(n, n, m_batch_size, m_cublasHandle, m_allocator, m_stream,
                   false);

    CUBLAS_CHECK(LinAlg::cublasgetrfBatched(m_cublasHandle, n, Acopy.data(), n,
                                            P, info, m_batch_size, m_stream));
    CUBLAS_CHECK(LinAlg::cublasgetriBatched(m_cublasHandle, n, Acopy.data(), n,
                                            P, Ainv.data(), n, info,
                                            m_batch_size, m_stream));

    m_allocator->deallocate(P, sizeof(int) * n * m_batch_size, m_stream);
    m_allocator->deallocate(info, sizeof(int) * m_batch_size, m_stream);

    return Ainv;
  }

  /**
   * @brief Compute alpha*A' for a batched matrix A
   *
   * @return alpha*A'
   */
  Matrix<T> transpose(double alpha = 1.0) const {
    int m = m_shape.first;
    int n = m_shape.second;

    Matrix<T> At(n, m, m_batch_size, m_cublasHandle, m_allocator, m_stream);

    const double* d_A = m_A_dense.get();
    double* d_At = At.m_A_dense.get();

    // Naive batched transpose ; TODO: improve
    auto counting = thrust::make_counting_iterator<int>(0);
    thrust::for_each(thrust::cuda::par.on(m_stream), counting,
                     counting + m_batch_size * m, [=] __device__(int tid) {
                       int bid = tid / m;
                       int i = tid % m;
                       const double* b_A = d_A + bid * m * n;
                       double* b_At = d_At + bid * m * n;
                       for (int j = 0; j < n; j++) {
                         b_At[i * n + j] = alpha * b_A[j * m + i];
                       }
                     });
    return At;
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
  static Matrix Identity(int m, int batch_size, cublasHandle_t cublasHandle,
                         std::shared_ptr<ML::deviceAllocator> allocator,
                         cudaStream_t stream) {
    Matrix I(m, m, batch_size, cublasHandle, allocator, stream, true);

    identity_matrix_kernel<T>
      <<<batch_size, std::min(256, m), 0, stream>>>(I.raw_data(), m);
    CUDA_CHECK(cudaPeekAtLastError());
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
  int m_batch_size;

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
void b_gemm(bool aT, bool bT, int m, int n, int k, T alpha, const Matrix<T>& A,
            const Matrix<T>& B, T beta, Matrix<T>& C) {
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
Matrix<T> b_gemm(const Matrix<T>& A, const Matrix<T>& B, bool aT = false,
                 bool bT = false) {
  // m = number of rows of matrix op(A) and C.
  int m = !aT ? A.shape().first : A.shape().second;
  // n = number of columns of matrix op(B) and C.
  int n = !bT ? B.shape().second : B.shape().first;

  // k = number of columns of op(A) and rows of op(B).
  int k = !aT ? A.shape().second : A.shape().first;
  int kB = !bT ? B.shape().first : B.shape().second;

  ASSERT(k == kB, "Matrix-Multiplication dimensions don't match!");

  // Create C(m,n)
  Matrix<T> C(m, n, A.batches(), A.cublasHandle(), A.allocator(), A.stream());

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
void b_gels(const Matrix<T>& A, Matrix<T>& C) {
  ASSERT(A.batches() == C.batches(),
         "A and C must have the same number of batches");
  int m = A.shape().first;
  ASSERT(C.shape().first == m, "Dimension mismatch: A rows, C rows");
  int n = A.shape().second;
  ASSERT(m > n, "Only overdetermined systems (m > n) are supported");
  int nrhs = C.shape().second;

  Matrix<T> Acopy(A);

  int info;
  CUBLAS_CHECK(LinAlg::cublasgelsBatched(
    A.cublasHandle(), CUBLAS_OP_N, m, n, nrhs, Acopy.data(), m, C.data(), m,
    &info, nullptr, A.batches(), A.stream()));
}

/**
 * @brief A utility method to implement a unary operation on a batched matrix
 * 
 * @param[in]  A          Batched matrix A
 * @param[in]  unary_op   The unary operation applied on the elements of A
 * @return A batched matrix, the result of unary_op A
 */
template <typename T, typename F>
Matrix<T> b_op_A(const Matrix<T>& A, F unary_op) {
  auto batch_size = A.batches();
  int m = A.shape().first;
  int n = A.shape().second;

  Matrix<T> C(m, n, batch_size, A.cublasHandle(), A.allocator(), A.stream());

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
Matrix<T> b_aA_op_B(const Matrix<T>& A, const Matrix<T>& B, F binary_op) {
  ASSERT(
    A.shape().first == B.shape().first && A.shape().second == B.shape().second,
    "ERROR: Matrices must be same size");

  ASSERT(A.batches() == B.batches(), "A & B must have same number of batches");

  auto batch_size = A.batches();
  int m = A.shape().first;
  int n = A.shape().second;

  Matrix<T> C(m, n, batch_size, A.cublasHandle(), A.allocator(), A.stream());

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
Matrix<T> operator*(const Matrix<T>& A, const Matrix<T>& B) {
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
Matrix<T> operator+(const Matrix<T>& A, const Matrix<T>& B) {
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
Matrix<T> operator-(const Matrix<T>& A, const Matrix<T>& B) {
  return b_aA_op_B(A, B, [] __device__(T a, T b) { return a - b; });
}

/**
 * @brief Unary substraction
 * 
 * @param[in]  A  Batched matrix A
 * @return -A
 */
template <typename T>
Matrix<T> operator-(const Matrix<T>& A) {
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
Matrix<T> b_solve(const Matrix<T>& A, const Matrix<T>& b) {
  Matrix<T> x = A.inv() * b;
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
Matrix<T> b_kron(const Matrix<T>& A, const Matrix<T>& B) {
  int m = A.shape().first;
  int n = A.shape().second;

  int p = B.shape().first;
  int q = B.shape().second;

  // Resulting shape
  int k_m = m * p;
  int k_n = n * q;

  Matrix<T> AkB(k_m, k_n, A.batches(), A.cublasHandle(), A.allocator(),
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
void b_lagged_mat(const Matrix<T>& vec, Matrix<T>& lagged_mat, int lags,
                  int lagged_height, int vec_offset, int mat_offset,
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
Matrix<T> b_lagged_mat(const Matrix<T>& vec, int lags) {
  ASSERT(vec.shape().first == 1 || vec.shape().second == 1,
         "The first argument must be a vector (either row or column)");
  int len = vec.shape().first * vec.shape().second;
  ASSERT(lags < len, "The number of lags can't exceed the vector length");
  int lagged_height = len - lags;

  // Create output matrix
  Matrix<T> lagged_mat(lagged_height, lags, vec.batches(), vec.cublasHandle(),
                       vec.allocator(), vec.stream(), false);
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
void b_2dcopy(const Matrix<T>& in, Matrix<T>& out, int starting_row,
              int starting_col, int rows, int cols) {
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
Matrix<T> b_2dcopy(const Matrix<T>& in, int starting_row, int starting_col,
                   int rows, int cols) {
  // Create output matrix
  Matrix<T> out(rows, cols, in.batches(), in.cublasHandle(), in.allocator(),
                in.stream(), false);

  // Call the other overload of the function
  b_2dcopy(in, out, starting_row, starting_col, rows, cols);

  return out;
}

/**
 * @todo: docs
 */
template <typename T>
DI void generate_householder_vector(T* d_uk, const T* d_xk, int m) {
  // Compute norm of the vectors x and u
  T x_norm = (T)0, u_norm = (T)0;
  for (int i = 1; i < m; i++) {
    u_norm += d_xk[i] * d_xk[i];
  }
  T x0 = d_xk[0];
  x_norm = sqrt(u_norm + x0 * x0);
  T u0 = x0 - (x0 < 0 ? 1 : -1) * x_norm;
  u_norm = sqrt(u_norm + u0 * u0);

  // Compute u
  d_uk[0] = u0 / u_norm;
  for (int i = 1; i < m; i++) {
    d_uk[i] = d_xk[i] / u_norm;
  }
}

/**
 * @brief Hessenberg decomposition A = UHU', where Q is unitary and H in
 *        Hessenberg form (no zeros below the subdiagonal), using Householder
 *        reflections
 * @todo: docs
 */
template <typename T>
void b_hessenberg(const Matrix<T>& A, Matrix<T>& U, Matrix<T>& H) {
  int n = A.shape().first;
  int n2 = n * n;
  int batch_size = A.batches();
  auto stream = A.stream();
  auto allocator = A.allocator();
  auto cublasHandle = A.cublasHandle();
  auto counting = thrust::make_counting_iterator(0);

  T* d_H = H.raw_data();
  T* d_U = U.raw_data();

  // Copy A in H
  copy(H.raw_data(), A.raw_data(), n2 * batch_size, stream);

  // Initialize U with the identity
  CUDA_CHECK(cudaMemsetAsync(d_U, 0, sizeof(T) * n2 * batch_size, stream));
  identity_matrix_kernel<T>
    <<<batch_size, std::min(256, n), 0, stream>>>(d_U, n);
  CUDA_CHECK(cudaPeekAtLastError());

  // Create an array to store the Householder vectors
  int u_size = (n * (n - 1)) / 2 - 1;
  T* d_u = (T*)allocator->allocate(u_size * batch_size * sizeof(T), stream);

  // Temporary buffer for intermediate results
  int temp_size = n;
  T* d_temp =
    (T*)allocator->allocate(temp_size * batch_size * sizeof(T), stream);

  // BLAS scalars
  T one = (T)1, zero = (T)0, m_two = (T)-2;
  CUBLAS_CHECK(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST));

  // Transform H to Hessenberg form in-place
  int u_offset = 0;
  for (int k = 0; k < n - 2; k++) {
    // Generate the reflector
    thrust::for_each(thrust::cuda::par.on(A.stream()), counting,
                     counting + batch_size, [=] __device__(int ib) {
                       T* b_uk = d_u + u_size * ib + u_offset;
                       T* b_xk = d_H + n2 * ib + (n + 1) * k + 1;
                       generate_householder_vector(b_uk, b_xk, n - k - 1);
                     });

    // H[k+1:, k:] = H[k+1:, k:] - 2 * uk * (uk' * H[k+1:, k:])
    CUBLAS_CHECK(LinAlg::cublasgemmStridedBatched<T>(
      cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1, n - k, n - k - 1, &one,
      d_u + u_offset, 1, u_size, d_H + (n + 1) * k + 1, n, n2, &zero, d_temp, 1,
      temp_size, batch_size, stream));
    CUBLAS_CHECK(LinAlg::cublasgemmStridedBatched<T>(
      cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n - k - 1, n - k, 1, &m_two,
      d_u + u_offset, n - k - 1, u_size, d_temp, 1, temp_size, &one,
      d_H + (n + 1) * k + 1, n, n2, batch_size, stream));

    // H[:, k+1:] = H[:, k+1:] - 2 * (H[:, k+1:] * uk) * uk'
    CUBLAS_CHECK(LinAlg::cublasgemmStridedBatched<T>(
      cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, 1, n - k - 1, &one,
      d_H + (k + 1) * n, n, n2, d_u + u_offset, n - k - 1, u_size, &zero,
      d_temp, n, temp_size, batch_size, stream));
    CUBLAS_CHECK(LinAlg::cublasgemmStridedBatched<T>(
      cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n - k - 1, 1, &m_two, d_temp,
      n, temp_size, d_u + u_offset, 1, u_size, &one, d_H + (k + 1) * n, n, n2,
      batch_size, stream));

    u_offset += n - k - 1;
  }

  // Update U
  u_offset = u_size - 2;
  for (int k = n - 3; k >= 0; k--) {
    // U[k+1:, k+1:] = U[k+1:, k+1:] - 2 * uk * (uk' * U[k+1:, k+1:])
    CUBLAS_CHECK(LinAlg::cublasgemmStridedBatched<T>(
      cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1, n - k - 1, n - k - 1, &one,
      d_u + u_offset, 1, u_size, d_U + (k + 1) * (n + 1), n, n2, &zero, d_temp,
      1, temp_size, batch_size, stream));
    CUBLAS_CHECK(LinAlg::cublasgemmStridedBatched<T>(
      cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n - k - 1, n - k - 1, 1, &m_two,
      d_u + u_offset, n - k - 1, u_size, d_temp, 1, temp_size, &one,
      d_U + (k + 1) * (n + 1), n, n2, batch_size, stream));

    u_offset -= n - k;
  }

  allocator->deallocate(d_u, u_size * batch_size * sizeof(T), stream);
  allocator->deallocate(d_temp, temp_size * batch_size * sizeof(T), stream);
}

/**
 * @todo: docs
 * @note From Golub and van Loan, Matrix Computations, 2nd ed., 1989
 */
template <typename T>
DI void generate_givens(T h0, T h1, T& c, T& s) {
  if (abs(h1) > abs(h0)) {
    T r = -h0 / h1;
    s = (T)1 / sqrt((T)1 + r * r);
    c = s * r;
  } else {
    T r = -h1 / h0;
    c = (T)1 / sqrt((T)1 + r * r);
    s = c * r;
  }
  // TODO: remove the if statements?
}

/**
 * @todo: docs
 */
template <typename T>
__global__ void hessenberg_qr_step_kernel(T* const* d_U, T* const* d_H, T* d_G,
                                          int n, int batch_size) {
  typedef typename std::conditional<std::is_same<T, float>::value, float4,
                                    double4>::type T4;
  int ib = blockDim.x * blockIdx.x + threadIdx.x;

  // TODO: don't store a 2x2 matrix, only store c and s!!

  if (ib < batch_size) {
    T* b_U = d_U[ib];
    T* b_H = d_H[ib];
    T* b_G = d_G + ib * 4 * (n - 1);

    T Gk[4];

    // Generate the Givens rotations and apply them to the left
    for (int k = 0; k < n - 1; k++) {
      // Givens rotation
      {
        // TODO: more stable way to compute the rotations?
        T h0 = b_H[(n + 1) * k];
        T h1 = b_H[(n + 1) * k + 1];
        T c, s;
        generate_givens(h0, h1, c, s);
        Gk[0] = c;
        Gk[1] = s;
        Gk[2] = -s;
        Gk[3] = c;
        reinterpret_cast<T4*>(b_G)[k] = reinterpret_cast<T4*>(Gk)[0];
      }

      // H[k:k+2, k:] = Gk * H[k:k+2, k:]
      for (int j = k; j < n; j++) {
        T h0 = b_H[n * j + k];
        T h1 = b_H[n * j + k + 1];
        b_H[n * j + k] = Gk[0] * h0 + Gk[2] * h1;
        b_H[n * j + k + 1] = Gk[1] * h0 + Gk[3] * h1;
      }
    }

    // Apply the rotations to the right and update U
    for (int k = 0; k < n - 1; k++) {
      reinterpret_cast<T4*>(Gk)[0] = reinterpret_cast<T4*>(b_G)[k];

      // H[:k+2, k:k+2] = H[:k+2, k:k+2] * Gk'
      for (int i = 0; i < k + 2; i++) {
        T h0 = b_H[n * k + i];
        T h1 = b_H[n * (k + 1) + i];
        b_H[n * k + i] = h0 * Gk[0] + h1 * Gk[2];
        b_H[n * (k + 1) + i] = h0 * Gk[1] + h1 * Gk[3];
      }

      // U[:n, k:k+2] = U[:n, k:k+2] * Gk'
      for (int i = 0; i < n; i++) {
        T u0 = b_U[n * k + i];
        T u1 = b_U[n * (k + 1) + i];
        b_U[n * k + i] = u0 * Gk[0] + u1 * Gk[2];
        b_U[n * (k + 1) + i] = u0 * Gk[1] + u1 * Gk[3];
      }
    }
  }
}

/**
 * @todo: docs
 */
template <typename T>
void _hessenberg_qr_step(Matrix<T>& U, Matrix<T>& H,
                         thrust::device_ptr<bool>& conv_thrust) {
  int n = U.shape().first;
  int n2 = n * n;
  int batch_size = U.batches();
  auto stream = U.stream();
  auto allocator = U.allocator();
  auto counting = thrust::make_counting_iterator(0);

  //
  // Extract a reduced batch with only non-converged matrices
  //

  // 1. cumulative sum of !converged that will be used to associate to the
  //    batch members their position in the reduced batch if relevant
  thrust::logical_not<int> unary_op;
  thrust::plus<int> binary_op;
  device_buffer<int> cumul_buffer(allocator, stream, batch_size);
  thrust::device_ptr<int> cumul_thrust =
    thrust::device_pointer_cast(cumul_buffer.data());
  thrust::transform(thrust::cuda::par.on(stream), conv_thrust,
                    conv_thrust + batch_size, cumul_thrust, unary_op);
  thrust::inclusive_scan(thrust::cuda::par.on(stream), cumul_thrust,
                         cumul_thrust + batch_size, cumul_thrust, binary_op);

  // 2. We can deduce the size of the reduced batch from that
  int reduced_batch_size;
  copy(&reduced_batch_size, cumul_buffer.data() + batch_size - 1, 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  // std::cout << reduced_batch_size << " ";

  // 3. Now we "reverse" the array and compute the offsets in the old batch
  //    of the matrices in the reduced batch
  device_buffer<T*> U_offsets_buffer(allocator, stream, reduced_batch_size);
  device_buffer<T*> H_offsets_buffer(allocator, stream, reduced_batch_size);
  int* d_cumul = cumul_buffer.data();
  T** d_U_ptrs = U_offsets_buffer.data();
  T** d_H_ptrs = H_offsets_buffer.data();
  T* d_U = U.raw_data();
  T* d_H = H.raw_data();
  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + batch_size, [=] __device__(int ib) {
                     int cumul_prv = ib == 0 ? -1 : (d_cumul[ib - 1] - 1);
                     int cumul_cur = d_cumul[ib] - 1;
                     if (cumul_cur > cumul_prv) {
                       d_U_ptrs[cumul_cur] = d_U + n2 * ib;
                       d_H_ptrs[cumul_cur] = d_H + n2 * ib;
                     }
                   });

  //
  // Compute a Hessenberg-QR step on the reduced batch
  //

  // Buffer for Givens rotation matrices (economic form)
  device_buffer<T> G(allocator, stream, 4 * (n - 1) * reduced_batch_size);

  constexpr int TPB = 256;
  hessenberg_qr_step_kernel<<<ceildiv<int>(reduced_batch_size, TPB), TPB, 0,
                              stream>>>(d_U_ptrs, d_H_ptrs, G.data(), n,
                                        reduced_batch_size);
  CUDA_CHECK(cudaPeekAtLastError());

  /// TODO: compare shared-mem and no-shared-mem kernels!
}

/**
 * @brief Schur decomposition A = USU', where U is unitary and S is a
 *        block-upper triangular matrix with block size <= 2
 * @todo: docs
 */
template <typename T>
void b_schur(const Matrix<T>& A, Matrix<T>& U, Matrix<T>& S) {
  int n = A.shape().first;
  int batch_size = A.batches();
  auto stream = A.stream();
  auto allocator = A.allocator();
  auto counting = thrust::make_counting_iterator(0);

  T near_zero = 1e-4;

  MLCommon::Matrix::print(A.raw_data() + n * n * 2, n, n, ',');

  //
  // Start with a Hessenberg decomposition
  //

  b_hessenberg(A, U, S);

  //
  // Hessenberg-QR algorithm (H stays in Hessenberg form)
  //

  bool converged;
  device_buffer<bool> conv_buffer(allocator, stream, batch_size);
  bool* d_conv = conv_buffer.data();
  thrust::device_ptr<bool> conv_thrust = thrust::device_pointer_cast(d_conv);
  CUDA_CHECK(cudaMemsetAsync(d_conv, 0, batch_size * sizeof(bool), stream));
  T* d_S = S.raw_data();

  int maxiter = 1000;  // TODO: remove

  int iter = 0;
  do {
    // Hessenberg QR step
    _hessenberg_qr_step(U, S, conv_thrust);

    // Convergence test
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + batch_size, [=] __device__(int ib) {
                       const double* b_S = d_S + ib * n * n;
                       bool conv = true;
                       bool prev = false;
                       for (int i = 0; i < n - 1; i++) {
                         bool nonzero = (abs(b_S[(n + 1) * i + 1]) > near_zero);
                         conv = conv && !(prev && nonzero);
                         prev = nonzero;
                       }
                       d_conv[ib] = conv;
                     });

    // Reduction of the convergence flags
    converged = thrust::reduce(thrust::cuda::par.on(stream), conv_thrust,
                               conv_thrust + batch_size, true,
                               thrust::logical_and<bool>());
    iter++;
    if (iter == maxiter) break;
  } while (!converged);
  std::cout << std::endl << "Iterations: " << iter << std::endl;
  if (iter == maxiter) {
    std::cout << "Iteration limit reached. Converged:" << std::endl;
    myPrintDevVector("conv", d_conv, batch_size);
  }
}

/**
 * @brief Solve discrete Lyapunov equation A*X*A' - X + Q = 0
 * 
 * @note The content of Q isn't modified, but can be reshaped into a vector
 *       and back into a matrix
 *
 * @param[in]  A       Batched matrix A
 * @param[in]  Q       Batched matrix Q
 * @return             Batched matrix X solving the Lyapunov equation
 */
template <typename T>
Matrix<T> b_lyapunov(const Matrix<T>& A, Matrix<T>& Q) {
  int batch_size = A.batches();
  auto stream = A.stream();
  auto allocator = A.allocator();
  int n = A.shape().first;
  int n2 = n * n;
  auto counting = thrust::make_counting_iterator(0);

  bool small = false;

  if (small) {
    //
    // Use direct solution with Kronecker product
    //
    Matrix<T> I_m_AxA = b_kron(-A, A);
    double* d_I_m_AxA = I_m_AxA.raw_data();
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + batch_size, [=] __device__(int ib) {
                       double* b_I_m_AxA = d_I_m_AxA + ib * n2 * n2;
                       for (int i = 0; i < n * n; i++) {
                         b_I_m_AxA[(n2 + 1) * i] += 1.0;
                       }
                     });
    Q.reshape(n2, 1);
    Matrix<T> X = b_solve(I_m_AxA, Q);
    Q.reshape(n, n);
    X.reshape(n, n);
    return X;
  } else {
    //
    // Transform to Sylvester equation (Popov, 1964)
    //
    Matrix<T> B(n, n, batch_size, A.cublasHandle(), allocator, stream, false);
    Matrix<T> C(n, n, batch_size, A.cublasHandle(), allocator, stream, false);
    {
      Matrix<T> ApI(A);
      Matrix<T> AmI(A);
      Matrix<T> AtpI = A.transpose();
      double* d_ApI = ApI.raw_data();
      double* d_AmI = AmI.raw_data();
      double* d_AtpI = AtpI.raw_data();
      thrust::for_each(thrust::cuda::par.on(stream), counting,
                       counting + batch_size, [=] __device__(int ib) {
                         int idx = ib * n2;
                         for (int i = 0; i < n; i++) {
                           d_ApI[idx] += 1.0;
                           d_AmI[idx] -= 1.0;
                           d_AtpI[idx] += 1.0;
                           idx += n + 1;
                         }
                       });
      Matrix<T> ApI_inv = ApI.inv();

      // B = (A-I)*(A+I)^{-1}
      b_gemm(false, false, n, n, n, 1.0, AmI, ApI_inv, 0.0, B);
      // C = 2*(A'+I)^{-1}*Q*(A+I)^{-1}
      b_gemm(false, false, n, n, n, -2.0, AtpI.inv(), Q * ApI_inv, 0.0, C);
    }

    //
    // Solve Sylvester equation BX + XB' = -C with Bartels-Stewart algorithm
    //

    // 1. Shur decomposition
    Matrix<T> R(n, n, batch_size, A.cublasHandle(), allocator, stream, false);
    Matrix<T> U(n, n, batch_size, A.cublasHandle(), allocator, stream, false);
    Matrix<T> S(n, n, batch_size, A.cublasHandle(), allocator, stream, false);
    Matrix<T> V(n, n, batch_size, A.cublasHandle(), allocator, stream, false);
    Matrix<T> mBt = B.transpose(-1.0);
    // b_schur(B, U, R);
    b_schur(mBt, V, S);
    /// TODO: clear prints

    // std::cout << "V" << std::endl;
    // MLCommon::Matrix::print(V.raw_data(), n, n * batch_size, ',');
    // std::cout << std::endl;

    // std::cout << "S" << std::endl;
    // MLCommon::Matrix::print(S.raw_data(), n, n * batch_size, ',');
    // std::cout << std::endl;

    // std::cout << "VV'" << std::endl;
    // MLCommon::Matrix::print(b_gemm(V, V, false, true).raw_data(), n,
    //                         n * batch_size, ',');
    // std::cout << std::endl;

    // std::cout << "-B' - VSV'" << std::endl;
    // MLCommon::Matrix::print(
    //   (mBt - b_gemm(b_gemm(V, S, false, false), V, false, true)).raw_data(), n,
    //   n * batch_size, ',');
    // std::cout << std::endl;

    // 2. F = -U'CV
    // Matrix<T> F(n, n, batch_size, A.cublasHandle(), allocator,
    //                    stream, false);
    // b_gemm(true, false, n, n, n, -1.0, U, C * V, 0.0, F);

    // 3. Solve RY-YS'=F (where Y=U'XV)
    /// TODO: solve using forward substitution on the blocks

    // 4. X = UYV'
    /// TODO: simple matmul

    Matrix<T> X =
      Matrix<T>::Identity(n, batch_size, A.cublasHandle(), allocator, stream);
    return X;
  }
}

}  // namespace Batched
}  // namespace LinAlg
}  // namespace MLCommon
