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

#include <cuml/common/cuml_allocator.hpp>
#include "coo.h"
#include <cuda_runtime.h>
#include "cuda_utils.h"

#include "linalg/unary_op.h"
#include "cusparse_wrappers.h"
#include "cublas_wrappers.h"
#include "cusolver_wrappers.h"
#include "matrix/matrix.h"

#include <type_traits>

using namespace MLCommon;

#define MIN(a, b) ((a) > (b)) ? (b) : (a)


namespace MLCommon {
namespace Sparse {


/**
 * @brief Randomized SVD on COO sparse matrices
 * @input param COO_Matrix: Pointer to COO sparse matrix.
 * @output param U: (n, n_components)
 * @output param S: (n_components)
 * @output param VT: (p, n_components)
 * @input param handle: cuML general GPU handle
 * @input n_components: How many SVD singular values / singular values you want (ie 2)
 * @input oversamples: Generally 10 or so. Too few will cause instability.
 * @input max_iter: Generally 3 or so. Minimum is 3 for stability.
 * @input random_state: Seed for random normal initialization. -1 is any seed.
 */
template <typename math_t> void
svd(const Sparse::COO<math_t> *__restrict COO_Matrix,
    math_t *__restrict U,    // (n, n_components) F-Contiguous
    math_t *__restrict S,    // (n_components)
    math_t *__restrict VT,   // (p, n_components) F-Contiguous
    std::shared_ptr<deviceAllocator> d_alloc,
    const cudaStream_t stream,
    const int n_components = 2,
    const int oversamples = 10,
    int max_iter = 3,
    long long random_state = -1)
{
  //===============================================================
  // Step 1: Setup Sparse Matrices, gather info and create handles

  ASSERT((std::is_same<math_t, float>::value or std::is_same<math_t, double>::value),
         "Sparse SVD only works on Float or Double type data!\n");

  ASSERT(n_components > 0, "Number of Components (%d) must be > 0\n", n_components);

  if (max_iter <= 0)
    max_iter = 3;

  const int n = COO_Matrix->n_rows;
  const int p = COO_Matrix->n_cols;
  const int nnz = COO_Matrix->nnz;
  const int k = MIN(n_components + oversamples, p);

  ASSERT(n > 0 and p > 0 and nnz > 0 and k > 0, "Size of sparse matrix must be > 0\n");

  cusolverDnHandle_t cusolverH; CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
  cublasHandle_t cublasH;       CUBLAS_CHECK(cublasCreate(&cublasH));
  cusparseHandle_t cusparseH;   CUSPARSE_CHECK(cusparseCreate(&cusparseH));

  // Create sparse matrix descriptor
  auto dtype = (std::is_same<math_t, float>::value) ? CUDA_R_32F : CUDA_R_64F;
  cusparseSpMatDescr_t X;
  CUSPARSE_CHECK(cusparseCreateCoo(&X, n, p, COO_Matrix->nnz,
                                   COO_Matrix->rows, COO_Matrix->cols, COO_Matrix->vals,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dtype));

  //===============================================================
  // Step 2: Create projection matrices and intialize random vectors

  // Create Y(n,k), Z(p,k) projection matrix buffers
  device_buffer<math_t> Y_(d_alloc, stream, n*k);
  math_t *Y = (math_t*) Y_.data();
  device_buffer<math_t> Z_(d_alloc, stream, p*k);
  math_t *Z = (math_t*) Z_.data();

  if (random_state < 0) {
    // Get random seed based on time of day
    struct timeval tp; gettimeofday(&tp, NULL);
    random_state = tp.tv_sec * 1000 + tp.tv_usec;
  }
  Random::Rng random(seed);
  random.normal<math_t>(Z, p*k, 0, 1, stream);

  // Create cuSPARSE descriptors
  cusparseDnMatDescr_t Y_cusparse;
  CUSPARSE_CHECK(cusparseCreateDnMat(&Y_cusparse, n, k, n, Y, dtype, CUSPARSE_ORDER_COL));
  cusparseDnMatDescr_t Z_cusparse;
  CUSPARSE_CHECK(cusparseCreateDnMat(&Z_cusparse, p, k, p, Z, dtype, CUSPARSE_ORDER_COL));


  //===============================================================
  // Step 3: Find buffers for QR Decomposition and spMM

  // Y, _ = np.linalg.qr(Y)
  int lwork_Y = 0;
  CUSOLVER_CHECK(cusolverDngeqrf_bufferSize(cusolverH, n, k, Y, n, &lwork_Y));

  // Z, _ = np.linalg.qr(Z)
  int lwork_Z = 0;
  CUSOLVER_CHECK(cusolverDngeqrf_bufferSize(cusolverH, p, k, Z, p, &lwork_Z));

  // Z = X.T @ Y
  math_t alpha = 1.0f;
  math_t beta = 0.0f;
  size_t lwork_XTY = 0;
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(cusparseH, CUSPARSE_OPERATION_TRANSPOSE,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                         A, Y_cusparse, &beta, Z_cusparse,
                                         dtype, CUSPARSE_COOMM_ALG3, &lwork_XTY));

  // Y = X @ Z
  size_t lwork_XZ = 0;
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                         A, Z_cusparse, &beta, Y_cusparse,
                                         dtype, CUSPARSE_COOMM_ALG3, &lwork_XZ));

  // Create massive buffer to hold everything
  size_t max = MAX(
                   MAX(lwork_Y, lwork_Z) * sizeof(math_t),
                   MAX(lwork_XTY, lwork_XZ),
                  );
  device_buffer<char> buffer_(d_alloc, stream, max);
  void *buffer = (void*) buffer_.data();

  // Tau for both QR factorizations
  device_buffer<math_t> tau_(d_alloc, stream, k);
  math_t *tau = (math_t*) tau_.data();
  device_buffer<int> info_(d_alloc, stream, 1);
  int *info = (int*) info_.data();


  //===============================================================
  // Step 4: Start iterations to get U, S, VT!

  // Y = X @ Z
  // TODO: Change to LU Decomposition and use cuBLAS STRSM to speed things up dramatically
  CUSPARSE_CHECK(cusparseSpMM(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                              A, Z_cusparse, &beta, Y_cusparse,
                              dtype, CUSPARSE_COOMM_ALG3, buffer));

  for (int i = 0; i < max_iter; i++)
  {
    // Y, _ = np.linalg.qr(Y)
    CUSOLVER_CHECK(cusolverDngeqrf(cusolverH, n, k, Y, n, tau, buffer, lwork_Y, info));
    CUSOLVER_CHECK(cusolverDnorgqr(cusolverH, n, k, k, Y, n, tau, buffer, lwork_Y, info));

    // Z = X.T @ Y
    CUSPARSE_CHECK(cusparseSpMM(cusparseH, CUSPARSE_OPERATION_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                A, Y_cusparse, &beta, Z_cusparse,
                                dtype, CUSPARSE_COOMM_ALG3, buffer));

    // Z, _ = np.linalg.qr(Z)
    CUSOLVER_CHECK(cusolverDngeqrf(cusolverH, p, k, Z, p, tau, buffer, lwork_Z, info));
    CUSOLVER_CHECK(cusolverDnorgqr(cusolverH, p, k, k, Z, p, tau, buffer, lwork_Z, info));

    // Y = X @ Z
    // TODO: Change to LU Decomposition and use cuBLAS STRSM to speed things up dramatically
    CUSPARSE_CHECK(cusparseSpMM(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                A, Z_cusparse, &beta, Y_cusparse,
                                dtype, CUSPARSE_COOMM_ALG3, buffer));
  }

  // Y(n,k), _ = np.linalg.qr(Y)
  // TODO: Change to Cholesky-QR and use cuBLAS STRSM to speed things up dramatically
  CUSOLVER_CHECK(cusolverDngeqrf(cusolverH, n, k, Y, n, tau, buffer, lwork_Y, info));
  CUSOLVER_CHECK(cusolverDnorgqr(cusolverH, n, k, k, Y, n, tau, buffer, lwork_Y, info));

  // Z(k,p) = Y.T @ X (or (X.T @ Y).T)
  CUSPARSE_CHECK(cusparseSpMM(cusparseH, CUSPARSE_OPERATION_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                              A, Y_cusparse, &beta, Z_cusparse,
                              dtype, CUSPARSE_COOMM_ALG3, buffer));

  buffer_.resize(0, stream);
  tau_.resize(0, stream);
  CUSPARSE_CHECK(cusparseDestroySpMat(A));
  CUSPARSE_CHECK(cusparseDestroyDnMat(Y_cusparse));
  CUSPARSE_CHECK(cusparseDestroyDnMat(Z_cusparse));
  CUSPARSE_CHECK(cusparseDestroy(cusparseH));


  //===============================================================
  // Step 5: Now extract the approximate eigenvectors for X.T @ X

  // T(k,k) = Z @ Z.T (or (Z.T @ Z))
  device_buffer<math_t> T_(d_alloc, stream, k * k);
  math_t *T = (math_t*) T_.data();
  CUBLAS_CHECK(cublassyrk(cublasH, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, k, p, &alpha, Z, p, &beta, T, k));


  // W(k), Uhat(k,k) = np.linalg.eigh(T)
  device_buffer<math_t> W_(d_alloc, stream, k);
  math_t *W = (math_t*) W_.data();
  int lwork_T = 0;


  // Syevj is faster on smaller matrices, whilst syevd is better when k -> inf
  syevjInfo_t params;
  if (k <= 200) {
    CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&params));
    CUSOLVER_CHECK(cusolverDnsyevj_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, k, T, k, W, &lwork_T, params));
  }
  else
    CUSOLVER_CHECK(cusolverDnsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, k, T, k, W, &lwork_T));

  // Reuse memory
  device_buffer<math_t> work_T_(d_alloc, stream, lwork_T);
  math_t *work_T = (math_t*) work_T_.data();

  if (k <= 200) {
    CUSOLVER_CHECK(cusolverDnsyevj(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, k, T, k, W, work_T, lwork_T, info, params));
    CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(params));
  }
  else
    CUSOLVER_CHECK(cusolverDnsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, k, T, k, W, work_T, lwork_T, info));


  //===============================================================
  // Step 6: Create U, S, VT

  // W(c), Uhat(k,k) = W[::-1], Uhat[:,::-1]
  float *Uhat = T;
  Matrix::colReverse(W, 1, k, stream);
  Matrix::colReverse(Uhat, k, k, stream);

  // U(n,c) = Y(n,k) @ Uhat[:,:n_components](k,c)
  CUBLAS_CHECK(cublasgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n_components, k, &alpha, Y, n, Uhat, k, &beta, U, n));


  // S(c) = np.sqrt(W)
  // Make all neg eigenvalues 0
  LinAlg::unaryOp(W, W, n_components, [] __device__(float x) { return (x>=0) ? sqrtf(x):0; }, stream);
  thrust::copy(thrust::cuda::par.on(stream), W, W + n_components, S);


  // VT = (Uhat / S).T @ Z // (but since Z = Z.T, we need to get trans(Z))
  // so instead we want:
  // VT(2,p) = Uhat.T(2,k) @ Z.T(k,p)
  LinAlg::unaryOp(W, W, n_components, [] __device__(float x) { return (x!=0) ? (1.0f/x):0; }, stream);

  Matrix::matrixVectorBinaryMult(Uhat, W, k, n_components, false, true, stream);

  //                                  Uhat.T       Z.T      Uhat.T([2],k) Z.T(k,[p]) Z.T([k],p)         
  CUBLAS_CHECK(cublasgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_T, n_components,    p,         k,     &alpha, Uhat, k, Z, p, &beta, VT, n_components));


  // Clean up all memory!
  CUBLAS_CHECK(cublasDestroy(cublasH));
  CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
}


};
};
