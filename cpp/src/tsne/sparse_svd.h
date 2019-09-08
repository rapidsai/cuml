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
#include "../../src_prims/utils.h"
#include "utils.h"
#include "unary_op.h"
#include "sparse/coo.h"


#define CHECK(x) ASSERT(x == 0, "Failed at line = %d file = %s\n", __LINE__, __FILE__)
#define MIN(a, b) ((a) > (b)) ? (b) : (a)


namespace ML {


/**
 * @brief Randomized SVD on COO sparse matrices
 * @input param COO_Matrix: Pointer to COO sparse matrix.
 * @output param U: (n, n_components)
 * @output param S: (n_components)
 * @output param VT: (p, n_components)
 * @input param handle: cuML general GPU handle
 * @input n_components: How many SVD singular values / singular values you want (ie 2)
 * @input oversamples: Generally 10 or so. Too few will cause instability.
 * @input max_iter: Generally 7 or so. Minimum is 3 for stability.
 * @input random_state: Seed for random normal initialization. -1 is any seed.
 */
template <typename math_t> void
SparseSVD(const MLCommon::Sparse::COO<math_t> *__restrict COO_Matrix,
          float *__restrict U,    // (n, n_components) F-Contiguous
          float *__restrict S,    // (n_components)
          float *__restrict VT,   // (p, n_components) F-Contiguous
          const cumlHandle &handle,
          const int n_components,
          const int oversamples,
          int max_iter,
          const long long random_state)
{
  auto d_alloc = handle.getDeviceAllocator();
  cudaStream_t stream = handle.getStream();

  if (n_components <= 0)
    return;

  if (max_iter <= 0)
    max_iter = 3;

  const int n = COO_Matrix->n_rows;
  const int p = COO_Matrix->n_cols;
  const int k = MIN(n_components + oversamples, p);
  cusolverDnHandle_t cusolverH; CHECK(cusolverDnCreate(&cusolverH));
  cublasHandle_t cublasH;       CHECK(cublasCreate(&cublasH));


  // Create Y, Z projection matrix buffers
  float *Y/*(n,k)*/ = (float*) d_alloc->allocate(sizeof(float) * n * k, stream);
  float *Z/*(p,k)*/ = (float*) d_alloc->allocate(sizeof(float) * p * k, stream);
  random_vector(Z, 0.0f, 1.0f, p * k, stream, random_state, true); // normal = true


  // Y, _ = np.linalg.qr(Y)
  int lwork_Y = 0;
  CHECK(cusolverDnSgeqrf_bufferSize(cusolverH, n, k, Y, n, &lwork_Y));
  float *work_Y = (float*) d_alloc->allocate(sizeof(float) * lwork_Y, stream);


  // Z, _ = np.linalg.qr(Z)
  int lwork_Z = 0;
  CHECK(cusolverDnSgeqrf_bufferSize(cusolverH, p, k, Z, p, &lwork_Z));
  // Reuse memory
  float *work_Z = (lwork_Z > lwork_Y) ? (float*) d_alloc->allocate(sizeof(float) * lwork_Z, stream) : work_Y; 


  // Tau for both QR factorizations
  float *tau = (float*) d_alloc->allocate(sizeof(float) * k, stream);
  int *info = (int*) d_alloc->allocate(sizeof(int), stream);

  // Y = X @ Z
  MLCommon::Sparse::coo_gemm(COO_Matrix, Z, k, Y, stream, false); // trans = false


  for (int i = 0; i < max_iter; i++)
  {
    // Y, _ = np.linalg.qr(Y)
    CHECK(cusolverDnSgeqrf(cusolverH, n, k, Y, n, tau, work_Y, lwork_Y, info));
    CHECK(cusolverDnSorgqr(cusolverH, n, k, k, Y, n, tau, work_Y, lwork_Y, info));

    // Z = X.T @ Y
    MLCommon::Sparse::coo_gemm(COO_Matrix, Y, k, Z, stream, true); // trans = true

    // Z, _ = np.linalg.qr(Z)
    CHECK(cusolverDnSgeqrf(cusolverH, p, k, Z, p, tau, work_Z, lwork_Z, info));
    CHECK(cusolverDnSorgqr(cusolverH, p, k, k, Z, p, tau, work_Z, lwork_Z, info));

    // Y = X @ Z
    MLCommon::Sparse::coo_gemm(COO_Matrix, Z, k, Y, stream, false); // trans = false
  }


  // Y(n,k), _ = np.linalg.qr(Y)
  CHECK(cusolverDnSgeqrf(cusolverH, n, k, Y, n, tau, work_Y, lwork_Y, info));
  CHECK(cusolverDnSorgqr(cusolverH, n, k, k, Y, n, tau, work_Y, lwork_Y, info));

  // Z(k,p) = Y.T @ X (or (X.T @ Y).T)
  MLCommon::Sparse::coo_gemm(COO_Matrix, Y, k, Z, stream, true); // trans = true


  // T(k,k) = Z @ Z.T (or (Z.T @ Z))
  float *T = (float*) d_alloc->allocate(sizeof(float) * k * k, stream);

  float alpha = 1.0f, beta = 0.0f;
  CHECK(cublasSsyrk(cublasH, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, k, p, &alpha, Z, p, &beta, T, k));


  // W(k), Uhat(k,k) = np.linalg.eigh(T)
  float *W = tau;   // Reuse memory tau(k)
  int lwork_T = 0;

  // Syevj is faster on smaller matrices, whilst syevd is better when k -> inf
  syevjInfo_t params;
  if (k <= 200) {
    cusolverDnCreateSyevjInfo(&params);
    CHECK(cusolverDnSsyevj_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, k, T, k, W, &lwork_T, params));
  }
  else
    CHECK(cusolverDnSsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, k, T, k, W, &lwork_T));

  // Reuse memory
  float *work_T = (lwork_T > lwork_Y) ? (float*) d_alloc->allocate(sizeof(float) * lwork_T, stream) : work_Y;

  if (k <= 200) {
    CHECK(cusolverDnSsyevj(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, k, T, k, W, work_T, lwork_T, info, params));
    cusolverDnDestroySyevjInfo(params);
  }
  else
    CHECK(cusolverDnSsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, k, T, k, W, work_T, lwork_T, info));


  // W(c), Uhat(k,k) = W[::-1], Uhat[:,::-1]
  float *Uhat = T;
  reverse(W, k, 0, stream);
  reverse(Uhat, k, k, stream);

  // U(n,c) = Y(n,k) @ Uhat[:,:n_components](k,c)
  CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n_components, k, &alpha, Y, n, Uhat, k, &beta, U, n));


  // S(c) = np.sqrt(W)
  // Make all neg eigenvalues 0
  MLCommon::LinAlg::unaryOp(W, W, n_components, [] __device__(float x) { return (x>=0) ? sqrtf(x):0; }, stream);
  thrust::copy(thrust::cuda::par.on(stream), W, W + n_components, S);


  // VT = (Uhat / S).T @ Z // (but since Z = Z.T, we need to get trans(Z))
  // so instead we want:
  // VT(2,p) = Uhat.T(2,k) @ Z.T(k,p)
  MLCommon::LinAlg::unaryOp(W, W, n_components, [] __device__(float x) { return (x!=0) ? (1.0f/x):0; }, stream);

  matrix_multiply_by_array(Uhat, k, n_components, W, stream);

  //                            Uhat.T       Z.T      Uhat.T([2],k) Z.T(k,[p]) Z.T([k],p)         
  CHECK(cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_T, n_components,    p,         k,     &alpha, Uhat, k, Z, p, &beta, VT, n_components));


  // Clean up all memory!
  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);

  d_alloc->deallocate(Y, sizeof(float) * n * k, stream);
  d_alloc->deallocate(Z, sizeof(float) * p * k, stream);

  d_alloc->deallocate(work_Y, sizeof(float) * lwork_Y, stream);
  if (lwork_Z > lwork_Y) d_alloc->deallocate(work_Z, sizeof(float) * lwork_Z, stream);

  d_alloc->deallocate(tau, sizeof(float) * k, stream);
  d_alloc->deallocate(info, sizeof(int), stream);

  d_alloc->deallocate(T, sizeof(float) * k * k, stream);
  if (lwork_T > lwork_Y) d_alloc->deallocate(work_T, sizeof(float) * lwork_T, stream);
}


};