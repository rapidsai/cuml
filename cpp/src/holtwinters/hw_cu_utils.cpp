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

#include "hw_cu_utils.hpp"
#include <iostream>

template <typename Dtype>
ML::HWStatus transpose_gpu(const Dtype *src, int m, int n, Dtype *dst) {
  // TODO(ahmad): check cublas return value
  Dtype a = 1.0;
  Dtype b = 0.0;
  ML::cublas::geam<Dtype>(
    CUBLAS_OP_T,  // 02/ transa
    CUBLAS_OP_N,  // 03/ transb
    m,            // 04/ m - number of rows of matrix op(A) and C
    n,            // 05/ n - number of columns of matrix op(B) and C
    &a,           // 06/ alpha
    src,          // 07/ A - lda x m (batch_size x n)
    n,  // 08/ lda - leading dimension of two-dimensional array used to store the matrix A.
    &b,       // 09/ beta
    nullptr,  // 10/ B - ldb x n ()
    m,  // 11/ ldb - leading dimension of two-dimensional array used to store matrix B.
    dst,  // 12/ C - ldc x n
    m);  // 13/ ldc - leading dimension of a two-dimensional array used to store the matrix C.
  return ML::HWStatus::HW_SUCCESS;
}

template ML::HWStatus transpose_gpu<float>(const float *src, int m, int n,
                                           float *dst);
template ML::HWStatus transpose_gpu<double>(const double *src, int m, int n,
                                            double *dst);

namespace ML {
thread_local cublasHandle_t cublas::m_handle = nullptr;

namespace {
cublasStatus_t cublas_axpy(cublasHandle_t handle, int n, const float *alpha,
                           const float *x, int incx, float *y, int incy) {
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}
cublasStatus_t cublas_axpy(cublasHandle_t handle, int n, const double *alpha,
                           const double *x, int incx, double *y, int incy) {
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublas_geam(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n,
                           const float *alpha, const float *A, int lda,
                           const float *beta, const float *B, int ldb, float *C,
                           int ldc) {
  return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb,
                     C, ldc);
}
cublasStatus_t cublas_geam(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n,
                           const double *alpha, const double *A, int lda,
                           const double *beta, const double *B, int ldb,
                           double *C, int ldc) {
  return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb,
                     C, ldc);
}

cublasStatus_t cublas_gemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const float *alpha, const float *A, int lda,
                           const float *B, int ldb, const float *beta, float *C,
                           int ldc) {
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}
cublasStatus_t cublas_gemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const double *alpha, const double *A, int lda,
                           const double *B, int ldb, const double *beta,
                           double *C, int ldc) {
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}
}  // namespace.

template <typename Dtype>
void cublas::axpy(int n, Dtype alpha, const Dtype *x, Dtype *y) {
  cublasHandle_t handle = cublas::get_handle();
  CHECK_CUBLAS(cublas_axpy(handle, n, &alpha, x, 1, y, 1));
}

template <typename Dtype>
void cublas::axpy(int n, Dtype alpha, const Dtype *x, int incx, Dtype *y,
                  int incy) {
  cublasHandle_t handle = cublas::get_handle();
  CHECK_CUBLAS(cublas_axpy(handle, n, &alpha, x, incx, y, incy));
}

template <typename Dtype>
void cublas::geam(cublasOperation_t transa, cublasOperation_t transb, int m,
                  int n, const Dtype *alpha, const Dtype *A, int lda,
                  const Dtype *beta, const Dtype *B, int ldb, Dtype *C,
                  int ldc) {
  cublasHandle_t handle = cublas::get_handle();
  CHECK_CUBLAS(cublas_geam(handle, transa, transb, m, n, alpha, A, lda, beta, B,
                           ldb, C, ldc));
}

template <typename Dtype>
void cublas::gemm(cublasOperation_t transa, cublasOperation_t transb, int m,
                  int n, int k, const Dtype *alpha, const Dtype *A, int lda,
                  const Dtype *B, int ldb, const Dtype *beta, Dtype *C,
                  int ldc) {
  cublasHandle_t handle = cublas::get_handle();
  CHECK_CUBLAS(cublas_gemm(handle, transa, transb, m, n, k, alpha, A, lda, B,
                           ldb, beta, C, ldc));
}

template void cublas::axpy(int n, float alpha, const float *x, float *y);
template void cublas::axpy(int n, double alpha, const double *x, double *y);
template void cublas::axpy(int n, float alpha, const float *x, int incx,
                           float *y, int incy);
template void cublas::axpy(int n, double alpha, const double *x, int incx,
                           double *y, int incy);

template void cublas::geam(cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, const float *alpha, const float *A,
                           int lda, const float *beta, const float *B, int ldb,
                           float *C, int ldc);
template void cublas::geam(cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, const double *alpha, const double *A,
                           int lda, const double *beta, const double *B,
                           int ldb, double *C, int ldc);

template void cublas::gemm(cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k, const float *alpha,
                           const float *A, int lda, const float *B, int ldb,
                           const float *beta, float *C, int ldc);
template void cublas::gemm(cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k, const double *alpha,
                           const double *A, int lda, const double *B, int ldb,
                           const double *beta, double *C, int ldc);

thread_local cusolverDnHandle_t cusolver::m_handle = nullptr;

namespace {
cusolverStatus_t cusolver_geqrf_bufferSize(cusolverDnHandle_t handle, int m,
                                           int n, float *A, int lda,
                                           int *Lwork) {
  return cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
}
cusolverStatus_t cusolver_geqrf_bufferSize(cusolverDnHandle_t handle, int m,
                                           int n, double *A, int lda,
                                           int *Lwork) {
  return cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t cusolver_geqrf(cusolverDnHandle_t handle, int m, int n,
                                float *A, int lda, float *TAU, float *Workspace,
                                int Lwork, int *devInfo) {
  return cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}
cusolverStatus_t cusolver_geqrf(cusolverDnHandle_t handle, int m, int n,
                                double *A, int lda, double *TAU,
                                double *Workspace, int Lwork, int *devInfo) {
  return cusolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}

cusolverStatus_t cusolver_orgqr_bufferSize(cusolverDnHandle_t handle, int m,
                                           int n, int k, const float *A,
                                           int lda, const float *tau,
                                           int *lwork) {
  return cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
}
cusolverStatus_t cusolver_orgqr_bufferSize(cusolverDnHandle_t handle, int m,
                                           int n, int k, const double *A,
                                           int lda, const double *tau,
                                           int *lwork) {
  return cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
}

cusolverStatus_t cusolver_orgqr(cusolverDnHandle_t handle, int m, int n, int k,
                                float *A, int lda, const float *tau,
                                float *work, int lwork, int *devInfo) {
  return cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}
cusolverStatus_t cusolver_orgqr(cusolverDnHandle_t handle, int m, int n, int k,
                                double *A, int lda, const double *tau,
                                double *work, int lwork, int *devInfo) {
  return cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}

// TODO(ahmad): report mismatch between doc and API in cusolver
// cusolverStatus_t cusolverDnSorgqr_bufferSize(cusolverDnHandle_t handle,
// int m, int n, int k, const float *A, int lda, int *lwork); // Doc
// cusolverStatus_t cusolverDnSorgqr_bufferSize(cusolverDnHandle_t handle,
// int m, int n, int k, const float *A, int lda, const float *tau, int *lwork); // API

}  // namespace.

template <typename Dtype>
void cusolver::geqrf_bufferSize(int m, int n, Dtype *A, int lda, int *Lwork) {
  cusolverDnHandle_t handle = cusolver::get_handle();
  CHECK_CUSOLVER(cusolver_geqrf_bufferSize(handle, m, n, A, lda, Lwork));
}

template <typename Dtype>
void cusolver::geqrf(int m, int n, Dtype *A, int lda, Dtype *TAU,
                     Dtype *Workspace, int Lwork, int *devInfo) {
  cusolverDnHandle_t handle = cusolver::get_handle();
  CHECK_CUSOLVER(
    cusolver_geqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo));
}

template <typename Dtype>
void cusolver::orgqr_bufferSize(int m, int n, int k, const Dtype *A, int lda,
                                const Dtype *tau, int *lwork) {
  cusolverDnHandle_t handle = cusolver::get_handle();
  CHECK_CUSOLVER(
    cusolver_orgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork));
}
template <typename Dtype>
void cusolver::orgqr(int m, int n, int k, Dtype *A, int lda, const Dtype *tau,
                     Dtype *work, int lwork, int *devInfo) {
  cusolverDnHandle_t handle = cusolver::get_handle();
  CHECK_CUSOLVER(
    cusolver_orgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo));
}

template void cusolver::geqrf_bufferSize(int m, int n, float *A, int lda,
                                         int *Lwork);
template void cusolver::geqrf_bufferSize(int m, int n, double *A, int lda,
                                         int *Lwork);
template void cusolver::geqrf(int m, int n, float *A, int lda, float *TAU,
                              float *Workspace, int Lwork, int *devInfo);
template void cusolver::geqrf(int m, int n, double *A, int lda, double *TAU,
                              double *Workspace, int Lwork, int *devInfo);

template void cusolver::orgqr_bufferSize(int m, int n, int k, const float *A,
                                         int lda, const float *tau, int *lwork);
template void cusolver::orgqr_bufferSize(int m, int n, int k, const double *A,
                                         int lda, const double *tau,
                                         int *lwork);
template void cusolver::orgqr(int m, int n, int k, float *A, int lda,
                              const float *tau, float *work, int lwork,
                              int *devInfo);
template void cusolver::orgqr(int m, int n, int k, double *A, int lda,
                              const double *tau, double *work, int lwork,
                              int *devInfo);

}  // namespace ML
