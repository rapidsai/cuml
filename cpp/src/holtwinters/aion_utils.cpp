/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#pragma once

#include "aion_utils.hpp"
#include <lapacke.h>

extern "C"{
#include <cblas.h>
}
#include <iostream>


template<typename Dtype>
aion::AionStatus transpose_cpu(const Dtype *src, int m, int n, Dtype *dst) {
  for (int j = 0; j < m; ++j) {
    for (int i = 0; i < n; ++i) {
      dst[j+i*m] = src[i+j*n];  // TODO(ahmad): cblas::geam ? : geam is not a cblas function, delete TODO ?
    }
  }
  return aion::AionStatus::AION_SUCCESS;
}

template<typename Dtype>
aion::AionStatus transpose_gpu(const Dtype *src, int m, int n, Dtype *dst) {
  // TODO(ahmad): check cublas return value
  Dtype a = 1.0;
  Dtype b = 0.0;
  aion::cublas::geam<Dtype>(
    CUBLAS_OP_T,    // 02/ transa
    CUBLAS_OP_N,    // 03/ transb
    m,              // 04/ m - number of rows of matrix op(A) and C
    n,              // 05/ n - number of columns of matrix op(B) and C
    &a,             // 06/ alpha
    src,            // 07/ A - lda x m (batch_size x n)
    n,              // 08/ lda - leading dimension of two-dimensional array used to store the matrix A.
    &b,             // 09/ beta
    nullptr,        // 10/ B - ldb x n ()
    m,              // 11/ ldb - leading dimension of two-dimensional array used to store matrix B.
    dst,            // 12/ C - ldc x n
    m);             // 13/ ldc - leading dimension of a two-dimensional array used to store the matrix C.
  return aion::AionStatus::AION_SUCCESS;
}

template aion::AionStatus transpose_cpu<float>(const float *src, int m, int n, float *dst);
template aion::AionStatus transpose_cpu<double>(const double *src, int m, int n, double *dst);
template aion::AionStatus transpose_gpu<float>(const float *src, int m, int n, float *dst);
template aion::AionStatus transpose_gpu<double>(const double *src, int m, int n, double *dst);

namespace aion {

namespace {
  void cblas_copy(int n, const float *x, int incx, float *y, int incy) {
    cblas_scopy(n, x, incx, y, incy);
  }
  void cblas_copy(int n, const double *x, int incx, double *y, int incy) {
    cblas_dcopy(n, x, incx, y, incy);
  }

  float cblas_dot(int n, const float *x, int incx, const float *y, int incy) {
    return cblas_sdot(n, x, incx, y, incy);
  }
  double cblas_dot(int n, const double *x, int incx, const double *y, int incy) {
    return cblas_ddot(n, x, incx, y, incy);
  }

  void cblas_axpy(int n, float alpha,
                  const float* x, int incx, float* y, int incy) {
    cblas_saxpy(n, alpha, x, incx, y, incy);
  }
  void cblas_axpy(int n, double alpha,
                  const double* x, int incx, double* y, int incy) {
    cblas_daxpy(n, alpha, x, incx, y, incy);
  }

  void cblas_gemv(CBLAS_TRANSPOSE trans,
                  int m, int n,
                  float alpha, const float *a, int lda, const float *x, int incx,
                  float beta, float *y, int incy) {
    cblas_sgemv(CblasRowMajor, trans,
      m, n,
      alpha, a, lda, x, incx,
      beta, y, incy);
  }
  void cblas_gemv(CBLAS_TRANSPOSE trans, int m, int n,
                  double alpha, const double *a, int lda, const double *x, int incx,
                  double beta, double *y, int incy) {
    cblas_dgemv(CblasRowMajor, trans, m, n,
      alpha, a, lda, x, incx,
      beta, y, incy);
  }

  void cblas_gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                  int m, int n, int k,
                  float alpha, float *a, int lda,
                  float *b, int ldb, float beta,
                  float *c, int ldc) {
    cblas_sgemm(CblasRowMajor, transa, transb,
      m, n, k,
      alpha, a, lda,
      b, ldb, beta,
      c, ldc);
  }
  void cblas_gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                  int m, int n, int k,
                  double alpha, double *a, int lda,
                  double *b, int ldb, double beta,
                  double *c, int ldc) {
    cblas_dgemm(CblasRowMajor, transa, transb,
      m, n, k,
      alpha, a, lda,
      b, ldb, beta,
      c, ldc);
  }

  void cblas_ger(int m, int n,
                 float alpha, const float *x, int incx,
                 const float *y, int incy, float *a, int lda) {
    cblas_sger(CblasRowMajor, m, n,
      alpha, x, incx,
      y, incy, a, lda);
  }
  void cblas_ger(int m, int n,
                 double alpha, const double *x, int incx,
                 const double *y, int incy, double *a, int lda) {
    cblas_dger(CblasRowMajor, m, n,
      alpha, x, incx,
      y, incy, a, lda);
  }


}  // namespace.

template <typename Dtype>
void cblas::copy(int n, const Dtype *x, int incx, Dtype *y, int incy) {
  cblas_copy(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype cblas::dot(int n, const Dtype *x, int incx, const Dtype *y, int incy) {
  return cblas_dot(n, x, incx, y, incy);
}

template <typename Dtype>
void cblas::axpy(int n, Dtype alpha, const Dtype* x, Dtype* y) {
  cblas_axpy(n, alpha, x, 1, y, 1);
}

template <typename Dtype>
void cblas::axpy(int n, Dtype alpha, const Dtype* x, int incx, Dtype* y, int incy) {
  cblas_axpy(n, alpha, x, incx, y, incy);
}

template <typename Dtype>
void cblas::gemv(CBLAS_TRANSPOSE trans, int m, int n,
                 Dtype alpha, const Dtype *a, int lda, const Dtype *x, int incx,
                 Dtype beta, Dtype *y, int incy) {
  cblas_gemv(trans, m, n,
    alpha, a, lda, x, incx,
    beta, y, incy);
}

template <typename Dtype>
void cblas::gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                 int m, int n, int k,
                 Dtype alpha, Dtype *a, int lda,
                 Dtype *b, int ldb, Dtype beta,
                 Dtype *c, int ldc) {
  cblas_gemm(transa, transb,
    m, n, k,
    alpha, a, lda,
    b, ldb, beta,
    c, ldc);
}
template <typename Dtype>
void cblas::ger(int m, int n,
               Dtype alpha, const Dtype *x, int incx,
               const Dtype *y, int incy, Dtype *a, int lda) {
  cblas_ger(m, n, alpha, x, incx, y, incy, a, lda);
}

template void cblas::copy(int n, const float *x, int incx, float *y, int incy);
template void cblas::copy(int n, const double *x, int incx, double *y, int incy);

template float cblas::dot(int n, const float *x, int incx, const float *y, int incy);
template double cblas::dot(int n, const double *x, int incx, const double *y, int incy);

template void cblas::axpy(int n, float alpha, const float* x, float* y);
template void cblas::axpy(int n, double alpha, const double* x, double* y);
template void cblas::axpy(int n, float alpha, const float* x, int incx, float* y, int incy);
template void cblas::axpy(int n, double alpha, const double* x, int incx, double* y, int incy);

template void cblas::gemv(CBLAS_TRANSPOSE trans, int m, int n,
                 float alpha, const float *a, int lda, const float *x, int incx,
                 float beta, float *y, int incy);
template void cblas::gemv(CBLAS_TRANSPOSE trans, int m, int n,
                 double alpha, const double *a, int lda, const double *x, int incx,
                 double beta, double *y, int incy);

template void cblas::gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                 int m, int n, int k,
                 float alpha, float *a, int lda,
                 float *b, int ldb, float beta,
                 float *c, int ldc);
template void cblas::gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                 int m, int n, int k,
                 double alpha, double *a, int lda,
                 double *b, int ldb, double beta,
                 double *c, int ldc);

template void cblas::ger(int m, int n,
                 float alpha, const float *x, int incx,
                 const float *y, int incy, float *a, int lda);
template void cblas::ger(int m, int n,
                 double alpha, const double *x, int incx,
                 const double *y, int incy, double *a, int lda);
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


namespace {
  int lapacke_geqrf(int m, int n, float* a, int lda, float* tau) {
    return LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, m, n, a, lda, tau);
  }
  int lapacke_geqrf(int m, int n, double* a, int lda, double* tau) {
    return LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, a, lda, tau);
  }

  int lapacke_orgqr(int m, int n, int k, float* a, int lda, const float* tau) {
    return LAPACKE_sorgqr(LAPACK_ROW_MAJOR, m, n, k, a, lda, tau);
  }
  int lapacke_orgqr(int m, int n, int k, double* a, int lda, const double* tau) {
    return LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, n, k, a, lda, tau);
  }
}  // namespace

template <typename Dtype>
int lapack::geqrf(int m, int n, Dtype* a, int lda, Dtype* tau) {
  return lapacke_geqrf(m, n, a, lda, tau);
}

template <typename Dtype>
int lapack::orgqr(int m, int n, int k, Dtype* a, int lda, const Dtype* tau) {
  return lapacke_orgqr(m, n, k, a, lda, tau);
}

template int lapack::geqrf(int m, int n, float* a, int lda, float* tau);
template int lapack::geqrf(int m, int n, double* a, int lda, double* tau);
template int lapack::orgqr(int m, int n, int k, float* a, int lda, const float* tau);
template int lapack::orgqr(int m, int n, int k, double* a, int lda, const double* tau);



}  // namespace aion
