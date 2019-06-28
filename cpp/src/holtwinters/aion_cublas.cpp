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

#include <util/aion_cublas.hpp>

namespace aion {

thread_local cublasHandle_t cublas::m_handle = nullptr;

namespace {
  cublasStatus_t cublas_axpy(cublasHandle_t handle, int n, const float* alpha,
      const float* x, int incx, float* y, int incy) {
    return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
  }
  cublasStatus_t cublas_axpy(cublasHandle_t handle, int n, const double* alpha,
      const double* x, int incx, double* y, int incy) {
    return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
  }

  cublasStatus_t cublas_geam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
      int m, int n,
      const float *alpha, const float *A, int lda,
      const float *beta, const float *B, int ldb,
      float *C, int ldc) {
    return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
  }
  cublasStatus_t cublas_geam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
      int m, int n,
      const double *alpha, const double *A, int lda,
      const double *beta, const double *B, int ldb,
      double *C, int ldc) {
    return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
  }

  cublasStatus_t cublas_gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
      int m, int n, int k,
      const float *alpha, const float *A, int lda,
      const float *B, int ldb,
      const float *beta, float *C, int ldc) {
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  cublasStatus_t cublas_gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
      int m, int n, int k,
      const double *alpha, const double *A, int lda,
      const double *B, int ldb,
      const double *beta, double *C, int ldc) {
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
}  // namespace.


template <typename Dtype>
void cublas::axpy(int n, Dtype alpha, const Dtype* x, Dtype* y) {
  cublasHandle_t handle = cublas::get_handle();
  CHECK_CUBLAS(cublas_axpy(handle, n, &alpha, x, 1, y, 1));
}

template <typename Dtype>
void cublas::axpy(int n, Dtype alpha, const Dtype* x, int incx, Dtype* y, int incy) {
  cublasHandle_t handle = cublas::get_handle();
  CHECK_CUBLAS(cublas_axpy(handle, n, &alpha, x, incx, y, incy));
}

template <typename Dtype>
void cublas::geam(cublasOperation_t transa, cublasOperation_t transb,
    int m, int n,
    const Dtype *alpha, const Dtype *A, int lda,
    const Dtype *beta, const Dtype *B, int ldb,
    Dtype *C, int ldc) {
  cublasHandle_t handle = cublas::get_handle();
  CHECK_CUBLAS(cublas_geam(handle, transa, transb, m, n,
    alpha, A, lda, beta, B, ldb, C, ldc));
}

template <typename Dtype>
void cublas::gemm(cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const Dtype *alpha, const Dtype *A, int lda,
    const Dtype *B, int ldb,
    const Dtype *beta, Dtype *C, int ldc) {
  cublasHandle_t handle = cublas::get_handle();
  CHECK_CUBLAS(cublas_gemm(handle, transa, transb, m, n, k,
    alpha, A, lda, B, ldb, beta, C, ldc));
}

template void cublas::axpy(int n, float alpha, const float* x, float* y);
template void cublas::axpy(int n, double alpha, const double* x, double* y);
template void cublas::axpy(int n, float alpha, const float* x, int incx, float* y, int incy);
template void cublas::axpy(int n, double alpha, const double* x, int incx, double* y, int incy);

template void cublas::geam(cublasOperation_t transa, cublasOperation_t transb,
  int m, int n,
  const float *alpha, const float *A, int lda,
  const float *beta, const float *B, int ldb,
  float *C, int ldc);
template void cublas::geam(cublasOperation_t transa, cublasOperation_t transb,
  int m, int n,
  const double *alpha, const double *A, int lda,
  const double *beta, const double *B, int ldb,
  double *C, int ldc);

template void cublas::gemm(cublasOperation_t transa, cublasOperation_t transb,
  int m, int n, int k,
  const float *alpha, const float *A, int lda,
  const float *B, int ldb,
  const float *beta, float *C, int ldc);
template void cublas::gemm(cublasOperation_t transa, cublasOperation_t transb,
  int m, int n, int k,
  const double *alpha, const double *A, int lda,
  const double *B, int ldb,
  const double *beta, double *C, int ldc);

}  // namespace aion

