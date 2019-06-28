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

#include "util/aion_cblas.hpp"

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
}  // namespace aion
