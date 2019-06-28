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

#include "aion_transpose.hpp"
#include "util/aion_cblas.hpp"
#include "util/aion_cublas.hpp"


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
