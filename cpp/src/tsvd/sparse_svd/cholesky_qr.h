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

#include "qr.h"
#include <linalg/cublas_wrappers.h>
#include <linalg/cusolver_wrappers.h>
#include <common/device_buffer.hpp>
#include "common/cumlHandle.hpp"

#define device_buffer    MLCommon::device_buffer

namespace ML {

/*
  [TODO] Change epsilon jitter cholesky to approximate cholesky
  Computes a QR factorization of X using cholesky decomposition.
  X = QR
  X.T @ X = R.T @ R = U.T @ U
  So Q = X @ R^-1
*/
template <typename math_t>
int prepare_cholesky_qr_onlyQ(math_t *__restrict R,
                              const int p,
                              const cumlHandle &handle)
{
  const cusolverDnHandle_t solver_h = handle.getImpl().getcusolverDnHandle();

  // X.T @ X workspace
  int lwork = 0;
  CUSOLVER_CHECK(MLCommon::LinAlg::cusolverDnpotrf_bufferSize(solver_h,
                 CUBLAS_FILL_MODE_UPPER, p, &R[0], p, &lwork));
  return lwork;
}



template <typename math_t>
static __global__
void correction(math_t *__restrict XTX,
                const int p)
{
  const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= p) return;

  if (XTX[i + i*p] == 0) XTX[i + i*p] = p*2 + i;
  XTX[i + i*p] += 1e-6;
}


template <typename math_t>
int cholesky_qr_onlyQ(math_t *__restrict X,
                      math_t *__restrict R,
                      const int n,
                      const int p,
                      const cumlHandle &handle,
                      int lwork = 0,
                      math_t *__restrict work = NULL,
                      int *__restrict info = NULL)
{
  auto d_alloc = handle.getDeviceAllocator();
  const cudaStream_t stream = handle.getStream();
  const cusolverDnHandle_t solver_h = handle.getImpl().getcusolverDnHandle();
  const cublasHandle_t blas_h = handle.getImpl().getCublasHandle();

  // Only allocate workspace if lwork or work is NULL
  device_buffer<math_t> work_(d_alloc, stream);
  if (work == NULL) {
    lwork = prepare_cholesky_qr_onlyQ(R, p, handle);
    work_.resize(lwork, stream);
    work = work_.data();
  }

  device_buffer<int> info_(d_alloc, stream);
  if (info == NULL) {
    info_.resize(1, stream);
    info = info_.data();
  }

  // Do X.T @ X
  const math_t alpha = 1;
  const math_t beta = 0;
  CUBLAS_CHECK(MLCommon::LinAlg::cublassyrk(blas_h,
               CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, p, n,
               &alpha, &X[0], n, &beta, &R[0], p, stream));

  // Check X.T @ X for ill conditioning.
  // (1) Change all 0s on the diagonal to p * 2 + i
  // (2) Add 1e-6 to diagonal.
  correction<<<MLCommon::ceildiv(p, 1024), 1024, 0, stream>>>(&R[0], p);
  CUDA_CHECK(cudaPeekAtLastError());

  // Cholesky factorization of XTX
  CUSOLVER_CHECK(MLCommon::LinAlg::cusolverDnpotrf(solver_h,
                 CUBLAS_FILL_MODE_UPPER, p, &R[0], p,
                 &work[0], lwork, &info[0], stream));

  int info_out = 0;
  MLCommon::updateHost(&info_out, &info[0], 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // If this fails, use QR decomposition
  if (info_out != 0) {
    return info_out;
  }

  // Now do triangular solve to get Q!
  CUBLAS_CHECK(MLCommon::LinAlg::cublastrsm(blas_h,
               CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
               CUBLAS_DIAG_NON_UNIT, n, p, &alpha, &R[0], p, &X[0], n, stream));

  return info_out;
}



template <typename math_t>
int prepare_fast_qr_onlyQ(math_t *__restrict X,
                          math_t *__restrict R,
                          const int n,
                          const int p,
                          const cumlHandle &handle,
                          math_t *__restrict tau = NULL)
{
  /*
    Use Cholesky-QR only on very tall skinny matrices.
    Otherwise, this can be slower! Since ssyrk is 1/2np^2 FLOPS
    QR is approx 2np^2 FLOPS, so approx only use cholesky is n > 4*p
  */
  int lwork = 0;
  if (n > 4*p)
    lwork = prepare_cholesky_qr_onlyQ(R, p, handle);
  else
    lwork = prepare_qr_onlyQ(R, p, handle, tau);
  return lwork;
}


template <typename math_t>
void fast_qr_onlyQ(math_t *__restrict X,
                   math_t *__restrict R,
                   const int n,
                   const int p,
                   const cumlHandle &handle,
                   const bool verbose = false,
                   int lwork = 0,
                   math_t *__restrict work = NULL,
                   math_t *__restrict tau = NULL,
                   int *__restrict info = NULL)
{
  if (n > 4*p)
  {
    if (verbose)
      fprintf(stdout, "Using Fast Cholesky QR!\n");

    if (cholesky_qr_onlyQ(X, R, n, p, handle, lwork, work, info) != 0)
    {
      if (verbose)
        fprintf(stdout, "Cholesky failed. Using Normal QR\n");

      qr_onlyQ(X, n, p, handle, lwork, work, tau, info);
    }
  }
  else
  {
    if (verbose)
      fprintf(stdout, "Using Normal QR\n");

    qr_onlyQ(X, n, p, handle, lwork, work, tau, info);
  }
}


} // namespace ML

#undef device_buffer
