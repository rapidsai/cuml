/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include "../matrix/matrix.h"
#include "cublas_wrappers.h"
#include "cusolver_wrappers.h"
#include "device_allocator.h"

namespace MLCommon {
namespace LinAlg {

/**
 * @defgroup QR decomposition, return the Q matrix
 * @param M: input matrix
 * @param Q: Q matrix to be returned (on GPU)
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param cusolverH cusolver handle
 * @param mgr device allocator for temporary buffers during computation
 * @{
 */
template <typename math_t>
void qrGetQ(math_t *M, math_t *Q, int n_rows, int n_cols,
            cusolverDnHandle_t cusolverH, cudaStream_t stream,
            DeviceAllocator &mgr) {
  int m = n_rows, n = n_cols;
  int k = min(m, n);
  CUDA_CHECK(cudaMemcpyAsync(Q, M, sizeof(math_t) * m * n,
                             cudaMemcpyDeviceToDevice, stream));

  math_t *tau = (math_t *)mgr.alloc(sizeof(math_t) * k);
  CUDA_CHECK(cudaMemsetAsync(tau, 0, sizeof(math_t) * k, stream));

  int *devInfo = (int *)mgr.alloc(sizeof(int));
  int Lwork;

  CUSOLVER_CHECK(cusolverDngeqrf_bufferSize(cusolverH, m, n, Q, m, &Lwork));
  math_t *workspace = (math_t *)mgr.alloc(sizeof(math_t) * Lwork);
  CUSOLVER_CHECK(
    cusolverDngeqrf(cusolverH, m, n, Q, m, tau, workspace, Lwork, devInfo, stream));
  mgr.free(workspace, stream);
  CUSOLVER_CHECK(
    cusolverDnorgqr_bufferSize(cusolverH, m, n, k, Q, m, tau, &Lwork));
  workspace = (math_t *)mgr.alloc(sizeof(math_t) * Lwork);
  CUSOLVER_CHECK(
    cusolverDnorgqr(cusolverH, m, n, k, Q, m, tau, workspace, Lwork, devInfo, stream));
  mgr.free(workspace, stream);
  mgr.free(devInfo, stream);

  // clean up
  mgr.free(tau, stream);
}

/**
 * @defgroup QR decomposition, return the Q matrix
 * @param M: input matrix
 * @param Q: Q matrix to be returned (on GPU)
 * @param R: R matrix to be returned (on GPU)
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param cusolverH cusolver handle
 * @param mgr device allocator for temporary buffers during computation
 * @{
 */
template <typename math_t>
void qrGetQR(math_t *M, math_t *Q, math_t *R, int n_rows, int n_cols,
             cusolverDnHandle_t cusolverH, cudaStream_t stream,
             DeviceAllocator &mgr) {
  int m = n_rows, n = n_cols;
  math_t *R_full = (math_t *)mgr.alloc(sizeof(math_t) * m * n);
  math_t *tau = (math_t *)mgr.alloc(sizeof(math_t) * min(m, n));
  CUDA_CHECK(cudaMemsetAsync(tau, 0, sizeof(math_t) * min(m, n), stream));
  int R_full_nrows = m, R_full_ncols = n;
  CUDA_CHECK(cudaMemcpyAsync(R_full, M, sizeof(math_t) * m * n,
                             cudaMemcpyDeviceToDevice, stream));

  int Lwork;
  int *devInfo = (int *)mgr.alloc(sizeof(int));

  CUSOLVER_CHECK(cusolverDngeqrf_bufferSize(
    cusolverH, R_full_nrows, R_full_ncols, R_full, R_full_nrows, &Lwork));
  math_t *workspace = (math_t *)mgr.alloc(sizeof(math_t) * Lwork);
  CUSOLVER_CHECK(cusolverDngeqrf(cusolverH, R_full_nrows, R_full_ncols, R_full,
                                 R_full_nrows, tau, workspace, Lwork, devInfo, stream));
  mgr.free(workspace, stream);

  Matrix::copyUpperTriangular(R_full, R, m, n, stream);

  CUDA_CHECK(cudaMemcpyAsync(Q, R_full, sizeof(math_t) * m * n,
                             cudaMemcpyDeviceToDevice, stream));
  int Q_nrows = m, Q_ncols = n;

  CUSOLVER_CHECK(cusolverDnorgqr_bufferSize(cusolverH, Q_nrows, Q_ncols,
                                            min(Q_ncols, Q_nrows), Q, Q_nrows,
                                            tau, &Lwork));
  workspace = (math_t *)mgr.alloc(sizeof(math_t) * Lwork);
  CUSOLVER_CHECK(cusolverDnorgqr(cusolverH, Q_nrows, Q_ncols,
                                 min(Q_ncols, Q_nrows), Q, Q_nrows, tau,
                                 workspace, Lwork, devInfo, stream));
  mgr.free(workspace, stream);
  mgr.free(devInfo, stream);

  // clean up
  mgr.free(R_full, stream);
  mgr.free(tau, stream);
}

}; // end namespace LinAlg
}; // end namespace MLCommon
