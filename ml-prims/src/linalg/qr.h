/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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
#include "common/cuml_allocator.hpp"
#include "common/device_buffer.hpp"

namespace MLCommon {
namespace LinAlg {

/**
 * @defgroup QR decomposition, return the Q matrix
 * @param M: input matrix
 * @param Q: Q matrix to be returned (on GPU)
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param cusolverH cusolver handle
 * @param stream cuda stream
 * @param allocate device allocator for temporary buffers during computation
 * @{
 */
template <typename math_t>
void qrGetQ(math_t *M, math_t *Q, int n_rows, int n_cols,
            cusolverDnHandle_t cusolverH, cudaStream_t stream,
            std::shared_ptr<deviceAllocator> allocator) {
  int m = n_rows, n = n_cols;
  int k = min(m, n);
  CUDA_CHECK(cudaMemcpyAsync(Q, M, sizeof(math_t) * m * n,
                             cudaMemcpyDeviceToDevice, stream));

  device_buffer<math_t> tau(allocator, stream, k);
  CUDA_CHECK(cudaMemsetAsync(tau.data(), 0, sizeof(math_t) * k, stream));

  device_buffer<int> devInfo(allocator, stream, 1);
  int Lwork;

  CUSOLVER_CHECK(cusolverDngeqrf_bufferSize(cusolverH, m, n, Q, m, &Lwork));
  device_buffer<math_t> workspace(allocator, stream, Lwork);
  CUSOLVER_CHECK(
      cusolverDngeqrf(cusolverH, m, n, Q, m, tau.data(), workspace.data(), Lwork, devInfo.data(), stream));
  /// @todo in v9.2, without deviceSynchronize *SquareMatrixNorm* ml-prims
  /// unit-tests fail. I have been able to use streamSynchronize successfully
  /// only from v10.0 onwards. Hence the following usage of macro
#if defined(CUDART_VERSION) && CUDART_VERSION > 9020
  CUDA_CHECK(cudaStreamSynchronize(stream));
#else
  CUDA_CHECK(cudaDeviceSynchronize());
#endif
  CUSOLVER_CHECK(
      cusolverDnorgqr_bufferSize(cusolverH, m, n, k, Q, m, tau.data(), &Lwork));
  workspace.resize(Lwork, stream);
  CUSOLVER_CHECK(
      cusolverDnorgqr(cusolverH, m, n, k, Q, m, tau.data(), workspace.data(), Lwork, devInfo.data(), stream));
}

/**
 * @defgroup QR decomposition, return the Q matrix
 * @param M: input matrix
 * @param Q: Q matrix to be returned (on GPU)
 * @param R: R matrix to be returned (on GPU)
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param cusolverH cusolver handle
 * @param stream cuda stream
 * @param allocator device allocator for temporary buffers during computation
 * @{
 */
template <typename math_t>
void qrGetQR(math_t *M, math_t *Q, math_t *R, int n_rows, int n_cols,
             cusolverDnHandle_t cusolverH, cudaStream_t stream,
             std::shared_ptr<deviceAllocator> allocator) {
  int m = n_rows, n = n_cols;
  device_buffer<math_t> R_full(allocator, stream, m * n);
  device_buffer<math_t> tau(allocator, stream, min(m, n));
  CUDA_CHECK(cudaMemsetAsync(tau.data(), 0, sizeof(math_t) * min(m, n), stream));
  int R_full_nrows = m, R_full_ncols = n;
  CUDA_CHECK(cudaMemcpyAsync(R_full.data(), M, sizeof(math_t) * m * n,
                             cudaMemcpyDeviceToDevice, stream));

  int Lwork;
  device_buffer<int> devInfo(allocator, stream, 1);

  CUSOLVER_CHECK(cusolverDngeqrf_bufferSize(
    cusolverH, R_full_nrows, R_full_ncols, R_full.data(), R_full_nrows, &Lwork));
  device_buffer<math_t> workspace(allocator, stream, Lwork);
  CUSOLVER_CHECK(cusolverDngeqrf(cusolverH, R_full_nrows, R_full_ncols, R_full.data(),
                                 R_full_nrows, tau.data(), workspace.data(), Lwork, devInfo.data(), stream));
  /// @todo in v9.2, without deviceSynchronize *SquareMatrixNorm* ml-prims
  /// unit-tests fail. I have been able to use streamSynchronize successfully
  /// only from v10.0 onwards. Hence the following usage of macro
#if defined(CUDART_VERSION) && CUDART_VERSION > 9020
  CUDA_CHECK(cudaStreamSynchronize(stream));
#else
  CUDA_CHECK(cudaDeviceSynchronize());
#endif

  Matrix::copyUpperTriangular(R_full.data(), R, m, n, stream);

  CUDA_CHECK(cudaMemcpyAsync(Q, R_full.data(), sizeof(math_t) * m * n,
                             cudaMemcpyDeviceToDevice, stream));
  int Q_nrows = m, Q_ncols = n;

  CUSOLVER_CHECK(cusolverDnorgqr_bufferSize(cusolverH, Q_nrows, Q_ncols,
                                            min(Q_ncols, Q_nrows), Q, Q_nrows,
                                            tau.data(), &Lwork));
  workspace.resize(Lwork, stream);
  CUSOLVER_CHECK(cusolverDnorgqr(cusolverH, Q_nrows, Q_ncols,
                                 min(Q_ncols, Q_nrows), Q, Q_nrows, tau.data(),
                                 workspace.data(), Lwork, devInfo.data(), stream));
}

}; // end namespace LinAlg
}; // end namespace MLCommon
