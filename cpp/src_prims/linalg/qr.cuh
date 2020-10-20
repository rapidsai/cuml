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

#include <raft/linalg/cublas_wrappers.h>
#include <raft/linalg/cusolver_wrappers.h>
#include <common/device_buffer.hpp>
#include <cuml/common/cuml_allocator.hpp>
#include <matrix/matrix.cuh>
#include <raft/mr/device/buffer.hpp>

namespace raft {
namespace linalg {

/**
 * @defgroup QRdecomp QR decomposition
 * @{
 */

/**
 * @brief compute QR decomp and return only Q matrix
 * @param handle: raft handle
 * @param M: input matrix
 * @param Q: Q matrix to be returned (on GPU)
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param stream cuda stream
 * @{
 */
template <typename math_t>
void qrGetQ(const raft::handle_t &handle, const math_t *M, math_t *Q,
            int n_rows, int n_cols, cudaStream_t stream) {
  auto allocator = handle.get_device_allocator();
  cusolverDnHandle_t cusolverH = handle.get_cusolver_dn_handle();

  int m = n_rows, n = n_cols;
  int k = min(m, n);
  CUDA_CHECK(cudaMemcpyAsync(Q, M, sizeof(math_t) * m * n,
                             cudaMemcpyDeviceToDevice, stream));

  raft::mr::device::buffer<math_t> tau(allocator, stream, k);
  CUDA_CHECK(cudaMemsetAsync(tau.data(), 0, sizeof(math_t) * k, stream));

  raft::mr::device::buffer<int> devInfo(allocator, stream, 1);
  int Lwork;

  CUSOLVER_CHECK(cusolverDngeqrf_bufferSize(cusolverH, m, n, Q, m, &Lwork));
  raft::mr::device::buffer<math_t> workspace(allocator, stream, Lwork);
  CUSOLVER_CHECK(cusolverDngeqrf(cusolverH, m, n, Q, m, tau.data(),
                                 workspace.data(), Lwork, devInfo.data(),
                                 stream));
  /// @note in v9.2, without deviceSynchronize *SquareMatrixNorm* ml-prims unit-tests fail.
#if defined(CUDART_VERSION) && CUDART_VERSION <= 9020
  CUDA_CHECK(cudaDeviceSynchronize());
#endif
  CUSOLVER_CHECK(
    cusolverDnorgqr_bufferSize(cusolverH, m, n, k, Q, m, tau.data(), &Lwork));
  workspace.resize(Lwork, stream);
  CUSOLVER_CHECK(cusolverDnorgqr(cusolverH, m, n, k, Q, m, tau.data(),
                                 workspace.data(), Lwork, devInfo.data(),
                                 stream));
}

/**
 * @brief compute QR decomp and return both Q and R matrices
 * @param handle: raft handle
 * @param M: input matrix
 * @param Q: Q matrix to be returned (on GPU)
 * @param R: R matrix to be returned (on GPU)
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param stream cuda stream
 */
template <typename math_t>
void qrGetQR(const raft::handle_t &handle, math_t *M, math_t *Q, math_t *R,
             int n_rows, int n_cols, cudaStream_t stream) {
  auto allocator = handle.get_device_allocator();
  cusolverDnHandle_t cusolverH = handle.get_cusolver_dn_handle();

  int m = n_rows, n = n_cols;
  raft::mr::device::buffer<math_t> R_full(allocator, stream, m * n);
  raft::mr::device::buffer<math_t> tau(allocator, stream, min(m, n));
  CUDA_CHECK(
    cudaMemsetAsync(tau.data(), 0, sizeof(math_t) * min(m, n), stream));
  int R_full_nrows = m, R_full_ncols = n;
  CUDA_CHECK(cudaMemcpyAsync(R_full.data(), M, sizeof(math_t) * m * n,
                             cudaMemcpyDeviceToDevice, stream));

  int Lwork;
  raft::mr::device::buffer<int> devInfo(allocator, stream, 1);

  CUSOLVER_CHECK(cusolverDngeqrf_bufferSize(cusolverH, R_full_nrows,
                                            R_full_ncols, R_full.data(),
                                            R_full_nrows, &Lwork));
  raft::mr::device::buffer<math_t> workspace(allocator, stream, Lwork);
  CUSOLVER_CHECK(cusolverDngeqrf(
    cusolverH, R_full_nrows, R_full_ncols, R_full.data(), R_full_nrows,
    tau.data(), workspace.data(), Lwork, devInfo.data(), stream));
  // @note in v9.2, without deviceSynchronize *SquareMatrixNorm* ml-prims unit-tests fail.
#if defined(CUDART_VERSION) && CUDART_VERSION <= 9020
  CUDA_CHECK(cudaDeviceSynchronize());
#endif

  raft::matrix::copyUpperTriangular(R_full.data(), R, m, n, stream);

  CUDA_CHECK(cudaMemcpyAsync(Q, R_full.data(), sizeof(math_t) * m * n,
                             cudaMemcpyDeviceToDevice, stream));
  int Q_nrows = m, Q_ncols = n;

  CUSOLVER_CHECK(cusolverDnorgqr_bufferSize(cusolverH, Q_nrows, Q_ncols,
                                            min(Q_ncols, Q_nrows), Q, Q_nrows,
                                            tau.data(), &Lwork));
  workspace.resize(Lwork, stream);
  CUSOLVER_CHECK(cusolverDnorgqr(
    cusolverH, Q_nrows, Q_ncols, min(Q_ncols, Q_nrows), Q, Q_nrows, tau.data(),
    workspace.data(), Lwork, devInfo.data(), stream));
}
/** @} */

};  // namespace linalg
};  // namespace raft
