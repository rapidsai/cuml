/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <raft/cudart_utils.h>
#include <raft/linalg/cublas_wrappers.h>
#include <raft/linalg/cusolver_wrappers.h>
#include <raft/linalg/gemv.h>
#include <raft/linalg/transpose.h>
#include <cuml/common/device_buffer.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/eig.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/qr.cuh>
#include <raft/linalg/svd.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/mr/device/allocator.hpp>
#include <raft/mr/device/buffer.hpp>
#include <raft/random/rng.cuh>
#include <rmm/device_uvector.hpp>

namespace MLCommon {
namespace LinAlg {

template <typename math_t>
void lstsq(const raft::handle_t& handle,
           math_t* A,
           int n_rows,
           int n_cols,
           math_t* b,
           math_t* w,
           int algo,
           cudaStream_t stream)
{
  cusolverDnHandle_t cusolverH = handle.get_cusolver_dn_handle();
  cublasHandle_t cublasH       = handle.get_cublas_handle();

  ASSERT(n_rows > 1, "lstsq: number of rows cannot be less than two");

  size_t U_len = n_rows * n_cols;
  size_t V_len = n_cols * n_cols;

  rmm::device_uvector<math_t> S(n_cols, stream);
  rmm::device_uvector<math_t> V(V_len, stream);
  rmm::device_uvector<math_t> U(U_len, stream);

  // we use a temporary vector to avoid doing re-using w in the last step, the
  // gemv, which could cause a very sporadic race condition in Pascal and
  // Turing GPUs that caused it to give the wrong results. Details:
  // https://github.com/rapidsai/cuml/issues/1739
  rmm::device_uvector<math_t> tmp_vector(n_cols, stream);

  if (algo == 0 || n_cols == 1) {
    raft::linalg::svdQR(
      handle, A, n_rows, n_cols, S.data(), U.data(), V.data(), true, true, true, stream);
  } else if (algo == 1) {
    raft::linalg::svdEig(handle, A, n_rows, n_cols, S.data(), U.data(), V.data(), true, stream);
  }

  raft::linalg::gemv(handle, U.data(), n_rows, n_cols, b, tmp_vector.data(), true, stream);

  raft::matrix::matrixVectorBinaryDivSkipZero(
    tmp_vector.data(), S.data(), 1, n_cols, false, true, stream);

  raft::linalg::gemv(handle, V.data(), n_cols, n_cols, tmp_vector.data(), w, false, stream);
}

template <typename math_t>
void lstsqQR(math_t* A,
             int n_rows,
             int n_cols,
             math_t* b,
             math_t* w,
             cusolverDnHandle_t cusolverH,
             cublasHandle_t cublasH,
             std::shared_ptr<raft::mr::device::allocator> allocator,
             cudaStream_t stream)
{
  int m = n_rows;
  int n = n_cols;

  int info = 0;
  device_buffer<math_t> d_tau(allocator, stream, n);
  device_buffer<int> d_info(allocator, stream, 1);

  const cublasSideMode_t side   = CUBLAS_SIDE_LEFT;
  const cublasOperation_t trans = CUBLAS_OP_T;

  int lwork_geqrf = 0;
  int lwork_ormqr = 0;
  int lwork       = 0;

  const int lda = m;
  const int ldb = m;

  CUSOLVER_CHECK(raft::linalg::cusolverDngeqrf_bufferSize(cusolverH, m, n, A, lda, &lwork_geqrf));

  CUSOLVER_CHECK(raft::linalg::cusolverDnormqr_bufferSize(cusolverH,
                                                          side,
                                                          trans,
                                                          m,
                                                          1,
                                                          n,
                                                          A,
                                                          lda,
                                                          d_tau.data(),
                                                          b,    // C,
                                                          lda,  // ldc,
                                                          &lwork_ormqr));

  lwork = (lwork_geqrf > lwork_ormqr) ? lwork_geqrf : lwork_ormqr;

  device_buffer<math_t> d_work(allocator, stream, lwork);

  CUSOLVER_CHECK(raft::linalg::cusolverDngeqrf(
    cusolverH, m, n, A, lda, d_tau.data(), d_work.data(), lwork, d_info.data(), stream));

  CUDA_CHECK(cudaMemcpyAsync(&info, d_info.data(), sizeof(int), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ASSERT(0 == info, "lstsq.h: QR wasn't successful");

  CUSOLVER_CHECK(raft::linalg::cusolverDnormqr(cusolverH,
                                               side,
                                               trans,
                                               m,
                                               1,
                                               n,
                                               A,
                                               lda,
                                               d_tau.data(),
                                               b,
                                               ldb,
                                               d_work.data(),
                                               lwork,
                                               d_info.data(),
                                               stream));

  CUDA_CHECK(cudaMemcpyAsync(&info, d_info.data(), sizeof(int), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ASSERT(0 == info, "lstsq.h: QR wasn't successful");

  const math_t one = 1;

  CUBLAS_CHECK(raft::linalg::cublastrsm(cublasH,
                                        side,
                                        CUBLAS_FILL_MODE_UPPER,
                                        CUBLAS_OP_N,
                                        CUBLAS_DIAG_NON_UNIT,
                                        n,
                                        1,
                                        &one,
                                        A,
                                        lda,
                                        b,
                                        ldb,
                                        stream));

  CUDA_CHECK(cudaMemcpyAsync(w, b, sizeof(math_t) * n, cudaMemcpyDeviceToDevice, stream));
}

};  // namespace LinAlg
// end namespace LinAlg
};  // namespace MLCommon
// end namespace MLCommon
