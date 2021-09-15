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
#include <common/nvtx.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/eig.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/qr.cuh>
#include <raft/linalg/svd.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/mr/device/buffer.hpp>
#include <raft/random/rng.cuh>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace MLCommon {
namespace LinAlg {

namespace {

struct DeviceStream {
 private:
  cudaStream_t s;

 public:
  DeviceStream() { CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
  ~DeviceStream() { CUDA_CHECK(cudaStreamDestroy(s)); }
  operator cudaStream_t() const { return s; }
  DeviceStream& operator=(const DeviceStream& other) = delete;
};

struct DeviceEvent {
 private:
  cudaEvent_t e;

 public:
  DeviceEvent() { CUDA_CHECK(cudaEventCreate(&e)); }
  ~DeviceEvent() { CUDA_CHECK(cudaEventDestroy(e)); }
  operator cudaEvent_t() const { return e; }
  void record(cudaStream_t& stream) { CUDA_CHECK(cudaEventRecord(e, stream)); }
  void record(cudaStream_t&& stream) { CUDA_CHECK(cudaEventRecord(e, stream)); }
  void wait(cudaStream_t& stream) { CUDA_CHECK(cudaStreamWaitEvent(stream, e, 0u)); }
  void wait() { CUDA_CHECK(cudaEventSynchronize(e)); }
  DeviceEvent& operator=(const DeviceEvent& other) = delete;
};

}  // namespace

/** Solves the linear ordinary least squares problem `Aw = b`
 *  Via SVD decomposition of `A = U S Vt` using default cuSOLVER routine.
 */
template <typename math_t>
void lstsqSvdQR(const raft::handle_t& handle,
                math_t* A,  // apparently, this must not be const, because cusolverDn<t>gesvd() says
                            // it's destroyed upon exit of the routine.
                const int n_rows,
                const int n_cols,
                const math_t* b,
                math_t* w,
                cudaStream_t stream)
{
  // we use some temporary vectors to avoid doing re-using w in the last step, the
  // gemv, which could cause a very sporadic race condition in Pascal and
  // Turing GPUs that caused it to give the wrong results. Details:
  // https://github.com/rapidsai/cuml/issues/1739

  // orig

  // size_t U_len = n_rows * n_cols;
  // size_t V_len = n_cols * n_cols;

  // rmm::device_uvector<math_t> S(n_cols, stream);
  // rmm::device_uvector<math_t> V(V_len, stream);
  // rmm::device_uvector<math_t> U(U_len, stream);

  // // we use a temporary vector to avoid doing re-using w in the last step, the
  // // gemv, which could cause a very sporadic race condition in Pascal and
  // // Turing GPUs that caused it to give the wrong results. Details:
  // // https://github.com/rapidsai/cuml/issues/1739
  // rmm::device_uvector<math_t> tmp_vector(n_cols, stream);
  // raft::linalg::svdQR(
  //   handle, A, n_rows, n_cols, S.data(), U.data(), V.data(), true, true, true, stream);
  // raft::linalg::gemv(handle, U.data(), n_rows, n_cols, b, tmp_vector.data(), true, stream);

  // raft::matrix::matrixVectorBinaryDivSkipZero(
  //   tmp_vector.data(), S.data(), 1, n_cols, false, true, stream);

  // raft::linalg::gemv(handle, V.data(), n_cols, n_cols, tmp_vector.data(), w, false, stream);

  // inlined

  const int minmn              = min(n_rows, n_cols);
  cusolverDnHandle_t cusolverH = handle.get_cusolver_dn_handle();
  int cusolverWorkSetSize      = 0;
  CUSOLVER_CHECK(raft::linalg::cusolverDngesvd_bufferSize<math_t>(
    cusolverH, n_rows, n_cols, &cusolverWorkSetSize));

  rmm::device_uvector<math_t> workset(cusolverWorkSetSize  // cuSolver
                                        + n_rows * minmn   // U
                                        + n_cols * n_cols  // V
                                        + minmn            // S
                                        + minmn            // U^T * b
                                        + 1                // devInfo
                                      ,
                                      stream);
  math_t* cusolverWorkSet = workset.data();
  math_t* U               = cusolverWorkSet + cusolverWorkSetSize;
  math_t* Vt              = U + n_rows * minmn;
  math_t* S               = Vt + n_cols * n_cols;
  math_t* Ub              = S + minmn;
  int* devInfo            = reinterpret_cast<int*>(Ub + minmn);

  CUSOLVER_CHECK(raft::linalg::cusolverDngesvd<math_t>(cusolverH,
                                                       'S',
                                                       'S',
                                                       n_rows,
                                                       n_cols,
                                                       A,
                                                       n_rows,
                                                       S,
                                                       U,
                                                       n_rows,
                                                       Vt,
                                                       n_cols,
                                                       cusolverWorkSet,
                                                       cusolverWorkSetSize,
                                                       nullptr,
                                                       devInfo,
                                                       stream));
  raft::linalg::gemv(handle, U, n_rows, minmn, b, Ub, true, stream);
  raft::linalg::eltwiseDivideCheckZero(Ub, Ub, S, minmn, stream);
  math_t alpha = math_t(1);
  math_t beta  = math_t(0);
  // wait on https://github.com/rapidsai/raft/pull/327 to enable this.
  // I have to use cublas here for its non-trivial lda not available in raft::linalg::gemv
  CUBLAS_CHECK(raft::linalg::cublasgemv<math_t>(handle.get_cublas_handle(),
                                                CUBLAS_OP_T,
                                                minmn,
                                                n_cols,
                                                &alpha,
                                                Vt,
                                                n_cols,
                                                Ub,
                                                1,
                                                &beta,
                                                w,
                                                1,
                                                stream));
}

/** Solves the linear ordinary least squares problem `Aw = b`
 *  Via SVD decomposition of `A = U S V^T` using Jacobi iterations (cuSOLVER).
 */
template <typename math_t>
void lstsqSvdJacobi(const raft::handle_t& handle,
                    math_t* A,  // apparently, this must not be const, because cusolverDn<t>gesvdj()
                                // says it's destroyed upon exit of the routine.
                    const int n_rows,
                    const int n_cols,
                    const math_t* b,
                    math_t* w,
                    cudaStream_t stream)
{
  const int minmn = min(n_rows, n_cols);
  gesvdjInfo_t gesvdj_params;
  CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));
  int cusolverWorkSetSize      = 0;
  cusolverDnHandle_t cusolverH = handle.get_cusolver_dn_handle();
  CUSOLVER_CHECK(raft::linalg::cusolverDngesvdj_bufferSize<math_t>(cusolverH,
                                                                   CUSOLVER_EIG_MODE_VECTOR,
                                                                   1,
                                                                   n_rows,
                                                                   n_cols,
                                                                   A,
                                                                   n_rows,
                                                                   nullptr,
                                                                   nullptr,
                                                                   n_rows,
                                                                   nullptr,
                                                                   n_cols,
                                                                   &cusolverWorkSetSize,
                                                                   gesvdj_params));
  rmm::device_uvector<math_t> workset(cusolverWorkSetSize  // cuSolver
                                        + n_rows * minmn   // U
                                        + n_cols * minmn   // V
                                        + minmn            // S
                                        + minmn            // U^T * b
                                        + 1                // devInfo
                                      ,
                                      stream);
  math_t* cusolverWorkSet = workset.data();
  math_t* U               = cusolverWorkSet + cusolverWorkSetSize;
  math_t* V               = U + n_rows * minmn;
  math_t* S               = V + n_cols * minmn;
  math_t* Ub              = S + minmn;
  int* devInfo            = reinterpret_cast<int*>(Ub + minmn);
  CUSOLVER_CHECK(raft::linalg::cusolverDngesvdj<math_t>(cusolverH,
                                                        CUSOLVER_EIG_MODE_VECTOR,
                                                        1,
                                                        n_rows,
                                                        n_cols,
                                                        A,
                                                        n_rows,
                                                        S,
                                                        U,
                                                        n_rows,
                                                        V,
                                                        n_cols,
                                                        cusolverWorkSet,
                                                        cusolverWorkSetSize,
                                                        devInfo,
                                                        gesvdj_params,
                                                        stream));
  raft::linalg::gemv(handle, U, n_rows, minmn, b, Ub, true, stream);
  raft::linalg::eltwiseDivideCheckZero(Ub, Ub, S, minmn, stream);

  // wait on https://github.com/rapidsai/raft/pull/327 to enable this.
  // raft::linalg::gemv(handle, V, n_cols, minmn, Ub, w, false, stream);
  math_t alpha = math_t(1);
  math_t beta  = math_t(0);
  CUBLAS_CHECK(raft::linalg::cublasgemv<math_t>(handle.get_cublas_handle(),
                                                CUBLAS_OP_N,
                                                n_cols,
                                                minmn,
                                                &alpha,
                                                V,
                                                n_cols,
                                                Ub,
                                                1,
                                                &beta,
                                                w,
                                                1,
                                                stream));
}

/** Solves the linear ordinary least squares problem `Aw = b`
 *  via eigenvalue decomposition of `A^T * A` (covariance matrix for dataset A).
 *  (`w = (A^T A)^-1  A^T b`)
 */
template <typename math_t>
void lstsqEig(const raft::handle_t& handle,
              const math_t* A,
              const int n_rows,
              const int n_cols,
              const math_t* b,
              math_t* w,
              cudaStream_t stream)
{
  rmm::device_uvector<math_t> workset(n_cols * n_cols * 3 + n_cols * 2, stream);
  math_t* Q    = workset.data();
  math_t* QS   = Q + n_cols * n_cols;
  math_t* covA = QS + n_cols * n_cols;
  math_t* S    = covA + n_cols * n_cols;
  math_t* Ab   = S + n_cols;

  // covA <- A* A
  math_t alpha = math_t(1);
  math_t beta  = math_t(0);
  raft::linalg::gemm(handle,
                     A,
                     n_rows,
                     n_cols,
                     A,
                     covA,
                     n_cols,
                     n_cols,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     alpha,
                     beta,
                     stream);

  // Ab <- A* b
  DeviceStream multAb;
  DeviceEvent multAbDone;
  raft::linalg::gemv(handle, A, n_rows, n_cols, b, Ab, true, multAb);
  multAbDone.record(multAb);

  // Q S Q* <- covA
  ML::PUSH_RANGE("Trace::MLCommon::LinAlg::lstsq::eigDC", stream);
  raft::linalg::eigDC(handle, covA, n_cols, n_cols, Q, S, stream);
  ML::POP_RANGE(stream);

  // QS  <- Q invS
  auto f = [] __device__(math_t a, math_t b) { return raft::myAbs(b) > math_t(1e-10) ? a / b : a; };
  raft::linalg::matrixVectorOp(QS, Q, S, n_cols, n_cols, false, true, f, stream);
  // covA <- QS Q* == Q invS Q* == inv(A* A)
  raft::linalg::gemm(handle,
                     QS,
                     n_cols,
                     n_cols,
                     Q,
                     covA,
                     n_cols,
                     n_cols,
                     CUBLAS_OP_N,
                     CUBLAS_OP_T,
                     alpha,
                     beta,
                     stream);
  multAbDone.wait(stream);
  // w <- covA Ab == Q invS Q* A b == inv(A* A) A b
  raft::linalg::gemv(handle, covA, n_cols, n_cols, Ab, w, false, stream);
}

/** Solves the linear ordinary least squares problem `Aw = b`
 *  via QR decomposition of `A = QR`.
 *  (triangular system of equations `Rw = Q^T b`)
 */
template <typename math_t>
void lstsqQR(const raft::handle_t& handle,
             math_t* A,
             const int n_rows,
             const int n_cols,
             math_t* b,
             math_t* w,
             cudaStream_t stream)
{
  cublasHandle_t cublasH       = handle.get_cublas_handle();
  cusolverDnHandle_t cusolverH = handle.get_cusolver_dn_handle();

  int m = n_rows;
  int n = n_cols;

  int info = 0;
  rmm::device_uvector<math_t> d_tau(n, stream);
  rmm::device_scalar<int> d_info(stream);

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

  rmm::device_uvector<math_t> d_work(lwork, stream);

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
};  // namespace MLCommon
