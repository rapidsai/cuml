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
  void wait(cudaStream_t& stream)
  {
    CUDA_CHECK(cudaStreamWaitEvent(stream, e, cudaEventWaitDefault));
  }
  void wait() { CUDA_CHECK(cudaEventSynchronize(e)); }
  DeviceEvent& operator=(const DeviceEvent& other) = delete;
};

}  // namespace

/** Solves the linear ordinary least squares problem `Aw = b`
 *  Via SVD decomposition of A or eigenvalue decomposition of `A^T * A`
 *  (covariance matrix for dataset A).
 */
template <typename math_t>
void lstsq(const raft::handle_t& handle,
           const math_t* A,
           const int n_rows,
           const int n_cols,
           const math_t* b,
           math_t* w,
           const int algo,
           cudaStream_t stream)
{
  ML::PUSH_RANGE("Trace::MLCommon::LinAlg::lstsq", stream);
  ASSERT(n_rows > 1, "lstsq: number of rows cannot be less than two");

  // we use some temporary vectors to avoid doing re-using w in the last step, the
  // gemv, which could cause a very sporadic race condition in Pascal and
  // Turing GPUs that caused it to give the wrong results. Details:
  // https://github.com/rapidsai/cuml/issues/1739
  if (algo == 0 || n_cols == 1 || n_cols > n_rows) {
    rmm::device_uvector<math_t> workset((n_cols + n_rows + 2) * n_cols, stream);
    math_t* U  = workset.data();
    math_t* V  = U + n_rows * n_cols;
    math_t* S  = V + n_cols * n_cols;
    math_t* Ub = S + n_cols;

    ML::PUSH_RANGE("Trace::MLCommon::LinAlg::lstsq::svdQR", stream);
    raft::linalg::svdQR(handle, (math_t*)A, n_rows, n_cols, S, U, V, true, true, true, stream);
    ML::POP_RANGE(stream);
    raft::linalg::gemv(handle, U, n_rows, n_cols, b, Ub, true, stream);

    raft::matrix::matrixVectorBinaryDivSkipZero(Ub, S, 1, n_cols, false, true, stream);

    raft::linalg::gemv(handle, V, n_cols, n_cols, Ub, w, false, stream);
  } else if (algo == 1) {
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
    auto f = [] __device__(math_t a, math_t b) {
      return raft::myAbs(b) > math_t(1e-10) ? a / b : a;
    };
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
  ML::POP_RANGE(stream);
}

template <typename math_t>
void lstsqQR(math_t* A,
             int n_rows,
             int n_cols,
             math_t* b,
             math_t* w,
             cusolverDnHandle_t cusolverH,
             cublasHandle_t cublasH,
             cudaStream_t stream)
{
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
// end namespace LinAlg
};  // namespace MLCommon
// end namespace MLCommon
