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
int prepare_cholesky_qr(math_t *__restrict R,
                         const int n,
                         const int p,
                         cusolverDnHandle_t solver_h)
{
  // X.T @ X workspace
  int lwork = 0;
  CUSOLVER_CHECK(cusolverDnpotrf_bufferSize(solver_h, CUBLAS_FILL_MODE_UPPER, p, R, p, &lwork));
  return lwork;
}


template <typename math_t>
void cholesky_qr(const math_t *__restrict X,
                 math_t *__restrict R,
                 const int n,
                 const int p,
                 const cumlHandle &handle,
                 const int lwork = 0,
                 math_t *__restrict work = NULL)
{
  auto d_alloc = handle.getDeviceAllocator();
  const cudaStream_t stream = handle.getStream();
  const cusolverDnHandle_t solver_h = handle.getImpl().getcusolverDnHandle();
  const cublasHandle_t blas_h = handle.getImpl().getCublasHandle();

  if (lwork == 0)
  {
    const int lwork = prepare_cholesky_qr(R, n, p, solver_h);
    device_buffer<math_t> work(d_alloc, stream, lwork);

    // Do X.T @ X
    const math_t alpha = 1;
    const math_t beta = 0;
    CUBLAS_CHECK(cublassyrk(blas_h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, p, n,
                            &alpha, &X[0], n, &beta, &R[0], p, stream));

    work.release(stream);
  }
}



}  // namespace ML

#undef device_buffer
#undef MIN
