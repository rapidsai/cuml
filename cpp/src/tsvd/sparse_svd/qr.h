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
#define MIN(x, y)        (((x) > (y)) ? (y) : (x))
#define MAX(x, y)        (((x) < (y)) ? (y) : (x))

namespace ML {

template <typename math_t>
int prepare_qr_onlyQ(math_t *__restrict X,
                     const int n,
                     const int p,
                     const cumlHandle &handle,
                     math_t *__restrict tau = NULL)
{
  auto d_alloc = handle.getDeviceAllocator();
  const cudaStream_t stream = handle.getStream();
  const cusolverDnHandle_t solver_h = handle.getImpl().getcusolverDnHandle();

  // Workspace for QR Decomposition
  int lwork1 = 0;
  CUSOLVER_CHECK(MLCommon::LinAlg::cusolverDngeqrf_bufferSize(solver_h,
                 n, p, &X[0], n, &lwork1));

  // Tau array
  // Is this needed to calculate workspace?
  const int K = MIN(n, p);
  device_buffer<math_t> tau_(d_alloc, stream);
  if (tau == NULL) {
    tau_.resize(K, stream);
    tau = tau_.data();
  }

  // Workspace to create Q
  int lwork2 = 0;
  CUSOLVER_CHECK(MLCommon::LinAlg::cusolverDnorgqr_bufferSize(solver_h,
                 n, p, K, &X[0], n, &tau[0], &lwork2));

  return MAX(lwork1, lwork2);
}


template <typename math_t>
void qr_onlyQ(math_t *__restrict X,
              const int n,
              const int p,
              const cumlHandle &handle,
              int lwork = 0,
              math_t *__restrict work = NULL,
              math_t *__restrict tau = NULL,
              int *__restrict info = NULL)
{
  auto d_alloc = handle.getDeviceAllocator();
  const cudaStream_t stream = handle.getStream();
  const cusolverDnHandle_t solver_h = handle.getImpl().getcusolverDnHandle();
  const cublasHandle_t blas_h = handle.getImpl().getCublasHandle();

  // Only allocate workspace if lwork or work is NULL
  device_buffer<math_t> work_(d_alloc, stream);
  if (work == NULL) {
    lwork = prepare_qr_onlyQ(X, n, p, handle);
    work_.resize(lwork, stream);
    work = work_.data();
  }

  device_buffer<int> info_(d_alloc, stream);
  if (info == NULL) {
    info_.resize(1, stream);
    info = info_.data();
  }

  // Tau array
  const int K = MIN(n, p);
  device_buffer<math_t> tau_(d_alloc, stream);
  if (tau == NULL) {
    tau_.resize(K, stream);
    tau = tau_.data();
  }

  // QR Decomposition
  CUSOLVER_CHECK(MLCommon::LinAlg::cusolverDngeqrf(solver_h,
                 n, p, &X[0], n, &tau[0], &work[0], lwork, &info[0], stream));

  // Get Q
  CUSOLVER_CHECK(MLCommon::LinAlg::cusolverDnorgqr(solver_h,
                 n, p, K, &X[0], n, &tau[0], &work[0], lwork, &info[0], stream));

  CUDA_CHECK(cudaGetLastError());
}


} // namespace ML

#undef device_buffer
#undef MIN
#undef MAX
