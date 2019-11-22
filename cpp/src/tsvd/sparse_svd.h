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

#include "sparse_svd.h"
#include "sparse_svd/cholesky_qr.h"
#include "sparse_svd/qr.h"
#include <linalg/gemm.h>
#include <random/rng.h>
#include <sys/time.h>

#define device_buffer    MLCommon::device_buffer
#define MIN(x, y)        (((x) > (y)) ? (y) : (x))
#define MAX(x, y)        (((x) < (y)) ? (y) : (x))

namespace ML {

template <typename math_t = float>
void SparseSVD_fit(const cumlHandle &handle,
                   const math_t *__restrict X,// (n, p)
                   const int n,
                   const int p,
                   math_t *__restrict U,      // (n, n_components)
                   math_t *__restrict S,      // (n_components)
                   math_t *__restrict VT,     // (n_components, p)
                   const int n_components = 2,
                   const int n_oversamples = 10,
                   const int max_iter = 3,
                   int random_state = -1,
                   const bool verbose = true)
{
  ASSERT(n > 0 and p > 0 and U != NULL and S != NULL and VT != NULL, "Bad input");
  ASSERT(n_components > 0 and n_oversamples > 0 and max_iter >= 0, "Bad input");

  const auto d_alloc = handle.getDeviceAllocator();
  const cudaStream_t stream = handle.getStream();
  const cusolverDnHandle_t solver_h = handle.getImpl().getcusolverDnHandle();
  const cublasHandle_t blas_h = handle.getImpl().getCublasHandle();

  const int K = MIN(n_components + n_oversamples, p);

  // Prepare device buffers
  device_buffer<math_t> Y_(d_alloc, stream, n*K); // Y(n, K)
  math_t *__restrict Y = Y_.data();
  device_buffer<math_t> Z_(d_alloc, stream, p*K); // Z(p, K)
  math_t *__restrict Z = Z_.data();
  device_buffer<math_t> T_(d_alloc, stream, K*K); // T(K, K)
  math_t *__restrict T = T_.data();


  // Fill Z with random normal(0, 1)
  if (random_state == -1) {
     struct timeval tp; gettimeofday(&tp, NULL);
     random_state = tp.tv_sec * 1000 + tp.tv_usec;
  }
  MLCommon::Random::Rng random(random_state);
  random.normal<math_t>(Z, p*K, 0, 1, stream);


  // Tau for QR(Y)
  const int min_dim = MIN(n, K);
  device_buffer<math_t> tau_(d_alloc, stream, min_dim);
  math_t *__restrict tau = tau_.data();

  // Prepare workspaces for QR decomposition
  const int lwork1 = prepare_fast_qr_onlyQ(&Y[0], &T[0], n, K, handle, tau);
  const int lwork2 = prepare_fast_qr_onlyQ(&Z[0], &T[0], p, K, handle, tau);
  const int lwork = MAX(lwork1, lwork2);
  device_buffer<math_t> work_(d_alloc, stream, lwork);
  math_t *__restrict work = work_.data();

  // Info
  device_buffer<int> info_(d_alloc, stream, 1);
  int *__restrict info = info_.data();


  // Y = X @ Z
  MLCommon::LinAlg::gemm(&X[0], n, p, &Z[0], &Y[0], n, K, CUBLAS_OP_N, CUBLAS_OP_N, blas_h, stream);
  // Y, _ = qr(Y)
  fast_qr_onlyQ(&Y[0], &T[0], n, K, handle, verbose, lwork, &work[0], &tau[0], &info[0]);

  for (int iter = 0; iter < max_iter; iter++)
  {
    // Z = X.T @ Y
    MLCommon::LinAlg::gemm(&X[0], n, p, &Y[0], &Z[0], p, K, CUBLAS_OP_T, CUBLAS_OP_N, blas_h, stream);
    // Z, _ = qr(Z)
    fast_qr_onlyQ(&Z[0], &T[0], p, K, handle, verbose, lwork, &work[0], &tau[0], &info[0]);

    // Y = X @ Z
    MLCommon::LinAlg::gemm(&X[0], n, p, &Z[0], &Y[0], n, K, CUBLAS_OP_N, CUBLAS_OP_N, blas_h, stream);
    // Y, _ = qr(Y)
    fast_qr_onlyQ(&Y[0], &T[0], n, K, handle, verbose, lwork, &work[0], &tau[0], &info[0]);
  }
  
}

}  // namespace ML

#undef device_buffer
#undef MIN
#undef MAX
