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
                   int random_state = -1)
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


  // Y = X @ Z
  MLCommon::LinAlg::gemm(&X[0], n, p, &Z[0], &Y[0], n, K, CUBLAS_OP_N, CUBLAS_OP_N, blas_h, stream);
  // Y, _ = qr(Y)
  SparseSVD::cholesky_qr(&Y[0], &T[0], n, K, handle);

  
}

}  // namespace ML

#undef device_buffer
#undef MIN
