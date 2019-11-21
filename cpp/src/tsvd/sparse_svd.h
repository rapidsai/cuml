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
#include <random/rng.h>
#include <sys/time.h>

#define device_buffer    MLCommon::device_buffer
#define MIN(x, y)        (((x) > (y)) ? (y) : (x))

namespace ML {

template <typename T = float>
void SparseSVD_fit(const cumlHandle &handle,
                   const T *__restrict X,// (n, p)
                   const int n,
                   const int p,
                   T *__restrict U,      // (n, n_components)
                   T *__restrict S,      // (n_components)
                   T *__restrict VT,     // (n_components, p)
                   const int n_components = 2,
                   const int n_oversamples = 10,
                   const int max_iter = 3)
{
  ASSERT(n > 0 and p > 0 and U != NULL and S != NULL and VT != NULL, "Bad input");
  ASSERT(n_components > 0 and n_oversamples > 0 and max_iter >= 0, "Bad input");

  const auto d_alloc = handle.getDeviceAllocator();
  const cudaStream_t stream = handle.getStream();
  const int K = MIN(n_components + n_oversamples, p);

  // Prepare device buffers
  device_buffer<T> Y(d_alloc, stream, n*K); // Y(n, K)
  device_buffer<T> Z(d_alloc, stream, p*K); // Z(p, K)

  // Fill Z with random normal(0, 1)
  struct timeval tp; gettimeofday(&tp, NULL);
  MLCommon::Random::Rng random(tp.tv_sec * 1000 + tp.tv_usec);
  random.normal<T>(Z, p*K, 0, 1, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  device_buffer<T> T(d_alloc, stream, K*K); // T(K, K)


}

}  // namespace ML

#undef device_buffer
#undef MIN
