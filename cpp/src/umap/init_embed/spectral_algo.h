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

#include <cuml/manifold/umapparams.h>
#include "common/device_buffer.hpp"

#include "sparse/coo.h"

#include "linalg/add.h"

#include "linalg/transpose.h"
#include "random/rng.h"

#include <cuml/cluster/spectral.hpp>
#include <iostream>

namespace UMAPAlgo {

namespace InitEmbed {

namespace SpectralInit {

using namespace ML;

/**
   * Performs a spectral layout initialization
   */
template <typename T>
void launcher(const cumlHandle &handle, const T *X, int n, int d,
              const int64_t *knn_indices, const T *knn_dists,
              MLCommon::Sparse::COO<float> *coo, UMAPParams *params,
              T *embedding) {
  cudaStream_t stream = handle.getStream();

  ASSERT(n > params->n_components,
         "Spectral layout requires n_samples > n_components");

  MLCommon::device_buffer<T> tmp_storage(handle.getDeviceAllocator(), stream,
                                         n * params->n_components);

  Spectral::fit_embedding(handle, coo->rows(), coo->cols(), coo->vals(),
                          coo->nnz, n, params->n_components,
                          tmp_storage.data());

  MLCommon::LinAlg::transpose(tmp_storage.data(), embedding, n,
                              params->n_components,
                              handle.getImpl().getCublasHandle(), stream);

  MLCommon::LinAlg::unaryOp<T>(
    tmp_storage.data(), tmp_storage.data(), n * params->n_components,
    [=] __device__(T input) { return fabsf(input); }, stream);

  thrust::device_ptr<T> d_ptr = thrust::device_pointer_cast(tmp_storage.data());
  T max = *(thrust::max_element(thrust::cuda::par.on(stream), d_ptr,
                                d_ptr + (n * params->n_components)));

  uint64_t seed = params->random_state;

  MLCommon::Random::Rng r(seed);
  r.normal(tmp_storage.data(), n * params->n_components, 0.0f, 0.0001f, stream);

  MLCommon::LinAlg::unaryOp<T>(
    embedding, embedding, n * params->n_components,
    [=] __device__(T input) { return (10.0f / max) * input; }, stream);

  MLCommon::LinAlg::add(embedding, embedding, tmp_storage.data(),
                        n * params->n_components, stream);

  CUDA_CHECK(cudaPeekAtLastError());
}
}  // namespace SpectralInit
}  // namespace InitEmbed
};  // namespace UMAPAlgo
