/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <raft/linalg/add.hpp>
#include <raft/sparse/coo.hpp>

#include <raft/linalg/transpose.hpp>
#include <raft/random/rng.hpp>

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
void launcher(const raft::handle_t& handle,
              int n,
              int d,
              raft::sparse::COO<float>* coo,
              UMAPParams* params,
              T* embedding)
{
  cudaStream_t stream = handle.get_stream();

  ASSERT(n > params->n_components, "Spectral layout requires n_samples > n_components");

  rmm::device_uvector<T> tmp_storage(n * params->n_components, stream);

  uint64_t seed = params->random_state;

  Spectral::fit_embedding(handle,
                          coo->rows(),
                          coo->cols(),
                          coo->vals(),
                          coo->nnz,
                          n,
                          params->n_components,
                          tmp_storage.data(),
                          seed);

  raft::linalg::transpose(handle, tmp_storage.data(), embedding, n, params->n_components, stream);

  raft::linalg::unaryOp<T>(
    tmp_storage.data(),
    tmp_storage.data(),
    n * params->n_components,
    [=] __device__(T input) { return fabsf(input); },
    stream);

  thrust::device_ptr<T> d_ptr = thrust::device_pointer_cast(tmp_storage.data());
  T max =
    *(thrust::max_element(thrust::cuda::par.on(stream), d_ptr, d_ptr + (n * params->n_components)));

  // Reuse tmp_storage to add random noise
  raft::random::Rng r(seed);
  r.normal(tmp_storage.data(), n * params->n_components, 0.0f, 0.0001f, stream);

  raft::linalg::unaryOp<T>(
    embedding,
    embedding,
    n * params->n_components,
    [=] __device__(T input) { return (10.0f / max) * input; },
    stream);

  raft::linalg::add(embedding, embedding, tmp_storage.data(), n * params->n_components, stream);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}
}  // namespace SpectralInit
}  // namespace InitEmbed
};  // namespace UMAPAlgo
