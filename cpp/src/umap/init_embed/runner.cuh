/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include "random_algo.cuh"
#include "spectral_algo.cuh"

#include <cuml/manifold/umapparams.h>

#include <raft/sparse/coo.hpp>

namespace UMAPAlgo {

namespace InitEmbed {

using namespace ML;

template <typename T, typename nnz_t>
void run(const raft::handle_t& handle,
         int n,
         int d,
         raft::sparse::COO<float>* coo,
         UMAPParams* params,
         T* embedding,
         cudaStream_t stream,
         int algo = 0)
{
  switch (algo) {
    /**
     * Initial algo uses FAISS indices
     */
    case 0: RandomInit::launcher<T, nnz_t>(n, d, params, embedding, handle.get_stream()); break;

    case 1: try { SpectralInit::launcher<T, nnz_t>(handle, n, d, coo, params, embedding);
      } catch (const raft::exception& e) {
        CUML_LOG_WARN("Spectral initialization failed, using random initialization instead.");
        RandomInit::launcher<T, nnz_t>(n, d, params, embedding, handle.get_stream());
      }
      break;
  }
}
}  // namespace InitEmbed
};  // namespace UMAPAlgo
