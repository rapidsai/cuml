/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <raft/sparse/coo.hpp>

#include "random_algo.cuh"
#include "spectral_algo.cuh"

namespace UMAPAlgo {

namespace InitEmbed {

using namespace ML;

template <typename T>
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
    case 0: RandomInit::launcher(n, d, params, embedding, handle.get_stream()); break;

    case 1: SpectralInit::launcher(handle, n, d, coo, params, embedding); break;
  }
}
}  // namespace InitEmbed
};  // namespace UMAPAlgo
