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
#include <raft/random/rng.cuh>
namespace UMAPAlgo {
namespace InitEmbed {
namespace RandomInit {

using namespace ML;

template <typename value_idx, typename T>
void launcher(int n, int d, const value_idx *knn_indices, const T *knn_dists,
              UMAPParams *params, T *embedding, cudaStream_t stream) {
  uint64_t seed = params->random_state;

  raft::random::Rng r(seed);
  r.uniform<T>(embedding, n * params->n_components, -10, 10, stream);
}
}  // namespace RandomInit
}  // namespace InitEmbed
};  // namespace UMAPAlgo
