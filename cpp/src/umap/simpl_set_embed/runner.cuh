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
#include "algo.cuh"

#include <raft/sparse/coo.cuh>

namespace UMAPAlgo {

namespace SimplSetEmbed {

using namespace ML;

template <int TPB_X, typename T>
void run(int m, int n, raft::sparse::COO<T> *coo, UMAPParams *params,
         T *embedding, std::shared_ptr<deviceAllocator> alloc,
         cudaStream_t stream, int algorithm = 0) {
  switch (algorithm) {
    case 0:
      SimplSetEmbed::Algo::launcher<TPB_X, T>(m, n, coo, params, embedding,
                                              alloc, stream);
  }
}
}  // namespace SimplSetEmbed
}  // namespace UMAPAlgo
