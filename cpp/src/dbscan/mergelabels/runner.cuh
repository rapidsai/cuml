/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <common/cumlHandle.hpp>
#include "naive.cuh"
#include "pack.h"

namespace ML {
namespace Dbscan {
namespace MergeLabels {

/// TODO: docs
template <typename Index_ = int>
void run(const raft::handle_t& handle, Index_* labelsA, const Index_* labelsB,
         const bool* mask, Index_* workBuffer, bool* m, Index_ N, int algo,
         cudaStream_t stream) {
  Pack<Index_> data = {labelsA, labelsB, mask, workBuffer, m, N};
  switch (algo) {
    case 0:
      Naive::launcher<Index_>(handle, data, stream);
      break;
    default:
      ASSERT(false, "Incorrect algo passed! '%d'", algo);
  }
}

}  // namespace MergeLabels
}  // namespace Dbscan
}  // namespace ML
