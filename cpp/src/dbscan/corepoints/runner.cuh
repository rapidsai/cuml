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
#include "algo.cuh"
#include "pack.h"

namespace ML {
namespace Dbscan {
namespace CorePoints {

template <typename Type_f, typename Index_ = int>
void run(const raft::handle_t& handle, const Index_* vd, bool* mask,
         Index_ minPts, Index_ startVertexId, Index_ batchSize,
         cudaStream_t stream) {
  Pack<Type_f, Index_> data = {vd, mask, minPts};
  Algo::launcher<Type_f, Index_>(handle, data, startVertexId, batchSize,
                                 stream);
}

}  // namespace CorePoints
}  // namespace Dbscan
}  // namespace ML
