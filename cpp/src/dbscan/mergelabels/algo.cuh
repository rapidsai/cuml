/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cuda_runtime.h>
#include <common/cumlHandle.hpp>
#include <label/merge_labels.cuh>

#include "pack.h"

namespace ML {
namespace Dbscan {
namespace MergeLabels {
namespace Algo {

/**
 * Launches the label merging computation
 */
template <typename Index_ = int, int TPB_X = 256>
void launcher(const raft::handle_t& handle, Pack<Index_> data,
              cudaStream_t stream) {
  MLCommon::Label::merge_labels<Index_, TPB_X>(data.labels_a, data.labels_b,
                                               data.mask, data.work_buffer,
                                               data.m, data.N, stream);
}

}  // namespace Algo
}  // namespace MergeLabels
}  // namespace Dbscan
}  // namespace ML
