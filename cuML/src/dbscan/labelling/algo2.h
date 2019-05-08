/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <cuda_utils.h>
#include "pack.h"
#include "dbscan/common.h"
#include <iostream>
#include <limits>
#include <common/cumlHandle.hpp>
#include <common/host_buffer.hpp>

#include "sparse/csr.h"

namespace Dbscan {
namespace Label {

/**
 * This implementation comes from [1] and solves component labeling problem in
 * parallel.
 *
 * todo: This might also be reusable as a more generalized connected component
 * labeling algorithm.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 */
namespace Algo2 {

using namespace thrust;
using namespace MLCommon;

static const int TPB_X = 256;

template <typename Type>
void launcher(const ML::cumlHandle_impl& handle, Pack<Type> data, Type N,
        int startVertexId, int batchSize, cudaStream_t stream) {

    bool *core_pts = data.core_pts;

    MLCommon::Sparse::weak_cc_batched<Type, TPB_X>(
            data.db_cluster, data.ex_scan, data.adj_graph, N,
            startVertexId, batchSize, [core_pts](int tid) {return core_pts[tid];},
            data.state, stream);
}

} // End Algo2
} // End Label
} // End Dbscan
