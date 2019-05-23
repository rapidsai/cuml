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

#include "cuda_runtime.h"
#include "distance/distance.h"
#include <math.h>
#include "cuda_utils.h"
#include <common/cumlHandle.hpp>
#include <common/device_buffer.hpp>

#include "pack.h"

namespace Dbscan {
namespace VertexDeg {
namespace Algo {


/**
 * Calculates the vertex degree array and the epsilon neighborhood adjacency matrix for the batch.
 */
template <typename value_t, typename index_t = int>
void launcher(const ML::cumlHandle_impl& handle, Pack<value_t, index_t> data,
              index_t startVertexId, index_t batchSize, cudaStream_t stream) {
    data.resetArray(stream, batchSize+1);

    index_t m = data.N;
    index_t n = min(data.N - startVertexId, batchSize);
    index_t k = data.D;

    index_t* vd = data.vd;

    value_t eps2 = data.eps * data.eps;

    MLCommon::device_buffer<char> workspace(handle.getDeviceAllocator(), stream);
    size_t workspaceSize = 0;

    constexpr auto distance_type = MLCommon::Distance::DistanceType::EucUnexpandedL2;

    workspaceSize =  MLCommon::Distance::getWorkspaceSize<distance_type, value_t, value_t,
                                                          bool, index_t>
            (data.x, data.x+startVertexId*k, m, n, k);

    if (workspaceSize != 0)
        workspace.resize(workspaceSize, stream);

    auto fused_op =  [vd, n] __device__ (index_t global_c_idx, bool in_neigh) {
                         // fused construction of vertex degree
                         index_t batch_vertex = global_c_idx - (n * (global_c_idx / n));
                         atomicAdd(vd+batch_vertex, in_neigh);
                         atomicAdd(vd+n, in_neigh);
                     };

    MLCommon::Distance::epsilon_neighborhood<distance_type, value_t,
                                             decltype(fused_op), index_t>
        (data.x, data.x+startVertexId*k, data.adj, m, n, k, eps2,
         (void*)workspace.data(), workspaceSize, stream, fused_op);

    CUDA_CHECK(cudaPeekAtLastError());
}
}  // end namespace Algo6
}  // end namespace VertexDeg
}; // end namespace Dbscan
