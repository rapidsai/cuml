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
 * Calculates both the vertex degree array and the epsilon neighborhood in a single kernel.
 *
 * Proposed API for this should be an epsilon neighborhood primitive that accepts a lambda and
 * executes the lambda with [n, acc, vertex].
 *
 * template<typename T, typename Lambda>
 * void epsilon_neighborhood(T *a, T *b, bool *adj, m, n, k, T eps,
 *      workspaceData, workspaceSize, fused_op, stream)
 *
 */
template <typename value_t>
void launcher(const ML::cumlHandle_impl& handle, Pack<value_t> data, int startVertexId, int batchSize, cudaStream_t stream) {
    data.resetArray(stream, batchSize+1);

    typedef cutlass::Shape<8, 128, 128> OutputTile_t;

    int m = data.N;
    int n = min(data.N - startVertexId, batchSize);
    int k = data.D;

    int* vd = data.vd;

    value_t eps2 = data.eps * data.eps;

    MLCommon::device_buffer<char> workspace(handle.getDeviceAllocator(), stream);
    size_t workspaceSize = 0;

    constexpr auto distance_type = MLCommon::Distance::DistanceType::EucUnexpandedL2;

    workspaceSize =  MLCommon::Distance::getWorkspaceSize<distance_type, value_t, value_t, bool>
            (data.x, data.x+startVertexId*k, m, n, k);

    if (workspaceSize != 0)
        workspace.resize(workspaceSize, stream);

    MLCommon::Distance::epsilon_neighborhood<distance_type, value_t, OutputTile_t>
        (data.x, data.x+startVertexId*k, data.adj, m, n, k, eps2,
         (void*)workspace.data(), workspaceSize,

         [vd, n] __device__ (int global_c_idx, bool in_neigh) {
             // fused construction of vertex degree
             int batch_vertex = global_c_idx - (n * (global_c_idx / n));
             atomicAdd(vd+batch_vertex, in_neigh);
             atomicAdd(vd+n, in_neigh);
         },
         stream
	);

    CUDA_CHECK(cudaPeekAtLastError());
}
}  // end namespace Algo6
}  // end namespace VertexDeg
}; // end namespace Dbscan
