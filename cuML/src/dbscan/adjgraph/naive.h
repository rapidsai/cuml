/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <cuda_utils.h>
#include "pack.h"
#include "dbscan/common.h"
#include <common/cumlHandle.hpp>
#include <common/host_buffer.hpp>

namespace Dbscan {
namespace AdjGraph { 
namespace Naive {

template <typename Type> 
void launcher(const ML::cumlHandle_impl& handle, Pack<Type> data, int batchSize, cudaStream_t stream) {
    int k = 0;
    int N = data.N;
    MLCommon::host_buffer<int> host_vd(handle.getHostAllocator(), stream, sizeof(Type)*(batchSize+1));
    MLCommon::host_buffer<bool> host_core_pts(handle.getHostAllocator(), stream, sizeof(bool)*(batchSize));
    MLCommon::host_buffer<bool> host_adj(handle.getHostAllocator(), stream, sizeof(bool)*batchSize*N);
    MLCommon::host_buffer<Type> host_ex_scan(handle.getHostAllocator(), stream, sizeof(Type)*batchSize);
    MLCommon::updateHostAsync(host_adj.data(), data.adj, batchSize*N, stream);
    MLCommon::updateHostAsync(host_vd.data(), data.vd, batchSize+1, stream);
    size_t adjgraph_size = size_t(host_vd[batchSize]);
    MLCommon::host_buffer<Type> host_adj_graph(handle.getHostAllocator(), stream, sizeof(Type)*adjgraph_size);
    for(int i=0; i<batchSize; i++) {
        for(int j=0; j<N; j++) {
            if(host_adj[i*N + j]) {
                host_adj_graph[k] = j;
                k = k + 1;
            }
        }
    }
    for(int i=0; i<batchSize; i++)
        host_core_pts[i] = (host_vd[i] >= data.minPts);
    host_ex_scan[0] = Type(0);
    for(int i=1; i<batchSize; i++) 
        host_ex_scan[i] = host_ex_scan[i-1] + host_vd[i-1];
    MLCommon::updateDeviceAsync(data.adj_graph, host_adj_graph.data(), adjgraph_size, stream);
    MLCommon::updateDeviceAsync(data.core_pts, host_core_pts.data(), batchSize, stream);
    MLCommon::updateDeviceAsync(data.ex_scan, host_ex_scan.data(), batchSize, stream);
}
} // End Naive
} // End AdjGraph
} // End Dbscan
