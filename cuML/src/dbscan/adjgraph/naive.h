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
#include <cuML.hpp>

namespace Dbscan {
namespace AdjGraph { 
namespace Naive {

template <typename Type> 
void launcher(const ML::cumlHandle& handle, Pack<Type> data, int batchSize) {
    cudaStream_t stream = handle.getStream();
    int k = 0;
    int N = data.N;
    int *host_vd = new Type[batchSize+1];
    bool *host_core_pts = new bool[batchSize];
    bool *host_adj = new bool[batchSize*N];
    Type *host_ex_scan = new Type[batchSize];
    MLCommon::updateHostAsync(host_adj, data.adj, batchSize*N, stream);
    MLCommon::updateHostAsync(host_vd, data.vd, batchSize+1, stream);
    size_t adjgraph_size = size_t(host_vd[batchSize]);
    Type *host_adj_graph = new Type[adjgraph_size];
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
    MLCommon::updateDeviceAsync(data.adj_graph, host_adj_graph, adjgraph_size, stream);
    MLCommon::updateDeviceAsync(data.core_pts, host_core_pts, batchSize, stream);
    MLCommon::updateDeviceAsync(data.ex_scan, host_ex_scan, batchSize, stream);
    delete [] host_vd;
    delete [] host_core_pts;
    delete [] host_adj;
    delete [] host_ex_scan;
    delete [] host_adj_graph;
}
} // End Naive
} // End AdjGraph
} // End Dbscan
