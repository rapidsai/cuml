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
#include "../common.h"
#include <queue>
#include <common/cumlHandle.hpp>
#include <common/host_buffer.hpp>

namespace Dbscan {
namespace Label {
namespace Naive {

using namespace std; 
template <typename Type>
void bfs(int id, Type *host_adj_graph,
	 Type *host_ex_scan, Type *host_vd, bool *host_visited,
         Type *host_db_cluster, Type cluster, bool *host_xa, size_t N) {
     queue<int> q;
     q.push(id);
     host_xa[id] = true;
     while(!q.empty()) {
         int f = q.front();
         q.pop();
         Type start = host_ex_scan[f];     
         for(int i = 0; i< host_vd[f]; i++) {
             if(!host_xa[host_adj_graph[start + i]]) {
                 q.push(host_adj_graph[start + i]);
                 host_xa[host_adj_graph[start + i]] = true;
             }
         }
     }
     
     for(int i=0; i<N; i++) {
         if(host_xa[i]) {
             host_db_cluster[i] = cluster; 
             host_visited[i] = true;
         }
     }
     memset(host_xa, false, N);
}

template <typename Type>
void launcher(const ML::cumlHandle_impl& handle, Pack<Type> data, int startVertexId, int batchSize, cudaStream_t stream) {
    size_t N = (size_t)data.N;
    MLCommon::host_buffer<Type> host_vd(handle.getHostAllocator(), stream, sizeof(Type)*(N+1));
    MLCommon::host_buffer<bool> host_core_pts(handle.getHostAllocator(), stream, sizeof(bool)*N);
    MLCommon::host_buffer<bool> host_visited(handle.getHostAllocator(), stream, sizeof(bool)*N);
    MLCommon::host_buffer<Type> host_ex_scan(handle.getHostAllocator(), stream, sizeof(Type)*N);
    MLCommon::host_buffer<Type> host_db_cluster(handle.getHostAllocator(), stream, sizeof(Type)*N);
    MLCommon::host_buffer<bool> host_xa(handle.getHostAllocator(), stream, sizeof(bool)*N);
    data.resetArray(stream);
    /** this line not in resetArray function because it interferes with algo2 */
    //CUDA_CHECK(cudaMemsetAsync(data.db_cluster, 0, sizeof(Type)*N, stream));
    MLCommon::updateHostAsync(host_core_pts.data(), data.core_pts, N, stream);
    MLCommon::updateHostAsync(host_vd.data(), data.vd, N+1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    size_t adjgraph_size = size_t(host_vd[N]);
    MLCommon::host_buffer<Type> host_adj_graph(handle.getHostAllocator(), stream, sizeof(Type)*adjgraph_size);
    MLCommon::updateHostAsync(host_ex_scan.data(), data.ex_scan, N, stream);
    MLCommon::updateHostAsync(host_adj_graph.data(), data.adj_graph, adjgraph_size, stream);
    MLCommon::updateHostAsync(host_xa.data(), data.xa, N, stream);
    MLCommon::updateHostAsync(host_visited.data(), data.visited, N, stream);
    MLCommon::updateHostAsync(host_db_cluster.data(), data.db_cluster, N, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    Type cluster = Type(1);
    for(int i=0; i<N; i++) { 
        if((!host_visited[i]) && host_core_pts[i]) {
	    host_visited[i] = true;
            host_db_cluster[i] = cluster;
            bfs(i, host_adj_graph.data(), host_ex_scan.data(), host_vd.data(), 
                host_visited.data(), host_db_cluster.data(), cluster, host_xa.data(), N);
            cluster++; 
	}
   } 
    MLCommon::updateDeviceAsync(data.visited, host_visited.data(), N, stream);
    MLCommon::updateDeviceAsync(data.db_cluster, host_db_cluster.data(), N, stream);
}
} // End Naive
} // End Label
} // End Dbscan
