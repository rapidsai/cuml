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
void launcher(Pack<Type> data, int startVertexId, int batchSize, cudaStream_t stream) {
    size_t N = (size_t)data.N;
    Type *host_vd = new Type[N+1];
    bool *host_core_pts = new bool[N];
    bool *host_visited = new bool[N];
    Type *host_ex_scan = new Type[N];
    Type *host_db_cluster = new Type[N];
    bool *host_xa = new bool[N]();
    data.resetArray(stream);
    /** this line not in resetArray function because it interferes with algo2 */
    //CUDA_CHECK(cudaMemsetAsync(data.db_cluster, 0, sizeof(Type)*N, stream));
    MLCommon::updateHost(host_core_pts, data.core_pts, N);
    MLCommon::updateHost(host_vd, data.vd, N+1);
    size_t adjgraph_size = size_t(host_vd[N]);
    Type *host_adj_graph = new Type[adjgraph_size];
    MLCommon::updateHost(host_ex_scan, data.ex_scan, N);
    MLCommon::updateHost(host_adj_graph, data.adj_graph, adjgraph_size);
    MLCommon::updateHost(host_xa, data.xa, N);
    MLCommon::updateHost(host_visited, data.visited, N);
    MLCommon::updateHost(host_db_cluster, data.db_cluster, N);
    Type cluster = Type(1);
    for(int i=0; i<N; i++) { 
        if((!host_visited[i]) && host_core_pts[i]) {
	    host_visited[i] = true;
            host_db_cluster[i] = cluster;
            bfs(i, host_adj_graph, host_ex_scan, host_vd, 
                host_visited, host_db_cluster, cluster, host_xa, N);
            cluster++; 
	}
   } 
    MLCommon::updateDevice(data.visited, host_visited, N);
    MLCommon::updateDevice(data.db_cluster, host_db_cluster, N);
    delete [] host_vd;
    delete [] host_core_pts;
    delete [] host_visited;
    delete [] host_ex_scan;
    delete [] host_db_cluster;
    delete [] host_xa;
    delete [] host_adj_graph;
}
} // End Naive
} // End Label
} // End Dbscan
