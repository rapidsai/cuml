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

#include <thrust/count.h>
#include <cuda_utils.h>
#include "pack.h"
#include "dbscan/common.h"
#include <common/cumlHandle.hpp>
#include <common/host_buffer.hpp>

namespace Dbscan {
namespace Label {
namespace Algo1 {

using namespace thrust;
using namespace MLCommon;

template <typename Type, int TPB_X>
__global__ void bfs_device(Pack<Type> data, int startVertexId, int batchSize) {
    int tid = threadIdx.x + blockIdx.x*TPB_X;
    if(tid < batchSize) {
        if(data.fa[tid + startVertexId]) {
            data.fa[tid + startVertexId] = false;
            data.xa[tid + startVertexId] = true;
        int start = int(data.ex_scan[tid]);
        for(int i=0; i< int(data.vd[tid]); i++) 
	    data.fa[startVertexId + data.adj_graph[start + i]] = 1 - data.xa[startVertexId + data.adj_graph[start + i]];          
        }
    } 
}

static const int TPB_X = 256;

template <typename Type>
void bfs(const ML::cumlHandle_impl& handle, int id, Pack<Type> data, Type *host_adj_graph, Type *host_ex_scan, int *host_vd,
         bool *host_visited, Type *host_db_cluster, Type cluster, size_t N,
         int startVertexId, int batchSize, cudaStream_t stream) {
    MLCommon::host_buffer<bool> host_xa(handle.getHostAllocator(), stream, sizeof(bool)*N);
    MLCommon::host_buffer<bool> host_fa(handle.getHostAllocator(), stream, sizeof(bool)*N);
    memset(host_xa.data(), false, sizeof(bool)*N);
    memset(host_fa.data(), false, sizeof(bool)*N);
    host_fa[id] = true;
    MLCommon::updateDeviceAsync(data.xa, host_xa.data(), N, stream);
    MLCommon::updateDeviceAsync(data.fa, host_fa.data(), N, stream);
    int countFa = 1;
    dim3 blocks(ceildiv(batchSize, TPB_X), 1, 1);
    dim3 threads(TPB_X, 1, 1);
    while(countFa > 0) {
        bfs_device<Type,TPB_X><<<blocks, threads, 0, stream>>>(data, startVertexId, batchSize);
        cudaStreamSynchronize(stream);
        ML::thrustAllocatorAdapter alloc( handle.getDeviceAllocator(), stream );
        auto execution_policy = thrust::cuda::par(alloc).on(stream);
        countFa = count(execution_policy, data.fa, data.fa + N, true);
    }
    MLCommon::updateHostAsync(host_xa.data(), data.xa, N, stream);
    cudaStreamSynchronize(stream);
    for(int i=0; i<N; i++) {
        if(host_xa[i]) {
            host_db_cluster[i] = cluster;
            host_visited[i] = true;
        }
    }
}

template <typename Type>
void identifyCluster(const ML::cumlHandle_impl& handle, Pack<Type> data, int startVertexId, int batchSize, cudaStream_t stream) {
    Type cluster = Type(1) + startVertexId;
    size_t N = (size_t)data.N;
    MLCommon::host_buffer<int> host_vd(handle.getHostAllocator(), stream, sizeof(int)*(batchSize+1));
    MLCommon::host_buffer<bool> host_core_pts(handle.getHostAllocator(), stream, sizeof(int)*batchSize);
    MLCommon::host_buffer<bool> host_visited(handle.getHostAllocator(), stream, sizeof(bool)*N);
    MLCommon::host_buffer<Type> host_ex_scan(handle.getHostAllocator(), stream, sizeof(Type)*batchSize);
    MLCommon::host_buffer<Type> host_db_cluster(handle.getHostAllocator(), stream, sizeof(Type)*N);

    MLCommon::updateHostAsync(host_core_pts.data(), data.core_pts, batchSize, stream);
    MLCommon::updateHostAsync(host_vd.data(), data.vd, batchSize+1, stream);
    size_t adjgraph_size = size_t(host_vd[batchSize]);
    MLCommon::host_buffer<Type> host_adj_graph(handle.getHostAllocator(), stream, sizeof(Type)*adjgraph_size);
    MLCommon::updateHostAsync(host_ex_scan.data(), data.ex_scan, batchSize, stream);
    MLCommon::updateHostAsync(host_adj_graph.data(), data.adj_graph, adjgraph_size, stream);
    MLCommon::updateHostAsync(host_visited.data(), data.visited, N, stream);
    MLCommon::updateHostAsync(host_db_cluster.data(), data.db_cluster, N, stream);

    for(int i=0; i<batchSize; i++) {
        if(!host_visited[i + startVertexId] && host_core_pts[i]) {
            host_visited[i + startVertexId] = true;
            host_db_cluster[i + startVertexId] = cluster;
            bfs(handle, i, data, host_adj_graph.data(), host_ex_scan.data(), host_vd.data(),
                host_visited.data(), host_db_cluster.data(), cluster, N, startVertexId, 
                batchSize, stream);
            cluster++;
        }
    }
    MLCommon::updateDeviceAsync(data.visited, host_visited.data(), N, stream);
    MLCommon::updateDeviceAsync(data.db_cluster, host_db_cluster.data(), N, stream);
}

template <typename Type>
void launcher(const ML::cumlHandle_impl& handle, Pack<Type> data, int startVertexId, int batchSize, cudaStream_t stream) {
    if(startVertexId == 0)
        data.resetArray(stream);
        CUDA_CHECK(cudaMemsetAsync(data.db_cluster, 0, sizeof(Type)*data.N, stream));
    identifyCluster(handle, data, startVertexId, batchSize, stream);
}

} //End Algo1
} //End Label
} //End Dbscan
