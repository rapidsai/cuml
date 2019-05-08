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

template <typename Type, int TPB_X>
__global__ void label_device(Pack<Type> data, int startVertexId, int batchSize) {
    int tid = threadIdx.x + blockIdx.x*TPB_X;
    if(tid<batchSize) {
        if(data.fa[tid + startVertexId]) {
            data.fa[tid + startVertexId] = false;
            int start = int(data.ex_scan[tid]);
            Type ci, cj;
            bool ci_mod = false;
            ci = data.db_cluster[tid + startVertexId];
            for(int j=0; j< int(data.vd[tid]); j++) {
                cj = data.db_cluster[data.adj_graph[start + j]];
                if(ci<cj) {
                    atomicMin(data.db_cluster + data.adj_graph[start +j], ci);
                    data.xa[data.adj_graph[start+j]] = true;
                    data.m[0] = true;
                }
                else if(ci>cj) {
                    ci = cj;
                    ci_mod = true;
                }
            }
            if(ci_mod) {
                atomicMin(data.db_cluster + startVertexId + tid, ci);
                data.xa[startVertexId + tid] = true;
                data.m[0] = true;
            }
        }
    }
}

template <typename Type, int TPB_X>
__global__ void init_label(Pack<Type> data, int startVertexId, int batchSize, Type MAX_LABEL) {
    /** F1 and F2 in the paper correspond to fa and xa */ 
    /** Cd in paper corresponds to db_cluster */
    int tid = threadIdx.x + blockIdx.x*TPB_X; 
    if(tid<batchSize) { 
        if(data.core_pts[tid] && data.db_cluster[tid + startVertexId]==MAX_LABEL) {
            data.db_cluster[startVertexId + tid] = Type(startVertexId + tid + 1);
        }
    }
}

template <typename Type, int TPB_X>
__global__ void init_all(Pack<Type> data, Type MAX_LABEL) {
    int tid = threadIdx.x + blockIdx.x*TPB_X;
    if(tid<data.N) { 
        data.db_cluster[tid] = MAX_LABEL;
        data.fa[tid] = true;
        data.xa[tid] = false;
    }
}
    
template <typename Type, int TPB_X>
__global__ void map_label(Pack<Type> data, Type MAX_LABEL) {
    int tid = threadIdx.x + blockIdx.x*TPB_X;
    if(tid<data.N) {
        if(data.db_cluster[tid] == MAX_LABEL)
            data.db_cluster[tid] = 0;
        for(int i=0; i< data.N; i++) {
            if((data.db_cluster[tid] == data.map_id[i]) 
                && (data.db_cluster[tid]!=0)) {
                data.db_cluster[tid] = i + 1;
                break;
            }
        }
    }
}

static const int TPB_X = 256;

template <typename Type>
void label(const ML::cumlHandle_impl& handle, Pack<Type> data, int startVertexId, int batchSize, cudaStream_t stream) {
    size_t N = data.N;
    bool host_m;
    MLCommon::host_buffer<bool> host_fa(handle.getHostAllocator(), stream, N);
    MLCommon::host_buffer<bool> host_xa(handle.getHostAllocator(), stream, N);

    dim3 blocks(ceildiv(batchSize, TPB_X));
    dim3 threads(TPB_X);
    Type MAX_LABEL = std::numeric_limits<Type>::max();
    
    init_label<Type, TPB_X><<<blocks, threads, 0, stream>>>(data, startVertexId, batchSize, MAX_LABEL); 
    do {
        CUDA_CHECK( cudaMemsetAsync(data.m, false, sizeof(bool), stream) ); 
        label_device<Type, TPB_X><<<blocks, threads, 0, stream>>>(data, startVertexId, batchSize);
        //** swapping F1 and F2
        MLCommon::updateHost(host_fa.data(), data.fa, N, stream);
        MLCommon::updateHost(host_xa.data(), data.xa, N, stream);
        MLCommon::updateDevice(data.fa, host_xa.data(), N, stream);
        MLCommon::updateDevice(data.xa, host_fa.data(), N, stream);
        //** Updating m *
        MLCommon::updateHost(&host_m, data.m, 1, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } while(host_m);
}

template <typename Type>
void launcher(const ML::cumlHandle_impl& handle, Pack<Type> data, Type N, int startVertexId, int batchSize, cudaStream_t stream) {
    //data.resetArray(stream);
    dim3 blocks(ceildiv(data.N, TPB_X));
    dim3 threads(TPB_X);
    Type MAX_LABEL = std::numeric_limits<Type>::max();
    if(startVertexId == 0)
        init_all<Type, TPB_X><<<blocks, threads, 0, stream>>>(data, MAX_LABEL); 
    label(handle, data, startVertexId, batchSize, stream);
}

template <typename Type>
void relabel(const ML::cumlHandle_impl& handle, Pack<Type> data, cudaStream_t stream) {
    dim3 blocks(ceildiv(data.N, TPB_X));
    dim3 threads(TPB_X);
    Type MAX_LABEL = std::numeric_limits<Type>::max();
    size_t N = data.N;
    MLCommon::host_buffer<Type> host_db_cluster(handle.getHostAllocator(), stream, N);
    MLCommon::host_buffer<Type> host_map_id(handle.getHostAllocator(), stream, N);
    memset(host_map_id.data(), 0, N*sizeof(Type));
    MLCommon::updateHost(host_db_cluster.data(), data.db_cluster, N, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    sort(host, host_db_cluster.data(), host_db_cluster.data() + N);
    Type *uid = unique(host, host_db_cluster.data(), host_db_cluster.data() + N, equal_to<Type>());
    Type num_clusters = uid - host_db_cluster.data();
    for(int i=0; i<num_clusters; i++)
        host_map_id[i] = host_db_cluster[i];
    MLCommon::updateDevice(data.map_id, host_map_id.data(), N, stream);
    map_label<Type,TPB_X><<<blocks, threads, 0, stream>>>(data, MAX_LABEL);
}

} // End Algo2
} // End Label
} // End Dbscan
