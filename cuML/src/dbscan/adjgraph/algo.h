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

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <cuda_utils.h>
#include "pack.h"
#include "../common.h"
#include <common/cumlHandle.hpp>
#include <common/allocatorAdapter.hpp>

using namespace thrust;

namespace Dbscan {
namespace AdjGraph {
namespace Algo {

using namespace MLCommon;

template <typename Type, int TPB_X>
__global__ void adj_graph_kernel(Pack<Type> data, int batchSize) {
    int row = blockIdx.x*TPB_X + threadIdx.x;
    int N = data.N;
    if(row < batchSize) {
        int k = 0;
        data.core_pts[row] = (data.vd[row] >= data.minPts);
        Type scan_id = data.ex_scan[row];
        for(int i=0; i<N; i++) {
            // @todo: uncoalesced mem accesses!
            if(data.adj[N*row + i]) {
                data.adj_graph[scan_id + k] = i;
                k = k + 1;
            }
        }
    }
    __syncthreads();
}


static const int TPB_X = 256;

template <typename Type>
void launcher(const ML::cumlHandle_impl& handle, Pack<Type> data, int batchSize, cudaStream_t stream) {
    dim3 blocks(ceildiv(batchSize, TPB_X));
    dim3 threads(TPB_X);
    device_ptr<int> dev_vd = device_pointer_cast(data.vd); 
    device_ptr<Type> dev_ex_scan = device_pointer_cast(data.ex_scan);

    ML::thrustAllocatorAdapter alloc( handle.getDeviceAllocator(), stream );
    auto execution_policy = thrust::cuda::par(alloc).on(stream);
    exclusive_scan(execution_policy, dev_vd, dev_vd + batchSize, dev_ex_scan);
    adj_graph_kernel<Type, TPB_X><<<blocks, threads, 0, stream>>>(data, batchSize);
}

}  // End Algo
}  // End AdjGraph
}  // End Dbscan   
