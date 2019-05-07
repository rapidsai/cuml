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

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <cuda_utils.h>
#include "pack.h"
#include "../common.h"
#include <common/cumlHandle.hpp>
#include <common/allocatorAdapter.hpp>

#include "sparse/csr.h"

using namespace thrust;

namespace Dbscan {
namespace AdjGraph {
namespace Algo {

using namespace MLCommon;

static const int TPB_X = 256;

/**
 * Takes vertex degree array (vd) and CSR row_ind array (ex_scan) to produce the
 * CSR row_ind_ptr array (adj_graph) and values array(core_pts). This could be
 * made into a reusable prim by providing a lambda for a fused op, given the
 */
template <typename Type>
void launcher(const ML::cumlHandle_impl& handle, Pack<Type> data, Type batchSize, cudaStream_t stream) {

    device_ptr<int> dev_vd = device_pointer_cast(data.vd); 
    device_ptr<Type> dev_ex_scan = device_pointer_cast(data.ex_scan);

    ML::thrustAllocatorAdapter alloc( handle.getDeviceAllocator(), stream );
    exclusive_scan(thrust::cuda::par(alloc).on(stream),
            dev_vd, dev_vd + batchSize, dev_ex_scan);

    bool *core_pts = data.core_pts;
    int minPts = data.minPts;
    int *vd = data.vd;

    MLCommon::Sparse::csr_adj_graph_batched<Type, TPB_X>(data.ex_scan, data.N, batchSize,
            data.adj, data.adj_graph,
            [core_pts, minPts, vd] __device__ (Type row, Type start_idx) {
        core_pts[row] = (vd[row] >= minPts);
    }, stream);

    CUDA_CHECK(cudaPeekAtLastError());
}

}  // End Algo
}  // End AdjGraph
}  // End Dbscan   
