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


template <typename value_t>
void launcher(const ML::cumlHandle_impl& handle, Pack<value_t> data, int startVertexId, int batchSize, cudaStream_t stream) {
    data.resetArray(stream, batchSize+1);

    typedef cutlass::Shape<8, 128, 128> OutputTile_t;

    int m = data.N;
    int n = min(data.N - startVertexId, batchSize);
    int k = data.D;

    MLCommon::device_buffer<char> workspace(handle.getDeviceAllocator(), stream);
    size_t workspaceSize = 0;

    value_t eps2 = data.eps * data.eps;

    int* vd = data.vd;
    bool* adj = data.adj;

    /**
     * Epilogue operator to fuse the construction of boolean eps neighborhood adjacency matrix, vertex degree array,
     * and the final distance matrix into a single kernel.
     */
    auto dbscan_op = [n, eps2, vd, adj] __device__
        (value_t val, 							// current value in gemm matrix
		int global_c_idx) {						// index of output in global memory
        int acc = val <= eps2;
        int vd_offset = global_c_idx - (n * (global_c_idx / n));   // bucket offset for the vertex degrees

        atomicAdd(vd+vd_offset, acc);
        atomicAdd(vd+n, acc);
        return bool(acc);
    };

    constexpr auto distance_type = MLCommon::Distance::DistanceType::EucUnexpandedL2;

    workspaceSize =  MLCommon::Distance::getWorkspaceSize<distance_type, value_t, value_t, bool>
    		(data.x, data.x+startVertexId*k, 					// x & y inputs
    		 m, n, k 											// Cutlass block params
    );

    CUDA_CHECK(cudaPeekAtLastError());

    if (workspaceSize != 0) {
        workspace.resize(workspaceSize, stream);
    }

    MLCommon::Distance::distance<distance_type, value_t, value_t, bool, OutputTile_t>
    		(data.x, data.x+startVertexId*k, 					// x & y inputs
             adj,
    		 m, n, k, 											// Cutlass block params
			 (void*)workspace.data(), workspaceSize, 					// workspace params
    		 dbscan_op, 										// epilogue operator
    		 stream												// cuda stream
	 );

    CUDA_CHECK(cudaPeekAtLastError());
}




}  // end namespace Algo6
}  // end namespace VertexDeg
}; // end namespace Dbscan
