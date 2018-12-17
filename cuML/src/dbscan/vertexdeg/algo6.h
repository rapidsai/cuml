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
#include "cuda_runtime.h"
#include "distance/distance.h"
#include <math.h>
#include "cuda_utils.h"
#include <stdio.h>


#include "pack.h"

namespace Dbscan {
namespace VertexDeg {
namespace Algo6 {


template <typename T>
struct OutStruct {
    bool* adj;
    int* vd;
    T* dist;
};

template <typename T>
struct InStruct {
    T eps2;
    int batchSize;
    int N;
};

template <typename value_t>
void launcher(Pack<value_t> data, cudaStream_t stream, int startVertexId, int batchSize) {
    data.resetArray(stream, batchSize+1);

    typedef cutlass::Shape<8, 128, 128> OutputTile_t;

    int m = data.N;
    int n = min(data.N - startVertexId, batchSize);
    int k = data.D;

    OutStruct<value_t> out = { data.adj, data.vd, data.dots };
    InStruct<value_t> in = { data.eps*data.eps, batchSize, n };

    char* workspace = nullptr;
    size_t workspaceSize = 0;

    /**
     * Epilogue operator to fuse the construction of boolean eps neighborhood adjacency matrix, vertex degree array,
     * and the final distance matrix into a single kernel.
     */
    auto dbscan_op = [] __device__
                        (value_t val, 							// current value in gemm matrix
			 int global_c_idx,						// index of output in global memory
                         const InStruct<value_t>& in_params,	// input parameters
                         OutStruct<value_t>& out_params) {		// output parameters

        int acc = val <= in_params.eps2;
        // This is bool, 1 byte. Need to divide global_c_index by the number of bytes in value_t
        
	int vd_offset = global_c_idx / in_params.N;   // calculate the bucket offset for the vertex degrees
	//printf("vd_offset=%d, global_c_idx=%d, in_params.N=%d\n", vd_offset, global_c_idx, in_params.N);
        atomicAdd(out_params.vd+vd_offset, acc);
        atomicAdd(out_params.vd+in_params.N, acc);

        out_params.adj[global_c_idx] = (bool)acc;

        return val;
    };

    MLCommon::Distance::distance<value_t, value_t, InStruct<value_t>, OutStruct<value_t>, OutputTile_t>
    		(data.x, data.x+startVertexId*k, 			// x & y inputs
    		 m, n, k, 											// Cutlass block params
    		 in, out, 											// input / output params
    		 MLCommon::Distance::DistanceType::EucExpandedL2, // distance metric type
    		 nullptr, workspaceSize, 					// workspace params
    		 dbscan_op, 										// epilogue operator
    		 stream												// cuda stream
	 );


    if (workspaceSize != 0) {
        MLCommon::allocate(workspace, workspaceSize);
    }

    MLCommon::Distance::distance<value_t, value_t, InStruct<value_t>, OutStruct<value_t>, OutputTile_t>
    		(data.x, data.x+startVertexId*k, 			// x & y inputs
    		 m, n, k, 											// Cutlass block params
    		 in, out, 											// input / output params
    		 MLCommon::Distance::DistanceType::EucExpandedL2, // distance metric type
    		 (void*)workspace, workspaceSize, 					// workspace params
    		 dbscan_op, 										// epilogue operator
    		 stream												// cuda stream
	 );

    CUDA_CHECK(cudaDeviceSynchronize());
}




}  // end namespace Algo6
}  // end namespace VertexDeg
}; // end namespace Dbscan
