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

#include <cuda_utils.h>
#include "vertexdeg/runner.h"
#include "adjgraph/runner.h"
#include "labelling/runner.h"
#include <common/cumlHandle.hpp>
#include <common/device_buffer.hpp>

namespace Dbscan {

using namespace MLCommon;

template <typename Type>
__global__ void relabelForSkl(Type* labels, Type N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < N) --labels[tid];
}

template<typename Type, typename Type_f>
/* @param N number of points
 * @param D dimensionality of the points
 * @param eps epsilon neighborhood criterion
 * @param minPts core points criterion
 * @param labels the output labels (should be of size N)
 * @param ....
 * @param temp temporary global memory buffer used to store intermediate computations
 *             If this is a null pointer, then this function will return the workspace size needed.
 *             It is the responsibility of the user to cudaMalloc and cudaFree this buffer!
 * @param stream the cudaStream where to launch the kernels
 * @return in case the temp buffer is null, this returns the size needed.
 */
size_t run(const ML::cumlHandle_impl& handle, Type_f* x, Type N, Type D, Type_f eps, Type minPts, Type* labels,
		int algoVd, int algoAdj, int algoCcl, void* workspace, int nBatches, cudaStream_t stream) {
    const size_t align = 256;
    int batchSize = ceildiv(N, nBatches);
    size_t adjSize = alignTo<size_t>(sizeof(bool) * N * batchSize, align);
    size_t corePtsSize = alignTo<size_t>(sizeof(bool) * N, align);
    size_t visitedSize = alignTo<size_t>(sizeof(bool) * N, align);
    size_t xaSize = alignTo<size_t>(sizeof(bool) * N, align);
    size_t mSize = alignTo<size_t>(sizeof(bool), align);
    size_t vdSize = alignTo<size_t>(sizeof(Type) * (batchSize + 1), align);
    size_t exScanSize = alignTo<size_t>(sizeof(Type) * batchSize, align);
    size_t mapIdSize = alignTo<size_t>(sizeof(Type) * N, align);

    if(workspace == NULL) {
        auto size = adjSize
            + corePtsSize
            + visitedSize
            + 2 * xaSize
            + mSize
            + vdSize
            + exScanSize
            + mapIdSize;
        return size;
    }
    // partition the temporary workspace needed for different stages of dbscan
    Type adjlen = 0;
    Type curradjlen = 0;
    char* temp = (char*)workspace;
    bool* adj = (bool*)temp;       temp += adjSize;
    bool* core_pts = (bool*)temp;  temp += corePtsSize;
    bool* visited = (bool*)temp;   temp += visitedSize;
    bool* xa = (bool*)temp;        temp += xaSize;
    bool* fa = (bool*)temp;        temp += xaSize;
    bool* m = (bool*)temp;         temp += mSize;
    int* vd = (int*)temp;        temp += vdSize;
    Type* ex_scan = (Type*)temp;   temp += exScanSize;
    Type* map_id = (Type*)temp;    temp += mapIdSize;

	// Running VertexDeg
	for (int i = 0; i < nBatches; i++) {
		MLCommon::device_buffer<Type> adj_graph(handle.getDeviceAllocator(), stream);
		int startVertexId = i * batchSize;
        int nPoints = min(N-startVertexId, batchSize);
        if(nPoints <= 0)
            continue;
		VertexDeg::run(handle, adj, vd, x, eps, N, D, algoVd,
				startVertexId, nPoints, stream);
		MLCommon::updateHost(&curradjlen, vd + nPoints, 1, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));

		// Running AdjGraph
		// TODO -: To come up with a mechanism as to reduce and reuse adjgraph mallocs
		if (curradjlen > adjlen || adj_graph.data() == NULL) {
			adjlen = curradjlen;
            adj_graph.resize(adjlen, stream);
		}
		AdjGraph::run(handle, adj, vd, adj_graph.data(), ex_scan, N, minPts, core_pts,
				algoAdj, nPoints, stream);
		// Running Labelling
		Label::run(handle, adj, vd, adj_graph.data(), ex_scan, N, minPts, core_pts, visited,
				labels, xa, fa, m, map_id, algoCcl, startVertexId,
				nPoints, stream);
	}
	if (algoCcl == 2) {
		Type *adj_graph = NULL;
		Label::final_relabel(handle, adj, vd, adj_graph, ex_scan, N, minPts, core_pts,
				visited, labels, xa, fa, m, map_id, stream);
	}

        static const int TPB = 256;
        int nblks = ceildiv(N, TPB);
        relabelForSkl<Type><<<nblks, TPB, 0, stream>>>(labels, N);

        CUDA_CHECK(cudaPeekAtLastError());


	return (size_t) 0;
}
} // namespace Dbscan
