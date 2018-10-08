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

#include "utils.h"
#include "vertexdeg/runner.h"
#include "adjgraph/runner.h"
#include "labelling/runner.h"

namespace Dbscan {
template<typename Type, typename Type_f>
void run(Type_f *x, Type N, Type minPts, Type D, Type_f eps, bool* adj,
		Type* vd, Type* adj_graph, Type* ex_scan, bool* core_pts, bool* visited,
		Type *db_cluster, bool *xa, bool *fa, bool *m, Type *map_id,
		Type_f* dots, cudaStream_t stream, int algoVd, int algoAdj,
		int algoCcl) {
	//Rynning VerexDeg
	VertexDeg::run(adj, vd, x, dots, eps, N, D, stream, algoVd);
	Type *host_vd = new Type[size_t(N + 1)];
	MLCommon::updateHost(host_vd, vd, N + 1);
	Type adjlen = host_vd[N];
	delete[] host_vd;
	// Running AdjGraph
	CUDA_CHECK(cudaMalloc((void** )&adj_graph, sizeof(Type) * adjlen));
	AdjGraph::run(adj, vd, adj_graph, ex_scan, N, minPts, core_pts, stream,
			algoAdj);
	// Running Labelling
	Label::run(adj, vd, adj_graph, ex_scan, N, minPts, core_pts, visited,
			db_cluster, xa, fa, m, map_id, stream, algoCcl);
	if (adj_graph != NULL)
		CUDA_CHECK(cudaFree(adj_graph));
}

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
size_t run(Type_f* x, Type N, Type D, Type_f eps, Type minPts, Type* labels,
		int algoVd, int algoAdj, int algoCcl, void* temp, int nBatches,
		cudaStream_t stream) {
	if (temp == NULL) {
		size_t size;
		int batchSize = ceildiv(N, nBatches);
		size = alignSize(N * batchSize * sizeof(bool), 256)
				+ alignSize(N * sizeof(Type), 256)
				+ alignSize(batchSize * sizeof(Type), 256)
				+ 3 * alignSize(N * sizeof(bool), 256)
				+ alignSize(sizeof(bool), 256)
				+ alignSize(batchSize * sizeof(bool), 256)
				+ alignSize((batchSize + 1) * sizeof(Type), 256)
				+ alignSize(N * sizeof(Type_f), 256);
		return size;
	}
	// partition the temporary workspace needed for different stages of dbscan
	Type adjlen = 0;
	Type curradjlen = 0;
	int batchSize = ceildiv(N, nBatches);
	size_t offset = 0;
	bool* adj = (bool*) (temp);
	offset += alignSize(N * batchSize * sizeof(bool), 256);
	bool* core_pts = (bool*) (temp) + offset;
	offset += alignSize(batchSize * sizeof(bool), 256);
	bool* visited = (bool*) (temp) + offset;
	offset += alignSize(N * sizeof(bool), 256);
	bool* xa = (bool*) (temp) + offset;
	offset += alignSize(N * sizeof(bool), 256);
	bool* fa = (bool*) (temp) + offset;
	offset += alignSize(N * sizeof(bool), 256);
	bool* m = (bool*) (temp) + offset;
	offset += alignSize(sizeof(bool), 256);
	Type* vd = (Type*) (temp) + offset / sizeof(Type);
	offset += alignSize((batchSize + 1) * sizeof(Type), 256);
	Type* ex_scan = (Type*) (temp) + offset / sizeof(Type);
	offset += alignSize(batchSize * sizeof(Type), 256);
	Type* map_id = (Type*) (temp) + offset / sizeof(Type);
	offset += alignSize(N * sizeof(Type), 256);
	Type_f* dots = (Type_f*) (temp) + offset / sizeof(Type_f);
	// Running VertexDeg
	for (int i = 0; i < nBatches; i++) {
		Type *adj_graph = NULL;
		int startVertexId = i * batchSize;
		VertexDeg::run(adj, vd, x, dots, eps, N, D, stream, algoVd,
				startVertexId, batchSize);
		MLCommon::updateHost(&curradjlen, vd + batchSize, 1);
		// Running AdjGraph
		// TODO -: To come up with a mechanism as to reduce and reuse adjgraph mallocs
		if (curradjlen > adjlen || adj_graph == NULL) {
			adjlen = curradjlen;
			CUDA_CHECK(cudaMalloc((void** )&adj_graph, sizeof(Type) * adjlen));
		}
		AdjGraph::run(adj, vd, adj_graph, ex_scan, N, minPts, core_pts, stream,
				algoAdj, batchSize);
		// Running Labelling
		Label::run(adj, vd, adj_graph, ex_scan, N, minPts, core_pts, visited,
				labels, xa, fa, m, map_id, stream, algoCcl, startVertexId,
				batchSize);
		if (adj_graph != NULL)
			CUDA_CHECK(cudaFree(adj_graph));
	}
	if (algoCcl == 2) {
		Type *adj_graph = NULL;
		Label::final_relabel(adj, vd, adj_graph, ex_scan, N, minPts, core_pts,
				visited, labels, xa, fa, m, map_id, stream);
	}

        static const int TPB = 256;
        int nblks = ceildiv(N, TPB);
        relabelForSkl<Type><<<nblks,TPB>>>(labels, N);

	return (size_t) 0;
}
} // namespace Dbscan
