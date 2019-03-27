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
#include <common/cumlHandle.hpp>

namespace Dbscan {
namespace VertexDeg {
namespace Naive {

using namespace MLCommon;

/** number of threads in a CTA along X dim */
static const int TPB_X = 32;
/** number of threads in a CTA along Y dim */
static const int TPB_Y = 8;

/**
 * @brief Naive distance matrix evaluation and epsilon neighborhood construction
 * @param adj eps-neighborhood (aka adjacent matrix)
 * @param x the input buffer
 * @param N number of rows
 * @param D number of columns
 */
template <typename Type>
__global__ void vertex_degree_kernel(Pack<Type> data, int startVertexId, int batchSize) {
    const Type Zero = (Type)0;
    int row = (blockIdx.y * TPB_Y) + threadIdx.y;
    int col = (blockIdx.x * TPB_X) + threadIdx.x;
    int N = data.N;
    if((row >= batchSize) || (col >= N))
        return;
    Type eps = data.eps;
    Type eps2 = eps * eps;
    Type sum = Zero;
    int D = data.D;
    Type *x = data.x;
    bool *adj = data.adj;
    int *vd = data.vd;
    for(int d=0;d<D;++d) {
        Type a = __ldg(x+(row+startVertexId)*D+d);
        Type b = __ldg(x+col*D+d);
        Type diff = a - b;
        sum += (diff * diff);
    }
    int res = (sum <= eps2);
    adj[row*N+col] = res;
    atomicAdd(vd+row, res);
    atomicAdd(vd+batchSize, res);
}

template <typename Type>
void launcher(Pack<Type> data, int startVertexId, int batchSize, cudaStream_t stream) {
    dim3 grid(ceildiv(data.N, TPB_X), ceildiv(batchSize, TPB_Y), 1);
    dim3 blk(TPB_X, TPB_Y, 1);
    data.resetArray(stream, batchSize+1);
    vertex_degree_kernel<<<grid, blk, 0, stream>>>(data, startVertexId, batchSize);
}

} // namespace Naive
} // namespace AdjGraph
} // namespace Dbscan
