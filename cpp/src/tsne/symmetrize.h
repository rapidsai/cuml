/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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


namespace ML {

struct COO_Matrix_t {
    float *VAL;
    int *COL;
    int *ROW;
    int NNZ;
};


//**********************************************
// Find how much space needed in each row
// We look through all datapoints and increment the count for each row.
__global__ static void
symmetric_find_size(const float *__restrict__ data,
                    const long *__restrict__ indices,
                    const int n, const int k,
                    int *__restrict__ row_sizes,
                    int *__restrict__ row_sizes2)
{
    const int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in row
    const int row = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every row
    if (row >= n || j >= k) return;

    const int col = indices[row*k + j];
    if (j % 2)  atomicAdd(&row_sizes[col], 1);
    else        atomicAdd(&row_sizes2[col], 1);
}

//**********************************************
// Reduce sum(row_sizes) + k
// We look through all datapoints and increment the count for each row.
__global__ static void
reduce_find_size(const int n, const int k,
                int *__restrict__ row_sizes,
                const int *__restrict__ row_sizes2)
{
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= n) return;
    row_sizes[i] += (row_sizes2[i] + k);
}


//**********************************************
// Perform final data + data.T operation
__global__ static void
symmetric_sum(int *__restrict__ edges,
                const float *__restrict__ data,
                const long *__restrict__ indices,
                float *__restrict__ VAL,
                int *__restrict__ COL,
                int *__restrict__ ROW,
                const int n, const int k)
{
    const int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in row
    const int row = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every row
    if (row >= n || j >= k) return;

    const int col = indices[row*k + j];
    const int original = atomicAdd(&edges[row], 1);
    const int transpose = atomicAdd(&edges[col], 1);

    VAL[transpose] = VAL[original] = data[row*k + j];
    // Notice swapped ROW, COL since transpose
    ROW[original] = row;
    COL[original] = col;

    ROW[transpose] = col;
    COL[transpose] = row;
}



//**********************************************
// (1) Find how much space needed in each row
// (2) Compute final space needed (n*k + sum(row_sizes)) == 2*n*k
// (3) Allocate new space
// (4) Prepare edges for each new row
// (5) Perform final data + data.T operation
// (6) Return summed up VAL, COL, ROW
template <int TPB_X = 32, int TPB_Y = 32> struct COO_Matrix_t
symmetrize_matrix(const float *__restrict__ data,
                    const long *__restrict__ indices,
                    const int n, const int k,
                    cudaStream_t stream)
{   
    // (1) Find how much space needed in each row
    // We look through all datapoints and increment the count for each row.
    const dim3 threadsPerBlock(TPB_X, TPB_Y);
    const dim3 numBlocks(ceil(k, threadsPerBlock.x), ceil(n, threadsPerBlock.y));

    int *row_sizes; CUDA_CHECK(cudaMalloc(&row_sizes, sizeof(int)*(n+1)));
    int *row_sizes2; CUDA_CHECK(cudaMalloc(&row_sizes2, sizeof(int)*(n+1)));
    // Notice n+1 since we can reuse these arrays for transpose_edges, original_edges in step (4)
    CUDA_CHECK(cudaMemsetAsync(row_sizes, 0, sizeof(int)*n, stream));
    CUDA_CHECK(cudaMemsetAsync(row_sizes2, 0, sizeof(int)*n, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    symmetric_find_size<<<numBlocks, threadsPerBlock, 0, stream>>>(data, indices, n, k, row_sizes, row_sizes2);
    CUDA_CHECK(cudaPeekAtLastError());

    reduce_find_size<<<ceil(n, 1024), 1024, 0, stream>>>(n, k, row_sizes, row_sizes2);
    CUDA_CHECK(cudaPeekAtLastError());


    // (2) Compute final space needed (n*k + sum(row_sizes)) == 2*n*k
    // Notice we don't do any merging and leave the result as 2*NNZ
    const int NNZ = 2*n*k;


    // (3) Allocate new space
    float *VAL; CUDA_CHECK(cudaMalloc(&VAL, sizeof(float)*NNZ));
    int *COL; CUDA_CHECK(cudaMalloc(&COL, sizeof(int)*NNZ));
    int *ROW; CUDA_CHECK(cudaMalloc(&ROW, sizeof(int)*NNZ));


    // (4) Prepare edges for each new row
    // This mirrors CSR matrix's row Pointer, were maximum bounds for each row
    // are calculated as the cumulative rolling sum of the previous rows.
    // Notice reusing old row_sizes2 memory
    int *edges = row_sizes2;
    thrust::device_ptr<int> __edges = thrust::device_pointer_cast(edges);
    thrust::device_ptr<int> __row_sizes = thrust::device_pointer_cast(row_sizes);

    // Rolling cumulative sum
    thrust::exclusive_scan(thrust::cuda::par.on(stream), __row_sizes, __row_sizes + n, __edges);
    // Set last to NNZ only if CSR needed
    // CUDA_CHECK(cudaMemcpy(edges + n, &NNZ, sizeof(int), cudaMemcpyHostToDevice));


    // (5) Perform final data + data.T operation in tandem with memcpying
    symmetric_sum<<<numBlocks, threadsPerBlock, 0, stream>>>(edges, data, indices, VAL, COL, ROW, n, k);
    CUDA_CHECK(cudaPeekAtLastError());


    CUDA_CHECK(cudaFree(row_sizes));
    CUDA_CHECK(cudaFree(edges));


    // (6) Return summed up VAL, COL, ROW
    struct COO_Matrix_t COO_Matrix = {.VAL = VAL, .COL = COL, .ROW = ROW, .NNZ = NNZ};
    return COO_Matrix;
}


}