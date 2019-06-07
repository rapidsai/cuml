
#pragma once
#include "utils.h"
#include "sparse.h"

//
namespace Attraction_ {


// Maps i to j for CSR to COO matrices
__global__
void map_i_to_j(int n, int NNZ, 
                volatile int * __restrict__ ROW,
                volatile int * __restrict__ COL,
                volatile int * __restrict__ mapping_i_to_j)
{
    int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= NNZ) return;

    int start = 0;
    int end = n + 1;
    int i = (n + 1) >> 1;
    int j;
    while (end - start > 1) {
        j = ROW[i];
        end = (j <= TID) ? end : i;
        start = (j > TID) ? start : i;
        i = (start + end) >> 1;
    }
    mapping_i_to_j[2*TID] = i;
    mapping_i_to_j[2*TID + 1] = COL[TID]; // = j
}



// P * Q
__global__
void PQ_Kernel(const int NNZ, const int N_NODES,
                volatile int * __restrict__ mapping_i_to_j,
                volatile float * __restrict__ P,            // Also is VAL in CSR matrix
                volatile float * __restrict__ PQ,           // force product
                volatile float * __restrict__ embedding)
{
    int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= NNZ) return;

    const int i = mapping_i_to_j[2*TID];
    const int j = mapping_i_to_j[2*TID + 1];
    const float ix = embedding[i];
    const float iy = embedding[N_NODES + 1 + i];
    const float jx = embedding[j];
    const float jy = embedding[N_NODES + 1 + j];
    const float dx = ix - jx;
    const float dy = iy - jy;
    PQ[TID] = P[TID] / (1.0f + dx*dx + dy*dy);   // P * Q
}



// Computes Attractive Forces with cuSPARSE
void attractionForces(
    const int n, const int NNZ, const int N_NODES,
    const int ATTRACT_GRIDSIZE, const int ATTRACT_BLOCKSIZE,

    cudaStream_t stream,
    Sparse_handle_t Sparse_Handle,

    const float * __restrict__ VAL,             // also is P
    const int * __restrict__ mapping_i_to_j,

    float * __restrict__ PQ,                    // force product
    const float * __restrict__ embedding,
    float * __restrict__ attraction,            // attraction forces

    CSR_t CSR_Matrix,
    Dense_t Ones_Matrix,
    Dense_t Embedding_Matrix,
    Dense_t Output_Matrix,
    void * __restrict__ buffer                  // For spMM usage
    )
{
    // Pij * Qij
    Attraction_::PQ_Kernel<<<ATTRACT_GRIDSIZE, ATTRACT_BLOCKSIZE>>>(
        n, NNZ, N_NODES,
        mapping_i_to_j, VAL, PQ, embedding);
    cuda_synchronize();


    // Z = sum (Qij)
    // Forces = sum ( Pij * Qij * Z * Yi)
    Sparse_::spMM(Sparse_Handle,
        CSR_Matrix,
        Ones_Matrix,
        1.0f, 0.0f,
        Output_Matrix,
        buffer);
    cuda_synchronize();


    // Do some vector multiplications
    thrust::device_ptr<float> attraction_begin = thrust::device_pointer_cast(attraction);
    thrust::device_ptr<float> embedding_begin = thrust::device_pointer_cast(embedding);

    thrust::transform(
        thrust::cuda::par.on(stream),
        attraction_begin, attraction_begin + n,
        embedding_begin, attraction_begin, thrust::multiplies<float>());

    thrust::transform(
        thrust::cuda::par.on(stream),
        attraction_begin + n, attraction_begin + 2*n,
        embedding_begin + N_NODES + 1, attraction_begin + n, thrust::multiplies<float>());


    // Remove mean (Yi - Yj) So Forces -= sum ( Pij * Qij * Z * Yj)
    Sparse_::spMM(Sparse_Handle,
        CSR_Matrix,
        Embedding_Matrix,
        -1.0f, 1.0f,
        Output_Matrix,
        buffer);
    cuda_synchronize();
}


// end namespace
}