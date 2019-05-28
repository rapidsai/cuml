
using namespace ML;
#include "utils.h"
#include "cuda_utils.h"

#pragma once

//
namespace Attraction_ {


__global__
void PQ_kernel(int N, int nnz, int nnodes,
                volatile int   * indices,
                volatile float * __restrict__ P,
                volatile float * __restrict__ Force,
                volatile float * __restrict__ embedding)
{
    int TID, i, j;
    float ix, iy, jx, jy, dx, dy;
    TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= nnz) return;

    i = indices[2*TID];
    j = indices[2*TID + 1];
    ix = embedding[i]; iy = embedding[nnodes + 1 + i];
    jx = embedding[j]; jy = embedding[nnodes + 1 + j];
    dx = ix - jx;
    dy = iy - jy;
    Force[TID] = P[TID] * 1 / (1 + dx*dx + dy*dy);
}



// computes unnormalized attractive forces
void computeAttrForce(  const int N,
                        const int nnz,
                        const int nnodes,
                        const int threads,
                        const int blocks,
                        const cusparseHandle_t &handle,
                        const cusparseMatDescr_t &descr,
                        const float * __restrict__ VAL,
                        const int * __restrict__ COL,
                        const int * __restrict__ ROW,
                        float * __restrict__ PQ,
                        const float * __restrict__ embedding,
                        


                        thrust::device_vector<float> &sparsePij,
                        thrust::device_vector<int>   &pijRowPtr, // (N + 1)-D vector, should be constant L
                        thrust::device_vector<int>   &pijColInd, // NxL matrix (same shape as sparsePij)
                        thrust::device_vector<float> &forceProd, // NxL matrix
                        thrust::device_vector<float> &pts,       // (nnodes + 1) x 2 matrix
                        thrust::device_vector<float> &forces,    // N x 2 matrix
                        thrust::device_vector<float> &ones,
                        thrust::device_vector<int> &indices)      // N x 2 matrix of ones
{
    // // Computes pij*qij for each i,j
    // PQ_kernel<<<threads, blocks>>>(
    //     N, nnz, nnodes,
    //     indices, thrust::raw_pointer_cast(indices.data()),
    //                                     thrust::raw_pointer_cast(sparsePij.data()),
    //                                     thrust::raw_pointer_cast(forceProd.data()),
    //                                     thrust::raw_pointer_cast(pts.data()));
    // // ComputePijxQijKernel<<<blocks*FACTOR7,THREADS7>>>(N, nnz, nnodes,
    // //                                     thrust::raw_pointer_cast(indices.data()),
    // //                                     thrust::raw_pointer_cast(sparsePij.data()),
    // //                                     thrust::raw_pointer_cast(forceProd.data()),
    // //                                     thrust::raw_pointer_cast(pts.data()));
    // GpuErrorCheck(cudaDeviceSynchronize());

    // // compute forces_i = sum_j pij*qij*normalization*yi
    // float alpha = 1.0f;
    // float beta = 0.0f;
    // CusparseSafeCall(cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                         N, 2, N, nnz, &alpha, descr,
    //                         thrust::raw_pointer_cast(forceProd.data()),
    //                         thrust::raw_pointer_cast(pijRowPtr.data()),
    //                         thrust::raw_pointer_cast(pijColInd.data()),
    //                         thrust::raw_pointer_cast(ones.data()),
    //                         N, &beta, thrust::raw_pointer_cast(forces.data()),
    //                         N));
    // GpuErrorCheck(cudaDeviceSynchronize());
    // thrust::transform(forces.begin(), forces.begin() + N, pts.begin(), forces.begin(), thrust::multiplies<float>());
    // thrust::transform(forces.begin() + N, forces.end(), pts.begin() + nnodes + 1, forces.begin() + N, thrust::multiplies<float>());

    // // compute forces_i = forces_i - sum_j pij*qij*normalization*yj
    // alpha = -1.0f;
    // beta = 1.0f;
    // CusparseSafeCall(cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                         N, 2, N, nnz, &alpha, descr,
    //                         thrust::raw_pointer_cast(forceProd.data()),
    //                         thrust::raw_pointer_cast(pijRowPtr.data()),
    //                         thrust::raw_pointer_cast(pijColInd.data()),
    //                         thrust::raw_pointer_cast(pts.data()),
    //                         nnodes + 1, &beta, thrust::raw_pointer_cast(forces.data()),
    //                         N));
    // GpuErrorCheck(cudaDeviceSynchronize());
    

}

// end namespace
}