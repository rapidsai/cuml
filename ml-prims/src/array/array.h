/*
 * array.h
 *
 *  Created on: May 8, 2019
 *      Author: cjnolet
 */

#pragma once

#include <thrust/sort.h>
#include <limits>

#include "cuda_utils.h"


namespace MLCommon {
namespace Array {

template <typename Type, int TPB_X, typename Lambda>
__global__ void map_label_kernel(Type *map_ids, Type *in, Type *out,
        Type N, Lambda filter_op) {
    int tid = threadIdx.x + blockIdx.x*TPB_X;
    if(tid < N) {

        if(!filter_op(in[tid])) {
            for(int i=0; i < N; i++) {
                if(in[tid] == map_ids[i]) {
                    out[tid] = i + 1;
                    break;
                }
            }
        }
    }
}


/**
 * Maps an input array containing a series of numbers into a new array
 * where numbers have been mapped to a monotonically increasing set
 * of labels. This can be useful in machine learning algorithms, for instance,
 * where a given set of labels is not taken from a monotonically increasing
 * set. This can happen if they are filtered or if only a subset of the
 * total labels are used in a dataset. This is also useful in graph algorithms
 * where a set of vertices need to be labeled in a monotonically increasing
 * order.
 * @tparam Type the numeric type of the input and output arrays
 * @tparam Lambda the type of an optional filter function, which determines
 * which items in the array to map.
 * @param N number of elements in the input array
 * @param stream cuda stream to use
 * @param filter_op an optional function for specifying which values
 * should have monotonically increasing labels applied to them.
 */
template <typename Type, typename Lambda>
void map_to_monotonic(Type *out, Type *in, Type N, cudaStream_t stream,
        Lambda filter_op) {

    static const int TPB_X = 256;

    dim3 blocks(ceildiv(N, TPB_X));
    dim3 threads(TPB_X);

    Type *map_ids;
    allocate(map_ids, N, stream);

    Type *host_in = (Type*)malloc(N*sizeof(Type));
    Type *host_map_ids = (Type*)malloc(N*sizeof(Type));

    memset(host_map_ids, 0, N*sizeof(Type));

    MLCommon::updateHost(host_in, in, N, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    thrust::sort(host, host_in, host_in + N);

    Type *uid = thrust::unique(host, host_in, host_in + N, equal_to<Type>());
    Type num_clusters = uid - host_in;
    for(int i=0; i<num_clusters; i++)
        host_map_ids[i] = host_in[i];

    MLCommon::updateDevice(map_ids, host_map_ids, N, stream);

    map_label_kernel<Type,TPB_X><<<blocks, threads, 0, stream>>>(map_ids, in, out, N, filter_op);
}

template <typename Type, typename Lambda>
void map_to_monotonic(Type *out, Type *in, Type N, cudaStream_t stream) {

    map_to_monotonic<Type, Lambda>(out, in, N, stream,
            [] __device__ (int val) {return false;});
}




};
};
