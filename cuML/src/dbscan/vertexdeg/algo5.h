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
#include "../common.h"
#include <cub/cub.cuh>

namespace Dbscan {
namespace VertexDeg {
namespace Algo5 {

using namespace MLCommon;

template <typename accum_t, typename output_t, typename scalar_t>
class vertexdeg_epilogue {
private:
    scalar_t* dots;
    scalar_t eps2;
    int* vdeg;
    int startDots;
    int dim_m;
    scalar_t two;

public:
    inline __device__ __host__ vertexdeg_epilogue(scalar_t* d, scalar_t eps,
                                                  int* vd, int startId, int m):
        dots(d), eps2(eps * eps), vdeg(vd), startDots(startId), dim_m(m), two((scalar_t)2.0) {
    }

    /// Epilogue operator
    inline __device__ __host__ output_t operator()(accum_t accumulator,
                                                   output_t c,
                                                   size_t x_idx,
						   size_t y_idx) const {
        scalar_t acc = -two * accumulator;
        acc += dots[startDots + x_idx];
        acc += dots[y_idx];
        output_t res = output_t(acc <= eps2);
        atomicAdd(vdeg+x_idx, res);
        atomicAdd(vdeg+dim_m, res);
        return res;
    }

    /// Epilogue operator
    inline __device__ __host__ output_t operator()(accum_t accumulator,
                                                   size_t x_idx,
	                                           size_t y_idx) const {
        scalar_t acc = -two * accumulator;
        acc += dots[startDots + x_idx];
        acc += dots[y_idx];
        output_t res = output_t(acc <= eps2);
        atomicAdd(vdeg+x_idx, res);
        atomicAdd(vdeg+dim_m, res);
	return res;
    }

    /**
     * Configure epilogue as to whether the thread block cheme  is a secondary
     * accumulator in an inter-block k-splitting scheme
     * @note: this is a no-op here!
     */
    inline __device__ void set_secondary_accumulator() {
    }

    /// Return whether the beta-scaled addend needs initialization
    /// @note: we don't need initialization
    inline __device__ bool must_init_addend() {
        return false;
    }
};

template <typename Type, int TPB>
__global__ void computeDotsKernel(Pack<Type> data, int limit) {
    typedef cub::BlockReduce<Type, TPB> BlockReduce;
     __shared__ typename BlockReduce::TempStorage temp_storage;
     Type thread_data = Type(0);
     Type acc = Type(0);
     for(int i=0; i<limit; i++) {
         if(threadIdx.x + i*TPB < data.D) {
             int tid = threadIdx.x + i*TPB + blockIdx.x*data.D;
             thread_data += data.x[tid]*data.x[tid];
         }
     }
     acc = BlockReduce(temp_storage).Sum(thread_data);
     if(threadIdx.x == 0) {
        data.dots[blockIdx.x] = acc;
     }
}

template <typename Type, int TPB>
void computeDotsImpl(Pack<Type>& data) {
    int limit = ceildiv(data.D, TPB);
    int nblks = data.N;
    computeDotsKernel<Type,TPB><<<nblks,TPB>>>(data, limit);
}

template <typename Type>
void computeDots(Pack<Type>& data) {
    if(data.D <= 32) {
        computeDotsImpl<Type, 32>(data);
    } else if(data.D <= 64) {
        computeDotsImpl<Type, 64>(data);
    } else if(data.D <= 128) {
        computeDotsImpl<Type, 128>(data);
    } else {
        computeDotsImpl<Type, 256>(data);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename value_t, cutlass::gemm::tiling_strategy::kind_t TilingStrategy>
void launcher(Pack<value_t> data, cudaStream_t stream, int startVertexId, int batchSize) {
    using epilogue_op_t = vertexdeg_epilogue<value_t, bool, value_t>;
    data.resetArray(stream, batchSize+1);
    computeDots<value_t>(data);
    epilogue_op_t epilogue(data.dots, data.eps, data.vd, startVertexId, batchSize);
    epsneigh_dispatch<TilingStrategy, value_t, bool,
                      cutlass::math_operation_scalar, epilogue_op_t,
                      cutlass::gemm::dp_accummulate<value_t, value_t> > edis;
    int m = data.N;
    int n = min(data.N - startVertexId, batchSize);
    int k = data.D;
    auto res = edis(m, n, k, data.x, data.x + startVertexId*data.D, data.adj,
                    epilogue, stream, false, false);
    CUDA_CHECK(res.result);
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace Algo5
} // namespace VertexDeg
} // namespace Dbscan
