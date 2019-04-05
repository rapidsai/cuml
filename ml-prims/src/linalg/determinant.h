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

#include <cublas_v2.h>

#include "cuda_utils.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/cusolver_wrappers.h"
#include "magma/magma_utils.h"


#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#define IDX(i,j,lda) ((i)+(j)*(lda))

namespace MLCommon {


template <typename T>
__global__
void diag_kernel(T* out, int n, T* dU, int lddu){
        int idxThread = threadIdx.x + blockDim.x * blockIdx.x;
        if(idxThread == 0) {
                // T sg = 1;
                T det = 0;
                for (size_t j = 0; j < n; j++) {
                        det += std::log(std::abs(dU[IDX(j, j, lddu)]));
                        // sg *= sign(dU[IDX(j, j, lddu)]);
                }
                *out = std::exp(det);
        }
}

template <typename T>
struct determinantHandleCublas_t {
        int* dipiv, *info, info_h;
        T* dA_cpy;
        int ldda;
        T* Ws;
};

template <typename T>
void createDeterminantHandleCublas_t_new(determinantHandleCublas_t<T>& handle, void* workspace){
        handle.Ws = (T *)((size_t)handle.Ws + (size_t)workspace);
        handle.dipiv = (int *)((size_t)handle.dipiv + (size_t)workspace);
        handle.info = (int *)((size_t)handle.info + (size_t)workspace);
        handle.dA_cpy = (T *)((size_t)handle.dA_cpy + (size_t)workspace);
}

template <typename T>
void determinantCublas_bufferSize(determinantHandleCublas_t<T> detHandle,
                                  cusolverDnHandle_t cusolverHandle,
                                  int n, int ldda,
                                  size_t& workspaceSize){
        workspaceSize = 0;
        const size_t granularity = 256;
        int cusolver_workspaceSize;

        detHandle.Ws = (T *)workspaceSize;
        CUSOLVER_CHECK(LinAlg::cusolverDngetrf_bufferSize(cusolverHandle, n, n, detHandle.dA_cpy, ldda, &cusolver_workspaceSize));
        workspaceSize = alignTo((size_t) cusolver_workspaceSize, granularity);
        detHandle.dipiv = (int *)workspaceSize;
        workspaceSize += alignTo(n * sizeof(int), granularity);
        detHandle.info = (int *)workspaceSize;
        workspaceSize += alignTo(sizeof(int), granularity);
        detHandle.dA_cpy = (T *)workspaceSize;
        workspaceSize += alignTo(ldda * n * sizeof(T), granularity);

}

template <typename T>
void det(T* out, int n, T* dA, int ldda,
         cusolverDnHandle_t cusolverHandle,
         determinantHandleCublas_t<T> detHandle,
         cudaStream_t stream=0){
        copy(detHandle.dA_cpy, dA, 1);
        // copy(detHandle.dA_cpy, dA, ldda * n);

        // CUSOLVER_CHECK(LinAlg::cusolverDngetrf(cusolverHandle,
        //                                        n,
        //                                        n,
        //                                        detHandle.dA_cpy,
        //                                        ldda,
        //                                        detHandle.Ws,
        //                                        detHandle.dipiv,
        //                                        detHandle.info));
        //
        // updateHost(&detHandle.info_h, detHandle.info, 1);
        // ASSERT(detHandle.info_h == 0,
        //        "sigma: error in determinant, info=%d | expected=0",
        //        detHandle.info_h);
        //
        // diag_kernel<<< 1, 1, 0, stream >>>(out, n, detHandle.dA_cpy, ldda);
}



}
