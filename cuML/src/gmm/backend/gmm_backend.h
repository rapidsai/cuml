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

#include <gmm/gmm_variables.h>

#include <magma/magma_utils.h>
#include <gmm/backend/normalize.h>

#include <cuda.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/sqrt.h>
#include <ml_utils.h>
#include <cublas_v2.h>

// using namespace MLCommon::HMM;
using namespace MLCommon::LinAlg;
using namespace MLCommon;

namespace gmm {

template <typename math_t>
void inverse(math_t *out, const math_t *in, int len,
             cudaStream_t stream = 0) {
        unaryOp(out, in, len,
                [] __device__(math_t in) {
                        return 1 / in;
                },
                stream);
}

template <typename math_t>
void exp(math_t *out, const math_t *in, int len,
         cudaStream_t stream = 0) {
        unaryOp(out, in, len,
                [] __device__(math_t in) {
                        return std::exp(in);
                },
                stream);
}

template <typename math_t>
void square(math_t *out, const math_t *in, int len,
            cudaStream_t stream = 0) {
        unaryOp(out, in, len,
                [] __device__(math_t in) {
                        return in * in;
                },
                stream);
}

template <typename Type>
__global__ void naiveAddElemKernel(Type *out, const Type *in1, const Type in2,
                                   int len) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < len) {
                out[idx] = in1[idx] + in2;
        }
}

template <typename Type>
void naiveAddElem(Type *out, const Type *in1, const Type in2, int len) {
        static const int TPB = 64;
        int nblks = ceildiv(len, TPB);
        naiveAddElemKernel<Type><<<nblks, TPB>>>(out, in1, in2, len);
        CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
void generate_trans_matrix(magma_int_t m, magma_int_t n, T* dA, magma_int_t lda, bool colwise){
        fill_matrix_gpu(m, n, dA, lda, false);
        normalize_matrix(m, n, dA, lda, colwise);
}

}
