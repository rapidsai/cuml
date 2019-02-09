#pragma once

#include "cuda_utils.h"
#include <cublas_v2.h>
#include "linalg/cublas_wrappers.h"
#include "hmm/utils.h"


using namespace MLCommon;
using namespace MLCommon::LinAlg;

namespace MLCommon {
namespace HMM {


template<typename T>
void bilinear(T* mat, int dim, T* x, cublasHandle_t cublas_handle,
              T* result) {

        T* temp;
        allocate(temp, dim);

        T alpha = 1, beta = 0;
        CUBLAS_CHECK(
                cublasgemv(cublas_handle, CUBLAS_OP_N, dim, dim, &alpha, mat, dim, x, 1, &beta, temp, 1));

        CUBLAS_CHECK(cublasdot(cublas_handle, dim, x, 1, temp, 1, result));
        CUDA_CHECK(cudaFree(temp));
}

}
;
// end namespace LinAlg
}
;
// end namespace MLCommon
