#pragma once

#include "cuda_utils.h"
#include <cublas_v2.h>
#include "linalg/cublas_wrappers.h"

using namespace MLCommon::LinAlg;

namespace MLCommon {
namespace LinAlg {


template<typename T>
void bilinear(const T* mat, int dim, const T* x,
              cublasHandle_t cublas_handle, T* result) {

        // Compute the squared sum
        T* temp;
        allocate(temp, dim);

        T alpha = 1, beta = 0;
        CUBLAS_CHECK(
                cublasgemv(cublas_handle, CUBLAS_OP_N, dim, dim, &alpha, mat, dim, x, dim,
                           &beta, temp, dim));

        CUBLAS_CHECK(cublasdot(cublas_handle, dim, x, dim, temp, dim, result));
        CUDA_CHECK(cudaFree(temp));
}

}
;
// end namespace LinAlg
}
;
// end namespace MLCommon
