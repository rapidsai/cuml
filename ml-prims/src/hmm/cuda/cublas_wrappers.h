#pragma once

#include <cublas_v2.h>
#include "cuda_utils.h"


namespace MLCommon {
namespace HMM {

/**
 * @defgroup dgmm cublas dgmm calls
 * @{
 */
template <typename T>
cublasStatus_t cublasdgmm(cublasHandle_t handle, cublasSideMode_t mode,
                          int m, int n, T *A, int lda, const T *x,
                          int incx, T *C, int ldc);

template <>
inline cublasStatus_t cublasdgmm(cublasHandle_t handle, cublasSideMode_t mode,
                                 int m, int n, float *A, int lda, const float *x,
                                 int incx, float *C, int ldc) {
        return cublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
}

template <>
inline cublasStatus_t cublasdgmm(cublasHandle_t handle, cublasSideMode_t mode,
                                 int m, int n, double *A, int lda, const double *x,
                                 int incx, double *C, int ldc) {
        return cublasDdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
}

}
}
