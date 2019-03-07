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

template <typename T>
cublasStatus_t cublasgemmBatched(cublasHandle_t handle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb,
                                 int m, int n, int k,
                                 T          *alpha,
                                 T          **Aarray, int lda,
                                 T          **Barray, int ldb,
                                 T          *beta,
                                 T          **Carray, int ldc,
                                 int batchCount);

template <>
inline cublasStatus_t cublasgemmBatched(cublasHandle_t handle,
                                        cublasOperation_t transa,
                                        cublasOperation_t transb,
                                        int m, int n, int k,
                                        float          *alpha,
                                        float          **Aarray, int lda,
                                        float          **Barray, int ldb,
                                        float          *beta,
                                        float          **Carray, int ldc,
                                        int batchCount)

{
        return cublasSgemmBatched( handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

template <>
inline cublasStatus_t cublasgemmBatched(cublasHandle_t handle,
                                        cublasOperation_t transa,
                                        cublasOperation_t transb,
                                        int m, int n, int k,
                                        double          *alpha,
                                        double          **Aarray, int lda,
                                        double          **Barray, int ldb,
                                        double          *beta,
                                        double          **Carray, int ldc,
                                        int batchCount)

{
        return cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}


}
}
