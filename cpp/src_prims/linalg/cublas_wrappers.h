/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
#include <cuml/common/logger.hpp>
#include <cuml/common/utils.hpp>

namespace MLCommon {
namespace LinAlg {

#define _CUBLAS_ERR_TO_STR(err) \
  case err:                     \
    return #err
inline const char *cublasErr2Str(cublasStatus_t err) {
  switch (err) {
    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_SUCCESS);
    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_NOT_INITIALIZED);
    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_ALLOC_FAILED);
    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_INVALID_VALUE);
    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_ARCH_MISMATCH);
    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_MAPPING_ERROR);
    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_EXECUTION_FAILED);
    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_INTERNAL_ERROR);
    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_NOT_SUPPORTED);
    _CUBLAS_ERR_TO_STR(CUBLAS_STATUS_LICENSE_ERROR);
    default:
      return "CUBLAS_STATUS_UNKNOWN";
  };
}
#undef _CUBLAS_ERR_TO_STR

/** check for cublas runtime API errors and assert accordingly */
#define CUBLAS_CHECK(call)                                         \
  do {                                                             \
    cublasStatus_t err = call;                                     \
    ASSERT(err == CUBLAS_STATUS_SUCCESS,                           \
           "CUBLAS call='%s' got errorcode=%d err=%s", #call, err, \
           MLCommon::LinAlg::cublasErr2Str(err));                  \
  } while (0)

/** check for cublas runtime API errors but do not assert */
#define CUBLAS_CHECK_NO_THROW(call)                                          \
  do {                                                                       \
    cublasStatus_t err = call;                                               \
    if (err != CUBLAS_STATUS_SUCCESS) {                                      \
      CUML_LOG_ERROR("CUBLAS call='%s' got errorcode=%d err=%s", #call, err, \
                     MLCommon::LinAlg::cublasErr2Str(err));                  \
    }                                                                        \
  } while (0)

/**
 * @defgroup Axpy cublas ax+y operations
 * @{
 */
template <typename T>
cublasStatus_t cublasaxpy(cublasHandle_t handle, int n, const T *alpha,
                          const T *x, int incx, T *y, int incy,
                          cudaStream_t stream);

template <>
inline cublasStatus_t cublasaxpy(cublasHandle_t handle, int n,
                                 const float *alpha, const float *x, int incx,
                                 float *y, int incy, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
inline cublasStatus_t cublasaxpy(cublasHandle_t handle, int n,
                                 const double *alpha, const double *x, int incx,
                                 double *y, int incy, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}
/** @} */

/**
 * @defgroup gemv cublas gemv calls
 * @{
 */
template <typename T>
cublasStatus_t cublasgemv(cublasHandle_t handle, cublasOperation_t transA,
                          int m, int n, const T *alfa, const T *A, int lda,
                          const T *x, int incx, const T *beta, T *y, int incy,
                          cudaStream_t stream);

template <>
inline cublasStatus_t cublasgemv(cublasHandle_t handle,
                                 cublasOperation_t transA, int m, int n,
                                 const float *alfa, const float *A, int lda,
                                 const float *x, int incx, const float *beta,
                                 float *y, int incy, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasSgemv(handle, transA, m, n, alfa, A, lda, x, incx, beta, y,
                     incy);
}

template <>
inline cublasStatus_t cublasgemv(cublasHandle_t handle,
                                 cublasOperation_t transA, int m, int n,
                                 const double *alfa, const double *A, int lda,
                                 const double *x, int incx, const double *beta,
                                 double *y, int incy, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasDgemv(handle, transA, m, n, alfa, A, lda, x, incx, beta, y,
                     incy);
}
/** @} */

/**
 * @defgroup ger cublas a(x*y.T) + A calls
 * @{
 */
template <typename T>
cublasStatus_t cublasger(cublasHandle_t handle, int m, int n, const T *alpha,
                         const T *x, int incx, const T *y, int incy, T *A,
                         int lda, cudaStream_t stream);
template <>
inline cublasStatus_t cublasger(cublasHandle_t handle, int m, int n,
                                const float *alpha, const float *x, int incx,
                                const float *y, int incy, float *A, int lda,
                                cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasSger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
inline cublasStatus_t cublasger(cublasHandle_t handle, int m, int n,
                                const double *alpha, const double *x, int incx,
                                const double *y, int incy, double *A, int lda,
                                cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasDger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}
/** @} */

/**
 * @defgroup gemm cublas gemm calls
 * @{
 */
template <typename T>
cublasStatus_t cublasgemm(cublasHandle_t handle, cublasOperation_t transA,
                          cublasOperation_t transB, int m, int n, int k,
                          const T *alfa, const T *A, int lda, const T *B,
                          int ldb, const T *beta, T *C, int ldc,
                          cudaStream_t stream);

template <>
inline cublasStatus_t cublasgemm(cublasHandle_t handle,
                                 cublasOperation_t transA,
                                 cublasOperation_t transB, int m, int n, int k,
                                 const float *alfa, const float *A, int lda,
                                 const float *B, int ldb, const float *beta,
                                 float *C, int ldc, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasSgemm(handle, transA, transB, m, n, k, alfa, A, lda, B, ldb,
                     beta, C, ldc);
}

template <>
inline cublasStatus_t cublasgemm(cublasHandle_t handle,
                                 cublasOperation_t transA,
                                 cublasOperation_t transB, int m, int n, int k,
                                 const double *alfa, const double *A, int lda,
                                 const double *B, int ldb, const double *beta,
                                 double *C, int ldc, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasDgemm(handle, transA, transB, m, n, k, alfa, A, lda, B, ldb,
                     beta, C, ldc);
}
/** @} */

/**
 * @defgroup gemmbatched cublas gemmbatched calls
 * @{
 */
template <typename T>
cublasStatus_t cublasgemmBatched(cublasHandle_t handle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb, int m, int n, int k,
                                 const T *alpha, const T *const Aarray[],
                                 int lda, const T *const Barray[], int ldb,
                                 const T *beta, T *Carray[], int ldc,
                                 int batchCount, cudaStream_t stream);

template <>
inline cublasStatus_t cublasgemmBatched(
  cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
  int m, int n, int k, const float *alpha, const float *const Aarray[], int lda,
  const float *const Barray[], int ldb, const float *beta, float *Carray[],
  int ldc, int batchCount, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                            Barray, ldb, beta, Carray, ldc, batchCount);
}

template <>
inline cublasStatus_t cublasgemmBatched(
  cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
  int m, int n, int k, const double *alpha, const double *const Aarray[],
  int lda, const double *const Barray[], int ldb, const double *beta,
  double *Carray[], int ldc, int batchCount, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                            Barray, ldb, beta, Carray, ldc, batchCount);
}
/** @} */

/**
 * @defgroup gemmbatched cublas gemmbatched calls
 * @{
 */
template <typename T>
cublasStatus_t cublasgemmStridedBatched(
  cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
  int m, int n, int k, const T *alpha, const T *const Aarray, int lda,
  long long int strideA, const T *const Barray, int ldb, long long int strideB,
  const T *beta, T *Carray, int ldc, long long int strideC, int batchCount,
  cudaStream_t stream);

template <>
inline cublasStatus_t cublasgemmStridedBatched(
  cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
  int m, int n, int k, const float *alpha, const float *const Aarray, int lda,
  long long int strideA, const float *const Barray, int ldb,
  long long int strideB, const float *beta, float *Carray, int ldc,
  long long int strideC, int batchCount, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha,
                                   Aarray, lda, strideA, Barray, ldb, strideB,
                                   beta, Carray, ldc, strideC, batchCount);
}

template <>
inline cublasStatus_t cublasgemmStridedBatched(
  cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
  int m, int n, int k, const double *alpha, const double *const Aarray, int lda,
  long long int strideA, const double *const Barray, int ldb,
  long long int strideB, const double *beta, double *Carray, int ldc,
  long long int strideC, int batchCount, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha,
                                   Aarray, lda, strideA, Barray, ldb, strideB,
                                   beta, Carray, ldc, strideC, batchCount);
}
/** @} */

/**
 * @defgroup solverbatched cublas getrf/gettribatched calls
 * @{
 */

template <typename T>
cublasStatus_t cublasgetrfBatched(cublasHandle_t handle, int n,
                                  T *const A[],    /*Device pointer*/
                                  int lda, int *P, /*Device Pointer*/
                                  int *info,       /*Device Pointer*/
                                  int batchSize, cudaStream_t stream);

template <>
inline cublasStatus_t cublasgetrfBatched(cublasHandle_t handle, int n,
                                         float *const A[], /*Device pointer*/
                                         int lda, int *P,  /*Device Pointer*/
                                         int *info,        /*Device Pointer*/
                                         int batchSize, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasSgetrfBatched(handle, n, A, lda, P, info, batchSize);
}

template <>
inline cublasStatus_t cublasgetrfBatched(cublasHandle_t handle, int n,
                                         double *const A[], /*Device pointer*/
                                         int lda, int *P,   /*Device Pointer*/
                                         int *info,         /*Device Pointer*/
                                         int batchSize, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasDgetrfBatched(handle, n, A, lda, P, info, batchSize);
}

template <typename T>
cublasStatus_t cublasgetriBatched(cublasHandle_t handle, int n,
                                  const T *const A[],    /*Device pointer*/
                                  int lda, const int *P, /*Device pointer*/
                                  T *const C[],          /*Device pointer*/
                                  int ldc, int *info, int batchSize,
                                  cudaStream_t stream);

template <>
inline cublasStatus_t cublasgetriBatched(
  cublasHandle_t handle, int n, const float *const A[], /*Device pointer*/
  int lda, const int *P,                                /*Device pointer*/
  float *const C[],                                     /*Device pointer*/
  int ldc, int *info, int batchSize, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasSgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize);
}

template <>
inline cublasStatus_t cublasgetriBatched(
  cublasHandle_t handle, int n, const double *const A[], /*Device pointer*/
  int lda, const int *P,                                 /*Device pointer*/
  double *const C[],                                     /*Device pointer*/
  int ldc, int *info, int batchSize, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasDgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize);
}

/** @} */

/**
 * @defgroup gelsbatched cublas gelsbatched calls
 * @{
 */

template <typename T>
inline cublasStatus_t cublasgelsBatched(cublasHandle_t handle,
                                        cublasOperation_t trans, int m, int n,
                                        int nrhs, T *Aarray[], int lda,
                                        T *Carray[], int ldc, int *info,
                                        int *devInfoArray, int batchSize,
                                        cudaStream_t stream = 0);

template <>
inline cublasStatus_t cublasgelsBatched(cublasHandle_t handle,
                                        cublasOperation_t trans, int m, int n,
                                        int nrhs, float *Aarray[], int lda,
                                        float *Carray[], int ldc, int *info,
                                        int *devInfoArray, int batchSize,
                                        cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasSgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc,
                            info, devInfoArray, batchSize);
}

template <>
inline cublasStatus_t cublasgelsBatched(cublasHandle_t handle,
                                        cublasOperation_t trans, int m, int n,
                                        int nrhs, double *Aarray[], int lda,
                                        double *Carray[], int ldc, int *info,
                                        int *devInfoArray, int batchSize,
                                        cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasDgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc,
                            info, devInfoArray, batchSize);
}

/** @} */

/**
 * @defgroup geam cublas geam calls
 * @{
 */
template <typename T>
cublasStatus_t cublasgeam(cublasHandle_t handle, cublasOperation_t transA,
                          cublasOperation_t transB, int m, int n, const T *alfa,
                          const T *A, int lda, const T *beta, const T *B,
                          int ldb, T *C, int ldc, cudaStream_t stream);

template <>
inline cublasStatus_t cublasgeam(cublasHandle_t handle,
                                 cublasOperation_t transA,
                                 cublasOperation_t transB, int m, int n,
                                 const float *alfa, const float *A, int lda,
                                 const float *beta, const float *B, int ldb,
                                 float *C, int ldc, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasSgeam(handle, transA, transB, m, n, alfa, A, lda, beta, B, ldb,
                     C, ldc);
}

template <>
inline cublasStatus_t cublasgeam(cublasHandle_t handle,
                                 cublasOperation_t transA,
                                 cublasOperation_t transB, int m, int n,
                                 const double *alfa, const double *A, int lda,
                                 const double *beta, const double *B, int ldb,
                                 double *C, int ldc, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasDgeam(handle, transA, transB, m, n, alfa, A, lda, beta, B, ldb,
                     C, ldc);
}
/** @} */

/**
 * @defgroup symm cublas symm calls
 * @{
 */
template <typename T>
cublasStatus_t cublassymm(cublasHandle_t handle, cublasSideMode_t side,
                          cublasFillMode_t uplo, int m, int n, const T *alpha,
                          const T *A, int lda, const T *B, int ldb,
                          const T *beta, T *C, int ldc, cudaStream_t stream);

template <>
inline cublasStatus_t cublassymm(cublasHandle_t handle, cublasSideMode_t side,
                                 cublasFillMode_t uplo, int m, int n,
                                 const float *alpha, const float *A, int lda,
                                 const float *B, int ldb, const float *beta,
                                 float *C, int ldc, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasSsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C,
                     ldc);
}

template <>
inline cublasStatus_t cublassymm(cublasHandle_t handle, cublasSideMode_t side,
                                 cublasFillMode_t uplo, int m, int n,
                                 const double *alpha, const double *A, int lda,
                                 const double *B, int ldb, const double *beta,
                                 double *C, int ldc, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasDsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C,
                     ldc);
}
/** @} */

/**
 * @defgroup syrk cublas syrk calls
 * @{
 */
template <typename T>
cublasStatus_t cublassyrk(cublasHandle_t handle, cublasFillMode_t uplo,
                          cublasOperation_t trans, int n, int k, const T *alpha,
                          const T *A, int lda, const T *beta, T *C, int ldc,
                          cudaStream_t stream);

template <>
inline cublasStatus_t cublassyrk(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, int n, int k,
                                 const float *alpha, const float *A, int lda,
                                 const float *beta, float *C, int ldc,
                                 cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasSsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

template <>
inline cublasStatus_t cublassyrk(cublasHandle_t handle, cublasFillMode_t uplo,
                                 cublasOperation_t trans, int n, int k,
                                 const double *alpha, const double *A, int lda,
                                 const double *beta, double *C, int ldc,
                                 cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasDsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}
/** @} */

/**
 * @defgroup nrm2 cublas nrm2 calls
 * @{
 */
template <typename T>
cublasStatus_t cublasnrm2(cublasHandle_t handle, int n, const T *x, int incx,
                          T *result, cudaStream_t stream);

template <>
inline cublasStatus_t cublasnrm2(cublasHandle_t handle, int n, const float *x,
                                 int incx, float *result, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasSnrm2(handle, n, x, incx, result);
}

template <>
inline cublasStatus_t cublasnrm2(cublasHandle_t handle, int n, const double *x,
                                 int incx, double *result,
                                 cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasDnrm2(handle, n, x, incx, result);
}
/** @} */

template <typename T>
cublasStatus_t cublastrsm(cublasHandle_t handle, cublasSideMode_t side,
                          cublasFillMode_t uplo, cublasOperation_t trans,
                          cublasDiagType_t diag, int m, int n, const T *alpha,
                          const T *A, int lda, T *B, int ldb,
                          cudaStream_t stream);

template <>
inline cublasStatus_t cublastrsm(cublasHandle_t handle, cublasSideMode_t side,
                                 cublasFillMode_t uplo, cublasOperation_t trans,
                                 cublasDiagType_t diag, int m, int n,
                                 const float *alpha, const float *A, int lda,
                                 float *B, int ldb, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                     ldb);
}

template <>
inline cublasStatus_t cublastrsm(cublasHandle_t handle, cublasSideMode_t side,
                                 cublasFillMode_t uplo, cublasOperation_t trans,
                                 cublasDiagType_t diag, int m, int n,
                                 const double *alpha, const double *A, int lda,
                                 double *B, int ldb, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                     ldb);
}

/**
 * @defgroup dot cublas dot calls
 * @{
 */
template <typename T>
cublasStatus_t cublasdot(cublasHandle_t handle, int n, const T *x, int incx,
                         const T *y, int incy, T *result, cudaStream_t stream);

template <>
inline cublasStatus_t cublasdot(cublasHandle_t handle, int n, const float *x,
                                int incx, const float *y, int incy,
                                float *result, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasSdot(handle, n, x, incx, y, incy, result);
}

template <>
inline cublasStatus_t cublasdot(cublasHandle_t handle, int n, const double *x,
                                int incx, const double *y, int incy,
                                double *result, cudaStream_t stream) {
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  return cublasDdot(handle, n, x, incx, y, incy, result);
}
/** @} */

};  // namespace LinAlg
};  // namespace MLCommon
