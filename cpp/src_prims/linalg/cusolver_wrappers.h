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

#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cuml/common/logger.hpp>
#include <cuml/common/utils.hpp>

namespace MLCommon {
namespace LinAlg {

#define _CUSOLVER_ERR_TO_STR(err) \
  case err:                       \
    return #err;
inline const char *cusolverErr2Str(cusolverStatus_t err) {
  switch (err) {
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_SUCCESS);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_NOT_INITIALIZED);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_ALLOC_FAILED);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_INVALID_VALUE);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_ARCH_MISMATCH);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_EXECUTION_FAILED);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_INTERNAL_ERROR);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_ZERO_PIVOT);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_NOT_SUPPORTED);
    default:
      return "CUSOLVER_STATUS_UNKNOWN";
  };
}
#undef _CUSOLVER_ERR_TO_STR

/** check for cusolver runtime API errors and assert accordingly */
#define CUSOLVER_CHECK(call)                                         \
  do {                                                               \
    cusolverStatus_t err = call;                                     \
    ASSERT(err == CUSOLVER_STATUS_SUCCESS,                           \
           "CUSOLVER call='%s' got errorcode=%d err=%s", #call, err, \
           MLCommon::LinAlg::cusolverErr2Str(err));                  \
  } while (0)

/** check for cusolver runtime API errors but do not assert */
#define CUSOLVER_CHECK_NO_THROW(call)                                          \
  do {                                                                         \
    cusolverStatus_t err = call;                                               \
    if (err != CUSOLVER_STATUS_SUCCESS) {                                      \
      CUML_LOG_ERROR("CUSOLVER call='%s' got errorcode=%d err=%s", #call, err, \
                     MLCommon::LinAlg::cusolverErr2Str(err));                  \
    }                                                                          \
  } while (0)

/**
 * @defgroup Getrf cusolver getrf operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDngetrf(cusolverDnHandle_t handle, int m, int n, T *A,
                                 int lda, T *Workspace, int *devIpiv,
                                 int *devInfo, cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDngetrf(cusolverDnHandle_t handle, int m, int n,
                                        float *A, int lda, float *Workspace,
                                        int *devIpiv, int *devInfo,
                                        cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

template <>
inline cusolverStatus_t cusolverDngetrf(cusolverDnHandle_t handle, int m, int n,
                                        double *A, int lda, double *Workspace,
                                        int *devIpiv, int *devInfo,
                                        cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

template <typename T>
cusolverStatus_t cusolverDngetrf_bufferSize(cusolverDnHandle_t handle, int m,
                                            int n, T *A, int lda, int *Lwork);

template <>
inline cusolverStatus_t cusolverDngetrf_bufferSize(cusolverDnHandle_t handle,
                                                   int m, int n, float *A,
                                                   int lda, int *Lwork) {
  return cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

template <>
inline cusolverStatus_t cusolverDngetrf_bufferSize(cusolverDnHandle_t handle,
                                                   int m, int n, double *A,
                                                   int lda, int *Lwork) {
  return cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

/**
 * @defgroup Getrs cusolver getrs operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDngetrs(cusolverDnHandle_t handle,
                                 cublasOperation_t trans, int n, int nrhs,
                                 const T *A, int lda, const int *devIpiv, T *B,
                                 int ldb, int *devInfo, cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDngetrs(cusolverDnHandle_t handle,
                                        cublasOperation_t trans, int n,
                                        int nrhs, const float *A, int lda,
                                        const int *devIpiv, float *B, int ldb,
                                        int *devInfo, cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb,
                          devInfo);
}

template <>
inline cusolverStatus_t cusolverDngetrs(cusolverDnHandle_t handle,
                                        cublasOperation_t trans, int n,
                                        int nrhs, const double *A, int lda,
                                        const int *devIpiv, double *B, int ldb,
                                        int *devInfo, cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb,
                          devInfo);
}
/** @} */

/**
 * @defgroup syevd cusolver syevd operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnsyevd_bufferSize(cusolverDnHandle_t handle,
                                            cusolverEigMode_t jobz,
                                            cublasFillMode_t uplo, int n,
                                            const T *A, int lda, const T *W,
                                            int *lwork);

template <>
inline cusolverStatus_t cusolverDnsyevd_bufferSize(cusolverDnHandle_t handle,
                                                   cusolverEigMode_t jobz,
                                                   cublasFillMode_t uplo, int n,
                                                   const float *A, int lda,
                                                   const float *W, int *lwork) {
  return cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
}

template <>
inline cusolverStatus_t cusolverDnsyevd_bufferSize(
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
  int n, const double *A, int lda, const double *W, int *lwork) {
  return cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
}
/** @} */

/**
 * @defgroup syevj cusolver syevj operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnsyevj(cusolverDnHandle_t handle,
                                 cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                 int n, T *A, int lda, T *W, T *work, int lwork,
                                 int *info, syevjInfo_t params,
                                 cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnsyevj(
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
  int n, float *A, int lda, float *W, float *work, int lwork, int *info,
  syevjInfo_t params, cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                          params);
}

template <>
inline cusolverStatus_t cusolverDnsyevj(
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
  int n, double *A, int lda, double *W, double *work, int lwork, int *info,
  syevjInfo_t params, cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info,
                          params);
}

template <typename T>
cusolverStatus_t cusolverDnsyevj_bufferSize(cusolverDnHandle_t handle,
                                            cusolverEigMode_t jobz,
                                            cublasFillMode_t uplo, int n,
                                            const T *A, int lda, const T *W,
                                            int *lwork, syevjInfo_t params);

template <>
inline cusolverStatus_t cusolverDnsyevj_bufferSize(cusolverDnHandle_t handle,
                                                   cusolverEigMode_t jobz,
                                                   cublasFillMode_t uplo, int n,
                                                   const float *A, int lda,
                                                   const float *W, int *lwork,
                                                   syevjInfo_t params) {
  return cusolverDnSsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                     params);
}

template <>
inline cusolverStatus_t cusolverDnsyevj_bufferSize(cusolverDnHandle_t handle,
                                                   cusolverEigMode_t jobz,
                                                   cublasFillMode_t uplo, int n,
                                                   const double *A, int lda,
                                                   const double *W, int *lwork,
                                                   syevjInfo_t params) {
  return cusolverDnDsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork,
                                     params);
}
/** @} */

/**
 * @defgroup syevd cusolver syevd operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnsyevd(cusolverDnHandle_t handle,
                                 cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                 int n, T *A, int lda, T *W, T *work, int lwork,
                                 int *devInfo, cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnsyevd(cusolverDnHandle_t handle,
                                        cusolverEigMode_t jobz,
                                        cublasFillMode_t uplo, int n, float *A,
                                        int lda, float *W, float *work,
                                        int lwork, int *devInfo,
                                        cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork,
                          devInfo);
}

template <>
inline cusolverStatus_t cusolverDnsyevd(cusolverDnHandle_t handle,
                                        cusolverEigMode_t jobz,
                                        cublasFillMode_t uplo, int n, double *A,
                                        int lda, double *W, double *work,
                                        int lwork, int *devInfo,
                                        cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork,
                          devInfo);
}
/** @} */

#if CUDART_VERSION >= 10010
/**
 * @defgroup syevdx cusolver syevdx operations
 * @{
*/
template <typename T>
cusolverStatus_t cusolverDnsyevdx_bufferSize(
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
  cublasFillMode_t uplo, int n, const T *A, int lda, T vl, T vu, int il, int iu,
  int *h_meig, const T *W, int *lwork);

template <>
inline cusolverStatus_t cusolverDnsyevdx_bufferSize(
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
  cublasFillMode_t uplo, int n, const float *A, int lda, float vl, float vu,
  int il, int iu, int *h_meig, const float *W, int *lwork) {
  return cusolverDnSsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl,
                                      vu, il, iu, h_meig, W, lwork);
}

template <>
inline cusolverStatus_t cusolverDnsyevdx_bufferSize(
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
  cublasFillMode_t uplo, int n, const double *A, int lda, double vl, double vu,
  int il, int iu, int *h_meig, const double *W, int *lwork) {
  return cusolverDnDsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl,
                                      vu, il, iu, h_meig, W, lwork);
}

template <typename T>
cusolverStatus_t cusolverDnsyevdx(
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
  cublasFillMode_t uplo, int n, T *A, int lda, T vl, T vu, int il, int iu,
  int *h_meig, T *W, T *work, int lwork, int *devInfo, cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnsyevdx(
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
  cublasFillMode_t uplo, int n, float *A, int lda, float vl, float vu, int il,
  int iu, int *h_meig, float *W, float *work, int lwork, int *devInfo,
  cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                           h_meig, W, work, lwork, devInfo);
}

template <>
inline cusolverStatus_t cusolverDnsyevdx(
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
  cublasFillMode_t uplo, int n, double *A, int lda, double vl, double vu,
  int il, int iu, int *h_meig, double *W, double *work, int lwork, int *devInfo,
  cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu,
                           h_meig, W, work, lwork, devInfo);
}
/** @} */
#endif

/**
 * @defgroup svd cusolver svd operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDngesvd_bufferSize(cusolverDnHandle_t handle, int m,
                                            int n, int *lwork) {
  if (typeid(T) == typeid(float)) {
    return cusolverDnSgesvd_bufferSize(handle, m, n, lwork);
  } else {
    return cusolverDnDgesvd_bufferSize(handle, m, n, lwork);
  }
}
template <typename T>
cusolverStatus_t cusolverDngesvd(cusolverDnHandle_t handle, signed char jobu,
                                 signed char jobvt, int m, int n, T *A, int lda,
                                 T *S, T *U, int ldu, T *VT, int ldvt, T *work,
                                 int lwork, T *rwork, int *devInfo,
                                 cudaStream_t stream);
template <>
inline cusolverStatus_t cusolverDngesvd(
  cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n,
  float *A, int lda, float *S, float *U, int ldu, float *VT, int ldvt,
  float *work, int lwork, float *rwork, int *devInfo, cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT,
                          ldvt, work, lwork, rwork, devInfo);
}
template <>
inline cusolverStatus_t cusolverDngesvd(
  cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n,
  double *A, int lda, double *S, double *U, int ldu, double *VT, int ldvt,
  double *work, int lwork, double *rwork, int *devInfo, cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT,
                          ldvt, work, lwork, rwork, devInfo);
}

template <typename T>
inline cusolverStatus_t CUSOLVERAPI cusolverDngesvdj_bufferSize(
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n,
  const T *A, int lda, const T *S, const T *U, int ldu, const T *V, int ldv,
  int *lwork, gesvdjInfo_t params);
template <>
inline cusolverStatus_t CUSOLVERAPI cusolverDngesvdj_bufferSize(
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n,
  const float *A, int lda, const float *S, const float *U, int ldu,
  const float *V, int ldv, int *lwork, gesvdjInfo_t params) {
  return cusolverDnSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U,
                                      ldu, V, ldv, lwork, params);
}
template <>
inline cusolverStatus_t CUSOLVERAPI cusolverDngesvdj_bufferSize(
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n,
  const double *A, int lda, const double *S, const double *U, int ldu,
  const double *V, int ldv, int *lwork, gesvdjInfo_t params) {
  return cusolverDnDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U,
                                      ldu, V, ldv, lwork, params);
}
template <typename T>
inline cusolverStatus_t CUSOLVERAPI cusolverDngesvdj(
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n,
  T *A, int lda, T *S, T *U, int ldu, T *V, int ldv, T *work, int lwork,
  int *info, gesvdjInfo_t params, cudaStream_t stream);
template <>
inline cusolverStatus_t CUSOLVERAPI cusolverDngesvdj(
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n,
  float *A, int lda, float *S, float *U, int ldu, float *V, int ldv,
  float *work, int lwork, int *info, gesvdjInfo_t params, cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                           work, lwork, info, params);
}
template <>
inline cusolverStatus_t CUSOLVERAPI
cusolverDngesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ,
                 int m, int n, double *A, int lda, double *S, double *U,
                 int ldu, double *V, int ldv, double *work, int lwork,
                 int *info, gesvdjInfo_t params, cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
                           work, lwork, info, params);
}
/** @} */

/**
 * @defgroup potrf cusolver potrf operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnpotrf_bufferSize(cusolverDnHandle_t handle,
                                            cublasFillMode_t uplo, int n, T *A,
                                            int lda, int *Lwork);

template <>
inline cusolverStatus_t cusolverDnpotrf_bufferSize(cusolverDnHandle_t handle,
                                                   cublasFillMode_t uplo, int n,
                                                   float *A, int lda,
                                                   int *Lwork) {
  return cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, Lwork);
}

template <>
inline cusolverStatus_t cusolverDnpotrf_bufferSize(cusolverDnHandle_t handle,
                                                   cublasFillMode_t uplo, int n,
                                                   double *A, int lda,
                                                   int *Lwork) {
  return cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, Lwork);
}

template <typename T>
inline cusolverStatus_t cusolverDnpotrf(cusolverDnHandle_t handle,
                                        cublasFillMode_t uplo, int n, T *A,
                                        int lda, T *Workspace, int Lwork,
                                        int *devInfo, cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnpotrf(cusolverDnHandle_t handle,
                                        cublasFillMode_t uplo, int n, float *A,
                                        int lda, float *Workspace, int Lwork,
                                        int *devInfo, cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

template <>
inline cusolverStatus_t cusolverDnpotrf(cusolverDnHandle_t handle,
                                        cublasFillMode_t uplo, int n, double *A,
                                        int lda, double *Workspace, int Lwork,
                                        int *devInfo, cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}
/** @} */

/**
 * @defgroup potrs cusolver potrs operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnpotrs(cusolverDnHandle_t handle,
                                 cublasFillMode_t uplo, int n, int nrhs,
                                 const T *A, int lda, T *B, int ldb,
                                 int *devInfo, cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnpotrs(cusolverDnHandle_t handle,
                                        cublasFillMode_t uplo, int n, int nrhs,
                                        const float *A, int lda, float *B,
                                        int ldb, int *devInfo,
                                        cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}

template <>
inline cusolverStatus_t cusolverDnpotrs(cusolverDnHandle_t handle,
                                        cublasFillMode_t uplo, int n, int nrhs,
                                        const double *A, int lda, double *B,
                                        int ldb, int *devInfo,
                                        cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}
/** @} */

/**
 * @defgroup geqrf cusolver geqrf operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDngeqrf(cusolverDnHandle_t handle, int m, int n, T *A,
                                 int lda, T *TAU, T *Workspace, int Lwork,
                                 int *devInfo, cudaStream_t stream);
template <>
inline cusolverStatus_t cusolverDngeqrf(cusolverDnHandle_t handle, int m, int n,
                                        float *A, int lda, float *TAU,
                                        float *Workspace, int Lwork,
                                        int *devInfo, cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}
template <>
inline cusolverStatus_t cusolverDngeqrf(cusolverDnHandle_t handle, int m, int n,
                                        double *A, int lda, double *TAU,
                                        double *Workspace, int Lwork,
                                        int *devInfo, cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}

template <typename T>
cusolverStatus_t cusolverDngeqrf_bufferSize(cusolverDnHandle_t handle, int m,
                                            int n, T *A, int lda, int *Lwork);
template <>
inline cusolverStatus_t cusolverDngeqrf_bufferSize(cusolverDnHandle_t handle,
                                                   int m, int n, float *A,
                                                   int lda, int *Lwork) {
  return cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
}
template <>
inline cusolverStatus_t cusolverDngeqrf_bufferSize(cusolverDnHandle_t handle,
                                                   int m, int n, double *A,
                                                   int lda, int *Lwork) {
  return cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
}
/** @} */

/**
 * @defgroup orgqr cusolver orgqr operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnorgqr(cusolverDnHandle_t handle, int m, int n, int k,
                                 T *A, int lda, const T *tau, T *work,
                                 int lwork, int *devInfo, cudaStream_t stream);
template <>
inline cusolverStatus_t cusolverDnorgqr(cusolverDnHandle_t handle, int m, int n,
                                        int k, float *A, int lda,
                                        const float *tau, float *work,
                                        int lwork, int *devInfo,
                                        cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}
template <>
inline cusolverStatus_t cusolverDnorgqr(cusolverDnHandle_t handle, int m, int n,
                                        int k, double *A, int lda,
                                        const double *tau, double *work,
                                        int lwork, int *devInfo,
                                        cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}

template <typename T>
cusolverStatus_t cusolverDnorgqr_bufferSize(cusolverDnHandle_t handle, int m,
                                            int n, int k, const T *A, int lda,
                                            const T *TAU, int *lwork);
template <>
inline cusolverStatus_t cusolverDnorgqr_bufferSize(cusolverDnHandle_t handle,
                                                   int m, int n, int k,
                                                   const float *A, int lda,
                                                   const float *TAU,
                                                   int *lwork) {
  return cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, TAU, lwork);
}
template <>
inline cusolverStatus_t cusolverDnorgqr_bufferSize(cusolverDnHandle_t handle,
                                                   int m, int n, int k,
                                                   const double *A, int lda,
                                                   const double *TAU,
                                                   int *lwork) {
  return cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, TAU, lwork);
}
/** @} */

/**
 * @defgroup ormqr cusolver ormqr operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnormqr(cusolverDnHandle_t handle,
                                 cublasSideMode_t side, cublasOperation_t trans,
                                 int m, int n, int k, const T *A, int lda,
                                 const T *tau, T *C, int ldc, T *work,
                                 int lwork, int *devInfo, cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnormqr(
  cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans,
  int m, int n, int k, const float *A, int lda, const float *tau, float *C,
  int ldc, float *work, int lwork, int *devInfo, cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc,
                          work, lwork, devInfo);
}

template <>
inline cusolverStatus_t cusolverDnormqr(
  cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans,
  int m, int n, int k, const double *A, int lda, const double *tau, double *C,
  int ldc, double *work, int lwork, int *devInfo, cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc,
                          work, lwork, devInfo);
}

template <typename T>
cusolverStatus_t cusolverDnormqr_bufferSize(cusolverDnHandle_t handle,
                                            cublasSideMode_t side,
                                            cublasOperation_t trans, int m,
                                            int n, int k, const T *A, int lda,
                                            const T *tau, const T *C, int ldc,
                                            int *lwork);

template <>
inline cusolverStatus_t cusolverDnormqr_bufferSize(
  cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans,
  int m, int n, int k, const float *A, int lda, const float *tau,
  const float *C, int ldc, int *lwork) {
  return cusolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau,
                                     C, ldc, lwork);
}

template <>
inline cusolverStatus_t cusolverDnormqr_bufferSize(
  cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans,
  int m, int n, int k, const double *A, int lda, const double *tau,
  const double *C, int ldc, int *lwork) {
  return cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau,
                                     C, ldc, lwork);
}
/** @} */

/**
 * @defgroup csrqrBatched cusolver batched
 * @{
 */
template <typename T>
cusolverStatus_t cusolverSpcsrqrBufferInfoBatched(
  cusolverSpHandle_t handle, int m, int n, int nnzA,
  const cusparseMatDescr_t descrA, const T *csrValA, const int *csrRowPtrA,
  const int *csrColIndA, int batchSize, csrqrInfo_t info,
  size_t *internalDataInBytes, size_t *workspaceInBytes);

template <>
inline cusolverStatus_t cusolverSpcsrqrBufferInfoBatched(
  cusolverSpHandle_t handle, int m, int n, int nnzA,
  const cusparseMatDescr_t descrA, const float *csrValA, const int *csrRowPtrA,
  const int *csrColIndA, int batchSize, csrqrInfo_t info,
  size_t *internalDataInBytes, size_t *workspaceInBytes) {
  return cusolverSpScsrqrBufferInfoBatched(
    handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, batchSize,
    info, internalDataInBytes, workspaceInBytes);
}

template <>
inline cusolverStatus_t cusolverSpcsrqrBufferInfoBatched(
  cusolverSpHandle_t handle, int m, int n, int nnzA,
  const cusparseMatDescr_t descrA, const double *csrValA, const int *csrRowPtrA,
  const int *csrColIndA, int batchSize, csrqrInfo_t info,
  size_t *internalDataInBytes, size_t *workspaceInBytes) {
  return cusolverSpDcsrqrBufferInfoBatched(
    handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, batchSize,
    info, internalDataInBytes, workspaceInBytes);
}

template <typename T>
cusolverStatus_t cusolverSpcsrqrsvBatched(
  cusolverSpHandle_t handle, int m, int n, int nnzA,
  const cusparseMatDescr_t descrA, const T *csrValA, const int *csrRowPtrA,
  const int *csrColIndA, const T *b, T *x, int batchSize, csrqrInfo_t info,
  void *pBuffer, cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverSpcsrqrsvBatched(
  cusolverSpHandle_t handle, int m, int n, int nnzA,
  const cusparseMatDescr_t descrA, const float *csrValA, const int *csrRowPtrA,
  const int *csrColIndA, const float *b, float *x, int batchSize,
  csrqrInfo_t info, void *pBuffer, cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverSpSetStream(handle, stream));
  return cusolverSpScsrqrsvBatched(handle, m, n, nnzA, descrA, csrValA,
                                   csrRowPtrA, csrColIndA, b, x, batchSize,
                                   info, pBuffer);
}

template <>
inline cusolverStatus_t cusolverSpcsrqrsvBatched(
  cusolverSpHandle_t handle, int m, int n, int nnzA,
  const cusparseMatDescr_t descrA, const double *csrValA, const int *csrRowPtrA,
  const int *csrColIndA, const double *b, double *x, int batchSize,
  csrqrInfo_t info, void *pBuffer, cudaStream_t stream) {
  CUSOLVER_CHECK(cusolverSpSetStream(handle, stream));
  return cusolverSpDcsrqrsvBatched(handle, m, n, nnzA, descrA, csrValA,
                                   csrRowPtrA, csrColIndA, b, x, batchSize,
                                   info, pBuffer);
}
/** @} */

};  // end namespace LinAlg
};  // end namespace MLCommon
