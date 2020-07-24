/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cusparse_v2.h>
#include <cuml/common/logger.hpp>
#include <cuml/common/utils.hpp>

namespace MLCommon {
namespace Sparse {

#define _CUSPARSE_ERR_TO_STR(err) \
  case err:                       \
    return #err;
inline const char* cusparseErr2Str(cusparseStatus_t err) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 10100
  return cusparseGetErrorString(status);
#else   // CUDART_VERSION
  switch (err) {
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_SUCCESS);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_NOT_INITIALIZED);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_ALLOC_FAILED);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_INVALID_VALUE);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_ARCH_MISMATCH);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_EXECUTION_FAILED);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_INTERNAL_ERROR);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    default:
      return "CUSPARSE_STATUS_UNKNOWN";
  };
#endif  // CUDART_VERSION
}
#undef _CUSPARSE_ERR_TO_STR

/** check for cusparse runtime API errors and assert accordingly */
#define CUSPARSE_CHECK(call)                                         \
  do {                                                               \
    cusparseStatus_t err = call;                                     \
    ASSERT(err == CUSPARSE_STATUS_SUCCESS,                           \
           "CUSPARSE call='%s' got errorcode=%d err=%s", #call, err, \
           MLCommon::Sparse::cusparseErr2Str(err));                  \
  } while (0)

/** check for cusparse runtime API errors but do not assert */
#define CUSPARSE_CHECK_NO_THROW(call)                                          \
  do {                                                                         \
    cusparseStatus_t err = call;                                               \
    if (err != CUSPARSE_STATUS_SUCCESS) {                                      \
      CUML_LOG_ERROR("CUSPARSE call='%s' got errorcode=%d err=%s", #call, err, \
                     MLCommon::Sparse::cusparseErr2Str(err));                  \
    }                                                                          \
  } while (0)

/**
 * @defgroup gthr cusparse gather methods
 * @{
 */
template <typename T>
cusparseStatus_t cusparsegthr(cusparseHandle_t handle, int nnz, const T* vals,
                              T* vals_sorted, int* d_P, cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsegthr(cusparseHandle_t handle, int nnz,
                                     const double* vals, double* vals_sorted,
                                     int* d_P, cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseDgthr(handle, nnz, vals, vals_sorted, d_P,
                       CUSPARSE_INDEX_BASE_ZERO);
}
template <>
inline cusparseStatus_t cusparsegthr(cusparseHandle_t handle, int nnz,
                                     const float* vals, float* vals_sorted,
                                     int* d_P, cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSgthr(handle, nnz, vals, vals_sorted, d_P,
                       CUSPARSE_INDEX_BASE_ZERO);
}
/** @} */

/**
 * @defgroup coo2csr cusparse COO to CSR converter methods
 * @{
 */
template <typename T>
void cusparsecoo2csr(cusparseHandle_t handle, const T* cooRowInd, int nnz,
                     int m, T* csrRowPtr, cudaStream_t stream);
template <>
inline void cusparsecoo2csr(cusparseHandle_t handle, const int* cooRowInd,
                            int nnz, int m, int* csrRowPtr,
                            cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  CUSPARSE_CHECK(cusparseXcoo2csr(handle, cooRowInd, nnz, m, csrRowPtr,
                                  CUSPARSE_INDEX_BASE_ZERO));
}
/** @} */

/**
 * @defgroup coosort cusparse coo sort methods
 * @{
 */
template <typename T>
size_t cusparsecoosort_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                     int nnz, const T* cooRows,
                                     const T* cooCols, cudaStream_t stream);
template <>
inline size_t cusparsecoosort_bufferSizeExt(cusparseHandle_t handle, int m,
                                            int n, int nnz, const int* cooRows,
                                            const int* cooCols,
                                            cudaStream_t stream) {
  size_t val;
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  CUSPARSE_CHECK(
    cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, cooRows, cooCols, &val));
  return val;
}

template <typename T>
void cusparsecoosortByRow(cusparseHandle_t handle, int m, int n, int nnz,
                          T* cooRows, T* cooCols, T* P, void* pBuffer,
                          cudaStream_t stream);
template <>
inline void cusparsecoosortByRow(cusparseHandle_t handle, int m, int n, int nnz,
                                 int* cooRows, int* cooCols, int* P,
                                 void* pBuffer, cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  CUSPARSE_CHECK(
    cusparseXcoosortByRow(handle, m, n, nnz, cooRows, cooCols, P, pBuffer));
}
/** @} */

/**
 * @defgroup Gemmi cusparse gemmi operations
 * @{
 */
inline cusparseStatus_t cusparsegemmi(
  cusparseHandle_t handle, int m, int n, int k, int nnz, const float* alpha,
  const float* A, int lda, const float* cscValB, const int* cscColPtrB,
  const int* cscRowIndB, const float* beta, float* C, int ldc) {
  return cusparseSgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB,
                        cscColPtrB, cscRowIndB, beta, C, ldc);
}
inline cusparseStatus_t cusparsegemmi(
  cusparseHandle_t handle, int m, int n, int k, int nnz, const double* alpha,
  const double* A, int lda, const double* cscValB, const int* cscColPtrB,
  const int* cscRowIndB, const double* beta, double* C, int ldc) {
  return cusparseDgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB,
                        cscColPtrB, cscRowIndB, beta, C, ldc);
}
/** @} */

#if __CUDACC_VER_MAJOR__ > 10
/**
 * @defgroup cusparse Create CSR operations
 * @{
 */
template <typename IndexT, typename ValueT>
cusparseStatus_t cusparsecreatecsr(cusparseSpMatDescr_t* spMatDescr,
                                   int64_t rows, int64_t cols, int64_t nnz,
                                   IndexT* csrRowOffsets, IndexT* csrColInd,
                                   ValueT* csrValues);
template <>
inline cusparseStatus_t cusparsecreatecsr(cusparseSpMatDescr_t* spMatDescr,
                                          int64_t rows, int64_t cols,
                                          int64_t nnz, int32_t* csrRowOffsets,
                                          int32_t* csrColInd,
                                          float* csrValues) {
  return cusparseCreateCsr(spMatDescr, rows, cols, nnz, csrRowOffsets,
                           csrColInd, csrValues, CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                           CUDA_R_32F);
}
template <>
inline cusparseStatus_t cusparsecreatecsr(cusparseSpMatDescr_t* spMatDescr,
                                          int64_t rows, int64_t cols,
                                          int64_t nnz, int32_t* csrRowOffsets,
                                          int32_t* csrColInd,
                                          double* csrValues) {
  return cusparseCreateCsr(spMatDescr, rows, cols, nnz, csrRowOffsets,
                           csrColInd, csrValues, CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                           CUDA_R_64F);
}
template <>
inline cusparseStatus_t cusparsecreatecsr(cusparseSpMatDescr_t* spMatDescr,
                                          int64_t rows, int64_t cols,
                                          int64_t nnz, int64_t* csrRowOffsets,
                                          int64_t* csrColInd,
                                          float* csrValues) {
  return cusparseCreateCsr(spMatDescr, rows, cols, nnz, csrRowOffsets,
                           csrColInd, csrValues, CUSPARSE_INDEX_64I,
                           CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO,
                           CUDA_R_32F);
}
template <>
inline cusparseStatus_t cusparsecreatecsr(cusparseSpMatDescr_t* spMatDescr,
                                          int64_t rows, int64_t cols,
                                          int64_t nnz, int64_t* csrRowOffsets,
                                          int64_t* csrColInd,
                                          double* csrValues) {
  return cusparseCreateCsr(spMatDescr, rows, cols, nnz, csrRowOffsets,
                           csrColInd, csrValues, CUSPARSE_INDEX_64I,
                           CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO,
                           CUDA_R_64F);
}
/** @} */


/**
 * @defgroup SpGEMM cusparse sparse gemm operations
 * @{
 */
template<typename T>
inline cusparseStatus_t cusparsespgemm_workestimation(cusparseHandle_t      handle,
                              cusparseOperation_t   opA,
                              cusparseOperation_t   opB,
                              const T*           alpha,
                              cusparseSpMatDescr_t  matA,
                              cusparseSpMatDescr_t  matB,
                              const T*           beta,
                              cusparseSpMatDescr_t  matC,
                              cudaDataType          computeType,
                              cusparseSpGEMMAlg_t   alg,
                              cusparseSpGEMMDescr_t spgemmDescr,
                              size_t*               bufferSize1,
                              void*                 externalBuffer1);

template<>
inline cusparseStatus_t cusparsespgemm_workestimation(cusparseHandle_t      handle,
                              cusparseOperation_t   opA,
                              cusparseOperation_t   opB,
                              const float*           alpha,
                              cusparseSpMatDescr_t  matA,
                              cusparseSpMatDescr_t  matB,
                              const float*           beta,
                              cusparseSpMatDescr_t  matC,
                              cudaDataType          computeType,
                              cusparseSpGEMMAlg_t   alg,
                              cusparseSpGEMMDescr_t spgemmDescr,
                              size_t*               bufferSize1,
                              void*                 externalBuffer1) {
	return cusparseSpGEMM_workEstimation(handle,
            opA, opB, alpha, matA, matB, beta,
            matC, computeType, alg, spgemmDescr,
            bufferSize1, externalBuffer1);
}

template<typename T>
inline cusparseStatus_t
cusparsespgemm_compute(cusparseHandle_t      handle,
                       cusparseOperation_t   opA,
                       cusparseOperation_t   opB,
                       const T*           alpha,
                       cusparseSpMatDescr_t  matA,
                       cusparseSpMatDescr_t  matB,
                       const T*           beta,
                       cusparseSpMatDescr_t  matC,
                       cudaDataType          computeType,
                       cusparseSpGEMMAlg_t   alg,
                       cusparseSpGEMMDescr_t spgemmDescr,
                       void*                 externalBuffer1,
                       size_t*               bufferSize2,
                       void*                 externalBuffer2);

template<>
inline cusparseStatus_t
cusparsespgemm_compute(cusparseHandle_t      handle,
                       cusparseOperation_t   opA,
                       cusparseOperation_t   opB,
                       const float*           alpha,
                       cusparseSpMatDescr_t  matA,
                       cusparseSpMatDescr_t  matB,
                       const float*           beta,
                       cusparseSpMatDescr_t  matC,
                       cudaDataType          computeType,
                       cusparseSpGEMMAlg_t   alg,
                       cusparseSpGEMMDescr_t spgemmDescr,
                       void*                 externalBuffer1,
                       size_t*               bufferSize2,
                       void*                 externalBuffer2) {

	return cusparseSpGEMM_compute(handle,
            opA, opB, alpha, matA, matB, beta,
            matC, computeType, alg, spgemmDescr,
            externalBuffer1, bufferSize2, externalBuffer2);
}

template<typename T>
inline cusparseStatus_t
cusparsespgemm_copy(cusparseHandle_t      handle,
                    cusparseOperation_t   opA,
                    cusparseOperation_t   opB,
                    const T*           alpha,
                    cusparseSpMatDescr_t  matA,
                    cusparseSpMatDescr_t  matB,
                    const T*           beta,
                    cusparseSpMatDescr_t  matC,
                    cudaDataType          computeType,
                    cusparseSpGEMMAlg_t   alg,
                    cusparseSpGEMMDescr_t spgemmDescr,
                    void*                 externalBuffer2);

template<>
inline cusparseStatus_t
cusparsespgemm_copy(cusparseHandle_t      handle,
                    cusparseOperation_t   opA,
                    cusparseOperation_t   opB,
                    const float*           alpha,
                    cusparseSpMatDescr_t  matA,
                    cusparseSpMatDescr_t  matB,
                    const float*           beta,
                    cusparseSpMatDescr_t  matC,
                    cudaDataType          computeType,
                    cusparseSpGEMMAlg_t   alg,
                    cusparseSpGEMMDescr_t spgemmDescr,
                    void*                 externalBuffer2) {
	cusparsespgemm_copy(handle,
			opA, opB, alpha, matA, matB, beta, matC,
	        computeType, alg, spgemmDescr, externalBuffer2);
}

/** @} */

/**
 * @defgroup setpointermode cusparse set pointer mode method
 * @{
 */
// no T dependency...
// template <typename T>
// cusparseStatus_t cusparsesetpointermode(  // NOLINT
//                                         cusparseHandle_t handle,
//                                         cusparsePointerMode_t mode,
//                                         cudaStream_t stream);

// template<>
inline cusparseStatus_t cusparsesetpointermode(cusparseHandle_t handle,
                                               cusparsePointerMode_t mode,
                                               cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSetPointerMode(handle, mode);
}
/** @} */


};  // namespace Sparse
};  // namespace MLCommon
