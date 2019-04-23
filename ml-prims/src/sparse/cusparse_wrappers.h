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

#include <cusparse_v2.h>

namespace MLCommon {

namespace Sparse {
#define CUSPARSE_CHECK(call)                                                   \
  do {                                                                         \
    cusparseStatus_t status = call;                                            \
    ASSERT(status == CUSPARSE_STATUS_SUCCESS, "FAIL: call='%s', status=%d\n", #call, status);     \
  } while (0)

template <typename T>
cusparseStatus_t cusparseGemmi(cusparseHandle_t handle, int m, int n, int k,
                               int nnz, const T *alpha, const T *A, int lda,
                               const T *cscValB, const int *cscColPtrB,
                               const int *cscRowIndB, const T *beta, T *C,
                               int ldc);

template <>
inline cusparseStatus_t cusparseGemmi<float>(
    cusparseHandle_t handle, int m, int n, int k, int nnz, const float *alpha,
    const float *A, int lda, const float *cscValB, const int *cscColPtrB,
    const int *cscRowIndB, const float *beta, float *C, int ldc) {
  return cusparseSgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB,
                        cscColPtrB, cscRowIndB, beta, C, ldc);
}

template <>
inline cusparseStatus_t cusparseGemmi<double>(
    cusparseHandle_t handle, int m, int n, int k, int nnz, const double *alpha,
    const double *A, int lda, const double *cscValB, const int *cscColPtrB,
    const int *cscRowIndB, const double *beta, double *C, int ldc) {
  return cusparseDgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB,
                        cscColPtrB, cscRowIndB, beta, C, ldc);
}

template <typename T>
cusparseStatus_t
cusparseCsrmm(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n,
              int k, int nnz, const T *alpha, const cusparseMatDescr_t descrA,
              const T *csrValA, const int *csrRowPtrA, const int *csrColIndA,
              const T *B, int ldb, const T *beta, T *C, int ldc);

template <>
inline cusparseStatus_t cusparseCsrmm<float>(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k,
    int nnz, const float *alpha, const cusparseMatDescr_t descrA,
    const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    const float *B, int ldb, const float *beta, float *C, int ldc) {
  return cusparseScsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA,
                        csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
}

template <>
inline cusparseStatus_t cusparseCsrmm<double>(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k,
    int nnz, const double *alpha, const cusparseMatDescr_t descrA,
    const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    const double *B, int ldb, const double *beta, double *C, int ldc) {
  return cusparseDcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA,
                        csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
}

template <typename T>
cusparseStatus_t
cusparseCsrmm2(cusparseHandle_t handle, cusparseOperation_t transA,
               cusparseOperation_t transB, int m, int n, int k, int nnz,
               const T *alpha, const cusparseMatDescr_t descrA,
               const T *csrValA, const int *csrRowPtrA, const int *csrColIndA,
               const T *B, int ldb, const T *beta, T *C, int ldc);

template <>
inline cusparseStatus_t
cusparseCsrmm2<float>(cusparseHandle_t handle, cusparseOperation_t transA,
                      cusparseOperation_t transB, int m, int n, int k, int nnz,
                      const float *alpha, const cusparseMatDescr_t descrA,
                      const float *csrValA, const int *csrRowPtrA,
                      const int *csrColIndA, const float *B, int ldb,
                      const float *beta, float *C, int ldc) {
  return cusparseScsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                         csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
}

template <>
inline cusparseStatus_t
cusparseCsrmm2<double>(cusparseHandle_t handle, cusparseOperation_t transA,
                       cusparseOperation_t transB, int m, int n, int k, int nnz,
                       const double *alpha, const cusparseMatDescr_t descrA,
                       const double *csrValA, const int *csrRowPtrA,
                       const int *csrColIndA, const double *B, int ldb,
                       const double *beta, double *C, int ldc) {

  return cusparseDcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                         csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
}
}
} // namespace MLCommon
