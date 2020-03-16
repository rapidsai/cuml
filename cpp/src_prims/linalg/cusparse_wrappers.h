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

#include <cusparse.h>
#include <cuml/common/utils.hpp>
#include <cuml/common/logger.hpp>

namespace MLCommon {
namespace LinAlg {

#define _CUSPARSE_ERR_TO_STR(err) case err: return #err;
inline const char* cusparseErr2Str(cusparseStatus_t err) {
  switch (err) {
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_SUCCESS);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_NOT_INITIALIZED);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_ALLOC_FAILED);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_INVALID_VALUE);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_ARCH_MISMATCH);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_EXECUTION_FAILED);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_INTERNAL_ERROR);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_NOT_SUPPORTED);
    default: return "CUSPARSE_STATUS_UNKNOWN";
  };
}
#undef _CUSPARSE_ERR_TO_STR

/** check for cusparse runtime API errors and assert accordingly */
#define CUSPARSE_CHECK(call)                                            \
  do {                                                                  \
    cusparseStatus_t err = call;                                        \
    ASSERT(err == CUSPARSE_STATUS_SUCCESS,                              \
           "CUSPARSE call='%s' got errorcode=%d err=%s", #call, err,    \
           MLCommon::LinAlg::cusparseErr2Str(err));                     \
  } while (0)

/** check for cusparse runtime API errors but do not assert */
#define CUSPARSE_CHECK_NO_THROW(call)                                   \
  do {                                                                  \
    cusparseStatus_t err = call;                                        \
    if (err != CUSPARSE_STATUS_SUCCESS) {                               \
      CUML_LOG_ERROR("CUSPARSE call='%s' got errorcode=%d err=%s", #call, err, \
                     MLCommon::LinAlg::cusparseErr2Str(err));           \
    }                                                                   \
  } while (0)

cusparseStatus_t cusparsegemmi(cusparseHandle_t handle, int m, int n, int k,
                               int nnz, const float *alpha, const float *A,
                               int lda, const float *cscValB,
                               const int *cscColPtrB, const int *cscRowIndB,
                               const float *beta, float *C, int ldc) {
  return cusparseSgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB,
                        cscColPtrB, cscRowIndB, beta, C, ldc);
}

cusparseStatus_t cusparsegemmi(cusparseHandle_t handle, int m, int n, int k,
                               int nnz, const double *alpha, const double *A,
                               int lda, const double *cscValB,
                               const int *cscColPtrB, const int *cscRowIndB,
                               const double *beta, double *C, int ldc) {
  return cusparseDgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB,
                        cscColPtrB, cscRowIndB, beta, C, ldc);
}

/** @} */

};  // namespace LinAlg
};  // namespace MLCommon
