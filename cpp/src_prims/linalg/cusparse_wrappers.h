/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

namespace MLCommon {
namespace LinAlg {

/** check for cusparse runtime API errors and assert accordingly */
#define CUSPARSE_CHECK(call)                                                   \
  {                                                                            \
    cusparseStatus_t err;                                                        \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS) {                           \
      fprintf(stderr, "Got CUSPARSE error %d at %s:%d\n", err, __FILE__,         \
              __LINE__);                                                       \
      switch (err) {                                                           \
        case CUSPARSE_STATUS_NOT_INITIALIZED:                                  \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_NOT_INITIALIZED");            \
          exit(1);                                                             \
        case CUSPARSE_STATUS_ALLOC_FAILED:                                     \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_ALLOC_FAILED");               \
          exit(1);                                                             \
        case CUSPARSE_STATUS_INVALID_VALUE:                                    \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_INVALID_VALUE");              \
          exit(1);                                                             \
        case CUSPARSE_STATUS_ARCH_MISMATCH:                                    \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_ARCH_MISMATCH");              \
          exit(1);                                                             \
        case CUSPARSE_STATUS_MAPPING_ERROR:                                    \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_MAPPING_ERROR");              \
          exit(1);                                                             \
        case CUSPARSE_STATUS_EXECUTION_FAILED:                                 \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_EXECUTION_FAILED");           \
          exit(1);                                                             \
        case CUSPARSE_STATUS_INTERNAL_ERROR:                                   \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_INTERNAL_ERROR");             \
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:                                   \
          fprintf(stderr, "%s\n", "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED");             \
      }                                                                        \
      exit(1);                                                                 \
      exit(1);                                                                 \
    }                                                                          \
  }

cusparseStatus_t cusparsegemmi(cusparseHandle_t handle, int m, int n, int k, int nnz,
									const float *alpha, const float *A, int lda,
									const float *cscValB, const int *cscColPtrB,
									const int *cscRowIndB, const float *beta,
									float *C, int ldc)
{
    return cusparseSgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB,
                                                cscRowIndB, beta, C, ldc);
}

cusparseStatus_t cusparsegemmi(cusparseHandle_t handle, int m, int n, int k, int nnz,
                                const double *alpha, const double *A, int lda,
                                const double *cscValB, const int *cscColPtrB,
                                const int *cscRowIndB, const double *beta,
                                double *C, int ldc)
{
    return cusparseDgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB,
                                                cscRowIndB, beta, C, ldc);
}


/** @} */

}; // namespace LinAlg
}; // namespace MLCommon