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

namespace MLCommon {
namespace Sparse {

#if defined(CUDART_VERSION) && CUDART_VERSION >= 10100
#define CUSPARSE_CHECK(call)                                                   \
  do {                                                                         \
    cusparseStatus_t status = call;                                            \
    ASSERT(status == CUSPARSE_STATUS_SUCCESS, "FAIL: call='%s' Reason='%s'\n", \
           #call, cusparseGetErrorString(status));                             \
  } while (0)
#else
#define CUSPARSE_CHECK(call)                                                 \
  do {                                                                       \
    cusparseStatus_t status = call;                                          \
    ASSERT(status == CUSPARSE_STATUS_SUCCESS, "FAIL: call='%s' Code='%d'\n", \
           #call, int(status));                                              \
  } while (0)
#endif  // CUDART_VERSION

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

};  // namespace Sparse
};  // namespace MLCommon
