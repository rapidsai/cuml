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
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "HoltWinters.hpp"

#define MAX_BLOCKS_PER_DIM 65535
#define GPU_LOOP(i, n)                                         \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)
#define GET_TID (blockIdx.x * blockDim.x + threadIdx.x)

inline int GET_THREADS_PER_BLOCK(const int n, const int max_threads = 512) {
  int ret;
  if (n <= 128)
    ret = 32;
  else if (n <= 1024)
    ret = 128;
  else
    ret = 512;
  return ret > max_threads ? max_threads : ret;
}

inline int GET_NUM_BLOCKS(const int n, const int max_threads = 512,
                          const int max_blocks = MAX_BLOCKS_PER_DIM) {
  int ret = (n - 1) / GET_THREADS_PER_BLOCK(n, max_threads) + 1;
  return ret > max_blocks ? max_blocks : ret;
}

#ifdef DEBUG
#define COUT() (std::cout)
#define CERR() (std::cerr)

#define WARNING(message)                                                  \
  do {                                                                    \
    std::stringstream ss;                                                 \
    ss << "Warning (" << __FILE__ << ":" << __LINE__ << "): " << message; \
    CERR() << ss.str() << std::endl;                                      \
  } while (0)

#define CASE_STR(CODE)            \
  case CODE:                      \
    CERR() << #CODE << std::endl; \
    break
#define CHECK_CUBLAS(call)                             \
  {                                                    \
    switch (call) {                                    \
      case CUBLAS_STATUS_SUCCESS:                      \
        break;                                         \
        CASE_STR(CUBLAS_STATUS_NOT_INITIALIZED);       \
        CASE_STR(CUBLAS_STATUS_ALLOC_FAILED);          \
        CASE_STR(CUBLAS_STATUS_INVALID_VALUE);         \
        CASE_STR(CUBLAS_STATUS_ARCH_MISMATCH);         \
        CASE_STR(CUBLAS_STATUS_MAPPING_ERROR);         \
        CASE_STR(CUBLAS_STATUS_EXECUTION_FAILED);      \
        CASE_STR(CUBLAS_STATUS_INTERNAL_ERROR);        \
      default:                                         \
        CERR() << "unknown CUBLAS error" << std::endl; \
    }                                                  \
  }

#define CHECK_CUSOLVER(call)                                 \
  {                                                          \
    switch (call) {                                          \
      case CUSOLVER_STATUS_SUCCESS:                          \
        break;                                               \
        CASE_STR(CUSOLVER_STATUS_NOT_INITIALIZED);           \
        CASE_STR(CUSOLVER_STATUS_ALLOC_FAILED);              \
        CASE_STR(CUSOLVER_STATUS_INVALID_VALUE);             \
        CASE_STR(CUSOLVER_STATUS_ARCH_MISMATCH);             \
        CASE_STR(CUSOLVER_STATUS_EXECUTION_FAILED);          \
        CASE_STR(CUSOLVER_STATUS_INTERNAL_ERROR);            \
        CASE_STR(CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED); \
      default:                                               \
        CERR() << "unknown CUSOLVER error" << std::endl;     \
    }                                                        \
  }

#else  // DEBUG
#define WARNING(message)
#define CHECK_CUBLAS(call) (call)
#define CHECK_CUSOLVER(call) (call)
#endif

template <typename Dtype>
aion::AionStatus transpose_gpu(const Dtype *src, int m, int n, Dtype *dst);

namespace aion {
class cublas;
class cusolver;

class cublas {
 private:
  thread_local static cublasHandle_t m_handle;
  cublas();
  ~cublas();

 public:
  // Get the handle.
  static cublasHandle_t get_handle() {
    if (m_handle == nullptr) CHECK_CUBLAS(cublasCreate(&m_handle));
    return m_handle;
  }

  static void destroy_handle() {
    if (m_handle != nullptr) CHECK_CUBLAS(cublasDestroy(m_handle));
    m_handle = nullptr;
  }

  template <typename Dtype>
  static void axpy(const int n, Dtype alpha, const Dtype *x, Dtype *y);

  template <typename Dtype>
  static void axpy(const int n, Dtype alpha, const Dtype *x, const int incx,
                   Dtype *y, const int incy);

  template <typename Dtype>
  static void geam(cublasOperation_t transa, cublasOperation_t transb, int m,
                   int n, const Dtype *alpha, const Dtype *A, int lda,
                   const Dtype *beta, const Dtype *B, int ldb, Dtype *C,
                   int ldc);

  template <typename Dtype>
  static void gemm(cublasOperation_t transa, cublasOperation_t transb, int m,
                   int n, int k, const Dtype *alpha, const Dtype *A, int lda,
                   const Dtype *B, int ldb, const Dtype *beta, Dtype *C,
                   int ldc);
};

class cusolver {
 private:
  thread_local static cusolverDnHandle_t m_handle;
  cusolver();
  ~cusolver();

 public:
  static cusolverDnHandle_t get_handle() {
    if (m_handle == nullptr) CHECK_CUSOLVER(cusolverDnCreate(&m_handle));
    return m_handle;
  }

  static void destroy_handle() {
    if (m_handle != nullptr) CHECK_CUSOLVER(cusolverDnDestroy(m_handle));
    m_handle = nullptr;
  }

  template <typename Dtype>
  static void geqrf_bufferSize(int m, int n, Dtype *A, int lda, int *Lwork);
  template <typename Dtype>
  static void geqrf(int m, int n, Dtype *A, int lda, Dtype *TAU,
                    Dtype *Workspace, int Lwork, int *devInfo);

  template <typename Dtype>
  static void orgqr_bufferSize(int m, int n, int k, const Dtype *A, int lda,
                               const Dtype *tau, int *lwork);
  template <typename Dtype>
  static void orgqr(int m, int n, int k, Dtype *A, int lda, const Dtype *tau,
                    Dtype *work, int lwork, int *devInfo);
};
}  // namespace aion
