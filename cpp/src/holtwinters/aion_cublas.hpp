/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
#include "macros.hpp"

namespace aion {
class cublas;

class cublas {
 private:
    thread_local static cublasHandle_t m_handle;
    cublas();
    ~cublas();

 public:
    // Get the handle.
    static cublasHandle_t get_handle() {
      if (m_handle == nullptr)
        CHECK_CUBLAS(cublasCreate(&m_handle));
      return m_handle;
    }

    static void destroy_handle() {
      if (m_handle != nullptr)
        CHECK_CUBLAS(cublasDestroy(m_handle));
      m_handle = nullptr;
    }

    template <typename Dtype>
    static void axpy(const int n, Dtype alpha, const Dtype* x, Dtype* y);

    template <typename Dtype>
    static void axpy(const int n, Dtype alpha, const Dtype* x,
        const int incx, Dtype* y, const int incy);

    template <typename Dtype>
    static void geam(cublasOperation_t transa, cublasOperation_t transb,
      int m, int n,
      const Dtype *alpha, const Dtype *A, int lda,
      const Dtype *beta, const Dtype *B, int ldb,
      Dtype *C, int ldc);

    template <typename Dtype>
    static void gemm(cublasOperation_t transa, cublasOperation_t transb,
      int m, int n, int k,
      const Dtype *alpha, const Dtype *A, int lda,
      const Dtype *B, int ldb,
      const Dtype *beta, Dtype *C, int ldc);
};

}  // namespace aion
