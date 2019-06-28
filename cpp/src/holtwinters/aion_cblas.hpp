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

extern "C" {
#include <cblas.h>
}

namespace aion {
class cblas;

class cblas {
 public:
    template <typename Dtype>
    static void copy(int n, const Dtype *x, int incx, Dtype *y, int incy);  // NOLINT

    template <typename Dtype>
    static Dtype dot(int n, const Dtype *x, int incx, const Dtype *y, int incy);

    template <typename Dtype>
    static void axpy(const int n, Dtype alpha, const Dtype* x, Dtype* y);

    template <typename Dtype>
    static void axpy(const int n, Dtype alpha, const Dtype* x, const int incx, Dtype* y, const int incy);

    template <typename Dtype>
    static void gemv(CBLAS_TRANSPOSE trans, int m, int n,
      Dtype alpha, const Dtype *a, int lda, const Dtype *x, int incx,
      Dtype beta, Dtype *y, int incy);

    template <typename Dtype>
    static void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
      int m, int n, int k,
      Dtype alpha, Dtype *a, int lda,
      Dtype *b, int ldb, Dtype beta,
      Dtype *c, int ldc);

    template <typename Dtype>
    static void ger(int m, int n,
      Dtype alpha, const Dtype *x, int incx,
      const Dtype *y, int incy, Dtype *a, int lda);
};

}  // namespace aion
