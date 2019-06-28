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

#include <cusolverDn.h>
#include "macros.hpp"

namespace aion {
class cusolver;

class cusolver {
 private:
    thread_local static cusolverDnHandle_t m_handle;
    cusolver();
    ~cusolver();

 public:
    // Get the handle.
    static cusolverDnHandle_t get_handle() {
      if (m_handle == nullptr)
        CHECK_CUSOLVER(cusolverDnCreate(&m_handle));
      return m_handle;
    }

    static void destroy_handle() {
      if (m_handle != nullptr)
        CHECK_CUSOLVER(cusolverDnDestroy(m_handle));
      m_handle = nullptr;
    }

    template <typename Dtype>
    static void geqrf_bufferSize(int m, int n, Dtype *A, int lda, int *Lwork);
    template <typename Dtype>
    static void geqrf(int m, int n, Dtype *A, int lda, Dtype *TAU, Dtype *Workspace, int Lwork, int *devInfo);

    template <typename Dtype>
    static void orgqr_bufferSize(int m, int n, int k, const Dtype *A, int lda, const Dtype *tau, int *lwork);
    template <typename Dtype>
    static void orgqr(int m, int n, int k, Dtype *A, int lda, const Dtype *tau, Dtype *work, int lwork, int *devInfo);
};

}  // namespace aion
