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

#include <thrust/device_vector.h>
#include "cublas_wrappers.h"

namespace MLCommon {
namespace LinAlg {

/**
 * @defgroup transpose on the column major input matrix using Jacobi method
 * @param in: input matrix
 * @param out: output. Transposed input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @{
 */
template <typename math_t>
void transpose(math_t *in, math_t *out, int n_rows, int n_cols,
               cublasHandle_t cublas_h, cudaStream_t stream) {
  int out_n_rows = n_cols;
  int out_n_cols = n_rows;

  const math_t alpha = 1.0;
  const math_t beta = 0.0;
  CUBLAS_CHECK(cublasgeam(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N, out_n_rows,
                          out_n_cols, &alpha, in, n_rows, &beta, out,
                          out_n_rows, out, out_n_rows, stream));
}

/**
 * @defgroup transpose on the column major input matrix using Jacobi method
 * @param inout: input and output matrix
 * @param n: number of rows and columns of input matrix
 * @{
 */
template <typename math_t>
void transpose(math_t *inout, int n, cudaStream_t stream) {
  auto m = n;
  auto size = n * n;
  auto d_inout = inout;
  auto counting = thrust::make_counting_iterator<int>(0);

  thrust::for_each(counting, counting + size, [=] __device__(int idx) {
    int s_row = idx % m;
    int s_col = idx / m;
    int d_row = s_col;
    int d_col = s_row;
    if (s_row < s_col) {
      auto temp = d_inout[d_col * m + d_row];
      d_inout[d_col * m + d_row] = d_inout[s_col * m + s_row];
      d_inout[s_col * m + s_row] = temp;
    }
  });
}

};  // end namespace LinAlg
};  // end namespace MLCommon
