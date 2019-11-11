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

namespace MLCommon {
namespace Matrix {

/**
 * @brief CPU sequential version of the Kronecker product
 * 
 * @note All the matrices are in column-major order
 * 
 * @tparam      DataT  Type of the data
 * @param[out]  K      Pointer to the result of the Kronecker product A (x) B
 * @param[in]   A      Matrix A
 * @param[in]   B      Matrix B
 * @param[in]   m      Rows of matrix A
 * @param[in]   n      Columns of matrix B
 * @param[in]   p      Rows of matrix A
 * @param[in]   q      Columns of matrix B
 */
template <typename DataT>
void naiveKronecker(DataT *K, const DataT *A, const DataT *B, int m, int n,
                    int p, int q) {
  int k_m = m * p;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      DataT a_ij = A[i + m * j];
      for (int v = 0; v < p; v++) {
        for (int w = 0; w < q; w++) {
          DataT b_vw = B[v + p * w];
          K[i * p + v + (j * q + w) * k_m] = a_ij * b_vw;
        }
      }
    }
  }
}

/**
 * @brief CPU sequential matrix multiplication A*B
 * 
 * @note All the matrices are in column-major order
 * 
 * @tparam      DataT  Type of the data
 * @param[out]  out    Pointer to the result
 * @param[in]   A      Matrix A
 * @param[in]   B      Matrix B
 * @param[in]   m      Rows of A
 * @param[in]   k      Columns of A / rows of B
 * @param[in]   n      Columns of B
 */
template <typename DataT>
void naiveMatMul(DataT *out, const DataT *A, const DataT *B, int m, int k,
                 int n) {
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      DataT s = 0.0;
      for (int r = 0; r < k; r++) {
        s += A[i + r * m] * B[r + j * k];
      }
      out[i + j * m] = s;
    }
  }
}

/**
 * @brief CPU sequential vector add (u + alpha * v)
 * 
 * @tparam      DataT  Type of the data
 * @param[out]  out    Pointer to the result
 * @param[in]   u      Vector u
 * @param[in]   v      Vector v
 * @param[in]   len    Length of the vectors to add
 * @param[in]   alpha  Coefficient to multiply the elements of v with
 */
template <typename DataT>
void naiveAdd(DataT *out, const DataT *u, const DataT *v, int len,
              DataT alpha = 1.0) {
  for (int i = 0; i < len; i++) {
    out[i] = u[i] + alpha * v[i];
  }
}

/**
 * TODO: docs
 */
template <typename DataT>
void naiveLaggedMat(DataT *out, const DataT *in, int len, int lags) {
  int lagged_len = len - lags;
  for (int lag = 1; lag <= lags; lag++) {
    DataT *out_ = out + (lag - 1) * lagged_len;
    const DataT *in_ = in + lags - lag;
    for (int i = 0; i < lagged_len; i++) {
      out_[i] = in_[i];
    }
  }
}

}  // namespace Matrix
}  // namespace MLCommon
