/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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
namespace LinAlg {
namespace Naive {

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
void kronecker(DataT* K, const DataT* A, const DataT* B, int m, int n, int p, int q)
{
  int k_m = m * p;
#pragma omp parallel for collapse(2)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      DataT a_ij = A[i + m * j];
      for (int v = 0; v < p; v++) {
        for (int w = 0; w < q; w++) {
          DataT b_vw                       = B[v + p * w];
          K[i * p + v + (j * q + w) * k_m] = a_ij * b_vw;
        }
      }
    }
  }
}

/**
 * @brief CPU sequential matrix multiplication out = alpha * A*B + beta * out
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
 * @param[in]   alpha  Scalar alpha
 * @param[in]   beta   Scalar beta
 */
template <typename DataT>
void matMul(
  DataT* out, const DataT* A, const DataT* B, int m, int k, int n, DataT alpha = 1, DataT beta = 0)
{
#pragma omp parallel for collapse(2)
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      DataT s = 0.0;
      for (int r = 0; r < k; r++) {
        s += A[i + r * m] * B[r + j * k];
      }
      out[i + j * m] = alpha * s + beta * out[i + j * m];
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
void add(DataT* out, const DataT* u, const DataT* v, int len, DataT alpha = 1.0)
{
#pragma omp parallel for
  for (int i = 0; i < len; i++) {
    out[i] = u[i] + alpha * v[i];
  }
}

/**
 * @brief CPU lagged matrix
 *
 * @tparam      DataT  Type of the data
 * @param[out]  out    Pointer to the result
 * @param[in]   in     Pointer to the input vector
 * @param[in]   len    Length or the vector
 * @param[in]   lags   Number of lags
 */
template <typename DataT>
void laggedMat(DataT* out, const DataT* in, int len, int lags)
{
  int lagged_len = len - lags;
#pragma omp parallel for
  for (int lag = 1; lag <= lags; lag++) {
    DataT* out_      = out + (lag - 1) * lagged_len;
    const DataT* in_ = in + lags - lag;
    for (int i = 0; i < lagged_len; i++) {
      out_[i] = in_[i];
    }
  }
}

/**
 * @brief CPU matrix 2D copy
 *
 * @tparam      DataT        Type of the data
 * @param[out]  out          Pointer to the result
 * @param[in]   in           Pointer to the input matrix
 * @param[in]   starting_row Starting row
 * @param[in]   starting_col Starting column
 * @param[in]   in_rows      Number of rows in the input matrix
 * @param[in]   out_rows     Number of rows in the output matrix
 * @param[in]   out_cols     Number of columns in the input matrix
 */
template <typename DataT>
void copy2D(DataT* out,
            const DataT* in,
            int starting_row,
            int starting_col,
            int in_rows,
            int out_rows,
            int out_cols)
{
#pragma omp parallel for collapse(2)
  for (int i = 0; i < out_rows; i++) {
    for (int j = 0; j < out_cols; j++) {
      out[i + j * out_rows] = in[starting_row + i + (starting_col + j) * in_rows];
    }
  }
}

/**
 * @brief CPU first difference of a vector
 *
 * @tparam      DataT        Type of the data
 * @param[out]  out          Pointer to the result
 * @param[in]   in           Pointer to the input vector
 * @param[in]   len          Length of the input vector
 */
template <typename DataT>
void diff(DataT* out, const DataT* in, int len)
{
#pragma omp parallel for
  for (int i = 0; i < len - 1; i++) {
    out[i] = in[i + 1] - in[i];
  }
}

}  // namespace Naive
}  // namespace LinAlg
}  // namespace MLCommon
