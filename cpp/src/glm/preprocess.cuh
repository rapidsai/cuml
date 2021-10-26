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

#include <raft/cudart_utils.h>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/matrix/math.hpp>
#include <raft/matrix/matrix.hpp>
#include <raft/stats/mean.hpp>
#include <raft/stats/mean_center.hpp>
#include <raft/stats/stddev.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace ML {
namespace GLM {

using namespace MLCommon;

template <typename math_t>
void preProcessData(const raft::handle_t& handle,
                    math_t* input,
                    int n_rows,
                    int n_cols,
                    math_t* labels,
                    math_t* intercept,
                    math_t* mu_input,
                    math_t* mu_labels,
                    math_t* norm2_input,
                    bool fit_intercept,
                    bool normalize,
                    cudaStream_t stream)
{
  ASSERT(n_cols > 0, "Parameter n_cols: number of columns cannot be less than one");
  ASSERT(n_rows > 1, "Parameter n_rows: number of rows cannot be less than two");

  if (fit_intercept) {
    raft::stats::mean(mu_input, input, n_cols, n_rows, false, false, stream);
    raft::stats::meanCenter(input, input, mu_input, n_cols, n_rows, false, true, stream);

    raft::stats::mean(mu_labels, labels, 1, n_rows, false, false, stream);
    raft::stats::meanCenter(labels, labels, mu_labels, 1, n_rows, false, true, stream);

    if (normalize) {
      raft::linalg::colNorm(norm2_input,
                            input,
                            n_cols,
                            n_rows,
                            raft::linalg::L2Norm,
                            false,
                            stream,
                            [] __device__(math_t v) { return raft::mySqrt(v); });
      raft::matrix::matrixVectorBinaryDivSkipZero(
        input, norm2_input, n_rows, n_cols, false, true, stream, true);
    }
  }
}

template <typename math_t>
void postProcessData(const raft::handle_t& handle,
                     math_t* input,
                     int n_rows,
                     int n_cols,
                     math_t* labels,
                     math_t* coef,
                     math_t* intercept,
                     math_t* mu_input,
                     math_t* mu_labels,
                     math_t* norm2_input,
                     bool fit_intercept,
                     bool normalize,
                     cudaStream_t stream)
{
  ASSERT(n_cols > 0, "Parameter n_cols: number of columns cannot be less than one");
  ASSERT(n_rows > 1, "Parameter n_rows: number of rows cannot be less than two");

  cublasHandle_t cublas_handle = handle.get_cublas_handle();
  rmm::device_scalar<math_t> d_intercept(stream);

  if (normalize) {
    raft::matrix::matrixVectorBinaryMult(input, norm2_input, n_rows, n_cols, false, true, stream);
    raft::matrix::matrixVectorBinaryDivSkipZero(
      coef, norm2_input, 1, n_cols, false, true, stream, true);
  }

  raft::linalg::gemm(
    handle, mu_input, 1, n_cols, coef, d_intercept.data(), 1, 1, CUBLAS_OP_N, CUBLAS_OP_N, stream);

  raft::linalg::subtract(d_intercept.data(), mu_labels, d_intercept.data(), 1, stream);
  *intercept = d_intercept.value(stream);

  raft::stats::meanAdd(input, input, mu_input, n_cols, n_rows, false, true, stream);
  raft::stats::meanAdd(labels, labels, mu_labels, 1, n_rows, false, true, stream);
}

};  // namespace GLM
};  // namespace ML
// end namespace ML
