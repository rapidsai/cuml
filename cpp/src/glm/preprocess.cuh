/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
#include <raft/linalg/gemm.hpp>
#include <raft/linalg/norm.hpp>
#include <raft/matrix/math.hpp>
#include <raft/matrix/matrix.hpp>
#include <raft/stats/mean.hpp>
#include <raft/stats/mean_center.hpp>
#include <raft/stats/meanvar.hpp>
#include <raft/stats/stddev.hpp>
#include <raft/stats/weighted_mean.cuh>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace ML {
namespace GLM {

/**
 * @brief Center and scale the data, depending on the flags fit_intercept and normalize
 *
 * @tparam math_t the element type
 * @param [inout] input the column-major data of size [n_rows, n_cols]
 * @param [in] n_rows
 * @param [in] n_cols
 * @param [inout] labels vector of size [n_rows]
 * @param [out] intercept
 * @param [out] mu_input the column-wise means of the input of size [n_cols]
 * @param [out] mu_labels the scalar mean of the target (labels vector)
 * @param [out] norm2_input the column-wise standard deviations of the input of size [n_cols];
 *                          note, the biased estimator is used to match sklearn's StandardScaler
 *                          (dividing by n_rows, not by (n_rows - 1)).
 * @param [in] fit_intercept whether to center the data / to fit the intercept
 * @param [in] normalize whether to normalize the data
 * @param [in] stream
 */
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
                    cudaStream_t stream,
                    math_t* sample_weight = nullptr)
{
  raft::common::nvtx::range fun_scope("ML::GLM::preProcessData-%d-%d", n_rows, n_cols);
  ASSERT(n_cols > 0, "Parameter n_cols: number of columns cannot be less than one");
  ASSERT(n_rows > 1, "Parameter n_rows: number of rows cannot be less than two");

  if (fit_intercept) {
    if (normalize && sample_weight == nullptr) {
      raft::stats::meanvar(mu_input, norm2_input, input, n_cols, n_rows, false, false, stream);
      raft::linalg::unaryOp(
        norm2_input,
        norm2_input,
        n_cols,
        [] __device__(math_t v) { return raft::mySqrt(v); },
        stream);
      raft::matrix::linewiseOp(
        input,
        input,
        n_rows,
        n_cols,
        false,
        [] __device__(math_t x, math_t m, math_t s) { return s > 1e-10 ? (x - m) / s : 0; },
        stream,
        mu_input,
        norm2_input);
    } else {
      if (sample_weight != nullptr) {
        raft::stats::weightedMean(
          mu_input, input, sample_weight, n_cols, n_rows, false, false, stream);
      } else {
        raft::stats::mean(mu_input, input, n_cols, n_rows, false, false, stream);
      }
      raft::stats::meanCenter(input, input, mu_input, n_cols, n_rows, false, true, stream);
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

    if (sample_weight != nullptr) {
      raft::stats::weightedMean(mu_labels, labels, sample_weight, 1, n_rows, true, false, stream);
    } else {
      raft::stats::mean(mu_labels, labels, 1, n_rows, false, false, stream);
    }
    raft::stats::meanCenter(labels, labels, mu_labels, 1, n_rows, false, true, stream);
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
  raft::common::nvtx::range fun_scope("ML::GLM::postProcessData-%d-%d", n_rows, n_cols);
  ASSERT(n_cols > 0, "Parameter n_cols: number of columns cannot be less than one");
  ASSERT(n_rows > 1, "Parameter n_rows: number of rows cannot be less than two");

  cublasHandle_t cublas_handle = handle.get_cublas_handle();
  rmm::device_scalar<math_t> d_intercept(stream);

  if (normalize) {
    raft::matrix::matrixVectorBinaryDivSkipZero(
      coef, norm2_input, 1, n_cols, false, true, stream, true);
  }

  raft::linalg::gemm(
    handle, mu_input, 1, n_cols, coef, d_intercept.data(), 1, 1, CUBLAS_OP_N, CUBLAS_OP_N, stream);

  raft::linalg::subtract(d_intercept.data(), mu_labels, d_intercept.data(), 1, stream);
  *intercept = d_intercept.value(stream);

  if (normalize) {
    raft::matrix::linewiseOp(
      input,
      input,
      n_rows,
      n_cols,
      false,
      [] __device__(math_t x, math_t m, math_t s) { return s * x + m; },
      stream,
      mu_input,
      norm2_input);
  } else {
    raft::stats::meanAdd(input, input, mu_input, n_cols, n_rows, false, true, stream);
  }
  raft::stats::meanAdd(labels, labels, mu_labels, 1, n_rows, false, true, stream);
}

};  // namespace GLM
};  // namespace ML
// end namespace ML
