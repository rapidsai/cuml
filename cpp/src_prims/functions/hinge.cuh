/*
 * Copyright (c) 2018-2025, NVIDIA CORPORATION.
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

#include "penalty.cuh"

#include <raft/core/handle.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/sum.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

namespace MLCommon {
namespace Functions {

template <typename math_t, typename idx_type = int>
void hingeLossGradMult(math_t* data,
                       const math_t* vec1,
                       const math_t* vec2,
                       idx_type n_row,
                       idx_type n_col,
                       cudaStream_t stream)
{
  raft::linalg::matrixVectorOp<false, false>(
    data,
    data,
    vec1,
    vec2,
    n_col,
    n_row,
    [] __device__(math_t a, math_t b, math_t c) {
      if (c < math_t(1))
        return -a * b;
      else
        return math_t(0);
    },
    stream);
}

template <typename math_t, typename idx_type = int>
void hingeLossSubtract(
  math_t* out, const math_t* in, math_t scalar, idx_type len, cudaStream_t stream)
{
  raft::linalg::unaryOp(
    out,
    in,
    len,
    [scalar] __device__(math_t in) {
      if (in < scalar)
        return math_t(1) - in;
      else
        return math_t(0);
    },
    stream);
}

template <typename math_t, typename idx_type = int>
void hingeH(const raft::handle_t& handle,
            const math_t* input,
            idx_type n_rows,
            idx_type n_cols,
            const math_t* coef,
            math_t* pred,
            math_t intercept,
            cudaStream_t stream)
{
  raft::linalg::gemm(
    handle, input, n_rows, n_cols, coef, pred, n_rows, 1, CUBLAS_OP_N, CUBLAS_OP_N, stream);

  if (intercept != math_t(0)) raft::linalg::addScalar(pred, pred, intercept, n_rows, stream);

  sign(pred, pred, math_t(1.0), n_rows, stream);
}

template <typename math_t>
void hingeLossGrads(const raft::handle_t& handle,
                    math_t* input,
                    int n_rows,
                    int n_cols,
                    const math_t* labels,
                    const math_t* coef,
                    math_t* grads,
                    penalty pen,
                    math_t alpha,
                    math_t l1_ratio,
                    cudaStream_t stream)
{
  rmm::device_uvector<math_t> labels_pred(n_rows, stream);

  raft::linalg::gemm(handle,
                     input,
                     n_rows,
                     n_cols,
                     coef,
                     labels_pred.data(),
                     n_rows,
                     1,
                     CUBLAS_OP_N,
                     CUBLAS_OP_N,
                     stream);

  raft::linalg::eltwiseMultiply(labels_pred.data(), labels_pred.data(), labels, n_rows, stream);
  hingeLossGradMult(input, labels, labels_pred.data(), n_rows, n_cols, stream);
  raft::stats::mean<false>(grads, input, n_cols, n_rows, false, stream);

  rmm::device_uvector<math_t> pen_grads(0, stream);

  if (pen != penalty::NONE) pen_grads.resize(n_cols, stream);

  if (pen == penalty::L1) {
    lassoGrad(pen_grads.data(), coef, n_cols, alpha, stream);
  } else if (pen == penalty::L2) {
    ridgeGrad(pen_grads.data(), coef, n_cols, alpha, stream);
  } else if (pen == penalty::ELASTICNET) {
    elasticnetGrad(pen_grads.data(), coef, n_cols, alpha, l1_ratio, stream);
  }

  if (pen != penalty::NONE) { raft::linalg::add(grads, grads, pen_grads.data(), n_cols, stream); }
}

template <typename math_t>
void hingeLoss(const raft::handle_t& handle,
               math_t* input,
               int n_rows,
               int n_cols,
               const math_t* labels,
               const math_t* coef,
               math_t* loss,
               penalty pen,
               math_t alpha,
               math_t l1_ratio,
               cudaStream_t stream)
{
  rmm::device_uvector<math_t> labels_pred(n_rows, stream);

  raft::linalg::gemm(handle,
                     input,
                     n_rows,
                     n_cols,
                     coef,
                     labels_pred.data(),
                     n_rows,
                     1,
                     CUBLAS_OP_N,
                     CUBLAS_OP_N,
                     stream);

  raft::linalg::eltwiseMultiply(labels_pred.data(), labels_pred.data(), labels, n_rows, stream);

  hingeLossSubtract(labels_pred.data(), labels_pred.data(), math_t(1), n_rows, stream);

  raft::stats::sum<false>(loss, labels_pred.data(), 1, n_rows, stream);

  rmm::device_uvector<math_t> pen_val(0, stream);

  if (pen != penalty::NONE) pen_val.resize(1, stream);

  if (pen == penalty::L1) {
    lasso(pen_val.data(), coef, n_cols, alpha, stream);
  } else if (pen == penalty::L2) {
    ridge(pen_val.data(), coef, n_cols, alpha, stream);
  } else if (pen == penalty::ELASTICNET) {
    elasticnet(pen_val.data(), coef, n_cols, alpha, l1_ratio, stream);
  }

  if (pen != penalty::NONE) { raft::linalg::add(loss, loss, pen_val.data(), 1, stream); }
}

};  // namespace Functions
};  // namespace MLCommon
// end namespace ML
