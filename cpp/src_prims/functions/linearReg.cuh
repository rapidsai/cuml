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

#include <raft/linalg/cublas_wrappers.h>
#include <raft/linalg/transpose.h>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/add.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/mr/device/buffer.hpp>
#include <raft/stats/mean.cuh>
#include <raft/stats/sum.cuh>
#include <rmm/device_uvector.hpp>
#include "penalty.cuh"

namespace MLCommon {
namespace Functions {

template <typename math_t>
void linearRegH(const raft::handle_t &handle, const math_t *input, int n_rows,
                int n_cols, const math_t *coef, math_t *pred, math_t intercept,
                cudaStream_t stream) {
  raft::linalg::gemm(handle, input, n_rows, n_cols, coef, pred, n_rows, 1,
                     CUBLAS_OP_N, CUBLAS_OP_N, stream);

  if (intercept != math_t(0))
    raft::linalg::addScalar(pred, pred, intercept, n_rows, stream);
}

template <typename math_t>
void linearRegLossGrads(const raft::handle_t &handle, math_t *input, int n_rows,
                        int n_cols, const math_t *labels, const math_t *coef,
                        math_t *grads, penalty pen, math_t alpha,
                        math_t l1_ratio, cudaStream_t stream) {
  rmm::device_uvector<math_t> labels_pred(n_rows, stream);

  linearRegH(handle, input, n_rows, n_cols, coef, labels_pred.data(), math_t(0),
             stream);
  raft::linalg::subtract(labels_pred.data(), labels_pred.data(), labels, n_rows,
                         stream);
  raft::matrix::matrixVectorBinaryMult(input, labels_pred.data(), n_rows,
                                       n_cols, false, false, stream);

  raft::stats::mean(grads, input, n_cols, n_rows, false, false, stream);
  raft::linalg::scalarMultiply(grads, grads, math_t(2), n_cols, stream);

  rmm::device_uvector<math_t> pen_grads(0, stream);

  if (pen != penalty::NONE) pen_grads.resize(n_cols, stream);

  if (pen == penalty::L1) {
    lassoGrad(pen_grads.data(), coef, n_cols, alpha, stream);
  } else if (pen == penalty::L2) {
    ridgeGrad(pen_grads.data(), coef, n_cols, alpha, stream);
  } else if (pen == penalty::ELASTICNET) {
    elasticnetGrad(pen_grads.data(), coef, n_cols, alpha, l1_ratio, stream);
  }

  if (pen != penalty::NONE) {
    raft::linalg::add(grads, grads, pen_grads.data(), n_cols, stream);
  }
}

template <typename math_t>
void linearRegLoss(const raft::handle_t &handle, math_t *input, int n_rows,
                   int n_cols, const math_t *labels, const math_t *coef,
                   math_t *loss, penalty pen, math_t alpha, math_t l1_ratio,
                   cudaStream_t stream) {
  rmm::device_uvector<math_t> labels_pred(n_rows, stream);

  linearRegH(handle, input, n_rows, n_cols, coef, labels_pred.data(), math_t(0),
             stream);

  raft::linalg::subtract(labels_pred.data(), labels, labels_pred.data(), n_rows,
                         stream);
  raft::matrix::power(labels_pred.data(), n_rows, stream);
  raft::stats::mean(loss, labels_pred.data(), 1, n_rows, false, false, stream);

  rmm::device_uvector<math_t> pen_val(0, stream);

  if (pen != penalty::NONE) pen_val.resize(1, stream);

  if (pen == penalty::L1) {
    lasso(pen_val.data(), coef, n_cols, alpha, stream);
  } else if (pen == penalty::L2) {
    ridge(pen_val.data(), coef, n_cols, alpha, stream);
  } else if (pen == penalty::ELASTICNET) {
    elasticnet(pen_val.data(), coef, n_cols, alpha, l1_ratio, stream);
  }

  if (pen != penalty::NONE) {
    raft::linalg::add(loss, loss, pen_val.data(), 1, stream);
  }
}

};  // namespace Functions
};  // namespace MLCommon
// end namespace ML
