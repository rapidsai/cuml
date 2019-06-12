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

#include <cuda_utils.h>
#include <linalg/add.h>
#include <linalg/binary_op.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/eltwise.h>
#include <linalg/gemm.h>
#include <linalg/subtract.h>
#include <linalg/transpose.h>
#include <matrix/math.h>
#include <matrix/matrix.h>
#include <stats/mean.h>
#include <stats/sum.h>
#include "penalty.h"
#include "sigmoid.h"

namespace MLCommon {
namespace Functions {

template <typename math_t>
void logisticRegH(const math_t *input, int n_rows, int n_cols,
                  const math_t *coef, math_t *pred, math_t intercept,
                  cublasHandle_t cublas_handle, cudaStream_t stream) {
  LinAlg::gemm(input, n_rows, n_cols, coef, pred, n_rows, 1, CUBLAS_OP_N,
               CUBLAS_OP_N, cublas_handle, stream);

  if (intercept != math_t(0))
    LinAlg::addScalar(pred, pred, intercept, n_rows, stream);

  sigmoid(pred, pred, n_rows, stream);
}

template <typename math_t>
void logisticRegLossGrads(math_t *input, int n_rows, int n_cols,
                          const math_t *labels, const math_t *coef,
                          math_t *grads, penalty pen, math_t alpha,
                          math_t l1_ratio, cublasHandle_t cublas_handle,
                          std::shared_ptr<deviceAllocator> allocator,
                          cudaStream_t stream) {
  device_buffer<math_t> labels_pred(allocator, stream, n_rows);

  logisticRegH(input, n_rows, n_cols, coef, labels_pred.data(), math_t(0),
               cublas_handle, stream);
  LinAlg::subtract(labels_pred.data(), labels_pred.data(), labels, n_rows,
                   stream);
  Matrix::matrixVectorBinaryMult(input, labels_pred.data(), n_rows, n_cols,
                                 false, false, stream);

  Stats::mean(grads, input, n_cols, n_rows, false, false, stream);

  device_buffer<math_t> pen_grads(allocator, stream, 0);

  if (pen != penalty::NONE) pen_grads.resize(n_cols, stream);

  if (pen == penalty::L1) {
    lassoGrad(pen_grads.data(), coef, n_cols, alpha, stream);
  } else if (pen == penalty::L2) {
    ridgeGrad(pen_grads.data(), coef, n_cols, alpha, stream);
  } else if (pen == penalty::ELASTICNET) {
    elasticnetGrad(pen_grads.data(), coef, n_cols, alpha, l1_ratio, stream);
  }

  if (pen != penalty::NONE) {
    LinAlg::add(grads, grads, pen_grads.data(), n_cols, stream);
  }
}

template <typename T>
void logLoss(T *out, T *label, T *label_pred, int len, cudaStream_t stream);

template <>
inline void logLoss(float *out, float *label, float *label_pred, int len,
                    cudaStream_t stream) {
  LinAlg::binaryOp(
    out, label, label_pred, len,
    [] __device__(float y, float y_pred) {
      return -y * logf(y_pred) - (1 - y) * logf(1 - y_pred);
    },
    stream);
}

template <>
inline void logLoss(double *out, double *label, double *label_pred, int len,
                    cudaStream_t stream) {
  LinAlg::binaryOp(
    out, label, label_pred, len,
    [] __device__(double y, double y_pred) {
      return -y * log(y_pred) - (1 - y) * logf(1 - y_pred);
    },
    stream);
}

template <typename math_t>
void logisticRegLoss(math_t *input, int n_rows, int n_cols, math_t *labels,
                     const math_t *coef, math_t *loss, penalty pen,
                     math_t alpha, math_t l1_ratio,
                     cublasHandle_t cublas_handle,
                     std::shared_ptr<deviceAllocator> allocator,
                     cudaStream_t stream) {
  device_buffer<math_t> labels_pred(allocator, stream, n_rows);

  logisticRegH(input, n_rows, n_cols, coef, labels_pred.data(), math_t(0),
               cublas_handle, stream);
  logLoss(labels_pred.data(), labels, labels_pred.data(), n_rows, stream);

  Stats::mean(loss, labels_pred.data(), 1, n_rows, false, false, stream);

  device_buffer<math_t> pen_val(allocator, stream, 0);

  if (pen != penalty::NONE) pen_val.resize(1, stream);

  if (pen == penalty::L1) {
    lasso(pen_val.data(), coef, n_cols, alpha, stream);
  } else if (pen == penalty::L2) {
    ridge(pen_val.data(), coef, n_cols, alpha, stream);
  } else if (pen == penalty::ELASTICNET) {
    elasticnet(pen_val.data(), coef, n_cols, alpha, l1_ratio, stream);
  }

  if (pen != penalty::NONE) {
    LinAlg::add(loss, loss, pen_val.data(), 1, stream);
  }
}

/** @} */
};  // namespace Functions
};  // namespace MLCommon
// end namespace ML
