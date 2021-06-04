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

#include <raft/linalg/gemv.h>
#include <linalg/lstsq.cuh>
#include <raft/linalg/add.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/mean_center.cuh>
#include <raft/stats/stddev.cuh>
#include <raft/stats/sum.cuh>
#include <rmm/device_uvector.hpp>
#include "preprocess.cuh"

namespace ML {
namespace GLM {

using namespace MLCommon;

/**
 * @brief fit an ordinary least squares model
 * @param handle        cuml handle
 * @param input         device pointer to feature matrix n_rows x n_cols
 * @param n_rows        number of rows of the feature matrix
 * @param n_cols        number of columns of the feature matrix
 * @param labels        device pointer to label vector of length n_rows
 * @param coef          device pointer to hold the solution for weights of size n_cols
 * @param intercept     device pointer to hold the solution for bias term of size 1
 * @param fit_intercept if true, fit intercept
 * @param normalize     if true, normalize data to zero mean, unit variance
 * @param stream        cuda stream
 * @param algo          specifies which solver to use (0: SVD, 1: Eigendecomposition, 2: QR-decomposition)
 */
template <typename math_t>
void olsFit(const raft::handle_t &handle, math_t *input, int n_rows, int n_cols,
            math_t *labels, math_t *coef, math_t *intercept, bool fit_intercept,
            bool normalize, cudaStream_t stream, int algo = 0) {
  auto cublas_handle = handle.get_cublas_handle();
  auto cusolver_handle = handle.get_cusolver_dn_handle();
  auto allocator = handle.get_device_allocator();

  ASSERT(n_cols > 0, "olsFit: number of columns cannot be less than one");
  ASSERT(n_rows > 1, "olsFit: number of rows cannot be less than two");

  rmm::device_uvector<math_t> mu_input(0, stream);
  rmm::device_uvector<math_t> norm2_input(0, stream);
  rmm::device_uvector<math_t> mu_labels(0, stream);

  if (fit_intercept) {
    mu_input.resize(n_cols, stream);
    mu_labels.resize(1, stream);
    if (normalize) {
      norm2_input.resize(n_cols, stream);
    }
    preProcessData(handle, input, n_rows, n_cols, labels, intercept,
                   mu_input.data(), mu_labels.data(), norm2_input.data(),
                   fit_intercept, normalize, stream);
  }

  if (algo == 0 || algo == 1) {
    LinAlg::lstsq(handle, input, n_rows, n_cols, labels, coef, algo, stream);
  } else if (algo == 2) {
    LinAlg::lstsqQR(input, n_rows, n_cols, labels, coef, cusolver_handle,
                    cublas_handle, allocator, stream);
  } else if (algo == 3) {
    ASSERT(false, "olsFit: no algorithm with this id has been implemented");
  } else {
    ASSERT(false, "olsFit: no algorithm with this id has been implemented");
  }

  if (fit_intercept) {
    postProcessData(handle, input, n_rows, n_cols, labels, coef, intercept,
                    mu_input.data(), mu_labels.data(), norm2_input.data(),
                    fit_intercept, normalize, stream);
  } else {
    *intercept = math_t(0);
  }
}

/**
 * @brief to make predictions with a fitted ordinary least squares and ridge regression model
 * @param handle        cuml ahndle
 * @param input         device pointer to feature matrix n_rows x n_cols
 * @param n_rows        number of rows of the feature matrix
 * @param n_cols        number of columns of the feature matrix
 * @param coef          coefficients of the model
 * @param intercept     bias term of the model
 * @param preds         device pointer to store predictions of size n_rows
 * @param stream        cuda stream
 */
template <typename math_t>
void gemmPredict(const raft::handle_t &handle, const math_t *input, int n_rows,
                 int n_cols, const math_t *coef, math_t intercept,
                 math_t *preds, cudaStream_t stream) {
  ASSERT(n_cols > 0, "gemmPredict: number of columns cannot be less than one");
  ASSERT(n_rows > 0, "gemmPredict: number of rows cannot be less than one");

  math_t alpha = math_t(1);
  math_t beta = math_t(0);
  raft::linalg::gemm(handle, input, n_rows, n_cols, coef, preds, n_rows, 1,
                     CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta, stream);

  if (intercept != math_t(0))
    raft::linalg::addScalar(preds, preds, intercept, n_rows, stream);
}

};  // namespace GLM
};  // namespace ML
// end namespace ML
