/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include "learning_rate.h"
#include "shuffle.h"

#include <cuml/solvers/params.hpp>

#include <raft/core/handle.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/gemv.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/mean_center.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <functions/hinge.cuh>
#include <functions/linearReg.cuh>
#include <functions/logisticReg.cuh>
#include <glm/preprocess.cuh>

namespace ML {
namespace Solver {

using namespace MLCommon;

/**
 * Fits a linear, lasso, and elastic-net regression model using Coordinate Descent solver
 * @param handle
 *        Reference of raft::handle_t
 * @param input
 *        pointer to an array in column-major format (size of n_rows, n_cols)
 * @param n_rows
 *        n_samples or rows in input
 * @param n_cols
 *        n_features or columns in X
 * @param labels
 *        pointer to an array for labels (size of n_rows)
 * @param coef
 *        pointer to an array for coefficients (size of n_cols). This will be filled with
 * coefficients once the function is executed.
 * @param intercept
 *        pointer to a scalar for intercept. This will be filled
 *        once the function is executed
 * @param fit_intercept
 *        boolean parameter to control if the intercept will be fitted or not
 * @param batch_size
 *        number of rows in the minibatch
 * @param epochs
 *        number of iterations that the solver will run
 * @param lr_type
 *        type of the learning rate function (i.e. OPTIMAL, CONSTANT, INVSCALING, ADAPTIVE)
 * @param eta0
 *        learning rate for constant lr_type. It's used to calculate learning rate function for
 * other types of lr_type
 * @param power_t
 *        power value in the INVSCALING lr_type
 * @param loss
 *        enum to use different loss functions.
 * @param penalty
 *        None, L1, L2, or Elastic-net penalty
 * @param alpha
 *        alpha value in L1
 * @param l1_ratio
 *        ratio of alpha will be used for L1. (1 - l1_ratio) * alpha will be used for L2.
 * @param shuffle
 *        boolean parameter to control whether coordinates will be picked randomly or not.
 * @param tol
 *        tolerance to stop the solver
 * @param n_iter_no_change
 *        solver stops if there is no update greater than tol after n_iter_no_change iterations
 * @param stream
 *        cuda stream
 */
template <typename math_t>
void sgdFit(const raft::handle_t& handle,
            math_t* input,
            int n_rows,
            int n_cols,
            math_t* labels,
            math_t* coef,
            math_t* intercept,
            bool fit_intercept,
            int batch_size,
            int epochs,
            ML::lr_type lr_type,
            math_t eta0,
            math_t power_t,
            ML::loss_funct loss,
            Functions::penalty penalty,
            math_t alpha,
            math_t l1_ratio,
            bool shuffle,
            math_t tol,
            int n_iter_no_change,
            cudaStream_t stream)
{
  ASSERT(n_cols > 0, "Parameter n_cols: number of columns cannot be less than one");
  ASSERT(n_rows > 1, "Parameter n_rows: number of rows cannot be less than two");

  cublasHandle_t cublas_handle = handle.get_cublas_handle();

  rmm::device_uvector<math_t> mu_input(0, stream);
  rmm::device_uvector<math_t> mu_labels(0, stream);
  rmm::device_uvector<math_t> norm2_input(0, stream);

  if (fit_intercept) {
    mu_input.resize(n_cols, stream);
    mu_labels.resize(1, stream);

    GLM::preProcessData(handle,
                        input,
                        n_rows,
                        n_cols,
                        labels,
                        intercept,
                        mu_input.data(),
                        mu_labels.data(),
                        norm2_input.data(),
                        fit_intercept,
                        false);
  }

  rmm::device_uvector<math_t> grads(n_cols, stream);
  rmm::device_uvector<int> indices(batch_size, stream);
  rmm::device_uvector<math_t> input_batch(batch_size * n_cols, stream);
  rmm::device_uvector<math_t> labels_batch(batch_size, stream);
  rmm::device_scalar<math_t> loss_value(stream);

  math_t prev_loss_value = math_t(0);
  math_t curr_loss_value = math_t(0);

  std::vector<int> rand_indices(n_rows);
  std::mt19937 g(rand());
  initShuffle(rand_indices, g);

  math_t t             = math_t(1);
  math_t learning_rate = math_t(0);
  if (lr_type == ML::lr_type::ADAPTIVE) {
    learning_rate = eta0;
  } else if (lr_type == ML::lr_type::OPTIMAL) {
    eta0 = calOptimalInit(alpha);
  }

  int n_iter_no_change_curr = 0;

  for (int i = 0; i < epochs; i++) {
    int cbs = 0;
    int j   = 0;

    if (i > 0 && shuffle) { Solver::shuffle(rand_indices, g); }

    while (j < n_rows) {
      if ((j + batch_size) > n_rows) {
        cbs = n_rows - j;
      } else {
        cbs = batch_size;
      }

      if (cbs == 0) break;

      raft::update_device(indices.data(), &rand_indices[j], cbs, stream);
      raft::matrix::copyRows(
        input, n_rows, n_cols, input_batch.data(), indices.data(), cbs, stream);
      raft::matrix::copyRows(labels, n_rows, 1, labels_batch.data(), indices.data(), cbs, stream);

      if (loss == ML::loss_funct::SQRD_LOSS) {
        Functions::linearRegLossGrads(handle,
                                      input_batch.data(),
                                      cbs,
                                      n_cols,
                                      labels_batch.data(),
                                      coef,
                                      grads.data(),
                                      penalty,
                                      alpha,
                                      l1_ratio,
                                      stream);
      } else if (loss == ML::loss_funct::LOG) {
        Functions::logisticRegLossGrads(handle,
                                        input_batch.data(),
                                        cbs,
                                        n_cols,
                                        labels_batch.data(),
                                        coef,
                                        grads.data(),
                                        penalty,
                                        alpha,
                                        l1_ratio,
                                        stream);
      } else if (loss == ML::loss_funct::HINGE) {
        Functions::hingeLossGrads(handle,
                                  input_batch.data(),
                                  cbs,
                                  n_cols,
                                  labels_batch.data(),
                                  coef,
                                  grads.data(),
                                  penalty,
                                  alpha,
                                  l1_ratio,
                                  stream);
      } else {
        ASSERT(false, "sgd.cuh: Other loss functions have not been implemented yet!");
      }

      if (lr_type != ML::lr_type::ADAPTIVE)
        learning_rate = calLearningRate(lr_type, eta0, power_t, alpha, t);

      raft::linalg::scalarMultiply(grads.data(), grads.data(), learning_rate, n_cols, stream);
      raft::linalg::subtract(coef, coef, grads.data(), n_cols, stream);

      j = j + cbs;
      t = t + 1;
    }

    if (tol > math_t(0)) {
      if (loss == ML::loss_funct::SQRD_LOSS) {
        Functions::linearRegLoss(handle,
                                 input,
                                 n_rows,
                                 n_cols,
                                 labels,
                                 coef,
                                 loss_value.data(),
                                 penalty,
                                 alpha,
                                 l1_ratio,
                                 stream);
      } else if (loss == ML::loss_funct::LOG) {
        Functions::logisticRegLoss(handle,
                                   input,
                                   n_rows,
                                   n_cols,
                                   labels,
                                   coef,
                                   loss_value.data(),
                                   penalty,
                                   alpha,
                                   l1_ratio,
                                   stream);
      } else if (loss == ML::loss_funct::HINGE) {
        Functions::hingeLoss(handle,
                             input,
                             n_rows,
                             n_cols,
                             labels,
                             coef,
                             loss_value.data(),
                             penalty,
                             alpha,
                             l1_ratio,
                             stream);
      }

      raft::update_host(&curr_loss_value, loss_value.data(), 1, stream);
      handle.sync_stream(stream);

      if (i > 0) {
        if (curr_loss_value > (prev_loss_value - tol)) {
          n_iter_no_change_curr = n_iter_no_change_curr + 1;
          if (n_iter_no_change_curr > n_iter_no_change) {
            if (lr_type == ML::lr_type::ADAPTIVE && learning_rate > math_t(1e-6)) {
              learning_rate         = learning_rate / math_t(5);
              n_iter_no_change_curr = 0;
            } else {
              break;
            }
          }
        } else {
          n_iter_no_change_curr = 0;
        }
      }

      prev_loss_value = curr_loss_value;
    }
  }

  if (fit_intercept) {
    GLM::postProcessData(handle,
                         input,
                         n_rows,
                         n_cols,
                         labels,
                         coef,
                         intercept,
                         mu_input.data(),
                         mu_labels.data(),
                         norm2_input.data(),
                         fit_intercept,
                         false);
  } else {
    *intercept = math_t(0);
  }
}

/**
 * Make predictions
 * @param handle
 *        Reference of raft::handle_t
 * @param input
 *        pointer to an array in column-major format (size of n_rows, n_cols)
 * @param n_rows
 *        n_samples or rows in input
 * @param n_cols
 *        n_features or columns in X
 * @param coef
 *        pointer to an array for coefficients (size of n_cols). Calculated in cdFit function.
 * @param intercept
 *        intercept value calculated in cdFit function
 * @param preds
 *        pointer to an array for predictions (size of n_rows). This will be fitted once functions
 * is executed.
 * @param loss
 *        enum to use different loss functions. Only linear regression loss functions is supported
 * right now.
 * @param stream
 *        cuda stream
 */
template <typename math_t>
void sgdPredict(const raft::handle_t& handle,
                const math_t* input,
                int n_rows,
                int n_cols,
                const math_t* coef,
                math_t intercept,
                math_t* preds,
                ML::loss_funct loss,
                cudaStream_t stream)
{
  ASSERT(n_cols > 0, "Parameter n_cols: number of columns cannot be less than one");
  ASSERT(n_rows > 1, "Parameter n_rows: number of rows cannot be less than two");

  if (loss == ML::loss_funct::SQRD_LOSS) {
    Functions::linearRegH(handle, input, n_rows, n_cols, coef, preds, intercept, stream);
  } else if (loss == ML::loss_funct::LOG) {
    Functions::logisticRegH(handle, input, n_rows, n_cols, coef, preds, intercept, stream);
  } else if (loss == ML::loss_funct::HINGE) {
    Functions::hingeH(handle, input, n_rows, n_cols, coef, preds, intercept, stream);
  }
}

/**
 * Make binary classifications
 * @param handle
 *        Reference of raft::handle_t
 * @param input
 *        pointer to an array in column-major format (size of n_rows, n_cols)
 * @param n_rows
 *        n_samples or rows in input
 * @param n_cols
 *        n_features or columns in X
 * @param coef
 *        pointer to an array for coefficients (size of n_cols). Calculated in cdFit function.
 * @param intercept
 *        intercept value calculated in cdFit function
 * @param preds
 *        pointer to an array for predictions (size of n_rows). This will be fitted once functions
 * is executed.
 * @param loss
 *        enum to use different loss functions. Only linear regression loss functions is supported
 * right now.
 * @param stream
 *        cuda stream
 */
template <typename math_t>
void sgdPredictBinaryClass(const raft::handle_t& handle,
                           const math_t* input,
                           int n_rows,
                           int n_cols,
                           const math_t* coef,
                           math_t intercept,
                           math_t* preds,
                           ML::loss_funct loss,
                           cudaStream_t stream)
{
  sgdPredict(handle, input, n_rows, n_cols, coef, intercept, preds, loss, stream);

  math_t scalar = math_t(1);
  if (loss == ML::loss_funct::SQRD_LOSS || loss == ML::loss_funct::LOG) {
    raft::linalg::unaryOp(
      preds,
      preds,
      n_rows,
      [scalar] __device__(math_t in) {
        if (in >= math_t(0.5))
          return math_t(1);
        else
          return math_t(0);
      },
      stream);
  } else if (loss == ML::loss_funct::HINGE) {
    raft::linalg::unaryOp(
      preds,
      preds,
      n_rows,
      [scalar] __device__(math_t in) {
        if (in >= math_t(0.0))
          return math_t(1);
        else
          return math_t(0);
      },
      stream);
  }
}

};  // namespace Solver
};  // namespace ML
// end namespace ML
