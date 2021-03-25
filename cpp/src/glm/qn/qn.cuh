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
#include <raft/matrix/math.cuh>
#include <rmm/device_uvector.hpp>
#include "glm_base.cuh"
#include "glm_linear.cuh"
#include "glm_logistic.cuh"
#include "glm_regularizer.cuh"
#include "glm_softmax.cuh"
#include "qn_solvers.cuh"

namespace ML {
namespace GLM {
template <typename T, typename LossFunction>
int qn_fit(const raft::handle_t &handle, LossFunction &loss, T *Xptr, T *yptr,
           T *zptr, int N, T l1, T l2, int max_iter, T grad_tol,
           int linesearch_max_iter, int lbfgs_memory, int verbosity,
           T *w0,  // initial value and result
           T *fx, int *num_iters, STORAGE_ORDER ordX, cudaStream_t stream) {
  LBFGSParam<T> opt_param;
  opt_param.epsilon = grad_tol;
  opt_param.max_iterations = max_iter;
  opt_param.m = lbfgs_memory;
  opt_param.max_linesearch = linesearch_max_iter;
  SimpleVec<T> w(w0, loss.n_param);

  // Scale the regularization strenght with the number of samples.
  l1 /= N;
  l2 /= N;

  if (l2 == 0) {
    GLMWithData<T, LossFunction> lossWith(&loss, Xptr, yptr, zptr, N, ordX);

    return qn_minimize(handle, w, fx, num_iters, lossWith, l1, opt_param,
                       stream, verbosity);

  } else {
    Tikhonov<T> reg(l2);
    RegularizedGLM<T, LossFunction, decltype(reg)> obj(&loss, &reg);
    GLMWithData<T, decltype(obj)> lossWith(&obj, Xptr, yptr, zptr, N, ordX);

    return qn_minimize(handle, w, fx, num_iters, lossWith, l1, opt_param,
                       stream, verbosity);
  }
}

template <typename T>
void qnFit(const raft::handle_t &handle, T *X, T *y, int N, int D, int C,
           bool fit_intercept, T l1, T l2, int max_iter, T grad_tol,
           int linesearch_max_iter, int lbfgs_memory, int verbosity, T *w0,
           T *f, int *num_iters, bool X_col_major, int loss_type,
           cudaStream_t stream, T *sample_weight = nullptr) {
  STORAGE_ORDER ord = X_col_major ? COL_MAJOR : ROW_MAJOR;
  int C_len = (loss_type == 0) ? (C - 1) : C;

  rmm::device_uvector<T> tmp(C_len * N, stream);
  SimpleMat<T> z(tmp.data(), C_len, N);

  switch (loss_type) {
    case 0: {
      ASSERT(C == 2, "qn.h: logistic loss invalid C");
      LogisticLoss<T> loss(handle, D, fit_intercept);
      if (sample_weight) loss.add_sample_weights(sample_weight, N, stream);
      qn_fit<T, decltype(loss)>(handle, loss, X, y, z.data, N, l1, l2, max_iter,
                                grad_tol, linesearch_max_iter, lbfgs_memory,
                                verbosity, w0, f, num_iters, ord, stream);
    } break;
    case 1: {
      ASSERT(C == 1, "qn.h: squared loss invalid C");
      SquaredLoss<T> loss(handle, D, fit_intercept);
      if (sample_weight) loss.add_sample_weights(sample_weight, N, stream);
      qn_fit<T, decltype(loss)>(handle, loss, X, y, z.data, N, l1, l2, max_iter,
                                grad_tol, linesearch_max_iter, lbfgs_memory,
                                verbosity, w0, f, num_iters, ord, stream);
    } break;
    case 2: {
      ASSERT(C > 2, "qn.h: softmax invalid C");
      Softmax<T> loss(handle, D, C, fit_intercept);
      if (sample_weight) loss.add_sample_weights(sample_weight, N, stream);
      qn_fit<T, decltype(loss)>(handle, loss, X, y, z.data, N, l1, l2, max_iter,
                                grad_tol, linesearch_max_iter, lbfgs_memory,
                                verbosity, w0, f, num_iters, ord, stream);
    } break;
    default: {
      ASSERT(false, "qn.h: unknown loss function.");
    }
  }
}

template <typename T>
void qnDecisionFunction(const raft::handle_t &handle, T *Xptr, int N, int D,
                        int C, bool fit_intercept, T *params, bool X_col_major,
                        int loss_type, T *scores, cudaStream_t stream) {
  // NOTE: While gtests pass X as row-major, and python API passes X as
  // col-major, no extensive testing has been done to ensure that
  // this function works correctly for both input types

  STORAGE_ORDER ordX = X_col_major ? COL_MAJOR : ROW_MAJOR;
  int C_len = (loss_type == 0) ? (C - 1) : C;

  GLMDims dims(C_len, D, fit_intercept);

  SimpleMat<T> X(Xptr, N, D, ordX);

  SimpleMat<T> W(params, C_len, dims.dims);
  SimpleMat<T> Z(scores, C_len, N);
  linearFwd(handle, Z, X, W, stream);
}

template <typename T>
void qnPredict(const raft::handle_t &handle, T *Xptr, int N, int D, int C,
               bool fit_intercept, T *params, bool X_col_major, int loss_type,
               T *preds, cudaStream_t stream) {
  int C_len = (loss_type == 0) ? (C - 1) : C;
  rmm::device_uvector<T> scores(C_len * N, stream);
  qnDecisionFunction<T>(handle, Xptr, N, D, C, fit_intercept, params,
                        X_col_major, loss_type, scores.data(), stream);
  SimpleMat<T> Z(scores.data(), C_len, N);
  SimpleMat<T> P(preds, 1, N);

  switch (loss_type) {
    case 0: {
      ASSERT(C == 2, "qn.h: logistic loss invalid C");
      auto thresh = [] __device__(const T z) {
        if (z > 0.0) return T(1);
        return T(0);
      };
      P.assign_unary(Z, thresh, stream);
    } break;
    case 1: {
      ASSERT(C == 1, "qn.h: squared loss invalid C");
      P.copy_async(Z, stream);
    } break;
    case 2: {
      ASSERT(C > 2, "qn.h: softmax invalid C");
      raft::matrix::argmax(Z.data, C, N, preds, stream);
    } break;
    default: {
      ASSERT(false, "qn.h: unknown loss function.");
    }
  }
}

};  // namespace GLM
};  // namespace ML
