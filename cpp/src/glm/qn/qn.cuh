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

#include "glm_base.cuh"
#include "glm_linear.cuh"
#include "glm_logistic.cuh"
#include "glm_regularizer.cuh"
#include "glm_softmax.cuh"
#include "glm_svm.cuh"
#include "qn_solvers.cuh"
#include "qn_util.cuh"

#include <cuml/linear_model/qn.h>

#include <raft/matrix/math.hpp>
#include <rmm/device_uvector.hpp>

namespace ML {
namespace GLM {

template <typename T, typename LossFunction>
int qn_fit(const raft::handle_t& handle,
           const qn_params& pams,
           LossFunction& loss,
           const SimpleMat<T>& X,
           const SimpleVec<T>& y,
           SimpleDenseMat<T>& Z,
           T* w0_data,  // initial value and result
           T* fx,
           int* num_iters,
           cudaStream_t stream)
{
  LBFGSParam<T> opt_param(pams);
  SimpleVec<T> w0(w0_data, loss.n_param);

  // Scale the regularization strenght with the number of samples.
  T l1 = pams.penalty_l1;
  T l2 = pams.penalty_l2;
  if (pams.penalty_normalized) {
    l1 /= X.m;
    l2 /= X.m;
  }

  if (l2 == 0) {
    GLMWithData<T, LossFunction> lossWith(&loss, X, y, Z);

    return qn_minimize(handle, w0, fx, num_iters, lossWith, l1, opt_param, stream, pams.verbose);

  } else {
    Tikhonov<T> reg(l2);
    RegularizedGLM<T, LossFunction, decltype(reg)> obj(&loss, &reg);
    GLMWithData<T, decltype(obj)> lossWith(&obj, X, y, Z);

    return qn_minimize(handle, w0, fx, num_iters, lossWith, l1, opt_param, stream, pams.verbose);
  }
}

template <typename T>
inline void qn_fit_x(const raft::handle_t& handle,
                     const qn_params& pams,
                     SimpleMat<T>& X,
                     T* y_data,
                     int C,
                     T* w0_data,
                     T* f,
                     int* num_iters,
                     cudaStream_t stream,
                     T* sample_weight = nullptr,
                     T svr_eps        = 0)
{
  /*
   NB:
    N - number of data rows
    D - number of data columns (features)
    C - number of output classes

    X in R^[N, D]
    w in R^[D, C]
    y in {0, 1}^[N, C] or {cat}^N

    Dimensionality of w0 depends on loss, so we initialize it later.
   */
  int N         = X.m;
  int D         = X.n;
  int n_targets = qn_is_classification(pams.loss) && C == 2 ? 1 : C;
  rmm::device_uvector<T> tmp(n_targets * N, stream);
  SimpleDenseMat<T> Z(tmp.data(), n_targets, N);
  SimpleVec<T> y(y_data, N);

  switch (pams.loss) {
    case QN_LOSS_LOGISTIC: {
      ASSERT(C == 2, "qn.h: logistic loss invalid C");
      LogisticLoss<T> loss(handle, D, pams.fit_intercept);
      if (sample_weight) loss.add_sample_weights(sample_weight, N, stream);
      qn_fit<T, decltype(loss)>(handle, pams, loss, X, y, Z, w0_data, f, num_iters, stream);
    } break;
    case QN_LOSS_SQUARED: {
      ASSERT(C == 1, "qn.h: squared loss invalid C");
      SquaredLoss<T> loss(handle, D, pams.fit_intercept);
      if (sample_weight) loss.add_sample_weights(sample_weight, N, stream);
      qn_fit<T, decltype(loss)>(handle, pams, loss, X, y, Z, w0_data, f, num_iters, stream);
    } break;
    case QN_LOSS_SOFTMAX: {
      ASSERT(C > 2, "qn.h: softmax invalid C");
      Softmax<T> loss(handle, D, C, pams.fit_intercept);
      if (sample_weight) loss.add_sample_weights(sample_weight, N, stream);
      qn_fit<T, decltype(loss)>(handle, pams, loss, X, y, Z, w0_data, f, num_iters, stream);
    } break;
    case QN_LOSS_SVC_L1: {
      ASSERT(C == 2, "qn.h: SVC-L1 loss invalid C");
      SVCL1Loss<T> loss(handle, D, pams.fit_intercept);
      if (sample_weight) loss.add_sample_weights(sample_weight, N, stream);
      qn_fit<T, decltype(loss)>(handle, pams, loss, X, y, Z, w0_data, f, num_iters, stream);
    } break;
    case QN_LOSS_SVC_L2: {
      ASSERT(C == 2, "qn.h: SVC-L2 loss invalid C");
      SVCL2Loss<T> loss(handle, D, pams.fit_intercept);
      if (sample_weight) loss.add_sample_weights(sample_weight, N, stream);
      qn_fit<T, decltype(loss)>(handle, pams, loss, X, y, Z, w0_data, f, num_iters, stream);
    } break;
    case QN_LOSS_SVR_L1: {
      ASSERT(C == 1, "qn.h: SVR-L1 loss invalid C");
      SVRL1Loss<T> loss(handle, D, pams.fit_intercept, svr_eps);
      if (sample_weight) loss.add_sample_weights(sample_weight, N, stream);
      qn_fit<T, decltype(loss)>(handle, pams, loss, X, y, Z, w0_data, f, num_iters, stream);
    } break;
    case QN_LOSS_SVR_L2: {
      ASSERT(C == 1, "qn.h: SVR-L2 loss invalid C");
      SVRL2Loss<T> loss(handle, D, pams.fit_intercept, svr_eps);
      if (sample_weight) loss.add_sample_weights(sample_weight, N, stream);
      qn_fit<T, decltype(loss)>(handle, pams, loss, X, y, Z, w0_data, f, num_iters, stream);
    } break;
    case QN_LOSS_ABS: {
      ASSERT(C == 1, "qn.h: abs loss (L1) invalid C");
      AbsLoss<T> loss(handle, D, pams.fit_intercept);
      if (sample_weight) loss.add_sample_weights(sample_weight, N, stream);
      qn_fit<T, decltype(loss)>(handle, pams, loss, X, y, Z, w0_data, f, num_iters, stream);
    } break;
    default: {
      ASSERT(false, "qn.h: unknown loss function type (id = %d).", pams.loss);
    }
  }
}

template <typename T>
void qnFit(const raft::handle_t& handle,
           const qn_params& pams,
           T* X_data,
           bool X_col_major,
           T* y_data,
           int N,
           int D,
           int C,
           T* w0_data,
           T* f,
           int* num_iters,
           cudaStream_t stream,
           T* sample_weight = nullptr,
           T svr_eps        = 0)
{
  SimpleDenseMat<T> X(X_data, N, D, X_col_major ? COL_MAJOR : ROW_MAJOR);
  qn_fit_x(handle, pams, X, y_data, C, w0_data, f, num_iters, stream, sample_weight, svr_eps);
}

template <typename T>
void qnFitSparse(const raft::handle_t& handle,
                 const qn_params& pams,
                 T* X_values,
                 int* X_cols,
                 int* X_row_ids,
                 int X_nnz,
                 T* y_data,
                 int N,
                 int D,
                 int C,
                 T* w0_data,
                 T* f,
                 int* num_iters,
                 cudaStream_t stream,
                 T* sample_weight = nullptr,
                 T svr_eps        = 0)
{
  SimpleSparseMat<T> X(X_values, X_cols, X_row_ids, X_nnz, N, D);
  qn_fit_x(handle, pams, X, y_data, C, w0_data, f, num_iters, stream, sample_weight, svr_eps);
}

template <typename T>
void qn_decision_function(const raft::handle_t& handle,
                          const qn_params& pams,
                          SimpleMat<T>& X,
                          int C,
                          T* params,
                          T* scores,
                          cudaStream_t stream)
{
  // NOTE: While gtests pass X as row-major, and python API passes X as
  // col-major, no extensive testing has been done to ensure that
  // this function works correctly for both input types
  int n_targets = qn_is_classification(pams.loss) && C == 2 ? 1 : C;
  GLMDims dims(n_targets, X.n, pams.fit_intercept);
  SimpleDenseMat<T> W(params, n_targets, dims.dims);
  SimpleDenseMat<T> Z(scores, n_targets, X.m);
  linearFwd(handle, Z, X, W, stream);
}

template <typename T>
void qnDecisionFunction(const raft::handle_t& handle,
                        const qn_params& pams,
                        T* Xptr,
                        bool X_col_major,
                        int N,
                        int D,
                        int C,
                        T* params,
                        T* scores,
                        cudaStream_t stream)
{
  SimpleDenseMat<T> X(Xptr, N, D, X_col_major ? COL_MAJOR : ROW_MAJOR);
  qn_decision_function(handle, pams, X, C, params, scores, stream);
}

template <typename T>
void qnDecisionFunctionSparse(const raft::handle_t& handle,
                              const qn_params& pams,
                              T* X_values,
                              int* X_cols,
                              int* X_row_ids,
                              int X_nnz,
                              int N,
                              int D,
                              int C,
                              T* params,
                              T* scores,
                              cudaStream_t stream)
{
  SimpleSparseMat<T> X(X_values, X_cols, X_row_ids, X_nnz, N, D);
  qn_decision_function(handle, pams, X, C, params, scores, stream);
}

template <typename T>
void qn_predict(const raft::handle_t& handle,
                const qn_params& pams,
                SimpleMat<T>& X,
                int C,
                T* params,
                T* preds,
                cudaStream_t stream)
{
  bool is_class = qn_is_classification(pams.loss);
  int n_targets = is_class && C == 2 ? 1 : C;
  rmm::device_uvector<T> scores(n_targets * X.m, stream);
  qn_decision_function(handle, pams, X, C, params, scores.data(), stream);
  SimpleDenseMat<T> Z(scores.data(), n_targets, X.m);
  SimpleDenseMat<T> P(preds, 1, X.m);

  if (is_class) {
    if (C == 2) {
      P.assign_unary(
        Z, [] __device__(const T z) { return z > 0.0 ? T(1) : T(0); }, stream);
    } else {
      raft::matrix::argmax(Z.data, C, X.m, preds, stream);
    }
  } else {
    P.copy_async(Z, stream);
  }
}

template <typename T>
void qnPredict(const raft::handle_t& handle,
               const qn_params& pams,
               T* Xptr,
               bool X_col_major,
               int N,
               int D,
               int C,
               T* params,
               T* preds,
               cudaStream_t stream)
{
  SimpleDenseMat<T> X(Xptr, N, D, X_col_major ? COL_MAJOR : ROW_MAJOR);
  qn_predict(handle, pams, X, C, params, preds, stream);
}

template <typename T>
void qnPredictSparse(const raft::handle_t& handle,
                     const qn_params& pams,
                     T* X_values,
                     int* X_cols,
                     int* X_row_ids,
                     int X_nnz,
                     int N,
                     int D,
                     int C,
                     T* params,
                     T* preds,
                     cudaStream_t stream)
{
  SimpleSparseMat<T> X(X_values, X_cols, X_row_ids, X_nnz, N, D);
  qn_predict(handle, pams, X, C, params, preds, stream);
}

};  // namespace GLM
};  // namespace ML
