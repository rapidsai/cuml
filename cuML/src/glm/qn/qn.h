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
#include <common/device_buffer.hpp>
#include <glm/qn/glm_base.h>
#include <glm/qn/glm_linear.h>
#include <glm/qn/glm_logistic.h>
#include <glm/qn/glm_regularizer.h>
#include <glm/qn/glm_softmax.h>
#include <glm/qn/qn_solvers.h>
#include <matrix/math.h>

namespace ML {
namespace GLM {

template <typename T, typename LossFunction>
size_t qn_workspace_size(T l1, int lbfgs_memory, LossFunction &loss) {
  LBFGSParam<T> opt_param;
  opt_param.m = lbfgs_memory;

  if (l1 == 0) { // lbfgs
    return lbfgs_workspace_size(opt_param, loss.n_param);
  } else { // owlqn
    return owlqn_workspace_size(opt_param, loss.n_param);
  }
}

template <typename T, typename LossFunction, typename MatX>
int qn_fit(LossFunction &loss, const MatX &X, const SimpleVec<T> &y,
           SimpleMat<T> &Z, T l1, T l2, int max_iter, T grad_tol,
           int linesearch_max_iter, int lbfgs_memory, int verbosity,
           T *w0, // initial value and result
           T *fx, int *num_iters, SimpleVec<T> &workspace,
           cudaStream_t stream) {

  LBFGSParam<T> opt_param;
  opt_param.epsilon = grad_tol;
  opt_param.max_iterations = max_iter;
  opt_param.m = lbfgs_memory;
  opt_param.max_linesearch = linesearch_max_iter;
  SimpleVec<T> w(w0, loss.n_param);

  if (l2 == 0) {
    GLMWithData<T, MatX, LossFunction> lossWith(&loss, X, y, Z);

    return min_owlqn(opt_param, lossWith, l1,
                     loss.D * loss.C, // number of params without bias
                     w, *fx, num_iters, workspace, stream, verbosity);

  } else {
    Tikhonov<T> reg(l2);
    RegularizedGLM<T, LossFunction, decltype(reg)> obj(&loss, &reg);
    GLMWithData<T, MatX, decltype(obj)> lossWith(&obj, X, y, Z);

    return min_lbfgs(opt_param, lossWith, w, *fx, num_iters, workspace, stream,
                     verbosity);
  }
}

template <typename T, typename MatX>
void qnFit(const cumlHandle_impl &handle, const MatX &X, const SimpleVec<T> &y,
           int C, bool fit_intercept, T l1, T l2, int max_iter, T grad_tol,
           int linesearch_max_iter, int lbfgs_memory, int verbosity, T *w0,
           T *f, int *num_iters, int loss_type, cudaStream_t stream) {
  typedef MLCommon::device_buffer<T> Buffer;

  const int D = X.n, N = X.m;
  ASSERT(y.len == N, "qn.h - qnFit(): inconsistent number of samples");

  Buffer tmp(handle.getDeviceAllocator(), stream, C * N);

  SimpleMat<T> z(tmp.data(), C, N);

  switch (loss_type) {
  case 0: {
    ASSERT(C == 1, "qn.h: logistic loss invalid C");
    LogisticLoss<T> loss(handle, D, fit_intercept, stream);

    Buffer ws_buf(handle.getDeviceAllocator(), stream,
                  qn_workspace_size(l1, lbfgs_memory, loss));
    SimpleVec<T> workspace(ws_buf.data(), ws_buf.size());

    qn_fit<T, decltype(loss)>(loss, X, y, z, l1, l2, max_iter, grad_tol,
                              linesearch_max_iter, lbfgs_memory, verbosity, w0,
                              f, num_iters, workspace, stream);
  } break;
  case 1: {

    ASSERT(C == 1, "qn.h: squared loss invalid C");
    SquaredLoss<T> loss(handle, D, fit_intercept, stream);

    Buffer ws_buf(handle.getDeviceAllocator(), stream,
                  qn_workspace_size(l1, lbfgs_memory, loss));
    SimpleVec<T> workspace(ws_buf.data(), ws_buf.size());

    qn_fit<T, decltype(loss)>(loss, X, y, z, l1, l2, max_iter, grad_tol,
                              linesearch_max_iter, lbfgs_memory, verbosity, w0,
                              f, num_iters, workspace, stream);
  } break;
  case 2: {

    ASSERT(C > 1, "qn.h: softmax invalid C");
    Softmax<T, MatX> loss(handle, D, C, fit_intercept, stream);

    Buffer ws_buf(handle.getDeviceAllocator(), stream,
                  qn_workspace_size(l1, lbfgs_memory, loss));
    SimpleVec<T> workspace(ws_buf.data(), ws_buf.size());

    qn_fit<T, decltype(loss)>(loss, X, y, z, l1, l2, max_iter, grad_tol,
                              linesearch_max_iter, lbfgs_memory, verbosity, w0,
                              f, num_iters, workspace, stream);
  } break;
  default: { ASSERT(false, "qn.h: unknown loss function."); }
  }
}

template <typename T>
void qnPredict(const cumlHandle_impl &handle, T *Xptr, int N, int D, int C,
               bool fit_intercept, T *params, bool X_col_major, int loss_type,
               T *preds, cudaStream_t stream) {

  STORAGE_ORDER ordX = X_col_major ? COL_MAJOR : ROW_MAJOR;

  GLMDims dims(C, D, fit_intercept);

  SimpleMat<T> X(Xptr, N, D, ordX);
  SimpleMat<T> P(preds, 1, N);

  MLCommon::device_buffer<T> tmp(handle.getDeviceAllocator(), stream, C * N);
  SimpleMat<T> Z(tmp.data(), C, N);

  SimpleMat<T> W(params, C, dims.dims);
  linearFwd(handle, Z, X, W, stream);

  switch (loss_type) {
  case 0: {
    ASSERT(C == 1, "qn.h: logistic loss invalid C");
    auto thresh = [] __device__(const T z) {
      if (z > 0.0)
        return T(1);
      return T(0);
    };
    P.assign_unary(Z, thresh, stream);
  } break;
  case 1: {
    ASSERT(C == 1, "qn.h: squared loss invalid C");
    P.copy_async(Z, stream);
  } break;
  case 2: {
    ASSERT(C > 1, "qn.h: softmax invalid C");
    MLCommon::Matrix::argmax(Z.data, C, N, preds, stream);
  } break;
  default: { ASSERT(false, "qn.h: unknown loss function."); }
  }
}

}; // namespace GLM
}; // namespace ML
