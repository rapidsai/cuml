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

#include <glm/glm_vectors.h>
#include <vector>
#include "cuda_utils.h"
#include "linalg/add.h"
#include "linalg/binary_op.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/map_then_reduce.h"
#include "stats/mean.h"

namespace ML {
namespace GLM {

template <typename T, class C, STORAGE_ORDER Storage>
struct GLM_BG_Loss {

  typedef SimpleVec<T> Vec;
  typedef SimpleMat<T, Storage> Mat;

  T lambda2; // l2 regularizer weight

  T invN; // 1/N

  int N; // rows: number of samples
  int D; // dims: parameter dimension

  int n_param; // number of parameters: D plus bias if has_bias
  bool has_bias;

  cublasHandle_t cublas;
  cudaStream_t stream;

  Mat X;
  Vec y;
  Vec eta;

  SimpleVec<T> loss_val; // not to deal with dealloc

  GLM_BG_Loss(T *Xptr, T *Yptr, T *EtaPtr, int N_, int D_, bool has_bias_,
              T lambda2_, cudaStream_t stream_ = 0)
    : lambda2(lambda2_), has_bias(has_bias_), N(N_), D(D_), invN(1.0 / N_),
      n_param(has_bias_ + D_), X(Xptr, N_, D_), y(Yptr, N_), eta(EtaPtr, N_),
      loss_val(1), stream(stream_) {
    cublasCreate(&cublas);
    cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST);
  }

  inline static void loss(C *loss_fn, const Vec &weights, T *loss_val) {
    const int D = loss_fn->D;
    const T invN = loss_fn->invN;
    const T lambda2 = loss_fn->lambda2;

    auto f_l = [=] __device__(const T y, const T eta) {
      return loss_fn->eval_l(y, eta) * invN;
    };

    MLCommon::LinAlg::mapThenSumReduce(loss_val, loss_fn->y.len, f_l,
                                       loss_fn->stream, loss_fn->y.data,
                                       loss_fn->eta.data);
 //Mapreduce memsets the output to 0. Otherwise, we could have saved this copy.
    T tmp;
    MLCommon::updateHost(&tmp, loss_val, 1);

    auto f_reg_2 = [=] __device__(const T w) {
      return tmp / D + 0.5 * lambda2 * w * w;
    };
    // reduce over D: do not penalize bias!
    MLCommon::LinAlg::mapThenSumReduce(loss_val, D, f_reg_2, loss_fn->stream,
                                       weights.data);
  }

  inline void eval_loss(const SimpleVec<T> &weights, T *loss_val) {
    eval_linear(X, weights, eta, cublas, stream);
    loss(static_cast<C *>(this), weights, loss_val);
  }

  inline T operator()(SimpleVec<T> &weights, SimpleVec<T> &grad) {
    eval_linear(X, weights, eta, cublas, stream);
    GLM_BG_Loss<T, C, Storage>::loss(static_cast<C *>(this), weights,
                                     loss_val.data);

    // penalty only affects weight vector part, not bias
    CUDA_CHECK(cudaMemcpy(grad.data, weights.data, D * sizeof(T),
                          cudaMemcpyDeviceToDevice));
    T beta = lambda2;
    static_cast<C *>(this)->eval_dl(y.data, eta.data);

    if (has_bias)
      MLCommon::Stats::mean(&grad.data[D], eta.data, 1, N, false, false,
                            stream);

    // grad_w = X' eta / N
    grad.assign_gemvT(invN, X, eta, beta, cublas);

    return loss_val[0];
  }

  static void eval_linear(const Mat &X, const Vec &weights, Vec &eta,
                          cublasHandle_t &cublas, cudaStream_t &stream = 0) {
    bool has_bias = weights.len != X.n;
    int D = X.n;

    if (has_bias) {
        T * tmp = weights.data;
      auto f = [=] __device__(const T x) { return tmp[D]; };
      eta.assign_unary(eta, f);
      cudaThreadSynchronize();

      eta.assign_gemv(T(1), X, weights, T(1), cublas);
    } else {
      eta.assign_gemv(T(1), X, weights, T(0), cublas);
    }
  }
};


template <typename T, STORAGE_ORDER Storage=COL_MAJOR>
struct LogisticLoss : GLM_BG_Loss<T, LogisticLoss<T, Storage>, Storage> {
  typedef GLM_BG_Loss<T, LogisticLoss<T, Storage>, Storage> Super;

  LogisticLoss(T *X, T *y, T *eta, int N, int D, bool has_bias, T lambda2)
    : Super(X, y, eta, N, D, has_bias, lambda2) {}

  inline __device__ T log_sigmoid(T x) const {
    T m = MLCommon::myMax<T>(T(0), x);
    return -MLCommon::myLog(MLCommon::myExp(-m) + MLCommon::myExp(-x - m)) - m;
  }

  inline __device__ T eval_l(const T y, const T eta) const {
    T ytil = 2 * y - 1;
    return -log_sigmoid(ytil * eta);
  }

  inline void eval_dl(const T *y, T *eta) {
    auto f = [] __device__(const T y, const T eta) {
      return T(1.0) / (T(1.0) + MLCommon::myExp(-eta)) - y;
    };
    MLCommon::LinAlg::binaryOp(eta, y, eta, Super::N, f);
  }
};


template <typename T, STORAGE_ORDER Storage=COL_MAJOR>
struct SquaredLoss : GLM_BG_Loss<T, SquaredLoss<T, Storage>, Storage> {
  typedef GLM_BG_Loss<T, SquaredLoss<T, Storage>, Storage> Super;

  SquaredLoss(T *X, T *y, T *eta, int N, int D, bool has_bias, T lambda2)
    : GLM_BG_Loss<T, SquaredLoss<T, Storage>, Storage>(X, y, eta, N, D, has_bias, lambda2) {}

  inline __device__ T eval_l(const T y, const T eta) const {
    T diff = y - eta;
    return diff * diff * 0.5;
  }

  inline void eval_dl(const T *y, T *eta) {
    auto f = [] __device__(const T y, const T eta) { return (eta - y); };
    MLCommon::LinAlg::binaryOp(eta, y, eta, Super::N, f);
  }

};


template <typename T>
__global__ void modKernel(T *w, const int tidx, const T h) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == tidx) {
    w[idx] += h;
  }
}

template <typename T, class Loss>
void numeric_grad(Loss &loss, const T *X, const T *y, const T *w,
                  T *grad_w_host, T *loss_val, T *eta, const T h = 1e-4) {
  int len = loss.n_param;
  SimpleVec<T> w_mod(len), grad(len);


  T lph = 0, lmh = 0;

  for (int d = 0; d < len; d++) {
    CUDA_CHECK(
      cudaMemcpy(w_mod.data, w, len * sizeof(T), cudaMemcpyDeviceToDevice));

    modKernel<<<MLCommon::ceildiv(len, 256), 256>>>(w_mod.data, d, h);
    cudaThreadSynchronize();

    lph = loss(w_mod, grad);

    modKernel<<<MLCommon::ceildiv(len, 256), 256>>>(w_mod.data, d, -2 * h);
    cudaThreadSynchronize();
    lmh = loss(w_mod, grad);
    grad_w_host[d] = (lph - lmh) / (2 * h);
  }
}

}; // namespace GLM
}; // namespace ML
