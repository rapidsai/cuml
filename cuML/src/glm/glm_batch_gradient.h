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

#include "cuda_utils.h"
#include "linalg/add.h"
#include "linalg/binary_op.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/map_then_reduce.h"
#include "stats/mean.h"
#include <glm/glm_vectors.h>
#include <linalg/matrix_vector_op.h>
#include <vector>

namespace ML {
namespace GLM {
/*
template <typename T, STORAGE_ORDER Storage>
inline void linearFwd(SimpleMat<T> &Z, const SimpleMat<T, Storage> &X,
                      const SimpleMat<T> &W, cublasHandle_t &cublas,
                      cudaStream_t stream = 0) {
  // Forward pass:  compute Z <- W * X.T + bias
  const bool has_bias = X.n != W.n;
  const int D = X.n;
  if (has_bias) {
    SimpleVec<T> bias;
    SimpleMat<T> weights;
    col_ref(W, bias, D);
    col_slice(W, weights, 0, D);
    // We implement Z <- W * X + b by
    // - Z <- b (broadcast): TODO reads Z unnecessarily atm
    // - Z <- W * X + Z    : TODO can be fused in CUTLASS?
    auto set_bias = [] __device__(const T z, const T b) { return b; };
    MLCommon::LinAlg::matrixVectorOp(Z.data, Z.data, bias.data, Z.m, Z.n, false,
                                     true, set_bias);

    Z.assign_gemmBT(1, weights, X, 1, cublas);
  } else {
    Z.assign_gemmBT(1, W, X, 0, cublas);
  }
}

template <typename T, STORAGE_ORDER Storage>
inline void linearBwd(SimpleMat<T> &G, const SimpleMat<T, Storage> &X,
                      const SimpleMat<T> &dZ, bool setZero,
                      cublasHandle_t &cublas, cudaStream_t stream = 0) {
  // Backward pass:
  // - compute G <- dZ * X.T
  // - for bias: Gb = mean(dZ, 1)

  const bool has_bias = X.n != G.n;
  const int D = X.n;
  const T beta = setZero ? T(0) : T(1);
  if (has_bias) {
    SimpleVec<T> Gbias;
    SimpleMat<T> Gweights;
    col_ref(G, Gbias, D);
    col_slice(G, Gweights, 0, D);

    //TODO can this be fused somehow?
    Gweights.assign_gemm(1.0 / X.m, dZ, X, beta, cublas);
    MLCommon::Stats::mean(Gbias.data, dZ.data, dZ.m, dZ.n, false, true, stream);
  } else {
    G.assign_gemm(1.0 / X.m, dZ, X, beta, cublas);
  }
}

template <typename T> struct Tikhonov {
  T l2_penalty;
  Tikhonov(T l2) : l2_penalty(l2) {}
  Tikhonov(const Tikhonov<T> &other) : l2_penalty(other.l2_penalty) {}

  HDI T operator()(const T w) const { return 0.5 * l2_penalty * w * w; }

  inline void loss_grad(T *reg_val, SimpleMat<T> &G, const SimpleMat<T> &W,
                        const bool has_bias, cudaStream_t stream=0) const {

    SimpleMat<T> Gweights;
    SimpleMat<T> Wweights;
    col_slice(G, Gweights, 0, G.n - has_bias);
    col_slice(W, Wweights, 0, G.n - has_bias);
    Gweights.ax(l2_penalty, Wweights);

    MLCommon::LinAlg::mapThenSumReduce(reg_val, Wweights.len, *this, stream, Wweights.data);
  }
};

struct GLMDims {

  bool fit_intercept;
  int C, D, dims, n_param;
  GLMDims(int C, int D, bool fit_intercept) : C(C), D(D), fit_intercept(fit_intercept) {
    dims = D + fit_intercept;
    n_param = dims * C;
  }
};

template <typename T, class Loss, STORAGE_ORDER Storage>
struct GLMBase : GLMDims {

  typedef SimpleMat<T, COL_MAJOR> Mat;
  typedef SimpleVec<T> Vec;

  cublasHandle_t cublas;
  cudaStream_t stream;

  GLMBase(int D, int C, bool fit_intercept, cudaStream_t stream = 0)
      : GLMDims(C, D, fit_intercept), stream(stream) {
    cublasCreate(&cublas);
  }

  /*
   * Computes the following:
   * 1. Z <- dL/DZ
   * 2. loss_val <- sum loss(Z)
   *
   * Default: elementwise application of loss and its derivative
   * /
  inline void getLossAndDZ(T *loss_val, SimpleMat<T> &Z,
                           const SimpleVec<T> &y) {
    // Base impl assumes simple case C = 1
    Loss *loss = static_cast<Loss *>(this);
    T invN = 1.0 / y.len;

    auto f_l = [=] __device__(const T y, const T z) {
      return loss->eval_l(y, z) * invN;
    };

    // TODO would be nice to have a kernel that fuses these two steps
    MLCommon::LinAlg::mapThenSumReduce(loss_val, y.len, f_l, stream, y.data,
                                       Z.data);

    loss->eval_dl(y.data, Z.data, y.len);
  }

  inline void loss_grad(T *loss_val, Mat &G, const Mat &W,
                        const SimpleMat<T, Storage> &Xb, const Vec &yb, Mat &Zb,
                        bool initGradZero = true) {
    // reshape data
    Loss *loss = static_cast<Loss *>(this); // polymorphism
    linearFwd(Zb, Xb, W, cublas, stream);   // linear part: forward pass
    loss->getLossAndDZ(loss_val, Zb, yb);   // loss specific part
    linearBwd(G, Xb, Zb, initGradZero, cublas,
              stream); // linear part: backward pass
  }
};

template <typename T, class Loss, class Reg, STORAGE_ORDER Storage = COL_MAJOR>
struct RegularizedGLM {
  Reg *reg;
  Loss *loss;
  RegularizedGLM(Loss *loss, Reg *reg) : reg(reg), loss(loss) {}
  inline void loss_grad(T *loss_val, SimpleMat<T> &G, const SimpleMat<T> &W,
                        const SimpleMat<T, Storage> &Xb, const SimpleVec<T> &yb,
                        SimpleMat<T> &Zb, bool initGradZero = true) {
    SimpleVec<T> lossVal(loss_val, 1);
    reg->loss_grad(lossVal.data, G, W, loss->stream);
    T reg = lossVal[0];
    loss->loss_grad(lossVal.data, G, W, Xb, yb, Zb, false);
    T loss = lossVal[0];
    lossVal.fill(loss + reg);
  }
};

template <typename T, STORAGE_ORDER Storage = COL_MAJOR>
struct LogisticLoss1 : GLMBase<T, LogisticLoss1<T, Storage>, Storage> {
  typedef GLMBase<T, LogisticLoss1<T, Storage>, Storage> Super;

  LogisticLoss1(int D, bool has_bias, cudaStream_t stream = 0)
      : Super(D, 1, has_bias, stream) {}

  inline __device__ T log_sigmoid(T x) const {
    T m = MLCommon::myMax<T>(T(0), x);
    return -MLCommon::myLog(MLCommon::myExp(-m) + MLCommon::myExp(-x - m)) - m;
  }

  inline __device__ T eval_l(const T y, const T eta) const {
    T ytil = 2 * y - 1;
    return -log_sigmoid(ytil * eta);
  }

  inline void eval_dl(const T *y, T *eta, const int N) {
    auto f = [] __device__(const T y, const T eta) {
      return T(1.0) / (T(1.0) + MLCommon::myExp(-eta)) - y;
    };
    MLCommon::LinAlg::binaryOp(eta, y, eta, N, f);
  }
};

template <typename T, class GLMObjective, STORAGE_ORDER Storage = COL_MAJOR>
struct GLMWithData :GLMDims {
  typedef SimpleMat<T> Mat;
  typedef SimpleVec<T> Vec;

  SimpleMat<T, Storage> X;
  Mat Z;
  Vec y;
  SimpleVec<T> lossVal;
  GLMObjective *objective;

  GLMWithData(GLMObjective *obj, T *Xptr, T *yptr, T *Zptr, int N, int D,
              bool fit_intercept)
      : objective(obj), X(Xptr, N, D), y(yptr, N), Z(Zptr, 1, N), lossVal(1),
        GLMDims(1,D, fit_intercept){}

  GLMWithData(GLMObjective *obj, T *Xptr, T *yptr, T *Zptr, int N, int D, int C,
              bool fit_intercept, cudaStream_t stream = 0)
      : objective(obj), X(Xptr, N, D), y(yptr, N), Z(Zptr, C, N), lossVal(1),
        GLMDims(C,D,fit_intercept){}

  T operator()(const Vec &wFlat, Vec &gradFlat) {
    Mat W(wFlat.data, C, dims);
    Mat G(gradFlat.data, C, dims);
    // optimizers often operate on vectors
    objective->loss_grad(lossVal.data, G, W, X, y, Z);
    return lossVal[0];
  }
};
/* */
template <typename T, class C, STORAGE_ORDER Storage> struct GLM_BG_Loss {

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
    // Mapreduce memsets the output to 0. Otherwise, we could have saved this
    // copy.
    T tmp;
    MLCommon::updateHost(&tmp, loss_val, 1);

    auto f_reg_2 = [=] __device__(const T w) {
      return tmp / D + 0.5 * lambda2 * w * w;
    };
    // reduce over D: do not penalize bias!
    MLCommon::LinAlg::mapThenSumReduce(loss_val, D, f_reg_2, loss_fn->stream,
                                       weights.data);

    T tmp2;
    MLCommon::updateHost(&tmp2, loss_val, 1);

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
      T *tmp = weights.data;
      auto f = [=] __device__(const T x) { return tmp[D]; };
      eta.assign_unary(eta, f);
      cudaThreadSynchronize();

      eta.assign_gemv(T(1), X, weights, T(1), cublas);
    } else {
      eta.assign_gemv(T(1), X, weights, T(0), cublas);
    }
  }
};

template <typename T, STORAGE_ORDER Storage = COL_MAJOR>
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

template <typename T, STORAGE_ORDER Storage = COL_MAJOR>
struct SquaredLoss : GLM_BG_Loss<T, SquaredLoss<T, Storage>, Storage> {
  typedef GLM_BG_Loss<T, SquaredLoss<T, Storage>, Storage> Super;

  SquaredLoss(T *X, T *y, T *eta, int N, int D, bool has_bias, T lambda2)
      : GLM_BG_Loss<T, SquaredLoss<T, Storage>, Storage>(X, y, eta, N, D,
                                                         has_bias, lambda2) {}

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
