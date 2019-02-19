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
   */
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

  inline T operator()(const Vec &wFlat, Vec &gradFlat) {
    Mat W(wFlat.data, C, dims);
    Mat G(gradFlat.data, C, dims);
    // optimizers often operate on vectors
    objective->loss_grad(lossVal.data, G, W, X, y, Z);
    return lossVal[0];
  }
};
}; // namespace GLM
}; // namespace ML
