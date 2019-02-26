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

template <typename T>
inline void linearFwd(SimpleMat<T> &Z, const SimpleMat<T> &X,
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
    MLCommon::LinAlg::matrixVectorOp(Z.data, Z.data, bias.data, Z.n, Z.m, false,
                                     false, set_bias);

    Z.assign_gemmBT(1, weights, X, 1, cublas);
  } else {
    Z.assign_gemmBT(1, W, X, 0, cublas);
  }
}

template <typename T>
inline void linearBwd(SimpleMat<T> &G, const SimpleMat<T> &X,
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

template <typename T, class Loss>
struct GLMBase : GLMDims {

  typedef SimpleMat<T> Mat;
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
      return loss->lz(y, z) * invN;
    };

    // TODO would be nice to have a kernel that fuses these two steps
    // This would be easy, if mapThenSumReduce allowed outputing the result of
    // map (supporting inplace)
    MLCommon::LinAlg::mapThenSumReduce(loss_val, y.len, f_l, stream, y.data,
                                       Z.data);

    auto f_dl = [=] __device__(const T y, const T z) {
      return loss->dlz(y, z);
    };
    MLCommon::LinAlg::binaryOp(Z.data, y.data, Z.data, y.len, f_dl);
  }

  template<typename XMat>
  inline void loss_grad(T *loss_val, Mat &G, const Mat &W,
                        const XMat &Xb, const Vec &yb, Mat &Zb,
                        bool initGradZero = true) {
    Loss *loss = static_cast<Loss *>(this); // static polymorphism

    linearFwd(Zb, Xb, W, cublas, stream);   // linear part: forward pass
    loss->getLossAndDZ(loss_val, Zb, yb);   // loss specific part
    linearBwd(G, Xb, Zb, initGradZero, cublas,
              stream); // linear part: backward pass
  }
};

template <typename T, class GLMObjective, STORAGE_ORDER Storage = COL_MAJOR>
struct GLMWithData : GLMDims {
  typedef SimpleMat<T> Mat;
  typedef SimpleVec<T> Vec;

  SimpleMat<T> X;
  Mat Z;
  Vec y;
  SimpleVec<T> lossVal;
  GLMObjective *objective;

  GLMWithData(GLMObjective *obj, T *Xptr, T *yptr, T *Zptr, int N)
      : objective(obj), X(Xptr, N, obj->D, Storage), y(yptr, N), Z(Zptr, obj->C, N), lossVal(1),
        GLMDims(obj->C,obj->D, obj->fit_intercept){}

  // interface exposed to typical non-linear optimizers
  inline T operator()(const Vec &wFlat, Vec &gradFlat) {
    Mat W(wFlat.data, C, dims);
    Mat G(gradFlat.data, C, dims);
    objective->loss_grad(lossVal.data, G, W, X, y, Z);

    return lossVal[0];
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
                  T *grad_w_host, T *loss_val, const T h = 1e-4) {
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
