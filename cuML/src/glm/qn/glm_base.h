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
#include <glm/qn/csr_mat.h>
#include <glm/qn/simple_mat.h>
#include <linalg/matrix_vector_op.h>
#include <vector>

namespace ML {
namespace GLM {

template <typename T, typename MatX>
inline void linearFwd(const cumlHandle_impl &handle, SimpleMat<T> &Z,
                      const MatX &X, const SimpleMat<T> &W,
                      cudaStream_t stream) {
  // Forward pass:  compute Z <- W * X.T + bias
  const bool has_bias = X.n != W.n;
  const int D = X.n;
  if (has_bias) {
    SimpleVec<T> bias;
    SimpleMat<T> weights;
    col_ref(W, bias, D);
    col_slice(W, weights, 0, D);
    // We implement Z <- W * X^T + b by
    // - Z <- b (broadcast): TODO reads Z unnecessarily atm
    // - Z <- W * X^T + Z    : TODO can be fused in CUTLASS?
    auto set_bias = [] __device__(const T z, const T b) { return b; };
    MLCommon::LinAlg::matrixVectorOp(Z.data, Z.data, bias.data, Z.n, Z.m, false,
                                     false, set_bias, stream);

    Z.assign_gemm(handle, 1, weights, false, X, true, 1, stream);
  } else {
    Z.assign_gemm(handle, 1, W, false, X, true, 0, stream);
  }
}

template <typename T, typename MatX>
inline void linearBwd(const cumlHandle_impl &handle, SimpleMat<T> &G,
                      const MatX &X, const SimpleMat<T> &dZ,
                      bool setZero, cudaStream_t stream) {
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

    // TODO can this be fused somehow?
    Gweights.assign_gemm(handle, 1.0 / X.m, dZ, false, X, false, beta, stream);
    MLCommon::Stats::mean(Gbias.data, dZ.data, dZ.m, dZ.n, false, true, stream);
  } else {
    G.assign_gemm(handle, 1.0 / X.m, dZ, false, X, false, beta, stream);
  }
}
struct GLMDims {

  bool fit_intercept;
  int C, D, dims, n_param;
  GLMDims(int C, int D, bool fit_intercept)
      : C(C), D(D), fit_intercept(fit_intercept) {
    dims = D + fit_intercept;
    n_param = dims * C;
  }
};

template <typename T, class Loss> struct GLMBase : GLMDims {

  typedef SimpleMat<T> Mat;
  typedef SimpleVec<T> Vec;

  const cumlHandle_impl &handle;

  GLMBase(const cumlHandle_impl &handle, int D, int C, bool fit_intercept)
      : GLMDims(C, D, fit_intercept), handle(handle) {}

  /*
   * Computes the following:
   * 1. Z <- dL/DZ
   * 2. loss_val <- sum loss(Z)
   *
   * Default: elementwise application of loss and its derivative
   */
  inline void getLossAndDZ(T *loss_val, SimpleMat<T> &Z, const SimpleVec<T> &y,
                           cudaStream_t stream) {

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
    MLCommon::LinAlg::binaryOp(Z.data, y.data, Z.data, y.len, f_dl, stream);
  }

  template <typename MatX> // matrix type for data matrix X
  inline void loss_grad(T *loss_val, Mat &G, const Mat &W, const MatX &Xb,
                        const Vec &yb, Mat &Zb, cudaStream_t stream,
                        bool initGradZero = true) {
    Loss *loss = static_cast<Loss *>(this); // static polymorphism

    linearFwd(handle, Zb, Xb, W, stream);         // linear part: forward pass
    loss->getLossAndDZ(loss_val, Zb, yb, stream); // loss specific part
    linearBwd(handle, G, Xb, Zb, initGradZero,
              stream); // linear part: backward pass
  }
};

template <typename T, typename MatX, class GLMObjective>
struct GLMWithData : GLMDims {

  MatX X;
  SimpleMat<T> Z;
  SimpleVec<T> y;
  GLMObjective *objective;

  GLMWithData(GLMObjective *obj, const MatX &X_, const SimpleVec<T> &y_,
              const SimpleMat<T> &Z_)
      : objective(obj), X(X_), y(y_), Z(Z_),
        GLMDims(obj->C, obj->D, obj->fit_intercept) {
    ASSERT(obj->C == Z.m,
           "glm_base.h: GLMBase(): inconsistent workspace size ");
    ASSERT(obj->D == X.n, "glm_base.h: GLMBase(): inconsistent dimensions");
    ASSERT(Z.n == X.m, "glm_base.h: GLMBase(): inconsistent sizes Z, X");
    ASSERT(y.len == X.m, "glm_base.h: GLMBase(): inconsistent sizes X, y");
  }

  // interface exposed to typical non-linear optimizers
  inline T operator()(const SimpleVec<T> &wFlat, SimpleVec<T> &gradFlat,
                      T *dev_scalar, cudaStream_t stream) {
    SimpleMat<T> W(wFlat.data, C, dims);
    SimpleMat<T> G(gradFlat.data, C, dims);
    objective->loss_grad(dev_scalar, G, W, X, y, Z, stream);
    T loss_host;
    MLCommon::updateHostAsync(&loss_host, dev_scalar, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return loss_host;
  }
};

}; // namespace GLM
}; // namespace ML
