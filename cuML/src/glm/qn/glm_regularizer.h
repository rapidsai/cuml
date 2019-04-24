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
#include "utils.h"
#include "linalg/binary_op.h"
#include "linalg/map_then_reduce.h"
#include "stats/mean.h"
#include <glm/qn/simple_mat.h>

namespace ML {
namespace GLM {

template <typename T> struct Tikhonov {
  T l2_penalty;
  Tikhonov(T l2) : l2_penalty(l2) {}
  Tikhonov(const Tikhonov<T> &other) : l2_penalty(other.l2_penalty) {}

  HDI T operator()(const T w) const { return 0.5 * l2_penalty * w * w; }

  inline void reg_grad(T *reg_val, SimpleMat<T> &G, const SimpleMat<T> &W,
                        const bool has_bias, cudaStream_t stream) const {

    // NOTE: scikit generally does not penalize biases
    SimpleMat<T> Gweights;
    SimpleMat<T> Wweights;
    col_slice(G, Gweights, 0, G.n - has_bias);
    col_slice(W, Wweights, 0, G.n - has_bias);
    Gweights.ax(l2_penalty, Wweights, stream);

    MLCommon::LinAlg::mapThenSumReduce(reg_val, Wweights.len, *this, stream,
                                       Wweights.data);
  }
};

template <typename T, class Loss, class Reg> struct RegularizedGLM : GLMDims {
  Reg *reg;
  Loss *loss;

  RegularizedGLM(Loss *loss, Reg *reg)
      : reg(reg), loss(loss), GLMDims(loss->C, loss->D, loss->fit_intercept) {}

  inline void loss_grad(T *loss_val, SimpleMat<T> &G, const SimpleMat<T> &W,
                        const SimpleMat<T> &Xb, const SimpleVec<T> &yb,
                        SimpleMat<T> &Zb, cudaStream_t stream, bool initGradZero = true) {
    SimpleVec<T> lossVal(loss_val, 1);
    G.fill(0, stream);
    reg->reg_grad(lossVal.data, G, W, loss->fit_intercept, stream);

    T reg_host; 
    MLCommon::updateHostAsync(&reg_host, lossVal.data, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    loss->loss_grad(lossVal.data, G, W, Xb, yb, Zb, stream, false);
    T loss_host;
    MLCommon::updateHostAsync(&loss_host, lossVal.data, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    lossVal.fill(loss_host + reg_host, stream);
  }
};
}; // namespace GLM
}; // namespace ML
