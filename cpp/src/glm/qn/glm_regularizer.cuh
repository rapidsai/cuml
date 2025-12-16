/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "simple_mat.cuh"

#include <raft/linalg/add.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/stats/mean.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

namespace ML {
namespace GLM {
namespace detail {

template <typename T>
struct Tikhonov {
  T l2_penalty;
  Tikhonov(T l2) : l2_penalty(l2) {}
  Tikhonov(const Tikhonov<T>& other) : l2_penalty(other.l2_penalty) {}

  HDI T operator()(const T w) const { return 0.5 * l2_penalty * w * w; }

  inline void reg_grad(T* reg_val,
                       SimpleDenseMat<T>& G,
                       const SimpleDenseMat<T>& W,
                       const bool has_bias,
                       cudaStream_t stream) const
  {
    // NOTE: scikit generally does not penalize biases
    SimpleDenseMat<T> Gweights;
    SimpleDenseMat<T> Wweights;
    col_slice(G, Gweights, 0, G.n - has_bias);
    col_slice(W, Wweights, 0, G.n - has_bias);
    Gweights.ax(l2_penalty, Wweights, stream);

    raft::linalg::mapThenSumReduce(reg_val, Wweights.len, *this, stream, Wweights.data);
  }
};

template <typename T, class Loss, class Reg>
struct RegularizedGLM : GLMDims {
  Reg* reg;
  Loss* loss;

  RegularizedGLM(Loss* loss, Reg* reg)
    : reg(reg), loss(loss), GLMDims(loss->C, loss->D, loss->fit_intercept)
  {
  }

  inline void loss_grad(T* loss_val,
                        SimpleDenseMat<T>& G,
                        const SimpleDenseMat<T>& W,
                        const SimpleMat<T>& Xb,
                        const SimpleVec<T>& yb,
                        SimpleDenseMat<T>& Zb,
                        cudaStream_t stream,
                        bool initGradZero = true)
  {
    T reg_host, loss_host;
    SimpleVec<T> lossVal(loss_val, 1);

    G.fill(0, stream);

    reg->reg_grad(lossVal.data, G, W, loss->fit_intercept, stream);
    raft::update_host(&reg_host, lossVal.data, 1, stream);

    loss->loss_grad(lossVal.data, G, W, Xb, yb, Zb, stream, false);
    raft::update_host(&loss_host, lossVal.data, 1, stream);

    raft::interruptible::synchronize(stream);

    lossVal.fill(loss_host + reg_host, stream);
  }

  inline T gradNorm(const SimpleVec<T>& grad, T* dev_scalar, cudaStream_t stream)
  {
    return loss->gradNorm(grad, dev_scalar, stream);
  }
};
};  // namespace detail
};  // namespace GLM
};  // namespace ML
