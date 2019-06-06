/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "umap/umapparams.h"

#include "linalg/add.h"
#include "linalg/binary_op.h"
#include "linalg/eltwise.h"
#include "linalg/multiply.h"
#include "linalg/power.h"
#include "linalg/unary_op.h"
#include "matrix/math.h"
#include "stats/mean.h"

#include <cuda_runtime.h>

#pragma once

namespace UMAPAlgo {

namespace Optimize {

using namespace ML;

template <typename T, int TPB_X, typename Lambda>
__global__ void map_kernel(T *output, T *X, int n_rows, T *coef, Lambda grad) {
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < n_rows) {
    T x = X[row];
    T a = coef[0];
    T b = coef[1];
    output[row] = grad(x, a, b);
    if (isnan(output[row])) output[row] = 0.0;
  }
}

/**
         * This works on a one-dimensional set of
         * x-values.
         */
template <typename T, int TPB_X>
void f(T *input, int n_rows, T *coef, T *preds) {
  dim3 grid(MLCommon::ceildiv(n_rows, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  // Function: 1/1+ax^(2b)
  map_kernel<T, TPB_X><<<grid, blk>>>(
    preds, input, n_rows, coef,
    [] __device__(T x, T a, T b) { return 1.0 / (1 + a * pow(x, 2.0 * b)); });
}

/**
         * Calculate the gradients for fitting parameters a and b
         * to a smooth function based on exponential decay
         */
template <typename T, int TPB_X>
void abLossGrads(T *input, int n_rows, const T *labels, T *coef, T *grads,
                 UMAPParams *params, cudaStream_t stream) {
  dim3 grid(MLCommon::ceildiv(n_rows, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  /**
             * Calculate residuals
             */
  T *residuals;
  MLCommon::allocate(residuals, n_rows);

  f<T, TPB_X>(input, n_rows, coef, residuals);
  MLCommon::LinAlg::eltwiseSub(residuals, residuals, labels, n_rows, stream);
  CUDA_CHECK(cudaPeekAtLastError());

  /**
             * Gradient w/ respect to a
             */
  T *a_deriv;
  MLCommon::allocate(a_deriv, n_rows);
  MLCommon::copy(a_deriv, input, n_rows, stream);
  map_kernel<T, TPB_X><<<grid, blk, 0, stream>>>(
    a_deriv, a_deriv, n_rows, coef, [] __device__ __host__(T x, T a, T b) {
      return -(pow(x, 2.0 * b)) / pow((1.0 + a * pow(x, 2.0 * b)), 2.0);
    });

  MLCommon::LinAlg::eltwiseMultiply(a_deriv, a_deriv, residuals, n_rows,
                                    stream);
  CUDA_CHECK(cudaPeekAtLastError());

  /**
             * Gradient w/ respect to b
             */
  T *b_deriv;
  MLCommon::allocate(b_deriv, n_rows);
  MLCommon::copy(b_deriv, input, n_rows, stream);
  map_kernel<T, TPB_X><<<grid, blk, 0, stream>>>(
    b_deriv, b_deriv, n_rows, coef, [] __device__ __host__(T x, T a, T b) {
      return -(2.0 * a * pow(x, 2.0 * b) * log(x)) /
             pow(1 + a * pow(x, 2.0 * b), 2.0);
    });

  /**
             * Multiply partial derivs by residuals
             */
  MLCommon::LinAlg::eltwiseMultiply(b_deriv, b_deriv, residuals, n_rows,
                                    stream);
  CUDA_CHECK(cudaPeekAtLastError());

  /**
             * Finally, take the mean
             */
  MLCommon::Stats::mean(grads, a_deriv, 1, n_rows, false, false, stream);
  MLCommon::Stats::mean(grads + 1, b_deriv, 1, n_rows, false, false, stream);

  CUDA_CHECK(cudaPeekAtLastError());

  CUDA_CHECK(cudaFree(residuals));
  CUDA_CHECK(cudaFree(a_deriv));
  CUDA_CHECK(cudaFree(b_deriv));
}

/**
         * Perform non-linear gradient descent
         */
template <typename T, int TPB_X>
void optimize_params(T *input, int n_rows, const T *labels, T *coef,
                     UMAPParams *params, cudaStream_t stream,
                     float tolerance = 1e-6, int max_epochs = 25000) {
  // Don't really need a learning rate since
  // we aren't using stochastic GD
  float learning_rate = 1.0;

  int num_iters = 0;
  int tol_grads = 0;
  do {
    tol_grads = 0;
    T *grads;
    MLCommon::allocate(grads, 2, true);

    abLossGrads<T, TPB_X>(input, n_rows, labels, coef, grads, params, stream);

    MLCommon::LinAlg::multiplyScalar(grads, grads, learning_rate, 2, stream);
    MLCommon::LinAlg::eltwiseSub(coef, coef, grads, 2, stream);

    T *grads_h = (T *)malloc(2 * sizeof(T));
    MLCommon::updateHost(grads_h, grads, 2, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < 2; i++) {
      if (abs(grads_h[i]) - tolerance <= 0) tol_grads += 1;
    }

    num_iters += 1;

    CUDA_CHECK(cudaFree(grads));
    delete grads_h;

  } while (tol_grads < 2 && num_iters < max_epochs);
}

void find_params_ab(UMAPParams *params, cudaStream_t stream) {
  float spread = params->spread;
  float min_dist = params->min_dist;

  float step = (spread * 3.0) / 300.0;

  float *X = (float *)malloc(300 * sizeof(float));
  float *y = (float *)malloc(300 * sizeof(float));

  for (int i = 0; i < 300; i++) {
    X[i] = i * step;
    y[i] = 0.0;
    if (X[i] >= min_dist)
      y[i] = exp(-(X[i] - min_dist) / spread);
    else if (X[i] < min_dist)
      y[i] = 1.0;
  }

  float *X_d;
  MLCommon::allocate(X_d, 300);
  MLCommon::updateDevice(X_d, X, 300, stream);

  float *y_d;
  MLCommon::allocate(y_d, 300);
  MLCommon::updateDevice(y_d, y, 300, stream);
  float *coeffs_h = (float *)malloc(2 * sizeof(float));
  coeffs_h[0] = 1.0;
  coeffs_h[1] = 1.0;

  float *coeffs;
  MLCommon::allocate(coeffs, 2, true);
  MLCommon::updateDevice(coeffs, coeffs_h, 2, stream);

  optimize_params<float, 256>(X_d, 300, y_d, coeffs, params, stream);

  MLCommon::updateHost(&(params->a), coeffs, 1, stream);
  MLCommon::updateHost(&(params->b), coeffs + 1, 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  if (params->verbose)
    std::cout << "a=" << params->a << ", " << params->b << std::endl;

  CUDA_CHECK(cudaFree(X_d));
  CUDA_CHECK(cudaFree(y_d));
  CUDA_CHECK(cudaFree(coeffs));

  delete coeffs_h;
}
}  // namespace Optimize
}  // namespace UMAPAlgo
