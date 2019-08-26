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
#pragma once

/** @file svc_impl.h
 * @brief Implementation of the stateless C++ functions to fit an SVM
 * classifier, and predict with it.
 */

#include <iostream>

#include <cublas_v2.h>
#include "common/cumlHandle.hpp"
#include "common/device_buffer.hpp"
#include "gram/kernelfactory.h"
#include "kernelcache.h"
#include "label/classlabels.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/unary_op.h"
#include "smosolver.h"
#include "svm_model.h"
#include "svm_parameter.h"

namespace ML {
namespace SVM {

/**
 * @brief Fit a support vector classifier to the training data.
 *
 * Each row of the input data stores a feature vector.
 * We use the SMO method to fit the SVM.
 *
 * The output dbuffers shall be unallocated on entry.
 * Note that n_support, n_classes and b are host scalars, all other output
 * pointers are device pointers.
 *
 * @tparam math_t floating point type
 * @param [in] handle the cuML handle
 * @param [in] input device pointer for the input data in column major format.
 *   Size n_rows x n_cols.
 * @param [in] n_rows number of rows
 * @param [in] n_cols number of colums
 * @param [in] labels device pointer for the labels. Size n_rows.
 * @param [in] param parameters for training
 * @param [in] kernel_params parameters for the kernel function
 * @param [out] model parameters of the trained model
 */
template <typename math_t>
void svcFit(const cumlHandle &handle, math_t *input, int n_rows, int n_cols,
            math_t *labels, const svmParameter &param,
            MLCommon::GramMatrix::KernelParams &kernel_params,
            svmModel<math_t> &model) {
  ASSERT(n_cols > 0,
         "Parameter n_cols: number of columns cannot be less than one");
  ASSERT(n_rows > 0,
         "Parameter n_rows: number of rows cannot be less than one");

  // KernelCache could use multiple streams, not implemented currently
  // See Issue #948.
  //ML::detail::streamSyncer _(handle_impl.getImpl());
  const cumlHandle_impl &handle_impl = handle.getImpl();

  cudaStream_t stream = handle_impl.getStream();
  MLCommon::Label::getUniqueLabels(labels, n_rows, &(model.unique_labels),
                                   &(model.n_classes), stream,
                                   handle_impl.getDeviceAllocator());

  ASSERT(model.n_classes == 2,
         "Only binary classification is implemented at the moment");

  MLCommon::device_buffer<math_t> y(handle_impl.getDeviceAllocator(), stream,
                                    n_rows);
  MLCommon::Label::getOvrLabels(labels, n_rows, model.unique_labels,
                                model.n_classes, y.data(), 1, stream);

  MLCommon::GramMatrix::GramMatrixBase<math_t> *kernel =
    MLCommon::GramMatrix::KernelFactory<math_t>::create(
      kernel_params, handle_impl.getCublasHandle());
  SmoSolver<math_t> smo(handle_impl, param.C, param.tol, kernel,
                        param.cache_size);
  smo.verbose = param.verbose;
  smo.Solve(input, n_rows, n_cols, y.data(), &(model.dual_coefs),
            &(model.n_support), &(model.x_support), &(model.support_idx),
            &(model.b), param.max_iter);
  model.n_cols = n_cols;
  delete kernel;
}

/**
 * @brief Predict classes for samples in input.
 *
 * The predictions are calculated according to the following formula:
 * pred(x_i) = sign(f(x_i)) where
 * f(x_i) = \sum_{j=1}^n_support K(x_i, x_j) * dual_coefs[j] + b)
 *
 * We evaluate f(x_i), and then instead of taking the sign to return +/-1 labels,
 * we map it to the original labels, and return those.
 *
 * @tparam math_t floating point type
 * @param handle the cuML handle
 * @param [in] input device pointer for the input data in column major format,
 *   size [n_rows x n_cols].
 * @param [in] n_rows number of rows (input vectors)
 * @param [in] n_cols number of colums (features)
 * @param [in] kernel_params parameters for the kernel function
 * @param [in] model SVM model parameters
 * @param [out] preds device pointer to store the predicted class labels.
 *    Size [n_rows]. Should be allocated on entry.
 */
template <typename math_t>
void svcPredict(const cumlHandle &handle, math_t *input, int n_rows, int n_cols,
                MLCommon::GramMatrix::KernelParams &kernel_params,
                const svmModel<math_t> &model, math_t *preds) {
  ASSERT(n_cols == model.n_cols,
         "Parameter n_cols: shall be the same that was used for fitting");
  //MLCommon::GramMatrix::KernelParams &kernel_params,
  //math_t *dual_coefs, int n_support, math_t b, math_t *x_support,
  //math_t *unique_labels, int n_classes, math_t *preds) {
  // We might want to query the available memory before selecting the batch size.
  // We will need n_batch * n_support floats for the kernel matrix K.
#define N_PRED_BATCH 4096
  int n_batch = N_PRED_BATCH < n_rows ? N_PRED_BATCH : n_rows;

  const cumlHandle_impl &handle_impl = handle.getImpl();
  cudaStream_t stream = handle_impl.getStream();

  MLCommon::device_buffer<math_t> K(handle_impl.getDeviceAllocator(), stream,
                                    n_batch * model.n_support);
  MLCommon::device_buffer<math_t> y(handle_impl.getDeviceAllocator(), stream,
                                    n_rows);

  cublasHandle_t cublas_handle = handle_impl.getCublasHandle();

  MLCommon::GramMatrix::GramMatrixBase<math_t> *kernel =
    MLCommon::GramMatrix::KernelFactory<math_t>::create(kernel_params,
                                                        cublas_handle);

  // We process the input data batchwise:
  //  - calculate the kernel values K[x_batch, x_support]
  //  - calculate y(x_batch) = K[x_batch, x_support] * dual_coeffs
  for (int i = 0; i < n_rows; i += n_batch) {
    if (i + n_batch >= n_rows) {
      n_batch = n_rows - i;
    }
    kernel->evaluate(input + i, n_batch, n_cols, model.x_support,
                     model.n_support, K.data(), stream, n_rows, model.n_support,
                     n_batch);
    math_t one = 1;
    math_t null = 0;
    CUBLAS_CHECK(MLCommon::LinAlg::cublasgemv(
      cublas_handle, CUBLAS_OP_N, n_batch, model.n_support, &one, K.data(),
      n_batch, model.dual_coefs, 1, &null, y.data() + i, 1, stream));
  }
  // Look up the label based on the value of the decision function:
  // f(x) = sign(y(x) + b)
  math_t *labels = model.unique_labels;
  math_t b = model.b;
  MLCommon::LinAlg::unaryOp(
    preds, y.data(), n_rows,
    [labels, b] __device__(math_t y) {
      return y + b < 0 ? labels[0] : labels[1];
    },
    stream);
  delete kernel;
}

template <typename math_t>
void svmFreeBuffers(const cumlHandle &handle, svmModel<math_t> &m) {
  auto allocator = handle.getImpl().getDeviceAllocator();
  cudaStream_t stream = handle.getStream();
  if (m.dual_coefs)
    allocator->deallocate(m.dual_coefs, m.n_support * sizeof(math_t), stream);
  if (m.support_idx)
    allocator->deallocate(m.support_idx, m.n_support * sizeof(int), stream);
  if (m.x_support)
    allocator->deallocate(m.x_support, m.n_support * m.n_cols * sizeof(math_t),
                          stream);
  if (m.unique_labels)
    allocator->deallocate(m.unique_labels, m.n_classes * sizeof(math_t),
                          stream);
  m.dual_coefs = nullptr;
  m.support_idx = nullptr;
  m.x_support = nullptr;
  m.unique_labels = nullptr;
}

};  // end namespace SVM
};  // end namespace ML
