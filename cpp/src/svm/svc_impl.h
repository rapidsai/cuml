/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cuml/svm/svm_model.h>
#include <cuml/svm/svm_parameter.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include "common/cumlHandle.hpp"
#include "common/device_buffer.hpp"
#include "kernelcache.h"
#include "label/classlabels.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/unary_op.h"
#include "matrix/kernelfactory.h"
#include "matrix/matrix.h"
#include "smosolver.h"

namespace ML {
namespace SVM {

/**
 * @brief Fit a support vector classifier to the training data.
 *
 * Each row of the input data stores a feature vector.
 * We use the SMO method to fit the SVM.
 *
 * The output device buffers in the model struct shall be unallocated on entry.
 *
 * @tparam math_t floating point type
 * @param [in] handle the cuML handle
 * @param [in] input device pointer for the input data in column major format.
 *   Size n_rows x n_cols.
 * @param [in] n_rows number of rows
 * @param [in] n_cols number of columns
 * @param [in] labels device pointer for the labels. Size n_rows.
 * @param [in] param parameters for training
 * @param [in] kernel_params parameters for the kernel function
 * @param [out] model parameters of the trained model
 */
template <typename math_t>
void svcFit(const cumlHandle &handle, math_t *input, int n_rows, int n_cols,
            math_t *labels, const svmParameter &param,
            MLCommon::Matrix::KernelParams &kernel_params,
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

  MLCommon::Matrix::GramMatrixBase<math_t> *kernel =
    MLCommon::Matrix::KernelFactory<math_t>::create(
      kernel_params, handle_impl.getCublasHandle());
  SmoSolver<math_t> smo(handle_impl, param, kernel);
  smo.Solve(input, n_rows, n_cols, y.data(), &(model.dual_coefs),
            &(model.n_support), &(model.x_support), &(model.support_idx),
            &(model.b), param.max_iter);
  model.n_cols = n_cols;
  delete kernel;
}

/**
 * @brief Predict classes or decision function value for samples in input.
 *
 * We evaluate the decision function f(x_i). Depending on the parameter
 * predict_class, we either return f(x_i) or the label corresponding to
 * sign(f(x_i)).
 *
 * The predictions are calculated according to the following formula:
 * f(x_i) = \sum_{j=1}^n_support K(x_i, x_j) * dual_coefs[j] + b)
 *
 * pred(x_i) = label[sign(f(x_i))], if predict_class==true, or
 * pred(x_i) = f(x_i),       if predict_class==false
 *
 * We process the input vectors batchwise, and evaluate the full rows of kernel
 * matrix K(x_i, x_j) for a batch (size n_batch * n_support). The maximum size
 * of this buffer (i.e. the maximum batch_size) is controlled by the
 * buffer_size input parameter. For models where n_support is large, increasing
 * buffer_size might improve prediction performance.
 *
 * @tparam math_t floating point type
 * @param handle the cuML handle
 * @param [in] input device pointer for the input data in column major format,
 *   size [n_rows x n_cols].
 * @param [in] n_rows number of rows (input vectors)
 * @param [in] n_cols number of columns (features)
 * @param [in] kernel_params parameters for the kernel function
 * @param [in] model SVM model parameters
 * @param [out] preds device pointer to store the output, size [n_rows].
 *     Should be allocated on entry.
 * @param [in] buffer_size size of temporary buffer in MiB
 * @param [in] predict_class whether to predict class label (true), or just
 *     return the decision function value (false)
 */
template <typename math_t>
void svcPredict(const cumlHandle &handle, math_t *input, int n_rows, int n_cols,
                MLCommon::Matrix::KernelParams &kernel_params,
                const svmModel<math_t> &model, math_t *preds,
                math_t buffer_size, bool predict_class) {
  ASSERT(n_cols == model.n_cols,
         "Parameter n_cols: shall be the same that was used for fitting");
  // We might want to query the available memory before selecting the batch size.
  // We will need n_batch * n_support floats for the kernel matrix K.
  const int N_PRED_BATCH = 4096;
  int n_batch = N_PRED_BATCH < n_rows ? N_PRED_BATCH : n_rows;

  // Limit the memory size of the prediction buffer
  buffer_size = buffer_size * 1024 * 1024;
  if (n_batch * model.n_support * sizeof(math_t) > buffer_size) {
    n_batch = buffer_size / (model.n_support * sizeof(math_t));
    if (n_batch < 1) n_batch = 1;
  }

  const cumlHandle_impl &handle_impl = handle.getImpl();
  cudaStream_t stream = handle_impl.getStream();

  MLCommon::device_buffer<math_t> K(handle_impl.getDeviceAllocator(), stream,
                                    n_batch * model.n_support);
  MLCommon::device_buffer<math_t> y(handle_impl.getDeviceAllocator(), stream,
                                    n_rows);
  MLCommon::device_buffer<math_t> x_rbf(handle_impl.getDeviceAllocator(),
                                        stream);
  MLCommon::device_buffer<int> idx(handle_impl.getDeviceAllocator(), stream);

  cublasHandle_t cublas_handle = handle_impl.getCublasHandle();

  MLCommon::Matrix::GramMatrixBase<math_t> *kernel =
    MLCommon::Matrix::KernelFactory<math_t>::create(kernel_params,
                                                    cublas_handle);
  if (kernel_params.kernel == MLCommon::Matrix::RBF) {
    // Temporary buffers for the RBF kernel, see below
    x_rbf.resize(n_batch * n_cols, stream);
    idx.resize(n_batch, stream);
  }
  // We process the input data batchwise:
  //  - calculate the kernel values K[x_batch, x_support]
  //  - calculate y(x_batch) = K[x_batch, x_support] * dual_coeffs
  for (int i = 0; i < n_rows; i += n_batch) {
    if (i + n_batch >= n_rows) {
      n_batch = n_rows - i;
    }
    math_t *x_ptr = nullptr;
    int ld1 = 0;
    if (kernel_params.kernel == MLCommon::Matrix::RBF) {
      // The RBF kernel does not support ld parameters (See issue #1172)
      // To come around this limitation, we copy the batch into a temporary
      // buffer.
      thrust::counting_iterator<int> first(i);
      thrust::counting_iterator<int> last = first + n_batch;
      thrust::device_ptr<int> idx_ptr(idx.data());
      thrust::copy(thrust::cuda::par.on(stream), first, last, idx_ptr);
      MLCommon::Matrix::copyRows(input, n_rows, n_cols, x_rbf.data(),
                                 idx.data(), n_batch, stream, false);
      x_ptr = x_rbf.data();
      ld1 = n_batch;
    } else {
      x_ptr = input + i;
      ld1 = n_rows;
    }
    kernel->evaluate(x_ptr, n_batch, n_cols, model.x_support, model.n_support,
                     K.data(), stream, ld1, model.n_support, n_batch);
    math_t one = 1;
    math_t null = 0;
    CUBLAS_CHECK(MLCommon::LinAlg::cublasgemv(
      cublas_handle, CUBLAS_OP_N, n_batch, model.n_support, &one, K.data(),
      n_batch, model.dual_coefs, 1, &null, y.data() + i, 1, stream));
  }
  math_t *labels = model.unique_labels;
  math_t b = model.b;
  if (predict_class) {
    // Look up the label based on the value of the decision function:
    // f(x) = sign(y(x) + b)
    MLCommon::LinAlg::unaryOp(
      preds, y.data(), n_rows,
      [labels, b] __device__(math_t y) {
        return y + b < 0 ? labels[0] : labels[1];
      },
      stream);
  } else {
    // Calculate the value of the decision function: f(x) = y(x) + b
    MLCommon::LinAlg::unaryOp(
      preds, y.data(), n_rows, [b] __device__(math_t y) { return y + b; },
      stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
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
