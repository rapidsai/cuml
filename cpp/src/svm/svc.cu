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

#include <iostream>

#include "common/device_buffer.hpp"
#include "gram/kernelfactory.h"
#include "kernelcache.h"
#include "label/classlabels.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/unary_op.h"
#include "smosolver.h"
#include "svc.h"

namespace ML {
namespace SVM {

using namespace MLCommon;

template <typename math_t>
SVC<math_t>::SVC(cumlHandle &handle, math_t C, math_t tol,
                 GramMatrix::KernelParams kernel_params, math_t cache_size,
                 int max_iter)
  : handle_impl(handle.getImpl()),
    C(C),
    tol(tol),
    kernel_params(kernel_params),
    cache_size(cache_size),
    max_iter(max_iter) {}

template <typename math_t>
SVC<math_t>::~SVC() {
  if (dual_coefs) CUDA_CHECK(cudaFree(dual_coefs));
  if (support_idx) CUDA_CHECK(cudaFree(support_idx));
  if (x_support) CUDA_CHECK(cudaFree(x_support));
  if (unique_labels) CUDA_CHECK(cudaFree(unique_labels));
}

template <typename math_t>
void SVC<math_t>::fit(math_t *input, int n_rows, int n_cols, math_t *labels) {
  ASSERT(n_cols > 0,
         "Parameter n_cols: number of columns cannot be less than one");
  ASSERT(n_rows > 0,
         "Parameter n_rows: number of rows cannot be less than one");

  this->n_cols = n_cols;
  // KernelCache can use multiple streams
  //ML::detail::streamSyncer _(handle.getImpl());

  cudaStream_t stream = handle_impl.getStream();
  Label::getUniqueLabels(labels, n_rows, &unique_labels, &n_classes, stream,
                         handle_impl.getDeviceAllocator());

  ASSERT(n_classes == 2,
         "Only binary classification is implemented at the moment");

  device_buffer<math_t> y(handle_impl.getDeviceAllocator(), stream, n_rows);
  Label::getOvrLabels(labels, n_rows, unique_labels, n_classes, y.data(), 1,
                      stream);

  GramMatrix::GramMatrixBase<math_t> *kernel =
    GramMatrix::KernelFactory<math_t>::create(kernel_params,
                                              handle_impl.getCublasHandle());
  SmoSolver<math_t> smo(handle_impl, C, tol, kernel, cache_size);
  smo.Solve(input, n_rows, n_cols, y.data(), &dual_coefs, &n_support,
            &x_support, &support_idx, &b, max_iter);
  delete kernel;
}

/** The predictions are calculated according to the following formula:
 * pred(x_i) = sign(f(x_i)) where
 * f(x_i) = \sum_{j=1}^n_support K(x_i, x_j) * dual_coefs[j] + b)
 *
 * We evaluate f(x_i), and then instead of taking the sign to return +/-1 labels,
 * we map it to the original labels, and return those.
 */
template <typename math_t>
void SVC<math_t>::predict(math_t *input, int n_rows, int n_cols,
                          math_t *preds) {
  ASSERT(n_cols == this->n_cols,
         "Parameter n_cols: shall be the same that was used for fitting");
  // We might want to query the available memory before selecting the batch size.
  // We will need n_batch * n_support floats for the kernel matrix K.
#define N_PRED_BATCH 4096
  int n_batch = N_PRED_BATCH < n_rows ? N_PRED_BATCH : n_rows;

  cudaStream_t stream = handle_impl.getStream();

  device_buffer<math_t> K(handle_impl.getDeviceAllocator(), stream,
                          n_batch * n_support);
  device_buffer<math_t> y(handle_impl.getDeviceAllocator(), stream, n_rows);

  cublasHandle_t cublas_handle = handle_impl.getCublasHandle();

  GramMatrix::GramMatrixBase<math_t> *kernel =
    GramMatrix::KernelFactory<math_t>::create(kernel_params, cublas_handle);

  // We process the input data batchwise:
  //  - calculate the kernel values K[x_batch, x_support]
  //  - calculate y(x_batch) = K[x_batch, x_support] * dual_coeffs
  for (int i = 0; i < n_rows; i += n_batch) {
    if (i + n_batch >= n_rows) {
      n_batch = n_rows - i;
    }
    kernel->evaluate(input + i, n_batch, n_cols, x_support, n_support, K.data(),
                     stream, n_rows, n_support, n_batch);
    math_t one = 1;
    math_t null = 0;
    CUBLAS_CHECK(LinAlg::cublasgemv(
      cublas_handle, CUBLAS_OP_N, n_batch, n_support, &one, K.data(), n_batch,
      dual_coefs, 1, &null, y.data() + i, 1, stream));
  }
  // Look up the label based on the value of the decision function: f(x) = sign(y(x) + b)
  math_t *labels = unique_labels;
  math_t b = this->b;
  LinAlg::unaryOp(
    preds, y.data(), n_rows,
    [labels, b] __device__(math_t y) {
      return y + b < 0 ? labels[0] : labels[1];
    },
    stream);
  delete kernel;
}

// Instantiate templates for the shared library
template class SVC<float>;
template class SVC<double>;

};  // end namespace SVM
};  // end namespace ML
