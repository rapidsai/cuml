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
#include "svc.hpp"

namespace ML {
namespace SVM {

using namespace MLCommon;

template <typename math_t>
SVC<math_t>::SVC(cumlHandle &handle, math_t C, math_t tol,
                 GramMatrix::KernelParams kernel_params, math_t cache_size,
                 int max_iter, bool verbose)
  : handle(handle),
    C(C),
    tol(tol),
    kernel_params(kernel_params),
    cache_size(cache_size),
    max_iter(max_iter),
    verbose(verbose) {}

template <typename math_t>
SVC<math_t>::~SVC() {
  free_buffers();
}

template <typename math_t>
void SVC<math_t>::free_buffers() {
  auto allocator = handle.getImpl().getDeviceAllocator();
  cudaStream_t stream = handle.getStream();
  if (dual_coefs)
    allocator->deallocate(dual_coefs, n_support * sizeof(math_t), stream);
  if (support_idx)
    allocator->deallocate(support_idx, n_support * sizeof(int), stream);
  if (x_support)
    allocator->deallocate(x_support, n_support * n_cols * sizeof(math_t),
                          stream);
  if (unique_labels)
    allocator->deallocate(unique_labels, n_classes * sizeof(math_t), stream);
  dual_coefs = nullptr;
  support_idx = nullptr;
  x_support = nullptr;
  unique_labels = nullptr;
}

template <typename math_t>
void SVC<math_t>::fit(math_t *input, int n_rows, int n_cols, math_t *labels) {
  this->n_cols = n_cols;
  if (dual_coefs) free_buffers();
  svcFit(handle, input, n_rows, n_cols, labels, C, tol, kernel_params,
         cache_size, max_iter, &dual_coefs, &n_support, &b, &x_support,
         &support_idx, &unique_labels, &n_classes, verbose);
}

template <typename math_t>
void SVC<math_t>::predict(math_t *input, int n_rows, int n_cols,
                          math_t *preds) {
  ASSERT(n_cols == this->n_cols,
         "Parameter n_cols: shall be the same that was used for fitting");
  svcPredict(handle, input, n_rows, n_cols, kernel_params, dual_coefs,
             n_support, b, x_support, unique_labels, n_classes, preds);
}

// Instantiate templates for the shared library
template class SVC<float>;
template class SVC<double>;

};  // end namespace SVM
};  // end namespace ML
