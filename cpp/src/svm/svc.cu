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
#include "svc.hpp"
#include "svc_impl.h"

namespace ML {
namespace SVM {

using namespace MLCommon;

// Explicit instantiation for the library
template void svcFit<float>(const cumlHandle &handle, float *input, int n_rows,
                            int n_cols, float *labels,
                            const svmParameter &param,
                            MLCommon::GramMatrix::KernelParams &kernel_params,
                            svmModel<float> &model);

template void svcFit<double>(const cumlHandle &handle, double *input,
                             int n_rows, int n_cols, double *labels,
                             const svmParameter &param,
                             MLCommon::GramMatrix::KernelParams &kernel_params,
                             svmModel<double> &model);

template void svcPredict<float>(
  const cumlHandle &handle, float *input, int n_rows, int n_cols,
  MLCommon::GramMatrix::KernelParams &kernel_params,
  const svmModel<float> &model, float *preds);

template void svcPredict<double>(
  const cumlHandle &handle, double *input, int n_rows, int n_cols,
  MLCommon::GramMatrix::KernelParams &kernel_params,
  const svmModel<double> &model, double *preds);

template void svmFreeBuffers(const cumlHandle &handle, svmModel<float> &m);

template void svmFreeBuffers(const cumlHandle &handle, svmModel<double> &m);

template <typename math_t>
SVC<math_t>::SVC(cumlHandle &handle, math_t C, math_t tol,
                 GramMatrix::KernelParams kernel_params, math_t cache_size,
                 int max_iter, bool verbose)
  : handle(handle),
    param(svmParameter{C, cache_size, max_iter, tol, verbose}),
    kernel_params(kernel_params) {
  model.n_support = 0;
  model.dual_coefs = nullptr;
  model.x_support = nullptr;
  model.support_idx = nullptr;
  model.unique_labels = nullptr;
}

template <typename math_t>
SVC<math_t>::~SVC() {
  svmFreeBuffers(handle, model);
}

template <typename math_t>
void SVC<math_t>::fit(math_t *input, int n_rows, int n_cols, math_t *labels) {
  model.n_cols = n_cols;
  if (model.dual_coefs) svmFreeBuffers(handle, model);
  svcFit(handle, input, n_rows, n_cols, labels, param, kernel_params, model);
}

template <typename math_t>
void SVC<math_t>::predict(math_t *input, int n_rows, int n_cols,
                          math_t *preds) {
  svcPredict(handle, input, n_rows, n_cols, kernel_params, model, preds);
}

// Instantiate templates for the shared library
template class SVC<float>;
template class SVC<double>;

};  // namespace SVM
};  // end namespace ML
