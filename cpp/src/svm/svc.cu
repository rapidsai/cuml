/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <raft/linalg/cublas_wrappers.h>
#include <cuml/svm/svc.hpp>
#include <label/classlabels.cuh>
#include <matrix/kernelfactory.cuh>
#include <raft/linalg/unary_op.cuh>
#include "kernelcache.cuh"
#include "smosolver.cuh"
#include "svc_impl.cuh"

namespace ML {
namespace SVM {

using namespace MLCommon;

// Explicit instantiation for the library
template void svcFit<float>(const raft::handle_t &handle, float *input,
                            int n_rows, int n_cols, float *labels,
                            const svmParameter &param,
                            MLCommon::Matrix::KernelParams &kernel_params,
                            svmModel<float> &model, const float *sample_weight);

template void svcFit<double>(const raft::handle_t &handle, double *input,
                             int n_rows, int n_cols, double *labels,
                             const svmParameter &param,
                             MLCommon::Matrix::KernelParams &kernel_params,
                             svmModel<double> &model,
                             const double *sample_weight);

template void svcPredict<float>(const raft::handle_t &handle, float *input,
                                int n_rows, int n_cols,
                                MLCommon::Matrix::KernelParams &kernel_params,
                                const svmModel<float> &model, float *preds,
                                float buffer_size, bool predict_class);

template void svcPredict<double>(const raft::handle_t &handle, double *input,
                                 int n_rows, int n_cols,
                                 MLCommon::Matrix::KernelParams &kernel_params,
                                 const svmModel<double> &model, double *preds,
                                 double buffer_size, bool predict_class);

template void svmFreeBuffers(const raft::handle_t &handle, svmModel<float> &m);

template void svmFreeBuffers(const raft::handle_t &handle, svmModel<double> &m);

template <typename math_t>
SVC<math_t>::SVC(raft::handle_t &handle, math_t C, math_t tol,
                 Matrix::KernelParams kernel_params, math_t cache_size,
                 int max_iter, int nochange_steps, int verbosity)
  : handle(handle),
    param(
      svmParameter{C, cache_size, max_iter, nochange_steps, tol, verbosity}),
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
void SVC<math_t>::fit(math_t *input, int n_rows, int n_cols, math_t *labels,
                      const math_t *sample_weight) {
  model.n_cols = n_cols;
  if (model.dual_coefs) svmFreeBuffers(handle, model);
  svcFit(handle, input, n_rows, n_cols, labels, param, kernel_params, model,
         sample_weight);
}

template <typename math_t>
void SVC<math_t>::predict(math_t *input, int n_rows, int n_cols,
                          math_t *preds) {
  math_t buffer_size = param.cache_size;
  svcPredict(handle, input, n_rows, n_cols, kernel_params, model, preds,
             buffer_size, true);
}

template <typename math_t>
void SVC<math_t>::decisionFunction(math_t *input, int n_rows, int n_cols,
                                   math_t *preds) {
  math_t buffer_size = param.cache_size;
  svcPredict(handle, input, n_rows, n_cols, kernel_params, model, preds,
             buffer_size, false);
}

// Instantiate templates for the shared library
template class SVC<float>;
template class SVC<double>;

};  // namespace SVM
};  // end namespace ML
