/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

/** @file svc_impl.cuh
 * @brief Implementation of the stateless C++ functions to fit an SVM
 * classifier, and predict with it.
 */

#include <iostream>

#include "kernelcache.cuh"
#include "smosolver.cuh"
#include <cublas_v2.h>
#include <cuml/svm/svm_model.h>
#include <cuml/svm/svm_parameter.h>
#include <label/classlabels.cuh>
#include <matrix/kernelfactory.cuh>
#include <raft/label/classlabels.hpp>
// #TODO: Replace with public header when ready
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/matrix/matrix.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

namespace ML {
namespace SVM {

template <typename math_t>
void svcFit(const raft::handle_t& handle,
            math_t* input,
            int n_rows,
            int n_cols,
            math_t* labels,
            const SvmParameter& param,
            MLCommon::Matrix::KernelParams& kernel_params,
            SvmModel<math_t>& model,
            const math_t* sample_weight)
{
  ASSERT(n_cols > 0, "Parameter n_cols: number of columns cannot be less than one");
  ASSERT(n_rows > 0, "Parameter n_rows: number of rows cannot be less than one");

  // KernelCache could use multiple streams, not implemented currently
  // See Issue #948.
  // ML::detail::streamSyncer _(handle_impl.getImpl());
  const raft::handle_t& handle_impl = handle;

  cudaStream_t stream = handle_impl.get_stream();
  {
    rmm::device_uvector<math_t> unique_labels(0, stream);
    model.n_classes = raft::label::getUniquelabels(unique_labels, labels, n_rows, stream);
    rmm::mr::device_memory_resource* rmm_alloc = rmm::mr::get_current_device_resource();
    model.unique_labels = (math_t*)rmm_alloc->allocate(model.n_classes * sizeof(math_t), stream);
    raft::copy(model.unique_labels, unique_labels.data(), model.n_classes, stream);
    handle_impl.sync_stream(stream);
  }

  ASSERT(model.n_classes == 2, "Only binary classification is implemented at the moment");

  rmm::device_uvector<math_t> y(n_rows, stream);
  MLCommon::Label::getOvrLabels(
    labels, n_rows, model.unique_labels, model.n_classes, y.data(), 1, stream);

  MLCommon::Matrix::GramMatrixBase<math_t>* kernel =
    MLCommon::Matrix::KernelFactory<math_t>::create(kernel_params, handle_impl.get_cublas_handle());
  SmoSolver<math_t> smo(handle_impl, param, kernel);
  smo.Solve(input,
            n_rows,
            n_cols,
            y.data(),
            sample_weight,
            &(model.dual_coefs),
            &(model.n_support),
            &(model.x_support),
            &(model.support_idx),
            &(model.b),
            param.max_iter);
  model.n_cols = n_cols;
  delete kernel;
}

template <typename math_t>
void svcPredict(const raft::handle_t& handle,
                math_t* input,
                int n_rows,
                int n_cols,
                MLCommon::Matrix::KernelParams& kernel_params,
                const SvmModel<math_t>& model,
                math_t* preds,
                math_t buffer_size,
                bool predict_class)
{
  ASSERT(n_cols == model.n_cols, "Parameter n_cols: shall be the same that was used for fitting");
  // We might want to query the available memory before selecting the batch size.
  // We will need n_batch * n_support floats for the kernel matrix K.
  const int N_PRED_BATCH = 4096;
  int n_batch            = N_PRED_BATCH < n_rows ? N_PRED_BATCH : n_rows;

  // Limit the memory size of the prediction buffer
  buffer_size = buffer_size * 1024 * 1024;
  if (n_batch * model.n_support * sizeof(math_t) > buffer_size) {
    n_batch = buffer_size / (model.n_support * sizeof(math_t));
    if (n_batch < 1) n_batch = 1;
  }

  const raft::handle_t& handle_impl = handle;
  cudaStream_t stream               = handle_impl.get_stream();

  rmm::device_uvector<math_t> K(n_batch * model.n_support, stream);
  rmm::device_uvector<math_t> y(n_rows, stream);
  if (model.n_support == 0) {
    RAFT_CUDA_TRY(cudaMemsetAsync(y.data(), 0, n_rows * sizeof(math_t), stream));
  }
  rmm::device_uvector<math_t> x_rbf(0, stream);
  rmm::device_uvector<int> idx(0, stream);

  cublasHandle_t cublas_handle = handle_impl.get_cublas_handle();

  MLCommon::Matrix::GramMatrixBase<math_t>* kernel =
    MLCommon::Matrix::KernelFactory<math_t>::create(kernel_params, cublas_handle);
  if (kernel_params.kernel == MLCommon::Matrix::RBF) {
    // Temporary buffers for the RBF kernel, see below
    x_rbf.resize(n_batch * n_cols, stream);
    idx.resize(n_batch, stream);
  }
  // We process the input data batchwise:
  //  - calculate the kernel values K[x_batch, x_support]
  //  - calculate y(x_batch) = K[x_batch, x_support] * dual_coeffs
  for (int i = 0; i < n_rows && model.n_support > 0; i += n_batch) {
    if (i + n_batch >= n_rows) { n_batch = n_rows - i; }
    math_t* x_ptr = nullptr;
    int ld1       = 0;
    if (kernel_params.kernel == MLCommon::Matrix::RBF) {
      // The RBF kernel does not support ld parameters (See issue #1172)
      // To come around this limitation, we copy the batch into a temporary
      // buffer.
      thrust::counting_iterator<int> first(i);
      thrust::counting_iterator<int> last = first + n_batch;
      thrust::device_ptr<int> idx_ptr(idx.data());
      thrust::copy(thrust::cuda::par.on(stream), first, last, idx_ptr);
      raft::matrix::copyRows(
        input, n_rows, n_cols, x_rbf.data(), idx.data(), n_batch, stream, false);
      x_ptr = x_rbf.data();
      ld1   = n_batch;
    } else {
      x_ptr = input + i;
      ld1   = n_rows;
    }
    kernel->evaluate(x_ptr,
                     n_batch,
                     n_cols,
                     model.x_support,
                     model.n_support,
                     K.data(),
                     false,
                     stream,
                     ld1,
                     model.n_support,
                     n_batch);
    math_t one  = 1;
    math_t null = 0;
    // #TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(cublas_handle,
                                                     CUBLAS_OP_N,
                                                     n_batch,
                                                     model.n_support,
                                                     &one,
                                                     K.data(),
                                                     n_batch,
                                                     model.dual_coefs,
                                                     1,
                                                     &null,
                                                     y.data() + i,
                                                     1,
                                                     stream));
  }
  math_t* labels = model.unique_labels;
  math_t b       = model.b;
  if (predict_class) {
    // Look up the label based on the value of the decision function:
    // f(x) = sign(y(x) + b)
    raft::linalg::unaryOp(
      preds,
      y.data(),
      n_rows,
      [labels, b] __device__(math_t y) { return y + b < 0 ? labels[0] : labels[1]; },
      stream);
  } else {
    // Calculate the value of the decision function: f(x) = y(x) + b
    raft::linalg::unaryOp(
      preds, y.data(), n_rows, [b] __device__(math_t y) { return y + b; }, stream);
  }
  handle_impl.sync_stream(stream);
  delete kernel;
}

template <typename math_t>
void svmFreeBuffers(const raft::handle_t& handle, SvmModel<math_t>& m)
{
  cudaStream_t stream                        = handle.get_stream();
  rmm::mr::device_memory_resource* rmm_alloc = rmm::mr::get_current_device_resource();
  if (m.dual_coefs) rmm_alloc->deallocate(m.dual_coefs, m.n_support * sizeof(math_t), stream);
  if (m.support_idx) rmm_alloc->deallocate(m.support_idx, m.n_support * sizeof(int), stream);
  if (m.x_support)
    rmm_alloc->deallocate(m.x_support, m.n_support * m.n_cols * sizeof(math_t), stream);
  if (m.unique_labels) rmm_alloc->deallocate(m.unique_labels, m.n_classes * sizeof(math_t), stream);
  m.dual_coefs    = nullptr;
  m.support_idx   = nullptr;
  m.x_support     = nullptr;
  m.unique_labels = nullptr;
}

};  // end namespace SVM
};  // end namespace ML
