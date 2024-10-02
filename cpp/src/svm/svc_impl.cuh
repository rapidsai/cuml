/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "kernelcache.cuh"
#include "smosolver.cuh"

#include <cuml/svm/svm_model.h>
#include <cuml/svm/svm_parameter.h>

#include <raft/core/handle.hpp>
#include <raft/distance/kernels.cuh>
#include <raft/label/classlabels.cuh>
#include <raft/linalg/gemv.cuh>

#include <rmm/aligned.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include <cublas_v2.h>

#include <iostream>

namespace ML {
namespace SVM {

template <typename math_t, typename MatrixViewType>
void svcFitX(const raft::handle_t& handle,
             MatrixViewType matrix,
             int n_rows,
             int n_cols,
             math_t* labels,
             const SvmParameter& param,
             raft::distance::kernels::KernelParams& kernel_params,
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
    model.unique_labels->resize(model.n_classes * sizeof(math_t), stream);
    raft::copy((math_t*)model.unique_labels->data(), unique_labels.data(), model.n_classes, stream);
    handle_impl.sync_stream(stream);
  }

  ASSERT(model.n_classes == 2, "Only binary classification is implemented at the moment");

  rmm::device_uvector<math_t> y(n_rows, stream);
  raft::label::getOvrlabels(
    labels, n_rows, (math_t*)model.unique_labels->data(), model.n_classes, y.data(), 1, stream);

  raft::distance::kernels::GramMatrixBase<math_t>* kernel =
    raft::distance::kernels::KernelFactory<math_t>::create(kernel_params);
  SmoSolver<math_t> smo(handle_impl, param, kernel_params.kernel, kernel);
  smo.Solve(matrix,
            n_rows,
            n_cols,
            y.data(),
            sample_weight,
            *(model.dual_coefs),
            model.n_support,
            model.support_matrix,
            *(model.support_idx),
            model.b,
            param.max_iter);
  model.n_cols = n_cols;
  handle_impl.sync_stream(stream);
  delete kernel;
}

template <typename math_t>
void svcFit(const raft::handle_t& handle,
            math_t* input,
            int n_rows,
            int n_cols,
            math_t* labels,
            const SvmParameter& param,
            raft::distance::kernels::KernelParams& kernel_params,
            SvmModel<math_t>& model,
            const math_t* sample_weight)
{
  auto dense_view = raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
    input, n_rows, n_cols, 0);
  svcFitX(handle, dense_view, n_rows, n_cols, labels, param, kernel_params, model, sample_weight);
}

template <typename math_t>
void svcFitSparse(const raft::handle_t& handle,
                  int* indptr,
                  int* indices,
                  math_t* data,
                  int n_rows,
                  int n_cols,
                  int nnz,
                  math_t* labels,
                  const SvmParameter& param,
                  raft::distance::kernels::KernelParams& kernel_params,
                  SvmModel<math_t>& model,
                  const math_t* sample_weight)
{
  auto csr_structure_view = raft::make_device_compressed_structure_view<int, int, int>(
    indptr, indices, n_rows, n_cols, nnz);
  auto csr_matrix_view = raft::make_device_csr_matrix_view(data, csr_structure_view);
  svcFitX(
    handle, csr_matrix_view, n_rows, n_cols, labels, param, kernel_params, model, sample_weight);
}

template <typename math_t, typename MatrixViewType>
void svcPredictX(const raft::handle_t& handle,
                 MatrixViewType matrix,
                 int n_rows,
                 int n_cols,
                 raft::distance::kernels::KernelParams& kernel_params,
                 const SvmModel<math_t>& model,
                 math_t* preds,
                 math_t buffer_size,
                 bool predict_class)
{
  ASSERT(n_cols == model.n_cols, "Parameter n_cols: shall be the same that was used for fitting");
  // We might want to query the available memory before selecting the batch size.
  // We will need n_batch * n_support floats for the kernel matrix K.
  int n_batch = n_rows;
  // Limit the memory size of the prediction buffer
  buffer_size = buffer_size * 1024 * 1024;
  if ((size_t)n_batch * model.n_support * sizeof(math_t) > buffer_size) {
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

  raft::distance::kernels::GramMatrixBase<math_t>* kernel =
    raft::distance::kernels::KernelFactory<math_t>::create(kernel_params);

  /*
    // kernel computation:
    //////////////////////////////////
    Dense input, dense support:
      * just multiply, expanded L2 norm for RBF
    Sparse Input, dense support
      * row ptr copy/shift for input csr, expanded L2 norm for RBF
    Dense input, sparse support
      * transpose kernel compute, expanded L2 norm for RBF
    Sparse input, sparse support
      * row ptr copy/shift for input csr
  */

  // store matrix dot product (l2 norm) for RBF kernels if applicable
  rmm::device_uvector<math_t> l2_input(0, stream);
  rmm::device_uvector<math_t> l2_support(0, stream);
  bool is_csr_input = !isDenseType<MatrixViewType>();

  bool is_csr_support   = model.support_matrix.data->size() > 0 && model.support_matrix.nnz >= 0;
  bool is_dense_support = model.support_matrix.data->size() > 0 && !is_csr_support;

  // Unfortunately we need runtime support for both types
  raft::device_matrix_view<math_t, int, raft::layout_stride> dense_support_matrix_view;
  if (is_dense_support) {
    dense_support_matrix_view =
      raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
        (math_t*)model.support_matrix.data->data(), model.n_support, n_cols, 0);
  }
  auto csr_structure_view =
    is_csr_support
      ? raft::make_device_compressed_structure_view<int, int, int>(
          (int*)model.support_matrix.indptr->data(),
          (int*)model.support_matrix.indices->data(),
          model.n_support,
          n_cols,
          model.support_matrix.nnz)
      : raft::make_device_compressed_structure_view<int, int, int>(nullptr, nullptr, 0, 0, 0);
  auto csr_support_matrix_view =
    is_csr_support
      ? raft::make_device_csr_matrix_view<math_t, int, int, int>(
          (math_t*)model.support_matrix.data->data(), csr_structure_view)
      : raft::make_device_csr_matrix_view<math_t, int, int, int>(nullptr, csr_structure_view);

  bool transpose_kernel = is_csr_support && !is_csr_input;
  if (model.n_support > 0 && kernel_params.kernel == raft::distance::kernels::RBF) {
    l2_input.resize(n_rows, stream);
    l2_support.resize(model.n_support, stream);
    ML::SVM::matrixRowNorm(handle, matrix, l2_input.data(), raft::linalg::NormType::L2Norm);
    if (model.n_support > 0)
      if (is_csr_support) {
        ML::SVM::matrixRowNorm(
          handle, csr_support_matrix_view, l2_support.data(), raft::linalg::NormType::L2Norm);
      } else {
        ML::SVM::matrixRowNorm(
          handle, dense_support_matrix_view, l2_support.data(), raft::linalg::NormType::L2Norm);
      }
  }

  // additional row pointer information needed for batched CSR access
  // copy matrix row pointer to host to compute partial nnz on the fly
  std::vector<int> host_indptr;
  rmm::device_uvector<int> indptr_batched(0, stream);
  if (model.n_support > 0 && is_csr_input) {
    host_indptr.resize(n_rows + 1);
    indptr_batched.resize(n_batch + 1, stream);
    copyIndptrToHost(matrix, host_indptr.data(), stream);
  }

  // We process the input data batchwise:
  //  - calculate the kernel values K[x_batch, x_support]
  //  - calculate y(x_batch) = K[x_batch, x_support] * dual_coeffs
  for (int i = 0; i < n_rows && model.n_support > 0; i += n_batch) {
    if (i + n_batch >= n_rows) { n_batch = n_rows - i; }
    math_t* l2_input1 = l2_input.data() != nullptr ? l2_input.data() + i : nullptr;
    math_t* l2_input2 = l2_support.data();

    auto batch_matrix =
      getMatrixBatch(matrix, n_batch, i, host_indptr.data(), indptr_batched.data(), stream);

    if (transpose_kernel) {
      KernelOp(
        handle_impl, kernel, csr_support_matrix_view, batch_matrix, K.data(), l2_input2, l2_input1);
    } else if (is_csr_support) {
      KernelOp(
        handle_impl, kernel, batch_matrix, csr_support_matrix_view, K.data(), l2_input1, l2_input2);
    } else {
      KernelOp(handle_impl,
               kernel,
               batch_matrix,
               dense_support_matrix_view,
               K.data(),
               l2_input1,
               l2_input2);
    }

    math_t one  = 1;
    math_t null = 0;
    raft::linalg::gemv(handle_impl,
                       transpose_kernel,
                       transpose_kernel ? model.n_support : n_batch,
                       transpose_kernel ? n_batch : model.n_support,
                       &one,
                       K.data(),
                       transpose_kernel ? model.n_support : n_batch,
                       (math_t*)model.dual_coefs->data(),
                       1,
                       &null,
                       y.data() + i,
                       1,
                       stream);

  }  // end of loop

  math_t* labels = (math_t*)model.unique_labels->data();
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
void svcPredict(const raft::handle_t& handle,
                math_t* input,
                int n_rows,
                int n_cols,
                raft::distance::kernels::KernelParams& kernel_params,
                const SvmModel<math_t>& model,
                math_t* preds,
                math_t buffer_size,
                bool predict_class)
{
  auto dense_view = raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
    input, n_rows, n_cols, 0);
  svcPredictX(
    handle, dense_view, n_rows, n_cols, kernel_params, model, preds, buffer_size, predict_class);
}

template <typename math_t>
void svcPredictSparse(const raft::handle_t& handle,
                      int* indptr,
                      int* indices,
                      math_t* data,
                      int n_rows,
                      int n_cols,
                      int nnz,
                      raft::distance::kernels::KernelParams& kernel_params,
                      const SvmModel<math_t>& model,
                      math_t* preds,
                      math_t buffer_size,
                      bool predict_class)
{
  auto csr_structure_view = raft::make_device_compressed_structure_view<int, int, int>(
    indptr, indices, n_rows, n_cols, nnz);
  auto csr_matrix_view = raft::make_device_csr_matrix_view(data, csr_structure_view);
  svcPredictX(handle,
              csr_matrix_view,
              n_rows,
              n_cols,
              kernel_params,
              model,
              preds,
              buffer_size,
              predict_class);
}

template <typename math_t>
void svmFreeBuffers(const raft::handle_t& handle, SvmModel<math_t>& m)
{
  cudaStream_t stream = handle.get_stream();

  m.n_support = 0;
  m.n_cols    = 0;
  m.b         = (math_t)0;
  m.dual_coefs->resize(0, stream);
  m.dual_coefs->shrink_to_fit(stream);
  m.support_idx->resize(0, stream);
  m.support_idx->shrink_to_fit(stream);
  m.support_matrix.indptr->resize(0, stream);
  m.support_matrix.indptr->shrink_to_fit(stream);
  m.support_matrix.indices->resize(0, stream);
  m.support_matrix.indices->shrink_to_fit(stream);
  m.support_matrix.data->resize(0, stream);
  m.support_matrix.data->shrink_to_fit(stream);
  m.support_matrix.nnz = -1;
  m.unique_labels->resize(0, stream);
  m.unique_labels->shrink_to_fit(stream);
}

};  // end namespace SVM
};  // end namespace ML
