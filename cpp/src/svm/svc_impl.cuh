/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <raft/core/handle.hpp>
#include <raft/distance/detail/matrix/matrix.hpp>
#include <raft/distance/kernels.cuh>
#include <raft/label/classlabels.cuh>
// #TODO: Replace with public header when ready
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/matrix/matrix.cuh>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

namespace ML {
namespace SVM {

using namespace raft::distance::matrix::detail;

template <typename math_t>
void svcFitX(const raft::handle_t& handle,
             const Matrix<math_t>& matrix,
             math_t* labels,
             const SvmParameter& param,
             raft::distance::kernels::KernelParams& kernel_params,
             SvmModel<math_t>& model,
             const math_t* sample_weight)
{
  int n_cols = matrix.n_cols;
  int n_rows = matrix.n_rows;

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
  raft::label::getOvrlabels(
    labels, n_rows, model.unique_labels, model.n_classes, y.data(), 1, stream);

  raft::distance::kernels::GramMatrixBase<math_t>* kernel =
    raft::distance::kernels::KernelFactory<math_t>::create(kernel_params, handle_impl);
  SmoSolver<math_t> smo(handle_impl, param, kernel_params.kernel, kernel);
  smo.Solve(matrix,
            n_rows,
            n_cols,
            y.data(),
            sample_weight,
            &(model.dual_coefs),
            &(model.n_support),
            &(model.support_matrix),
            &(model.support_idx),
            &(model.b),
            param.max_iter);
  model.n_cols = n_cols;
  delete kernel;
}

// TODO remove layer once matrix object can be passed from python
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
  DenseMatrix<math_t> dense_matrix(input, n_rows, n_cols);
  svcFitX(handle, dense_matrix, labels, param, kernel_params, model, sample_weight);
}

template <typename math_t>
void svcPredictX(const raft::handle_t& handle,
                 const Matrix<math_t>& matrix,
                 raft::distance::kernels::KernelParams& kernel_params,
                 const SvmModel<math_t>& model,
                 math_t* preds,
                 math_t buffer_size,
                 bool predict_class)
{
  int n_rows = matrix.n_rows;
  int n_cols = matrix.n_cols;

  ASSERT(n_cols == model.n_cols, "Parameter n_cols: shall be the same that was used for fitting");
  // We might want to query the available memory before selecting the batch size.
  // We will need n_batch * n_support floats for the kernel matrix K.
  // FIXME: why choose such a small max? Why choose a max at all when we limit by the buffer_size?
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

  raft::distance::kernels::GramMatrixBase<math_t>* kernel =
    raft::distance::kernels::KernelFactory<math_t>::create(kernel_params, handle_impl);

  /*
    // kernel computation:
    //////////////////////////////////
    Dense input, dense support:
      * just multiply, expanded L2 norm for RBF
    Sparse Input, dense support
      * row ptr copy/shift for input csr, expanded L2 norm for RBF
    Dense input, sparse support
      * transpose kernel compute, expanded L2 norm for RBF
    Sparse intput, sparse support
      * row ptr copy/shift for input csr

    Note: RBF with expanded euclidean only possible with single norm vector for both matrices
  */

  // store matrix dot product for RBF kernels if applicable
  rmm::device_uvector<math_t> dot_input(0, stream);
  rmm::device_uvector<math_t> dot_support(0, stream);
  bool is_csr_input     = !matrix.isDense();
  bool is_csr_support   = !model.support_matrix->isDense();
  bool transpose_kernel = is_csr_support && !is_csr_input;
  if (model.n_support > 0 && kernel_params.kernel == raft::distance::kernels::RBF) {
    dot_input.reserve(n_rows, stream);
    dot_support.reserve(model.n_support, stream);
    ML::SVM::matrixRowNorm(matrix, dot_input.data(), raft::linalg::NormType::L2Norm, stream);
    if (model.n_support > 0)
      ML::SVM::matrixRowNorm(
        *model.support_matrix, dot_support.data(), raft::linalg::NormType::L2Norm, stream);
  }

  // additional row pointer information needed for batched CSR access
  // copy matrix row pointer to host to compute partial nnz on the fly
  std::vector<int> host_indptr;
  rmm::device_uvector<int> indptr_batched(0, stream);
  if (model.n_support > 0 && is_csr_input) {
    auto csr_matrix = matrix.asCsr();
    host_indptr.reserve(n_rows + 1);
    indptr_batched.reserve(n_batch + 1, stream);
    raft::update_host(host_indptr.data(), csr_matrix->indptr, n_rows + 1, stream);
  }

  // We process the input data batchwise:
  //  - calculate the kernel values K[x_batch, x_support]
  //  - calculate y(x_batch) = K[x_batch, x_support] * dual_coeffs
  for (int i = 0; i < n_rows && model.n_support > 0; i += n_batch) {
    if (i + n_batch >= n_rows) { n_batch = n_rows - i; }
    Matrix<math_t>* kernel_input1 = nullptr;
    Matrix<math_t>* kernel_input2 = nullptr;
    DenseMatrix<math_t> kernel_out(K.data(),
                                   transpose_kernel ? model.n_support : n_batch,
                                   transpose_kernel ? n_batch : model.n_support);
    math_t* dot_input1 = dot_input.data() + i;
    math_t* dot_input2 = dot_support.data();

    if (is_csr_input) {
      // we have csr input to batch over
      // create indptr array for batch interval
      // indptr_batched = indices[i, i+n_batch+1] - indices[i]
      // allows for batched csr access on original col/data
      auto csr_input = matrix.asCsr();
      int batch_nnz  = host_indptr[i + n_batch] - host_indptr[i];
      {
        thrust::device_ptr<int> inptr_src(csr_input->indptr + i);
        thrust::device_ptr<int> inptr_tgt(indptr_batched.data());
        thrust::transform(thrust::cuda::par.on(stream),
                          inptr_src,
                          inptr_src + n_batch + 1,
                          thrust::make_constant_iterator(host_indptr[i]),
                          inptr_tgt,
                          thrust::minus<int>());
      }
      kernel_input1 = new CsrMatrix(indptr_batched.data(),
                                    csr_input->indices + host_indptr[i],
                                    csr_input->data + host_indptr[i],
                                    batch_nnz,
                                    n_batch,
                                    n_cols);
    } else {
      kernel_input1 = new DenseMatrix(matrix.asDense()->data + i, n_batch, n_cols, false, n_rows);
    }

    if (is_csr_support) {
      kernel_input2 = new CsrMatrix(*(model.support_matrix->asCsr()));
    } else {
      kernel_input2 = new DenseMatrix(*(model.support_matrix->asDense()));
    }

    (*kernel)(transpose_kernel ? *kernel_input2 : *kernel_input1,
              transpose_kernel ? *kernel_input1 : *kernel_input2,
              kernel_out,
              stream,
              transpose_kernel ? dot_input2 : dot_input1,
              transpose_kernel ? dot_input1 : dot_input2);

    delete kernel_input1;
    delete kernel_input2;

    math_t one  = 1;
    math_t null = 0;
    // #TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(handle_impl.get_cublas_handle(),
                                                     transpose_kernel ? CUBLAS_OP_T : CUBLAS_OP_N,
                                                     transpose_kernel ? model.n_support : n_batch,
                                                     transpose_kernel ? n_batch : model.n_support,
                                                     &one,
                                                     K.data(),
                                                     n_batch,
                                                     model.dual_coefs,
                                                     1,
                                                     &null,
                                                     y.data() + i,
                                                     1,
                                                     stream));

  }  // end of loop

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

// TODO remove layer once matrix object can be passed from python
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
  DenseMatrix<math_t> dense_matrix(input, n_rows, n_cols);
  svcPredictX(handle, dense_matrix, kernel_params, model, preds, buffer_size, predict_class);
}

template <typename math_t>
void svmFreeBuffers(const raft::handle_t& handle, SvmModel<math_t>& m)
{
  cudaStream_t stream                        = handle.get_stream();
  rmm::mr::device_memory_resource* rmm_alloc = rmm::mr::get_current_device_resource();
  if (m.dual_coefs) rmm_alloc->deallocate(m.dual_coefs, m.n_support * sizeof(math_t), stream);
  if (m.support_idx) rmm_alloc->deallocate(m.support_idx, m.n_support * sizeof(int), stream);
  if (m.support_matrix) {
    // matrix is either DenseMatrix or ResizableCsr
    // dense is only a wrapper and needs manual free
    if (m.support_matrix->isDense()) {
      rmm_alloc->deallocate(m.support_matrix->asDense()->data, 
                            m.support_matrix->n_rows * m.support_matrix->n_cols * sizeof(math_t), 
                            stream);
    }
    delete m.support_matrix;
  }
  if (m.unique_labels) rmm_alloc->deallocate(m.unique_labels, m.n_classes * sizeof(math_t), stream);
  m.dual_coefs     = nullptr;
  m.support_idx    = nullptr;
  m.support_matrix = nullptr;
  m.unique_labels  = nullptr;
}

};  // end namespace SVM
};  // end namespace ML
