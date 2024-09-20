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

#include <common/cumlHandle.hpp>

#include <cuml/svm/svc.hpp>
#include <cuml/svm/svm_api.h>

#include <raft/distance/distance_types.hpp>

#include <tuple>

extern "C" {

cumlError_t cumlSpSvcFit(cumlHandle_t handle,
                         float* input,
                         int n_rows,
                         int n_cols,
                         float* labels,
                         float C,
                         float cache_size,
                         int max_iter,
                         int nochange_steps,
                         float tol,
                         int verbosity,
                         cumlSvmKernelType kernel,
                         int degree,
                         float gamma,
                         float coef0,
                         int* n_support,
                         float* b,
                         float** dual_coefs,
                         float** x_support,
                         int** support_idx,
                         int* n_classes,
                         float** unique_labels)
{
  ML::SVM::SvmParameter param;
  param.C              = C;
  param.cache_size     = cache_size;
  param.max_iter       = max_iter;
  param.nochange_steps = nochange_steps;
  param.tol            = tol;
  param.verbosity      = verbosity;

  raft::distance::kernels::KernelParams kernel_param;
  kernel_param.kernel = (raft::distance::kernels::KernelType)kernel;
  kernel_param.degree = degree;
  kernel_param.gamma  = gamma;
  kernel_param.coef0  = coef0;

  ML::SVM::SvmModel<float> model;

  rmm::device_async_resource_ref rmm_alloc = rmm::mr::get_current_device_resource();

  cumlError_t status;
  raft::handle_t* handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  cudaStream_t stream          = handle_ptr->get_stream();
  if (status == CUML_SUCCESS) {
    try {
      ML::SVM::svcFit(*handle_ptr,
                      input,
                      n_rows,
                      n_cols,
                      labels,
                      param,
                      kernel_param,
                      model,
                      static_cast<float*>(nullptr));
      *n_support = model.n_support;
      *b         = model.b;
      if (model.dual_coefs.size() > 0) {
        *dual_coefs = (float*)rmm_alloc.allocate_async(
          model.dual_coefs.size(), rmm::CUDA_ALLOCATION_ALIGNMENT, stream);
        RAFT_CUDA_TRY(cudaMemcpyAsync(
          dual_coefs, model.dual_coefs.data(), model.dual_coefs.size(), cudaMemcpyDefault, stream));
      } else {
        *dual_coefs = nullptr;
      }
      *x_support     = model.support_matrix.data;
      *support_idx   = model.support_idx;
      *n_classes     = model.n_classes;
      *unique_labels = model.unique_labels;

    }
    // TODO: Implement this
    // catch (const MLCommon::Exception& e)
    //{
    //    //log e.what()?
    //    status =  e.getErrorCode();
    //}
    catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}

cumlError_t cumlDpSvcFit(cumlHandle_t handle,
                         double* input,
                         int n_rows,
                         int n_cols,
                         double* labels,
                         double C,
                         double cache_size,
                         int max_iter,
                         int nochange_steps,
                         double tol,
                         int verbosity,
                         cumlSvmKernelType kernel,
                         int degree,
                         double gamma,
                         double coef0,
                         int* n_support,
                         double* b,
                         double** dual_coefs,
                         double** x_support,
                         int** support_idx,
                         int* n_classes,
                         double** unique_labels)
{
  ML::SVM::SvmParameter param;
  param.C              = C;
  param.cache_size     = cache_size;
  param.max_iter       = max_iter;
  param.nochange_steps = nochange_steps;
  param.tol            = tol;
  param.verbosity      = verbosity;

  raft::distance::kernels::KernelParams kernel_param;
  kernel_param.kernel = (raft::distance::kernels::KernelType)kernel;
  kernel_param.degree = degree;
  kernel_param.gamma  = gamma;
  kernel_param.coef0  = coef0;

  ML::SVM::SvmModel<double> model;

  rmm::device_async_resource_ref rmm_alloc = rmm::mr::get_current_device_resource();

  cumlError_t status;
  raft::handle_t* handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  cudaStream_t stream          = handle_ptr->get_stream();
  if (status == CUML_SUCCESS) {
    try {
      ML::SVM::svcFit(*handle_ptr,
                      input,
                      n_rows,
                      n_cols,
                      labels,
                      param,
                      kernel_param,
                      model,
                      static_cast<double*>(nullptr));
      *n_support = model.n_support;
      *b         = model.b;
      if (model.dual_coefs.size() > 0) {
        *dual_coefs = (double*)rmm_alloc.allocate_async(
          model.dual_coefs.size(), rmm::CUDA_ALLOCATION_ALIGNMENT, stream);
        RAFT_CUDA_TRY(cudaMemcpyAsync(
          dual_coefs, model.dual_coefs.data(), model.dual_coefs.size(), cudaMemcpyDefault, stream));
      } else {
        *dual_coefs = nullptr;
      }
      *x_support     = model.support_matrix.data;
      *support_idx   = model.support_idx;
      *n_classes     = model.n_classes;
      *unique_labels = model.unique_labels;
    }
    // TODO: Implement this
    // catch (const MLCommon::Exception& e)
    //{
    //    //log e.what()?
    //    status =  e.getErrorCode();
    //}
    catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}

cumlError_t cumlSpSvcPredict(cumlHandle_t handle,
                             float* input,
                             int n_rows,
                             int n_cols,
                             cumlSvmKernelType kernel,
                             int degree,
                             float gamma,
                             float coef0,
                             int n_support,
                             float b,
                             float* dual_coefs,
                             float* x_support,
                             int n_classes,
                             float* unique_labels,
                             float* preds,
                             float buffer_size,
                             int predict_class)
{
  cumlError_t status;
  raft::handle_t* handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  cudaStream_t stream          = handle_ptr->get_stream();

  raft::distance::kernels::KernelParams kernel_param;
  kernel_param.kernel = (raft::distance::kernels::KernelType)kernel;
  kernel_param.degree = degree;
  kernel_param.gamma  = gamma;
  kernel_param.coef0  = coef0;

  ML::SVM::SvmModel<float> model;
  model.n_support = n_support;
  model.b         = b;
  if (n_support > 0) {
    model.dual_coefs.resize(n_support * sizeof(float), stream);
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      model.dual_coefs.data(), dual_coefs, n_support * sizeof(float), cudaMemcpyDefault, stream));
  }

  model.support_matrix = {.data = x_support};
  model.support_idx    = nullptr;
  model.n_classes      = n_classes;
  model.unique_labels  = unique_labels;

  if (status == CUML_SUCCESS) {
    try {
      ML::SVM::svcPredict(
        *handle_ptr, input, n_rows, n_cols, kernel_param, model, preds, buffer_size, predict_class);
    }
    // TODO: Implement this
    // catch (const MLCommon::Exception& e)
    //{
    //    //log e.what()?
    //    status =  e.getErrorCode();
    //}
    catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}

cumlError_t cumlDpSvcPredict(cumlHandle_t handle,
                             double* input,
                             int n_rows,
                             int n_cols,
                             cumlSvmKernelType kernel,
                             int degree,
                             double gamma,
                             double coef0,
                             int n_support,
                             double b,
                             double* dual_coefs,
                             double* x_support,
                             int n_classes,
                             double* unique_labels,
                             double* preds,
                             double buffer_size,
                             int predict_class)
{
  cumlError_t status;
  raft::handle_t* handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  cudaStream_t stream          = handle_ptr->get_stream();

  raft::distance::kernels::KernelParams kernel_param;
  kernel_param.kernel = (raft::distance::kernels::KernelType)kernel;
  kernel_param.degree = degree;
  kernel_param.gamma  = gamma;
  kernel_param.coef0  = coef0;

  ML::SVM::SvmModel<double> model;
  model.n_support = n_support;
  model.b         = b;
  if (n_support > 0) {
    model.dual_coefs.resize(n_support * sizeof(double), stream);
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      model.dual_coefs.data(), dual_coefs, n_support * sizeof(double), cudaMemcpyDefault, stream));
  }

  model.support_matrix = {.data = x_support};
  model.support_idx    = nullptr;
  model.n_classes      = n_classes;
  model.unique_labels  = unique_labels;

  if (status == CUML_SUCCESS) {
    try {
      ML::SVM::svcPredict(
        *handle_ptr, input, n_rows, n_cols, kernel_param, model, preds, buffer_size, predict_class);
    }
    // TODO: Implement this
    // catch (const MLCommon::Exception& e)
    //{
    //    //log e.what()?
    //    status =  e.getErrorCode();
    //}
    catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}
}
