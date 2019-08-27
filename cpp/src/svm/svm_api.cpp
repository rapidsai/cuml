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

#include <tuple>
#include "cuML_api.h"
#include "common/cumlHandle.hpp"
#include "gram/kernelparams.h"
#include "svm_api.h"
#include "svc.hpp"
#include "svm_parameter.h"
#include "svm_model.h"

cumlError_t cumlSpSvcFit(
   cumlHandle_t handle, float *input, int n_rows, int n_cols, float *labels,
   float C, float cache_size, int max_iter, float tol, int verbose,
   cumlSvmKernelType kernel, int degree, float gamma, float coef0,
   int *n_support, float *b, float **dual_coefs, float **x_support,
   int **support_idx, int *n_classes,  float **unique_labels) {

   ML::SVM::svmParameter param;
   param.C = C;
   param.cache_size = cache_size;
   param.max_iter = max_iter;
   param.tol = tol;
   param.verbose = verbose;

   MLCommon::GramMatrix::KernelParams kernel_param;
   kernel_param.kernel = (MLCommon::GramMatrix::KernelType) kernel;
   kernel_param.degree = degree;
   kernel_param.gamma = gamma;
   kernel_param.coef0 = coef0;

   ML::SVM::svmModel<float> model;

   cumlError_t status;
   ML::cumlHandle *handle_ptr;
   std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
   if (status == CUML_SUCCESS) {
     try {
       ML::SVM::svcFit(*handle_ptr, input, n_rows, n_cols, labels, param,
                       kernel_param, model);
       *n_support = model.n_support;
       *b = model.b;
       *dual_coefs = model.dual_coefs;
       *x_support = model.x_support;
       *support_idx = model.support_idx;
       *n_classes = model.n_classes;
       *unique_labels = model.unique_labels;
     }
     //TODO: Implement this
     //catch (const MLCommon::Exception& e)
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

cumlError_t cumlDpSvcFit(
   cumlHandle_t handle, double *input, int n_rows, int n_cols, double *labels,
   double C, double cache_size, int max_iter, double tol, int verbose,
   cumlSvmKernelType kernel, int degree, double gamma, double coef0,
   int *n_support, double *b, double **dual_coefs, double **x_support,
   int **support_idx,  int *n_classes, double **unique_labels) {

  ML::SVM::svmParameter param;
  param.C = C;
  param.cache_size = cache_size;
  param.max_iter = max_iter;
  param.tol = tol;
  param.verbose = verbose;

  MLCommon::GramMatrix::KernelParams kernel_param;
  kernel_param.kernel = (MLCommon::GramMatrix::KernelType) kernel;
  kernel_param.degree = degree;
  kernel_param.gamma = gamma;
  kernel_param.coef0 = coef0;

  ML::SVM::svmModel<double> model;

  cumlError_t status;
  ML::cumlHandle *handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      ML::SVM::svcFit(*handle_ptr, input, n_rows, n_cols, labels, param,
                      kernel_param, model);
      *n_support = model.n_support;
      *b = model.b;
      *dual_coefs = model.dual_coefs;
      *x_support = model.x_support;
      *support_idx = model.support_idx;
      *n_classes = model.n_classes;
      *unique_labels = model.unique_labels;
    }
    //TODO: Implement this
    //catch (const MLCommon::Exception& e)
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

cumlError_t cumlSpSvcPredict(
  cumlHandle_t handle, float *input, int n_rows, int n_cols,
  cumlSvmKernelType kernel, int degree, float gamma, float coef0,
  int n_support, float b, float *dual_coefs, float *x_support,
  int n_classes, float *unique_labels, float *preds) {

  MLCommon::GramMatrix::KernelParams kernel_param;
  kernel_param.kernel = (MLCommon::GramMatrix::KernelType) kernel;
  kernel_param.degree = degree;
  kernel_param.gamma = gamma;
  kernel_param.coef0 = coef0;

  ML::SVM::svmModel<float> model;
  model.n_support = n_support;
  model.b = b;
  model.dual_coefs = dual_coefs;
  model.x_support = x_support;
  model.support_idx = nullptr;
  model.n_classes = n_classes;
  model.unique_labels = unique_labels;

  cumlError_t status;
  ML::cumlHandle *handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      ML::SVM::svcPredict(*handle_ptr, input, n_rows, n_cols, kernel_param,
                          model, preds);
    }
    //TODO: Implement this
    //catch (const MLCommon::Exception& e)
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

cumlError_t cumlDpSvcPredict(
  cumlHandle_t handle, double *input, int n_rows, int n_cols,
  cumlSvmKernelType kernel, int degree, double gamma, double coef0,
  int n_support, double b, double *dual_coefs, double *x_support,
  int n_classes, double *unique_labels, double *preds) {

  MLCommon::GramMatrix::KernelParams kernel_param;
  kernel_param.kernel = (MLCommon::GramMatrix::KernelType) kernel;
  kernel_param.degree = degree;
  kernel_param.gamma = gamma;
  kernel_param.coef0 = coef0;

  ML::SVM::svmModel<double> model;
  model.n_support = n_support;
  model.b = b;
  model.dual_coefs = dual_coefs;
  model.x_support = x_support;
  model.support_idx = nullptr;
  model.n_classes = n_classes;
  model.unique_labels = unique_labels;

  cumlError_t status;
  ML::cumlHandle *handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      ML::SVM::svcPredict(*handle_ptr, input, n_rows, n_cols, kernel_param,
                          model, preds);
    }
    //TODO: Implement this
    //catch (const MLCommon::Exception& e)
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
