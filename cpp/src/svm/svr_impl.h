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

/** @file svr_impl.h
 * @brief Implementation of the stateless C++ functions to fit an SVM regressor.
 */

#include <iostream>

#include <cublas_v2.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include "common/cumlHandle.hpp"
#include "common/device_buffer.hpp"
#include "cuml/svm/svm_model.h"
#include "cuml/svm/svm_parameter.h"
#include "kernelcache.h"
#include "label/classlabels.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/unary_op.h"
#include "matrix/kernelfactory.h"
#include "matrix/matrix.h"
#include "smosolver.h"
#include "svc_impl.h"

namespace ML {
namespace SVM {

/**
 * @brief Fit a support vector regressor to the training data.
 *
 * Each row of the input data stores a feature vector.
 *
 * The output buffers in the model struct shall be unallocated on entry.
 *
 * @tparam math_t floating point type
 * @param [in] handle the cuML handle
 * @param [in] X device pointer for the input data in column major format.
 *   Size n_rows x n_cols.
 * @param [in] n_rows number of rows
 * @param [in] n_cols number of columns
 * @param [in] y device pointer for the labels. Size n_rows.
 * @param [in] param parameters for training
 * @param [in] kernel_params parameters for the kernel function
 * @param [out] model parameters of the trained model
 */
template <typename math_t>
void svrFit(const cumlHandle &handle, math_t *X, int n_rows, int n_cols,
            math_t *y, const svmParameter &param,
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

  MLCommon::Matrix::GramMatrixBase<math_t> *kernel =
    MLCommon::Matrix::KernelFactory<math_t>::create(
      kernel_params, handle_impl.getCublasHandle());

  SmoSolver<math_t> smo(handle_impl, param, kernel);
  smo.Solve(X, n_rows, n_cols, y, &(model.dual_coefs), &(model.n_support),
            &(model.x_support), &(model.support_idx), &(model.b),
            param.max_iter);
  model.n_cols = n_cols;
  delete kernel;
}

};  // end namespace SVM
};  // end namespace ML
