/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <cuml/matrix/kernel_params.hpp>

#include <cublas_v2.h>

namespace ML {
namespace SVM {

template <typename math_t>
struct SvmModel;
struct SvmParameter;

// Forward declarations of the stateless API
/**
 * @brief Fit a support vector regressor to the training data.
 *
 * Each row of the input data stores a feature vector.
 *
 * The output buffers in model shall be unallocated on entry.
 *
 * @tparam math_t floating point type
 * @param [in] handle the cuML handle
 * @param [in] X device pointer for the input data in column major format.
 *   Size n_rows x n_cols.
 * @param [in] n_rows number of rows
 * @param [in] n_cols number of columns
 * @param [in] y device pointer for target values. Size [n_rows].
 * @param [in] param parameters for training
 * @param [in] kernel_params parameters for the kernel function
 * @param [out] model parameters of the trained model
 * @param [in] sample_weight optional sample weights, size [n_rows]
 */
template <typename math_t>
void svrFit(const raft::handle_t& handle,
            math_t* X,
            int n_rows,
            int n_cols,
            math_t* y,
            const SvmParameter& param,
            ML::matrix::KernelParams& kernel_params,
            SvmModel<math_t>& model,
            const math_t* sample_weight = nullptr);

/**
 * @brief Fit a support vector regressor to the training data.
 *
 * Each row of the input data stores a feature vector.
 *
 * The output buffers in model shall be unallocated on entry.
 *
 * @tparam math_t floating point type
 * @param [in] handle the cuML handle
 * @param [in] indptr device pointer for CSR row positions. Size [n_rows + 1].
 * @param [in] indices device pointer for CSR column indices. Size [nnz].
 * @param [in] data device pointer for the CSR data. Size [nnz].
 * @param [in] n_rows number of rows
 * @param [in] n_cols number of columns
 * @param [in] nnz number of stored entries.
 * @param [in] y device pointer for target values. Size [n_rows].
 * @param [in] param parameters for training
 * @param [in] kernel_params parameters for the kernel function
 * @param [out] model parameters of the trained model
 * @param [in] sample_weight optional sample weights, size [n_rows]
 */
template <typename math_t>
void svrFitSparse(const raft::handle_t& handle,
                  int* indptr,
                  int* indices,
                  math_t* data,
                  int n_rows,
                  int n_cols,
                  int nnz,
                  math_t* y,
                  const SvmParameter& param,
                  ML::matrix::KernelParams& kernel_params,
                  SvmModel<math_t>& model,
                  const math_t* sample_weight = nullptr);

// For prediction we use svcPredict

};  // end namespace SVM
};  // end namespace ML
