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

#include <rmm/device_buffer.hpp>

namespace ML {
namespace SVM {

// Contains array(s) for matrix storage
struct SupportStorage {
  int nnz = -1;
  rmm::device_buffer* indptr;
  rmm::device_buffer* indices;
  rmm::device_buffer* data;
};

/**
 * Parameters that describe a trained SVM model.
 * All pointers are device pointers.
 */
template <typename math_t>
struct SvmModel {
  int n_support;  //!< Number of support vectors
  int n_cols;     //!< Number of features
  math_t b;       //!< Constant used in the decision function

  //! Non-zero dual coefficients ( dual_coef[i] = \f$ y_i \alpha_i \f$).
  //! Size [n_support].
  rmm::device_buffer* dual_coefs;

  //! Support vector storage - can contain either CSR or dense
  SupportStorage support_matrix;

  //! Indices (from the training set) of the support vectors, size [n_support].
  rmm::device_buffer* support_idx;

  int n_classes;  //!< Number of classes found in the input labels
  //! Device pointer for the unique classes. Size [n_classes]
  rmm::device_buffer* unique_labels;
};

/**
 * Helper container that allows a SvmModel+buffer construction on the stack
 */
template <typename math_t>
struct SvmModelContainer {
  SvmModelContainer()
    : dual_coef_bf(),
      support_idx_bf(),
      unique_labels_bf(),
      support_matrix_indptr_bf(),
      support_matrix_indices_bf(),
      support_matrix_data_bf(),
      model({0,
             0,
             0,
             &dual_coef_bf,
             SupportStorage{
               -1, &support_matrix_indptr_bf, &support_matrix_indices_bf, &support_matrix_data_bf},
             &support_idx_bf,
             0,
             &unique_labels_bf})
  {
  }

  rmm::device_buffer dual_coef_bf;
  rmm::device_buffer support_idx_bf;
  rmm::device_buffer unique_labels_bf;
  rmm::device_buffer support_matrix_indptr_bf;
  rmm::device_buffer support_matrix_indices_bf;
  rmm::device_buffer support_matrix_data_bf;
  SvmModel<math_t> model;
};

};  // namespace SVM
};  // namespace ML
