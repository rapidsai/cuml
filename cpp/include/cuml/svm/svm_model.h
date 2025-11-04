/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace ML {
namespace SVM {

// Contains array(s) for matrix storage
template <typename math_t>
struct SupportStorage {
  int nnz      = -1;
  int* indptr  = nullptr;
  int* indices = nullptr;
  math_t* data = nullptr;
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
  math_t* dual_coefs;

  //! Support vector storage - can contain either CSR or dense
  SupportStorage<math_t> support_matrix;

  //! Indices (from the training set) of the support vectors, size [n_support].
  int* support_idx;

  int n_classes;  //!< Number of classes found in the input labels
  //! Device pointer for the unique classes. Size [n_classes]
  math_t* unique_labels;
};

};  // namespace SVM
};  // namespace ML
