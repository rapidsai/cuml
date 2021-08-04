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
#pragma once

#include <raft/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace ML {
namespace SVM {

/**
 * Parameters that describe a trained SVM model.
 * All pointers are device pointers.
 */
template <typename math_t>
struct svmModel {
  svmModel(int n_support_,
           int n_cols_,
           math_t b_,
           int n_classes_,
           rmm::cuda_stream_view stream = rmm::cuda_stream_default)
    : n_support(n_support_),
      n_cols(n_cols_),
      b(b_),
      n_classes(n_classes_),
      dual_coefs(0, stream.value()),
      x_support(0, stream.value()),
      support_idx(0, stream.value()),
      unique_labels(0, stream.value())
  {
  }

  svmModel(rmm::cuda_stream_view stream = rmm::cuda_stream_default)
    : dual_coefs(0, stream.value()),
      x_support(0, stream.value()),
      support_idx(0, stream.value()),
      unique_labels(0, stream.value())
  {
  }

  svmModel(const svmModel& m)
    : n_support(m.n_support),
      n_cols(m.n_cols),
      b(m.b),
      n_classes(m.n_classes),
      dual_coefs_ptr(m.dual_coefs_ptr),
      x_support_ptr(m.x_support_ptr),
      support_idx_ptr(m.support_idx_ptr),
      unique_labels_ptr(m.unique_labels_ptr),
      dual_coefs(0, rmm::cuda_stream_default),
      x_support(0, rmm::cuda_stream_default),
      support_idx(0, rmm::cuda_stream_default),
      unique_labels(0, rmm::cuda_stream_default)
  {
    if (m.dual_coefs.size() > 0) {
      dual_coefs.resize(m.dual_coefs.size(), rmm::cuda_stream_default);
      raft::copy(
        dual_coefs.data(), m.dual_coefs.data(), m.dual_coefs.size(), rmm::cuda_stream_default);
    }
    if (m.x_support.size() > 0) {
      x_support.resize(m.x_support.size(), rmm::cuda_stream_default);
      raft::copy(
        x_support.data(), m.x_support.data(), m.x_support.size(), rmm::cuda_stream_default);
    }
    if (m.support_idx.size() > 0) {
      support_idx.resize(m.support_idx.size(), rmm::cuda_stream_default);
      raft::copy(
        support_idx.data(), m.support_idx.data(), m.support_idx.size(), rmm::cuda_stream_default);
    }
    if (m.unique_labels.size() > 0) {
      unique_labels.resize(m.unique_labels.size(), rmm::cuda_stream_default);
      raft::copy(unique_labels.data(),
                 m.unique_labels.data(),
                 m.unique_labels.size(),
                 rmm::cuda_stream_default);
    }
  }

  math_t* get_dual_coefs() { return dual_coefs_ptr ? dual_coefs_ptr : dual_coefs.data(); }
  math_t* get_x_support() { return x_support_ptr ? x_support_ptr : x_support.data(); }
  int* get_support_idx() { return support_idx_ptr ? support_idx_ptr : support_idx.data(); }
  math_t* get_unique_labels()
  {
    return unique_labels_ptr ? unique_labels_ptr : unique_labels.data();
  }

  const math_t* get_dual_coefs() const
  {
    return dual_coefs_ptr ? dual_coefs_ptr : dual_coefs.data();
  }
  const math_t* get_x_support() const { return x_support_ptr ? x_support_ptr : x_support.data(); }
  const int* get_support_idx() const
  {
    return support_idx_ptr ? support_idx_ptr : support_idx.data();
  }
  const math_t* get_unique_labels() const
  {
    return unique_labels_ptr ? unique_labels_ptr : unique_labels.data();
  }

  int n_support;  //!< Number of support vectors
  int n_cols;     //!< Number of features
  math_t b;       //!< Constant used in the decision function

  //! Non-zero dual coefficients ( dual_coef[i] = \f$ y_i \alpha_i \f$).
  //! Size [n_support].
  rmm::device_uvector<math_t> dual_coefs;
  math_t* dual_coefs_ptr = nullptr;

  //! Support vectors in column major format. Size [n_support x n_cols].
  rmm::device_uvector<math_t> x_support;
  math_t* x_support_ptr = nullptr;

  //! Indices (from the training set) of the support vectors, size [n_support].
  rmm::device_uvector<int> support_idx;
  int* support_idx_ptr = nullptr;

  int n_classes;  //!< Number of classes found in the input labels
  //! Device pointer for the unique classes. Size [n_classes]
  rmm::device_uvector<math_t> unique_labels;
  math_t* unique_labels_ptr = nullptr;
};

};  // namespace SVM
};  // namespace ML
