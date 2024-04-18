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

#include <cuml/common/logger.hpp>
#include <cuml/svm/svm_parameter.h>

#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>

namespace ML {
namespace SVM {

/**
 * Working set selection for the SMO algorithm.
 *
 * The working set is a subset of the training vectors, by default it has 1024 elements.
 * At every outer iteration in SmoSolver::Solve, we select a different working set, and
 * optimize the dual coefficients for the working set.
 *
 * The vectors are selected based on the f values, which is the difference between the
 * target label and the decision function value.
 */
template <typename math_t>
class WorkingSet {
 public:
  //!> Workspace selection strategy, note that only FIFO is tested so far
  bool FIFO_strategy = true;

  /**
   * @brief Manage a working set.
   *
   * @param handle cuml handle implementation
   * @param stream cuda stream for working set operations
   * @param n_rows number of training vectors
   * @param n_ws number of elements in the working set (default 1024)
   * @param svmType classification or regression
   */
  WorkingSet(const raft::handle_t& handle,
             cudaStream_t stream,
             int n_rows      = 0,
             int n_ws        = 0,
             SvmType svmType = C_SVC)
    : handle(handle),
      stream(stream),
      svmType(svmType),
      n_rows(n_rows),
      available(0, stream),
      available_sorted(0, stream),
      cub_storage(0, stream),
      f_idx(0, stream),
      f_idx_sorted(0, stream),
      f_sorted(0, stream),
      idx_tmp(0, stream),
      idx(0, stream),
      ws_idx_sorted(0, stream),
      ws_idx_selected(0, stream),
      ws_idx_save(0, stream),
      ws_priority(0, stream),
      ws_priority_sorted(0, stream),
      d_num_selected(stream)
  {
    n_train = (svmType == EPSILON_SVR) ? n_rows * 2 : n_rows;
    SetSize(n_train, n_ws);
  }

  ~WorkingSet() {}

  /**
   * @brief Set the size of the working set and allocate buffers accordingly.
   *
   * @param n_train number of training vectors
   * @param n_ws working set size (default min(1024, n_train))
   */
  void SetSize(int n_train, int n_ws = 0)
  {
    if (n_ws == 0 || n_ws > n_train) { n_ws = n_train; }
    n_ws       = std::min(1024, n_ws);
    this->n_ws = n_ws;
    CUML_LOG_DEBUG("Creating working set with %d elements", n_ws);
    AllocateBuffers();
  }

  /** Return the size of the working set. */
  int GetSize() { return n_ws; }

  /**
   * @brief Return a device pointer to the the working set indices.
   *
   * The returned array is owned by WorkingSet.
   */
  int* GetIndices() { return idx.data(); }

  /**
   * @brief Select new elements for a working set.
   *
   * Here we follow the working set selection strategy by Joachims [1], we
   * select n training instances as:
   *   - select n/2 element of upper set, where f is largest
   *   - select n/2 from lower set, where f is smallest
   *
   * The difference compared to Joachims' strategy is that we can already have
   * some elements selected by a different strategy, therefore we select only
   * n = n_ws - n_already_selected.
   *
   * References:
   * [1] Joachims, T. (1998). Making large-scale support vector machine learning
   *     practical. In B. Scholkopf, C. Burges, & A. Smola (Eds.), Advances in
   *     kernel methods: Support vector machines. Cambridge, MA: MIT Press
   *
   * @param f optimality indicator vector, size [n_train]
   * @param alpha dual coefficients, size [n_train]
   * @param y target labels (+/- 1)
   * @param C penalty parameter vector size [n_train]
   * @param n_already_selected
   */

  void SimpleSelect(
    math_t* f, math_t* alpha, math_t* y, const math_t* C, int n_already_selected = 0);

  /**
   * @brief Select working set indices.
   *
   * To avoid training vectors oscillating in and out of the working set, we
   * keep half of the previous working set, and fill new elements only to the
   * other half.
   *
   * We can have a FIFO retention policy, or we can
   * consider the time (=ws_priority) a vector already spent in the ws.
   * References:
   * [1] Z. Wen et al. ThunderSVM: A Fast SVM Library on GPUs and CPUs, Journal
   *     of Machine Learning Research, 19, 1-5 (2018)
   *
   * @param f optimality indicator vector, size [n_train]
   * @param alpha dual coefficients, size [n_train]
   * @param y class labels, size [n_train]
   * @param C penalty parameter vector, size [n_train]
   */
  void Select(math_t* f, math_t* alpha, math_t* y, const math_t* C)
  {
    if (n_ws >= n_train) {
      // All elements are selected, we have initialized idx to cover this case
      return;
    }
    int nc         = n_ws / 4;
    int n_selected = 0;
    if (firstcall) {
      if (nc >= 1) {
        firstcall = false;
      } else {
        // This can only happen for n_ws < 4.
        // We keep the calculation always in firstcall mode (only SimpleSelect
        // is used, no advanced strategies because we do not have enough elements)
        //
        // Nothing to do, firstcall is already true
      }
    } else {
      // keep 1/2 of the old working set
      if (FIFO_strategy) {
        // FIFO selection following ThunderSVM
        raft::copy(idx.data(), ws_idx_save.data() + 2 * nc, 2 * nc, stream);
        n_selected = nc * 2;
      } else {
        // priority based selection preferring to keep newer elements in ws
        n_selected = PrioritySelect(alpha, C, nc);
      }
    }
    SimpleSelect(f, alpha, y, C, n_selected);
    raft::copy(ws_idx_save.data(), idx.data(), n_ws, stream);
  }

  /**
   * @brief Select elements from the previous working set based on their priority.
   *
   * We sort the old working set based on their priority in ascending order,
   * and then select nc elements from free, and then lower/upper bound vectors.
   * For details see [2].
   *
   * See Issue #946.
   *
   * References:
   * [2] T Serafini, L Zanni: On the Working Set selection in grad. projection
   *     based decomposition techniques for Support Vector Machines
   *     DOI: 10.1080/10556780500140714
   *
   * @param [in] alpha device vector of dual coefficients, size [n_train]
   * @param [in] C penalty parameter
   * @param [in] nc number of elements to select
   */
  int PrioritySelect(math_t* alpha, const math_t* C, int nc);

 private:
  const raft::handle_t& handle;
  cudaStream_t stream;

  bool firstcall = true;
  int n_train    = 0;  ///< number of training vectors (including duplicates for SVR)
  int n_rows     = 0;  ///< number of original training vectors (no duplicates)
  int n_ws       = 0;

  SvmType svmType;

  int TPB = 256;  //!< Threads per block for workspace selection kernels

  // Buffers for the domain size [n_train]
  rmm::device_uvector<int> f_idx;  //!< Arrays used for sorting for sorting
  rmm::device_uvector<int> f_idx_sorted;
  //! Temporary buffer for index manipulation
  rmm::device_uvector<int> idx_tmp;
  rmm::device_uvector<math_t> f_sorted;
  //! Flag vectors available for selection
  rmm::device_uvector<bool> available;
  rmm::device_uvector<bool> available_sorted;

  // working set buffers size [n_ws]
  rmm::device_uvector<int> idx;  //!< Indices of the worknig set
  rmm::device_uvector<int> ws_idx_sorted;
  rmm::device_uvector<int> ws_idx_selected;
  rmm::device_uvector<int> ws_idx_save;

  rmm::device_uvector<int> ws_priority;
  rmm::device_uvector<int> ws_priority_sorted;

  rmm::device_scalar<int> d_num_selected;
  std::size_t cub_bytes = 0;
  rmm::device_uvector<char> cub_storage;

  void AllocateBuffers();

  /**
   * @brief Gather available elements from the working set.
   *
   * We select the first (last) n_needed element from the front (end) of
   * f_idx_sorted. We ignore the elements that are already selected, and those
   * where this->available is false.
   *
   * @param n_already_selected number of element already selected (their indices
   *   are stored in idx[0:n_already_selected])
   * @param n_needed number of elements to be selected
   * @param copy_front if true, then copy the elements from the front of the
   *        selected list, otherwise copy from the end of the list
   * @return the number of elements copied (which might be less than n_needed)
   */
  int GatherAvailable(int n_already_selected, int n_needed, bool copy_front);

  void Initialize();

  /**
   * @brief Select the first n_needed elements from ws_idx_sorted where op is true.
   *
   * The selected elements are appended to this->idx.
   *
   * @param n_needed number of elements that should be selected
   * @param n_already_selected number of already selected elements
   * @param op selection condition
   * @return the number of elements selected
   */
  template <typename select_op>
  int SelectPrevWs(int n_needed, int n_already_selected, select_op op);
};

};  // end namespace SVM
};  // end namespace ML
