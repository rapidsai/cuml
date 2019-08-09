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

#pragma once

#include <cuda_utils.h>
#include <limits.h>
#include <cub/cub.cuh>
#include "common/cumlHandle.hpp"
#include "common/device_buffer.hpp"
#include "linalg/add.h"
#include "ml_utils.h"
#include "smo_sets.h"
#include "ws_util.h"

namespace ML {
namespace SVM {

__device__ bool dummy_select_op(int idx) { return true; }

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
  bool verbose = false;

  //!> Workspace selection strategy, note that only FIFO is tested so far
  bool FIFO_strategy = true;

  /** Create a working set
   * @param handle cuml handle implementation
   * @stream cuda stream for working set operations
   * @param n_rows number of training vectors
   * @param n_ws number of elements in the working set (default 1024)
   */
  WorkingSet(const cumlHandle_impl &handle, cudaStream_t stream, int n_rows = 0,
             int n_ws = 0)
    : handle(handle),
      stream(stream),
      available(handle.getDeviceAllocator(), stream),
      available_sorted(handle.getDeviceAllocator(), stream),
      cub_storage(handle.getDeviceAllocator(), stream),
      f_idx(handle.getDeviceAllocator(), stream),
      f_idx_sorted(handle.getDeviceAllocator(), stream),
      f_sorted(handle.getDeviceAllocator(), stream),
      idx_tmp(handle.getDeviceAllocator(), stream),
      idx(handle.getDeviceAllocator(), stream),
      ws_idx_sorted(handle.getDeviceAllocator(), stream),
      ws_idx_selected(handle.getDeviceAllocator(), stream),
      ws_idx_save(handle.getDeviceAllocator(), stream),
      ws_priority(handle.getDeviceAllocator(), stream),
      ws_priority_sorted(handle.getDeviceAllocator(), stream) {
    SetSize(n_rows, n_ws);
  }

  ~WorkingSet() {
    handle.getDeviceAllocator()->deallocate(d_num_selected, 1 * sizeof(int),
                                            stream);
  }

  /**
   * Set the size of the working set and allocate buffers accordingly.
   *
   * @param n_rows number of training vectors
   * @param n_ws working set size (default min(1024, n_rows))
   */
  void SetSize(int n_rows, int n_ws = 0) {
    if (n_ws == 0 || n_ws > n_rows) {
      n_ws = n_rows;
    }
    n_ws = min(1024, n_ws);
    this->n_ws = n_ws;
    this->n_rows = n_rows;
    AllocateBuffers();
  }

  /** Return the size of the working set. */
  int GetSize() { return n_ws; }

  /** Return a pointer the the working set indices */
  int *GetIndices() { return idx.data(); }

  /**
   * Select new elements for a working set.
   *
   * Here we follow the working set selection strategy by Joachims [1], we
   * select n traning instances as:
   *   - select n/2 element of upper set, where f is largest
   *   - select n/2 from lower set, wher f is smallest
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
   * @param f optimality indicator vector, size [n_rows]
   * @param alpha dual coefficients, size [n_rows]
   * @param y target labels (+/- 1)
   * @param C penalty parameter
   * @param n_already_selected
   */

  void SimpleSelect(math_t *f, math_t *alpha, math_t *y, math_t C,
                    int n_already_selected = 0) {
    // We are not using the topK kernel, because of the additional lower/upper
    // constraint
    int n_needed = n_ws - n_already_selected;

    // Zero the priority of the elements that will be newly selected
    CUDA_CHECK(cudaMemsetAsync(ws_priority.data() + n_already_selected, 0,
                               n_needed * sizeof(int), stream));

    cub::DeviceRadixSort::SortPairs(
      (void *)cub_storage.data(), cub_bytes, f, f_sorted.data(), f_idx.data(),
      f_idx_sorted.data(), n_rows, 0, (int)8 * sizeof(int), stream);

    if (verbose && n_rows < 20) {
      MLCommon::myPrintDevVector("idx_sorted", f_idx_sorted.data(), n_rows,
                                 std::cout);
    }
    // Select top n_ws/2 elements
    bool *available = this->available.data();
    set_upper<<<MLCommon::ceildiv(n_rows, TPB), TPB, 0, stream>>>(
      available, n_rows, alpha, y, C);
    CUDA_CHECK(cudaPeekAtLastError());
    n_already_selected +=
      GatherAvailable(n_already_selected, n_needed / 2, true);

    // Select bottom n_ws/2 elements
    set_lower<<<MLCommon::ceildiv(n_rows, TPB), TPB, 0, stream>>>(
      available, n_rows, alpha, y, C);
    CUDA_CHECK(cudaPeekAtLastError());
    n_already_selected +=
      GatherAvailable(n_already_selected, n_ws - n_already_selected, false);

    // In case we could not find enough elements, then we just fill using the
    // still available elements.
    if (n_already_selected < n_ws) {
      if (verbose)
        std::cout << "Warning: could not fill working set, found only "
                  << n_already_selected << " elements.\n";
      if (verbose) std::cout << "Filling up with unused elements\n";
      CUDA_CHECK(cudaMemset(available, 1, sizeof(bool) * n_rows));
      n_already_selected +=
        GatherAvailable(n_already_selected, n_ws - n_already_selected, true);
    }
  }

  /**
  * Select working set indices.
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
  */
  void Select(math_t *f, math_t *alpha, math_t *y, math_t C) {
    if (n_ws >= n_rows) {
      // All elements are selected, we have initialized idx to cover this case
      return;
    }
    int nc = n_ws / 4;
    int n_selected = 0;
    if (firstcall) {
      firstcall = false;
    } else {
      // keep 1/2 of the old working set
      if (FIFO_strategy) {
        // FIFO selection following ThunderSVM
        MLCommon::copy(idx.data(), ws_idx_save.data() + 2 * nc, 2 * nc, stream);
        n_selected = nc * 2;
      } else {
        // priority based selection preferring to keep newer elements in ws
        n_selected = PrioritySelect(alpha, C, nc);
      }
    }
    SimpleSelect(f, alpha, y, C, n_selected);
    MLCommon::copy(ws_idx_save.data(), idx.data(), n_ws, stream);
  }

  /**
   * Select elements from the previous working set based on their priority and
   * dual coefficients.
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
   * @param [in] alpha device vector of dual coefficients, size [n_rows]
   * @param [in] C penalty parameter
   * @param [in] nc number of elements to select
   */
  int PrioritySelect(math_t *alpha, math_t C, int nc) {
    int n_selected = 0;

    cub::DeviceRadixSort::SortPairs(
      (void *)cub_storage.data(), cub_bytes, ws_priority.data(),
      ws_priority_sorted.data(), idx.data(), ws_idx_sorted.data(), n_ws);

    //Select first from free vectors (0<alpha<C)
    n_selected += SelectPrevWs(2 * nc, n_selected, [alpha, C] HD(int idx) {
      return 0 < alpha[idx] && alpha[idx] < C;
    });

    //then from lower bound (alpha=0)
    n_selected += SelectPrevWs(2 * nc, n_selected,
                               [alpha] HD(int idx) { return alpha[idx] <= 0; });
    // and in the end from upper bound vectors (alpha=c)
    n_selected += SelectPrevWs(
      2 * nc, n_selected, [alpha, C] HD(int idx) { return alpha[idx] >= C; });
    // we have now idx[0:n_selected] indices from the old working set
    // we need to update their priority.
    update_priority<<<MLCommon::ceildiv(n_selected, TPB), TPB, 0, stream>>>(
      ws_priority.data(), n_selected, idx.data(), n_ws, ws_idx_sorted.data(),
      ws_priority_sorted.data());
    return n_selected;
  }

 private:
  const cumlHandle_impl &handle;
  cudaStream_t stream;

  bool firstcall = true;
  int n_rows = 0;
  int n_ws = 0;

  int TPB = 256;  //!< Threads per block for workspace selection kernels

  // Buffers for the domain size [n_rows]
  MLCommon::device_buffer<int> f_idx;  //!< Arrays used for sorting for sorting
  MLCommon::device_buffer<int> f_idx_sorted;
  //! Temporary buffer for index manipulation
  MLCommon::device_buffer<int> idx_tmp;
  MLCommon::device_buffer<math_t> f_sorted;
  //! Flag vectors available for selection
  MLCommon::device_buffer<bool> available;
  MLCommon::device_buffer<bool> available_sorted;

  // working set buffers size [n_ws]
  MLCommon::device_buffer<int> idx;  //!< Indices of the worknig set
  MLCommon::device_buffer<int> ws_idx_sorted;
  MLCommon::device_buffer<int> ws_idx_selected;
  MLCommon::device_buffer<int> ws_idx_save;

  MLCommon::device_buffer<int> ws_priority;
  MLCommon::device_buffer<int> ws_priority_sorted;

  int *d_num_selected = nullptr;
  size_t cub_bytes = 0;
  MLCommon::device_buffer<char> cub_storage;

  void AllocateBuffers() {
    if (n_ws > 0) {
      f_idx.resize(n_rows, stream);
      f_idx_sorted.resize(n_rows, stream);
      idx_tmp.resize(n_rows, stream);
      f_sorted.resize(n_rows, stream);
      available.resize(n_rows, stream);
      available_sorted.resize(n_rows, stream);

      idx.resize(n_ws, stream);  //allocate(idx, n_ws, stream);
      ws_idx_sorted.resize(n_ws, stream);
      ws_idx_save.resize(n_ws, stream);
      ws_idx_selected.resize(n_ws, stream);
      ws_priority.resize(n_ws, stream);
      ws_priority_sorted.resize(n_ws, stream);

      d_num_selected =
        (int *)handle.getDeviceAllocator()->allocate(1 * sizeof(int), stream);

      // Determine temporary device storage requirements for cub
      size_t cub_bytes2 = 0;
      cub::DeviceRadixSort::SortPairs(
        NULL, cub_bytes, f_idx.data(), f_idx_sorted.data(), f_sorted.data(),
        f_sorted.data(), n_rows, 0, 8 * sizeof(int), stream);
      cub::DeviceSelect::If(NULL, cub_bytes2, f_idx.data(), f_idx.data(),
                            d_num_selected, n_rows, dummy_select_op, stream);
      cub_bytes = max(cub_bytes, cub_bytes2);
      cub_storage.resize(cub_bytes, stream);
      Initialize();
    }
  }

  /**
   * Gather available elements from the working set.
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
  int GatherAvailable(int n_already_selected, int n_needed, bool copy_front) {
    // First we update the mask to ignores already selected elements
    bool *available = this->available.data();
    if (n_already_selected > 0) {
      set_unavailable<<<MLCommon::ceildiv(n_rows, TPB), TPB, 0, stream>>>(
        available, n_rows, idx.data(), n_already_selected);
      CUDA_CHECK(cudaPeekAtLastError());
    }
    if (verbose && n_rows < 20) {
      MLCommon::myPrintDevVector("avail", available, n_rows, std::cout);
    }

    // Map the mask to the sorted indices
    map_to_sorted<<<MLCommon::ceildiv(n_rows, TPB), TPB, 0, stream>>>(
      available, n_rows, available_sorted.data(), f_idx_sorted.data());
    CUDA_CHECK(cudaPeekAtLastError());
    if (verbose && n_rows < 20) {
      MLCommon::myPrintDevVector("avail_sorted", available_sorted.data(),
                                 n_rows, std::cout);
    }

    // Select the available elements
    cub::DeviceSelect::Flagged((void *)cub_storage.data(), cub_bytes,
                               f_idx_sorted.data(), available_sorted.data(),
                               idx_tmp.data(), d_num_selected, n_rows);
    int n_selected;
    MLCommon::updateHost(&n_selected, d_num_selected, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy to output
    int n_copy = n_selected > n_needed ? n_needed : n_selected;
    if (copy_front) {
      MLCommon::copy(idx.data() + n_already_selected, idx_tmp.data(), n_copy,
                     stream);
    } else {
      MLCommon::copy(idx.data() + n_already_selected,
                     idx_tmp.data() + n_selected - n_copy, n_copy, stream);
    }
    if (verbose && n_rows < 20) {
      MLCommon::myPrintDevVector("selected", idx.data(),
                                 n_already_selected + n_copy, std::cout);
    }
    return n_copy;
  }

  void Initialize() {
    range<<<MLCommon::ceildiv(n_rows, TPB), TPB>>>(f_idx.data(), n_rows);
    CUDA_CHECK(cudaPeekAtLastError());
    range<<<MLCommon::ceildiv(n_ws, TPB), TPB>>>(idx.data(), n_rows);
    CUDA_CHECK(cudaPeekAtLastError());
  }

  /**
   * Select the first n_needed elements from ws_idx_sorted where op is true.
   *
   * The selected elements are appended to this->idx.
   *
   * @param n_needed number of elements that should be selected
   * @param n_already_selected number of already selected elements
   * @param op selection condition
   * @return the number of elements selected
   */
  template <typename select_op>
  int SelectPrevWs(int n_needed, int n_already_selected, select_op op) {
    n_needed -= n_already_selected;
    if (n_needed <= 0) {
      return 0;
    }
    cub::DeviceSelect::If(cub_storage.data(), cub_bytes, ws_idx_sorted.data(),
                          ws_idx_selected.data(), d_num_selected, n_ws, op);
    int n_selected;
    MLCommon::updateHost(&n_selected, d_num_selected, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int n_copy = n_selected < n_needed ? n_selected : n_needed;
    MLCommon::copy(idx.data() + n_already_selected, ws_idx_selected.data(),
                   n_copy, stream);
    return n_copy;
  }
};

};  // end namespace SVM
};  // end namespace ML
