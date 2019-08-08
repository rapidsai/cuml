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
#include <math.h>
#include <iostream>
#include <limits>
#include <memory>

#include <cub/device/device_select.cuh>
#include "common/cumlHandle.hpp"
#include "common/cuml_allocator.hpp"
#include "common/device_buffer.hpp"
#include "common/host_buffer.hpp"
#include "kernelcache.h"
#include "linalg/binary_op.h"
#include "linalg/map_then_reduce.h"
#include "linalg/unary_op.h"
#include "ws_util.h"

namespace ML {
namespace SVM {

using namespace MLCommon;

template <typename math_t, typename Lambda>
__global__ void set_flag(bool *flag, const math_t *alpha, int n, Lambda op) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) flag[tid] = op(alpha[tid]);
}

template <typename math_t>
class Results {
 public:
  /*
   * Helper class to collect the parameters of the SVC classifier after it is
   * fitted using SMO.
   *
   * @param handle cuML handle implementation
   * @param x training vectors in column major format, size [n_rows x n_cols]
   * @param y target labels (values +/-1), size [n_rows]
   * @param n_rows number of training vectors
   * @param n_cols number of features
   * @param C penalty parameter
   */
  Results(const cumlHandle_impl &handle, const math_t *x, const math_t *y,
          int n_rows, int n_cols, math_t C)
    : allocator(handle.getDeviceAllocator()),
      stream(handle.getStream()),
      handle(handle),
      n_rows(n_rows),
      n_cols(n_cols),
      x(x),
      y(y),
      C(C),
      cub_storage(handle.getDeviceAllocator(), stream),
      d_num_selected(handle.getDeviceAllocator(), stream, 1),
      d_val_reduced(handle.getDeviceAllocator(), stream, 1),
      f_idx(handle.getDeviceAllocator(), stream, n_rows),
      idx_selected(handle.getDeviceAllocator(), stream, n_rows),
      val_selected(handle.getDeviceAllocator(), stream, n_rows),
      flag(handle.getDeviceAllocator(), stream, n_rows) {
    InitCubBuffers();
    range<<<ceildiv(n_rows, TPB), TPB, 0, stream>>>(f_idx.data(), n_rows);
    CUDA_CHECK(cudaPeekAtLastError());
  }

  /**
   * Collect the parameters found during training.
   *
   * After fitting, the non-zero dual coefs, the corresponding support vectors,
   * and the constant b represent the parameters of the support vector classifier.
   *
   * On entry the output arrays should not be allocated.
   * All output arrays will be allocated on the device.
   * Note that b is not an array but a host scalar.
   *
   * @param [in] alpha dual coefficients, size [n_rows]
   * @param [in] f optimality indicator vector, size [n_rows]
   * @param [out] dual_coefs size [n_support]
   * @param [out] n_support number of support vectors
   * @param [out] idx the original training set indices of the support vectors, size [n_support]
   * @param [out] x_support support vectors in column major format, size [n_support, n_cols]
   * @param [out] b scalar constant in the decision function
   */
  void Get(const math_t *alpha, const math_t *f, math_t **dual_coefs,
           int *n_support, int **idx, math_t **x_support, math_t *b) {
    GetSupportVectorIndices(alpha, n_support, idx);
    if (*n_support > 0) {
      *x_support = CollectSupportVectors(*idx, *n_support);
      *dual_coefs = GetDualCoefs(*n_support, alpha);
      *b = CalcB(alpha, f);
    }
  }

  /**
   * Collect support vectors into a contiguous buffer
   *
   * @param [in] idx indices of support vectors, size [n_support]
   * @param [in] n_support number of support vectors
   * @return pointer to a newly allocated device buffer that stores the support
   *   vectors, size [n_suppor*n_cols]
  */
  math_t *CollectSupportVectors(const int *idx, int n_support) {
    math_t *x_support = (math_t *)allocator->allocate(
      n_support * n_cols * sizeof(math_t), stream);
    // Collect support vectors into a contiguous block
    get_rows<<<ceildiv(n_support * n_cols, TPB), TPB, 0, stream>>>(
      x, n_rows, n_cols, x_support, n_support, idx);
    CUDA_CHECK(cudaPeekAtLastError());
    return x_support;
  }

  /* Calculate coefficients = alpha * y
   * @param n_support number of support vertors
   * @param alpha
   * @return buffer with dual coefficients, size [n_support]
   */
  math_t *GetDualCoefs(int n_support, const math_t *alpha) {
    auto allocator = handle.getDeviceAllocator();
    math_t *dual_coefs =
      (math_t *)allocator->allocate(n_support * sizeof(math_t), stream);
    device_buffer<math_t> math_tmp(allocator, stream, n_rows);
    // Calculate dual coefficients = alpha * y
    LinAlg::binaryOp(
      math_tmp.data(), alpha, y, n_rows,
      [] __device__(math_t a, math_t y) { return a * y; }, stream);
    // Return only the non-zero coefficients
    // Flag.data() still contains the mask for selecting support vectors
    cub::DeviceSelect::Flagged(cub_storage.data(), cub_bytes, math_tmp.data(),
                               flag.data(), dual_coefs, d_num_selected.data(),
                               n_rows, stream);
    return dual_coefs;
  }

  /**
   * Flag support vectors and also collect their indices.
   * Support vectors are the vectors where alpha > 0.
   *
   * On exit, the flag member variable will be initialized as
   *  flag[i] = alpha[i] > 0
   *
   * @param [in] alpha dual coefficients, size [n_rows]
   * @param [out] n_support number of support vectors
   * @param [out] idx indices of the suport vectors, size [n_support]
   */
  void GetSupportVectorIndices(const math_t *alpha, int *n_support, int **idx) {
    *n_support = SelectByAlpha(
      alpha, n_rows, f_idx.data(),
      [] __device__(math_t a) -> bool { return 0 < a; }, idx_selected.data());
    if (*n_support > 0) {
      *idx = (int *)allocator->allocate((*n_support) * sizeof(int), stream);
      copy(*idx, idx_selected.data(), *n_support, stream);
    }
  }

  /**
   * Calculate the b constant in the decision function.
   *
   * @param [in] alpha dual coefficients, size [n_rows]
   * @param [in] f optimality indicator vector, size [n_rows]
   * @return the value of b
 */
  math_t CalcB(const math_t *alpha, const math_t *f) {
    // We know that for an unbound support vector i, the decision function
    // (before taking the sign) has value F(x_i) = y_i, where
    // F(x_i) = \sum_j y_j \alpha_j K(x_j, x_i) + b, and j runs through all
    // support vectors. The constant b can be expressed from these formulas.
    // Note that F and f denote different quantities. The lower case f is the
    // optimality indicator vector defined as
    // f_i = y_i - \sum_j y_j \alpha_j K(x_j, x_i).
    // For unbound support vectors f_i = -b.

    // Select f for unbound support vectors (0 < alpha < C)
    math_t C = this->C;
    auto select = [C] __device__(math_t a) -> bool { return 0 < a && a < C; };
    int n_free = SelectByAlpha(alpha, n_rows, f, select, val_selected.data());
    if (n_free > 0) {
      cub::DeviceReduce::Sum(cub_storage.data(), cub_bytes, val_selected.data(),
                             d_val_reduced.data(), n_free, stream);
      math_t sum;
      updateHost(&sum, d_val_reduced.data(), 1, stream);
      return -sum / n_free;
    } else {
      // All support vectors are bound. Let's define
      // b_up = min {f_i | i \in I_upper} and
      // b_low = max {f_i | i \in I_lower}
      // Any value in the interval [b_low, b_up] would be allowable for b,
      // we will select in the middle point b = -(b_low + b_up)/2
      math_t b_up = SelectReduce(alpha, f, true, set_upper);
      math_t b_low = SelectReduce(alpha, f, false, set_lower);
      return -(b_up + b_low) / 2;
    }
  }

  std::shared_ptr<deviceAllocator> allocator;

 private:
  const cumlHandle_impl &handle;
  cudaStream_t stream;

  int n_rows;       //!< number of training vectors
  int n_cols;       //!< number of features
  const math_t *x;  //!< training vectors
  const math_t *y;  //!< labels
  math_t C;

  const int TPB = 256;  // threads per block
  // Temporary variables used by cub in GetResults
  device_buffer<int> d_num_selected;
  device_buffer<math_t> d_val_reduced;
  device_buffer<char> cub_storage;
  size_t cub_bytes = 0;

  // Helper arrays for collecting the results
  device_buffer<int> f_idx;
  device_buffer<int> idx_selected;
  device_buffer<math_t> val_selected;
  device_buffer<bool> flag;

  /* Allocate cub temporary buffers for GetResults
    */
  void InitCubBuffers() {
    size_t cub_bytes2 = 0;
    // Query the size of required workspace buffer
    math_t *p = nullptr;
    cub::DeviceSelect::Flagged(NULL, cub_bytes, f_idx.data(), flag.data(),
                               f_idx.data(), d_num_selected.data(), n_rows,
                               stream);
    cub::DeviceSelect::Flagged(NULL, cub_bytes2, p, flag.data(), p,
                               d_num_selected.data(), n_rows, stream);
    cub_bytes = max(cub_bytes, cub_bytes2);
    cub::DeviceReduce::Sum(NULL, cub_bytes2, val_selected.data(),
                           d_val_reduced.data(), n_rows, stream);
    cub_bytes = max(cub_bytes, cub_bytes2);
    cub::DeviceReduce::Min(NULL, cub_bytes2, val_selected.data(),
                           d_val_reduced.data(), n_rows, stream);
    cub_bytes = max(cub_bytes, cub_bytes2);
    cub_storage.resize(cub_bytes, stream);
  }

  /**
  * Filter values based on the corresponding alpha values.
  * @tparam select_op lambda selection criteria
  * @tparam valType type of values that will be selected
  * @param [in] alpha dual coefficients, size [n]
  * @param [in] n number of dual coefficients
  * @param [in] val values to filter, size [n]
  * @param [out] out buffer size [n]
  * @return number of selected elements
  */
  template <typename select_op, typename valType>
  int SelectByAlpha(const math_t *alpha, int n, const valType *val,
                    select_op op, valType *out) {
    set_flag<<<ceildiv(n, TPB), TPB, 0, stream>>>(flag.data(), alpha, n, op);
    CUDA_CHECK(cudaPeekAtLastError());
    cub::DeviceSelect::Flagged(cub_storage.data(), cub_bytes, val, flag.data(),
                               out, d_num_selected.data(), n, stream);
    int n_selected;
    updateHost(&n_selected, d_num_selected.data(), 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return n_selected;
  }

  /** Select values from f, and do a min or max reduction on them.
   * @param [in] alpha dual coefficients, size [n_rows]
   * @param [in] f optimality indicator vector, size [n_rows]
   * @param flag_op operation to flag values for selection (set_upper/lower)
   * @param return the reduced value.
   */
  math_t SelectReduce(const math_t *alpha, const math_t *f, bool min,
                      void (*flag_op)(bool *, int, const math_t *,
                                      const math_t *, math_t)) {
    flag_op<<<ceildiv(n_rows, TPB), TPB, 0, stream>>>(flag.data(), n_rows,
                                                      alpha, y, C);
    CUDA_CHECK(cudaPeekAtLastError());
    cub::DeviceSelect::Flagged(cub_storage.data(), cub_bytes, f, flag.data(),
                               val_selected.data(), d_num_selected.data(),
                               n_rows, stream);
    int n_selected;
    updateHost(&n_selected, d_num_selected.data(), 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    math_t res = 0;
    if (n_selected > 0) {
      if (min) {
        cub::DeviceReduce::Min(cub_storage.data(), cub_bytes,
                               val_selected.data(), d_val_reduced.data(),
                               n_selected, stream);
      } else {
        cub::DeviceReduce::Max(cub_storage.data(), cub_bytes,
                               val_selected.data(), d_val_reduced.data(),
                               n_selected, stream);
      }
      updateHost(&res, d_val_reduced.data(), 1, stream);
    } else {
      std::cerr << "Error: empty set in calcB\n";
    }
    return res;
  }
};  // namespace SVM

};  // namespace SVM
};  // end namespace ML
