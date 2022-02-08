/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <iostream>
#include <limits>
#include <math.h>
#include <memory>
#include <raft/cuda_utils.cuh>

#include "ws_util.cuh"
#include <cub/device/device_select.cuh>
#include <linalg/init.h>
#include <raft/cudart_utils.h>
#include <raft/linalg/add.hpp>
#include <raft/linalg/map_then_reduce.hpp>
#include <raft/linalg/unary_op.hpp>
#include <raft/matrix/matrix.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace ML {
namespace SVM {

template <typename math_t, typename Lambda>
__global__ void set_flag(bool* flag, const math_t* alpha, int n, Lambda op)
{
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
   * @param y target labels (values +/-1), size [n_train]
   * @param n_rows number of training vectors
   * @param n_cols number of features
   * @param C penalty parameter
   */
  Results(const raft::handle_t& handle,
          const math_t* x,
          const math_t* y,
          int n_rows,
          int n_cols,
          const math_t* C,
          SvmType svmType)
    : rmm_alloc(rmm::mr::get_current_device_resource()),
      stream(handle.get_stream()),
      handle(handle),
      n_rows(n_rows),
      n_cols(n_cols),
      x(x),
      y(y),
      C(C),
      svmType(svmType),
      n_train(svmType == EPSILON_SVR ? n_rows * 2 : n_rows),
      cub_storage(0, stream),
      d_num_selected(stream),
      d_val_reduced(stream),
      f_idx(n_train, stream),
      idx_selected(n_train, stream),
      val_selected(n_train, stream),
      val_tmp(n_train, stream),
      flag(n_train, stream)
  {
    InitCubBuffers();
    MLCommon::LinAlg::range(f_idx.data(), n_train, stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
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
   * @param [in] alpha dual coefficients, size [n_train]
   * @param [in] f optimality indicator vector, size [n_train]
   * @param [out] dual_coefs size [n_support]
   * @param [out] n_support number of support vectors
   * @param [out] idx the original training set indices of the support vectors, size [n_support]
   * @param [out] x_support support vectors in column major format, size [n_support, n_cols]
   * @param [out] b scalar constant in the decision function
   */
  void Get(const math_t* alpha,
           const math_t* f,
           math_t** dual_coefs,
           int* n_support,
           int** idx,
           math_t** x_support,
           math_t* b)
  {
    CombineCoefs(alpha, val_tmp.data());
    GetDualCoefs(val_tmp.data(), dual_coefs, n_support);
    *b = CalcB(alpha, f, *n_support);
    if (*n_support > 0) {
      *idx       = GetSupportVectorIndices(val_tmp.data(), *n_support);
      *x_support = CollectSupportVectors(*idx, *n_support);
    } else {
      *dual_coefs = nullptr;
      *idx        = nullptr;
      *x_support  = nullptr;
    }
    // Make sure that all pending GPU calculations finished before we return
    handle.sync_stream(stream);
  }

  /**
   * Collect support vectors into a contiguous buffer
   *
   * @param [in] idx indices of support vectors, size [n_support]
   * @param [in] n_support number of support vectors
   * @return pointer to a newly allocated device buffer that stores the support
   *   vectors, size [n_suppor*n_cols]
   */
  math_t* CollectSupportVectors(const int* idx, int n_support)
  {
    math_t* x_support = (math_t*)rmm_alloc->allocate(n_support * n_cols * sizeof(math_t), stream);
    // Collect support vectors into a contiguous block
    raft::matrix::copyRows(x, n_rows, n_cols, x_support, idx, n_support, stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    return x_support;
  }

  /**
   * @brief Combine alpha parameters and labels to get SVM coefficients.
   *
   * The output coefficients are the values that can be used directly
   * to calculate the decision function:
   * \f[ f(\bm(x)) = \sum_{i=1}^{n_rows} coef_i K(\bm{x}_i,\bm{x}) + b. \f]
   *
   * Here coefs includes coefficients with zero value.
   *
   * For a classifier, \f$ coef_i = y_i * \alpha_i (i \in [0..n-1])\f$,
   * For a regressor \f$ coef_i = y_i * \alpha_i  + y_{i+n/2} * alpha_{i+n/2},
   * (i \in [0..n/2-1]) \f$
   *
   * @param [in] alpha device array of dual coefficients, size [n_train]
   * @param [out] coef device array of SVM coefficients size [n_rows]
   */
  void CombineCoefs(const math_t* alpha, math_t* coef)
  {
    // Calculate dual coefficients = alpha * y
    raft::linalg::binaryOp(
      coef, alpha, y, n_train, [] __device__(math_t a, math_t y) { return a * y; }, stream);

    if (svmType == EPSILON_SVR) {
      // for regression the final coefficients are
      // coef[0..n-rows-1] = alpha[0..nrows-1] - alpha[nrows..2*n_rows-1]
      raft::linalg::add(coef, coef, coef + n_rows, n_rows, stream);
    }
  }

  /** Return non zero dual coefficients.
   *
   * @param [in] val_tmp device pointer with dual coefficients
   * @param [out] dual_coefs device pointer of non-zero dual coefficiens,
   *   unallocated on entry, on exit size [n_support]
   * @param [out] n_support number of support vectors
   */
  void GetDualCoefs(const math_t* val_tmp, math_t** dual_coefs, int* n_support)
  {
    // Return only the non-zero coefficients
    auto select_op = [] __device__(math_t a) { return 0 != a; };
    *n_support     = SelectByCoef(val_tmp, n_rows, val_tmp, select_op, val_selected.data());
    *dual_coefs    = (math_t*)rmm_alloc->allocate(*n_support * sizeof(math_t), stream);
    raft::copy(*dual_coefs, val_selected.data(), *n_support, stream);
    handle.sync_stream(stream);
  }

  /**
   * Flag support vectors and also collect their indices.
   * Support vectors are the vectors where alpha > 0.
   *
   * @param [in] coef dual coefficients, size [n_rows]
   * @param [in] n_support number of support vectors
   * @return indices of the support vectors, size [n_support]
   */
  int* GetSupportVectorIndices(const math_t* coef, int n_support)
  {
    auto select_op = [] __device__(math_t a) -> bool { return 0 != a; };
    SelectByCoef(coef, n_rows, f_idx.data(), select_op, idx_selected.data());
    int* idx = (int*)rmm_alloc->allocate(n_support * sizeof(int), stream);
    raft::copy(idx, idx_selected.data(), n_support, stream);
    return idx;
  }

  /**
   * Calculate the b constant in the decision function.
   *
   * @param [in] alpha dual coefficients, size [n_rows]
   * @param [in] f optimality indicator vector, size [n_rows]
   * @return the value of b
   */
  math_t CalcB(const math_t* alpha, const math_t* f, int n_support)
  {
    if (n_support == 0) {
      math_t f_sum;
      cub::DeviceReduce::Sum(
        cub_storage.data(), cub_bytes, f, d_val_reduced.data(), n_train, stream);
      raft::update_host(&f_sum, d_val_reduced.data(), 1, stream);
      return -f_sum / n_train;
    }
    // We know that for an unbound support vector i, the decision function
    // (before taking the sign) has value F(x_i) = y_i, where
    // F(x_i) = \sum_j y_j \alpha_j K(x_j, x_i) + b, and j runs through all
    // support vectors. The constant b can be expressed from these formulas.
    // Note that F and f denote different quantities. The lower case f is the
    // optimality indicator vector defined as
    // f_i = - y_i + \sum_j y_j \alpha_j K(x_j, x_i).
    // For unbound support vectors f_i = -b.

    // Select f for unbound support vectors (0 < alpha < C)
    int n_free = SelectUnboundSV(alpha, n_train, f, val_selected.data());
    if (n_free > 0) {
      cub::DeviceReduce::Sum(
        cub_storage.data(), cub_bytes, val_selected.data(), d_val_reduced.data(), n_free, stream);
      math_t sum;
      raft::update_host(&sum, d_val_reduced.data(), 1, stream);
      return -sum / n_free;
    } else {
      // All support vectors are bound. Let's define
      // b_up = min {f_i | i \in I_upper} and
      // b_low = max {f_i | i \in I_lower}
      // Any value in the interval [b_low, b_up] would be allowable for b,
      // we will select in the middle point b = -(b_low + b_up)/2
      math_t b_up  = SelectReduce(alpha, f, true, set_upper);
      math_t b_low = SelectReduce(alpha, f, false, set_lower);
      return -(b_up + b_low) / 2;
    }
  }

  /**
   * @brief Select values for unbound support vectors (not bound by C).
   * @tparam valType type of values that will be selected
   * @param [in] alpha dual coefficients, size [n]
   * @param [in] n number of dual coefficients
   * @param [in] val values to filter, size [n]
   * @param [out] out buffer size [n]
   * @return number of selected elements
   */
  template <typename valType>
  int SelectUnboundSV(const math_t* alpha, int n, const valType* val, valType* out)
  {
    auto select = [] __device__(math_t a, math_t C) -> bool { return 0 < a && a < C; };
    raft::linalg::binaryOp(flag.data(), alpha, C, n, select, stream);
    cub::DeviceSelect::Flagged(
      cub_storage.data(), cub_bytes, val, flag.data(), out, d_num_selected.data(), n, stream);
    int n_selected;
    raft::update_host(&n_selected, d_num_selected.data(), 1, stream);
    handle.sync_stream(stream);
    return n_selected;
  }

  rmm::mr::device_memory_resource* rmm_alloc;

 private:
  const raft::handle_t& handle;
  cudaStream_t stream;

  int n_rows;       //!< number of rows in the training vector matrix
  int n_cols;       //!< number of features
  const math_t* x;  //!< training vectors
  const math_t* y;  //!< labels
  const math_t* C;  //!< penalty parameter
  SvmType svmType;  //!< SVM problem type: SVC or SVR
  int n_train;      //!< number of training vectors (including duplicates for SVR)

  const int TPB = 256;  // threads per block
  // Temporary variables used by cub in GetResults
  rmm::device_scalar<int> d_num_selected;
  rmm::device_scalar<math_t> d_val_reduced;
  rmm::device_uvector<char> cub_storage;
  size_t cub_bytes = 0;

  // Helper arrays for collecting the results
  rmm::device_uvector<int> f_idx;
  rmm::device_uvector<int> idx_selected;
  rmm::device_uvector<math_t> val_selected;
  rmm::device_uvector<math_t> val_tmp;
  rmm::device_uvector<bool> flag;

  /* Allocate cub temporary buffers for GetResults
   */
  void InitCubBuffers()
  {
    size_t cub_bytes2 = 0;
    // Query the size of required workspace buffer
    math_t* p = nullptr;
    cub::DeviceSelect::Flagged(NULL,
                               cub_bytes,
                               f_idx.data(),
                               flag.data(),
                               f_idx.data(),
                               d_num_selected.data(),
                               n_train,
                               stream);
    cub::DeviceSelect::Flagged(
      NULL, cub_bytes2, p, flag.data(), p, d_num_selected.data(), n_train, stream);
    cub_bytes = max(cub_bytes, cub_bytes2);
    cub::DeviceReduce::Sum(
      NULL, cub_bytes2, val_selected.data(), d_val_reduced.data(), n_train, stream);
    cub_bytes = max(cub_bytes, cub_bytes2);
    cub::DeviceReduce::Min(
      NULL, cub_bytes2, val_selected.data(), d_val_reduced.data(), n_train, stream);
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
  int SelectByCoef(const math_t* coef, int n, const valType* val, select_op op, valType* out)
  {
    set_flag<<<raft::ceildiv(n, TPB), TPB, 0, stream>>>(flag.data(), coef, n, op);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    cub::DeviceSelect::Flagged(
      cub_storage.data(), cub_bytes, val, flag.data(), out, d_num_selected.data(), n, stream);
    int n_selected;
    raft::update_host(&n_selected, d_num_selected.data(), 1, stream);
    handle.sync_stream(stream);
    return n_selected;
  }

  /** Select values from f, and do a min or max reduction on them.
   * @param [in] alpha dual coefficients, size [n_train]
   * @param [in] f optimality indicator vector, size [n_train]
   * @param flag_op operation to flag values for selection (set_upper/lower)
   * @param return the reduced value.
   */
  math_t SelectReduce(const math_t* alpha,
                      const math_t* f,
                      bool min,
                      void (*flag_op)(bool*, int, const math_t*, const math_t*, const math_t*))
  {
    flag_op<<<raft::ceildiv(n_train, TPB), TPB, 0, stream>>>(flag.data(), n_train, alpha, y, C);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    cub::DeviceSelect::Flagged(cub_storage.data(),
                               cub_bytes,
                               f,
                               flag.data(),
                               val_selected.data(),
                               d_num_selected.data(),
                               n_train,
                               stream);
    int n_selected;
    raft::update_host(&n_selected, d_num_selected.data(), 1, stream);
    handle.sync_stream(stream);
    math_t res = 0;
    ASSERT(n_selected > 0,
           "Incorrect training: cannot calculate the constant in the decision "
           "function");
    if (min) {
      cub::DeviceReduce::Min(cub_storage.data(),
                             cub_bytes,
                             val_selected.data(),
                             d_val_reduced.data(),
                             n_selected,
                             stream);
    } else {
      cub::DeviceReduce::Max(cub_storage.data(),
                             cub_bytes,
                             val_selected.data(),
                             d_val_reduced.data(),
                             n_selected,
                             stream);
    }
    raft::update_host(&res, d_val_reduced.data(), 1, stream);
    return res;
  }
};  // namespace SVM

};  // namespace SVM
};  // end namespace ML
