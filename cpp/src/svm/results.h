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
#include "gram/grammatrix.h"
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
   * @tparam math_t
   * @param handle cuML handle implementation
   * @param x training vectors in column major format, size [n_rows x n_cols]
   * @param y target labels (values +/-1), size [n_rows]
   * @param n_rows number of training vectors
   * @param n_cols number of features
   * @param C penalty parameter
   * @param kernel pointer to a kernel class
   */
  Results(const cumlHandle_impl &handle, const math_t *x, const math_t *y,
          int n_rows, int n_cols, math_t C,
          GramMatrix::GramMatrixBase<math_t> *kernel)
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
      f_idx(handle.getDeviceAllocator(), stream, n_rows),
      idx_selected(handle.getDeviceAllocator(), stream, n_rows),
      flag(handle.getDeviceAllocator(), stream, n_rows),
      kernel(kernel) {
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
   * @param [out] dual_coefs size [n_support]
   * @param [out] n_support number of support vectors
   * @param [out] idx the original training set indices of the support vectors, size [n_support]
   * @param [out] x_support support vectors in column major format, size [n_support, n_cols]
   * @param [out] b scalar constant in the decision function
   */
  void Get(const math_t *alpha, math_t **dual_coefs, int *n_support, int **idx,
           math_t **x_support, math_t *b) {
    GetSupportVectorIndices(alpha, n_support, idx);
    if (*n_support > 0) {
      *x_support = CollectSupportVectors(*idx, *n_support);
      *dual_coefs = GetDualCoefs(*n_support, alpha);
      *b = CalcB(*x_support, *n_support, *dual_coefs, *idx);
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
    cub::DeviceSelect::Flagged(cub_storage.data(), cub_bytes, math_tmp.data(),
                               flag.data(), dual_coefs, d_num_selected.data(),
                               n_rows, stream);
    return dual_coefs;
  }

  /**
   * Flag support vectors and also collect their indices.
   * Support vectors are the vectors where alpha > 0.
   *
   * @param [in] alpha dual coefficients, size [n_rows]
   * @param [in] n_rows number of traning vectors
   * @param [out] n_support number of support vectors
   * @param [out] idx indices of the suport vectors, size [n_support]
   * @param [out] flag[i] = alpha[i] > 0, size [n_rows]
   */
  void GetSupportVectorIndices(const math_t *alpha, int *n_support, int **idx) {
    //Set flags true for non-zero alpha
    set_flag<<<ceildiv(n_rows, TPB), TPB, 0, stream>>>(
      flag.data(), alpha, n_rows, [] __device__(math_t a) { return a > 0; });
    CUDA_CHECK(cudaPeekAtLastError());
    // This would be better, but we have different input and output types:
    //LinAlg::unaryOp(flag, alpha, n_rows,
    //  []__device__(math_t a) { return a > 0;}, stream);

    // Select indices for non-zero dual coefficient
    cub::DeviceSelect::Flagged(cub_storage.data(), cub_bytes, f_idx.data(),
                               flag.data(), idx_selected.data(),
                               d_num_selected.data(), n_rows, stream);

    updateHost(n_support, d_num_selected.data(), 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (*n_support > 0) {
      *idx = (int *)allocator->allocate((*n_support) * sizeof(int), stream);
      copy(*idx, idx_selected.data(), *n_support, stream);
    }
  }

  /**
  * Find an unbound support vector.
  * A support vector is unbound if its dual coefficient is smaller than C.
  * This subroutine fills idx with indices of support vectors, and also returns
  * the value of the first index.
  * @param dual_coefs dual coefficients (=alpha*y), size [n_support]
  * @param n_support number of dual coefficients
  * @param [out] idx device buffer used to select unbound indices, size [n_support]
  * @return an index of the first unbound support vector
  */
  int get_unbound_idx(const math_t *dual_coefs, int n_support, int *idx) {
    // Set flags true for 0 < alpha < C, these are the unbound support vectors
    // Note that abs(dual_coefs) > 0
    math_t C = this->C;
    set_flag<<<ceildiv(n_support, TPB), TPB, 0, stream>>>(
      flag.data(), dual_coefs, n_support,
      [C] __device__(math_t a) -> bool { return abs(a) < C; });
    CUDA_CHECK(cudaPeekAtLastError());
    //LinAlg::unaryOp(flag, dual_coefs, n_support,
    //  [C]__device__(math_t a) {  abs(a) < C;}, stream);

    // Select the first the unbound support vector
    cub::DeviceSelect::Flagged(cub_storage.data(), cub_bytes, f_idx.data(),
                               flag.data(), idx, d_num_selected.data(),
                               n_support, stream);
    int n_selected;
    updateHost(&n_selected, d_num_selected.data(), 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (n_selected == 0) {
      // should not really happen in practice, that woud mean that all support
      // vectors are bound (alpha==C). If it does happen, then we
      // just use one of the bound support vectors
      copy(idx, f_idx.data(), 1, stream);
    }
    int idx_unbound;
    updateHost(&idx_unbound, idx, 1, stream);
    return idx_unbound;
  }
  /**
   * Calculate the b constant in the decision function.
   *
   * @param [in] x_support support vectors, size [n_support*n_cols]
   * @param [in] n_support number of support vectors
   * @param [in] dual_coefs dual coefficients, size [n_support]
   * @param [in] support_idx indices of support vectors size [n_support]
   * @return the value of b
   */
  math_t CalcB(const math_t *x_support, int n_support, const math_t *dual_coefs,
               const int *support_idx) {
    // To calculate b, we know that for an unbound support vector i, the
    // decision function has value f(x_i) = y_i.
    // We also know that f(x_i) = s + b, where
    // s = \sum_j y_j \alpha_j K(x_j, x_i), here j runs through all support
    // vectors. We will calculate b as  b = y_i - s;

    int idx_val = get_unbound_idx(dual_coefs, n_support, idx_selected.data());

    device_buffer<math_t> K(allocator, stream, n_support);
    kernel->evaluate(x_support, n_support, n_cols, x_support + idx_val, 1,
                     K.data(), stream, n_support, n_support, n_support);

    // Calculate s = \sum y_j \alpha_j K(x_j, x_i)
    device_buffer<math_t> s(allocator, stream, 1);
    auto mult = [] __device__(const math_t x, const math_t y) { return x * y; };
    LinAlg::mapThenSumReduce(s.data(), n_support, mult, stream, dual_coefs,
                             K.data());
    math_t s_val;
    updateHost(&s_val, s.data(), 1, stream);

    // Get y that corresponds to the unbound support vector
    int sv_idx;
    updateHost(&sv_idx, support_idx + idx_val, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    math_t y_val;
    updateHost(&y_val, y + sv_idx, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return y_val - s_val;
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
  GramMatrix::GramMatrixBase<math_t> *kernel;

  const int TPB = 256;  // threads per block
  // Temporary variables used by cub in GetResults
  device_buffer<int> d_num_selected;
  device_buffer<char> cub_storage;
  size_t cub_bytes = 0;

  // Helper arrays for collecting the results
  device_buffer<int> f_idx;
  device_buffer<int> idx_selected;
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
    cub_storage.resize(cub_bytes, stream);
  }
};

};  // end namespace SVM
};  // end namespace ML
