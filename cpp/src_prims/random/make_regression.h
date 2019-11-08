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

/* Adapted from scikit-learn
 * https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/_samples_generator.py
 */

#pragma once

#include <algorithm>
#include <cuml/common/cuml_allocator.hpp>

#include "linalg/add.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/init.h"
#include "linalg/qr.h"
#include "linalg/transpose.h"
#include "matrix/matrix.h"
#include "permute.h"
#include "rng.h"
#include "utils.h"

namespace MLCommon {
namespace Random {

/* Internal auxiliary function to help build the singular profile */
template <typename DataT, typename IdxT>
static __global__ void _singular_profile_kernel(DataT* out, IdxT n,
                                                DataT tail_strength,
                                                IdxT rank) {
  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    DataT sval = static_cast<DataT>(tid) / rank;
    DataT low_rank = ((DataT)1.0 - tail_strength) * myExp(-sval * sval);
    DataT tail = tail_strength * myExp((DataT)-0.1 * sval);
    out[tid] = low_rank + tail;
  }
}

/* Internal auxiliary function to generate a low-rank matrix */
template <typename DataT, typename IdxT>
static void _make_low_rank_matrix(DataT* out, IdxT n_rows, IdxT n_cols,
                                  IdxT effective_rank, DataT tail_strength,
                                  Rng& r, cublasHandle_t cublas_handle,
                                  cusolverDnHandle_t cusolver_handle,
                                  std::shared_ptr<deviceAllocator> allocator,
                                  cudaStream_t stream) {
  IdxT n = std::min(n_rows, n_cols);

  // Generate random (ortho normal) vectors with QR decomposition
  device_buffer<DataT> rd_mat_0(allocator, stream);
  device_buffer<DataT> rd_mat_1(allocator, stream);
  rd_mat_0.resize(n_rows * n, stream);
  rd_mat_1.resize(n_cols * n, stream);
  r.normal(rd_mat_0.data(), n_rows * n, (DataT)0.0, (DataT)1.0, stream);
  r.normal(rd_mat_1.data(), n_cols * n, (DataT)0.0, (DataT)1.0, stream);
  device_buffer<DataT> q0(allocator, stream);
  device_buffer<DataT> q1(allocator, stream);
  q0.resize(n_rows * n, stream);
  q1.resize(n_cols * n, stream);
  LinAlg::qrGetQ(rd_mat_0.data(), q0.data(), n_rows, n, cusolver_handle, stream,
                 allocator);
  LinAlg::qrGetQ(rd_mat_1.data(), q1.data(), n_cols, n, cusolver_handle, stream,
                 allocator);

  // Build the singular profile by assembling signal and noise components
  device_buffer<DataT> singular_vec(allocator, stream);
  device_buffer<DataT> singular_mat(allocator, stream);
  singular_vec.resize(n, stream);
  _singular_profile_kernel<<<ceildiv<IdxT>(n, 256), 256, 0, stream>>>(
    singular_vec.data(), n, tail_strength, effective_rank);
  CUDA_CHECK(cudaPeekAtLastError());
  singular_mat.resize(n * n, stream);
  CUDA_CHECK(
    cudaMemsetAsync(singular_mat.data(), 0, n * n * sizeof(DataT), stream));
  Matrix::initializeDiagonalMatrix(singular_vec.data(), singular_mat.data(), n,
                                   n, stream);

  // Generate the column-major matrix
  device_buffer<DataT> temp_q0s(allocator, stream);
  device_buffer<DataT> temp_out(allocator, stream);
  temp_q0s.resize(n_rows * n, stream);
  temp_out.resize(n_rows * n_cols, stream);
  DataT alpha = 1.0, beta = 0.0;
  LinAlg::cublasgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n_rows, n, n,
                     &alpha, q0.data(), n_rows, singular_mat.data(), n, &beta,
                     temp_q0s.data(), n_rows, stream);
  LinAlg::cublasgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n_rows, n_cols, n,
                     &alpha, temp_q0s.data(), n_rows, q1.data(), n_cols, &beta,
                     temp_out.data(), n_rows, stream);

  // Transpose from column-major to row-major
  LinAlg::transpose(temp_out.data(), out, n_rows, n_cols, cublas_handle,
                    stream);
}

/* Internal auxiliary function to permute rows in the given matrix according
 * to a given permutation vector */
template <typename DataT, typename IdxT>
static __global__ void _gather2d_kernel(DataT* out, const DataT* in,
                                        const IdxT* perms, IdxT n_rows,
                                        IdxT n_cols) {
  IdxT tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n_rows) {
    const DataT* row_in = in + n_cols * perms[tid];
    DataT* row_out = out + n_cols * tid;

    for (IdxT i = 0; i < n_cols; i++) {
      row_out[i] = row_in[i];
    }
  }
}

/**
 * @brief GPU-equivalent of sklearn.datasets.make_regression as documented at:
 * https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html
 * 
 * @tparam  DataT  Scalar type
 * @tparam  IdxT   Index type
 * 
 * @param[out]  out             Row-major (samples, features) matrix to store
 *                              the problem data
 * @param[out]  values          Row-major (samples, targets) matrix to store
 *                              the values for the regression problem
 * @param[in]   n_rows          Number of samples
 * @param[in]   n_cols          Number of features
 * @param[in]   n_informative   Number of informative features (non-zero
 *                              coefficients)
 * @param[in]   cublas_handle   cuBLAS handle
 * @param[in]   cusolver_handle cuSOLVER handle
 * @param[in]   allocator       Device memory allocator
 * @param[in]   stream          CUDA stream
 * @param[out]  coef            Row-major (features, targets) matrix to store
 *                              the coefficients used to generate the values
 *                              for the regression problem. If nullptr is
 *                              given, nothing will be written
 * @param[in]   n_targets       Number of targets (generated values per sample)
 * @param[in]   bias            A scalar that will be added to the values
 * @param[in]   effective_rank  The approximate rank of the data matrix (used
 *                              to create correlations in the data). -1 is the
 *                              code to use well-conditioned data
 * @param[in]   tail_strength   The relative importance of the fat noisy tail
 *                              of the singular values profile if
 *                              effective_rank is not -1
 * @param[in]   noise           Standard deviation of the gaussian noise
 *                              applied to the output
 * @param[in]   shuffle         Shuffle the samples and the features
 * @param[in]   seed            Seed for the random number generator
 * @param[in]   type            Random generator type
 */
template <typename DataT, typename IdxT>
void make_regression(DataT* out, DataT* values, IdxT n_rows, IdxT n_cols,
                     IdxT n_informative, cublasHandle_t cublas_handle,
                     cusolverDnHandle_t cusolver_handle,
                     std::shared_ptr<deviceAllocator> allocator,
                     cudaStream_t stream, DataT* coef = nullptr,
                     IdxT n_targets = (IdxT)1, DataT bias = (DataT)0.0,
                     IdxT effective_rank = (IdxT)-1,
                     DataT tail_strength = (DataT)0.5, DataT noise = (DataT)0.0,
                     bool shuffle = true, uint64_t seed = 0ULL,
                     GeneratorType type = GenPhilox) {
  n_informative = std::min(n_informative, n_cols);
  cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
  Rng r(seed, type);

  if (effective_rank < 0) {
    // Randomly generate a well conditioned input set
    r.normal(out, n_rows * n_cols, (DataT)0.0, (DataT)1.0, stream);
  } else {
    // Randomly generate a low rank, fat tail input set
    _make_low_rank_matrix(out, n_rows, n_cols, effective_rank, tail_strength, r,
                          cublas_handle, cusolver_handle, allocator, stream);
  }

  // Use the right output buffer for the values
  device_buffer<DataT> tmp_values(allocator, stream);
  DataT* _values;
  if (shuffle) {
    tmp_values.resize(n_rows * n_targets, stream);
    _values = tmp_values.data();
  } else {
    _values = values;
  }
  // Create a column-major matrix of output values only if it has more
  // than 1 column
  device_buffer<DataT> values_col(allocator, stream);
  DataT* _values_col;
  if (n_targets > 1) {
    values_col.resize(n_rows * n_targets, stream);
    _values_col = values_col.data();
  } else {
    _values_col = _values;
  }

  // Use the right buffer for the coefficients
  device_buffer<DataT> tmp_coef(allocator, stream);
  DataT* _coef;
  if (coef != nullptr && !shuffle) {
    _coef = coef;
  } else {
    tmp_coef.resize(n_cols * n_targets, stream);
    _coef = tmp_coef.data();
  }

  // Generate a ground truth model with only n_informative features
  r.uniform(_coef, n_informative * n_targets, (DataT)1.0, (DataT)100.0, stream);
  if (coef && n_informative != n_cols) {
    CUDA_CHECK(cudaMemsetAsync(
      _coef + n_informative * n_targets, 0,
      (n_cols - n_informative) * n_targets * sizeof(DataT), stream));
  }

  // Compute the output values
  DataT alpha = (DataT)1.0, beta = (DataT)0.0;
  CUBLAS_CHECK(LinAlg::cublasgemm(
    cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, n_rows, n_targets, n_informative,
    &alpha, out, n_cols, _coef, n_targets, &beta, _values_col, n_rows, stream));

  // Transpose the values from column-major to row-major if needed
  if (n_targets > 1) {
    LinAlg::transpose(_values_col, _values, n_rows, n_targets, cublas_handle,
                      stream);
  }

  if (bias != 0.0) {
    // Add bias
    LinAlg::addScalar(_values, _values, bias, n_rows * n_targets, stream);
  }

  device_buffer<DataT> white_noise(allocator, stream);
  if (noise != 0.0) {
    // Add white noise
    white_noise.resize(n_rows * n_targets, stream);
    r.normal(white_noise.data(), n_rows * n_targets, (DataT)0.0, noise, stream);
    LinAlg::add(_values, _values, white_noise.data(), n_rows * n_targets,
                stream);
  }

  if (shuffle) {
    device_buffer<DataT> tmp_out(allocator, stream);
    device_buffer<IdxT> perms_samples(allocator, stream);
    device_buffer<IdxT> perms_features(allocator, stream);
    tmp_out.resize(n_rows * n_cols, stream);
    perms_samples.resize(n_rows, stream);
    perms_features.resize(n_cols, stream);

    constexpr IdxT Nthreads = 256;

    // Shuffle the samples from out to tmp_out
    permute<DataT, IdxT, IdxT>(perms_samples.data(), tmp_out.data(), out,
                               n_cols, n_rows, true, stream);
    IdxT nblks_rows = ceildiv<IdxT>(n_rows, Nthreads);
    _gather2d_kernel<<<nblks_rows, Nthreads, 0, stream>>>(
      values, _values, perms_samples.data(), n_rows, n_targets);
    CUDA_CHECK(cudaPeekAtLastError());

    // Shuffle the features from tmp_out to out
    permute<DataT, IdxT, IdxT>(perms_features.data(), out, tmp_out.data(),
                               n_rows, n_cols, false, stream);

    // Shuffle the coefficients accordingly
    if (coef != nullptr) {
      IdxT nblks_cols = ceildiv<IdxT>(n_cols, Nthreads);
      _gather2d_kernel<<<nblks_cols, Nthreads, 0, stream>>>(
        coef, _coef, perms_features.data(), n_cols, n_targets);
      CUDA_CHECK(cudaPeekAtLastError());
    }
  }
}

}  // namespace Random
}  // namespace MLCommon
