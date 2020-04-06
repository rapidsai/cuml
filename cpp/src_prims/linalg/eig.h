/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <cuda_runtime_api.h>
#include <cuml/common/cuml_allocator.hpp>
#include "common/device_buffer.hpp"
#include "cuda_utils.h"
#include "cusolver_wrappers.h"
#include "matrix/matrix.h"

namespace MLCommon {
namespace LinAlg {

/**
 * @defgroup eig decomp with divide and conquer method for the column-major
 * symmetric matrices
 * @param in the input buffer (symmetric matrix that has real eig values and
 * vectors.
 * @param n_rows: number of rows of the input
 * @param n_cols: number of cols of the input
 * @param eig_vectors: eigenvectors
 * @param eig_vals: eigen values
 * @param cusolverH cusolver handle
 * @param stream cuda stream
 * @param allocator device allocator for temporary buffers during computation
 * @{
 */
template <typename math_t>
void eigDC(const math_t *in, int n_rows, int n_cols, math_t *eig_vectors,
           math_t *eig_vals, cusolverDnHandle_t cusolverH, cudaStream_t stream,
           std::shared_ptr<deviceAllocator> allocator) {
  int lwork;
  CUSOLVER_CHECK(cusolverDnsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
                                            CUBLAS_FILL_MODE_UPPER, n_rows, in,
                                            n_cols, eig_vals, &lwork));

  device_buffer<math_t> d_work(allocator, stream, lwork);
  device_buffer<int> d_dev_info(allocator, stream, 1);

  MLCommon::Matrix::copy(in, eig_vectors, n_rows, n_cols, stream);

  CUSOLVER_CHECK(cusolverDnsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
                                 CUBLAS_FILL_MODE_UPPER, n_rows, eig_vectors,
                                 n_cols, eig_vals, d_work.data(), lwork,
                                 d_dev_info.data(), stream));
  CUDA_CHECK(cudaGetLastError());

  int dev_info;
  updateHost(&dev_info, d_dev_info.data(), 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ASSERT(dev_info == 0,
         "eig.h: eigensolver couldn't converge to a solution. "
         "This usually occurs when some of the features do not vary enough.");
}

enum EigVecMemUsage { OVERWRITE_INPUT, COPY_INPUT };

#if CUDART_VERSION >= 10010

/**
 * @defgroup eig decomp with divide and conquer method for the column-major
 * symmetric matrices
 * @param in the input buffer (symmetric matrix that has real eig values and
 * vectors.
 * @param n_rows: number of rows of the input
 * @param n_cols: number of cols of the input
 * @param n_eig_vals: number of eigenvectors to be generated
 * @param eig_vectors: eigenvectors
 * @param eig_vals: eigen values
 * @param cusolverH cusolver handle
 * @param stream cuda stream
 * @param allocator device allocator for temporary buffers during computation
 * @{
 */
template <typename math_t>
void eigSelDC(math_t *in, int n_rows, int n_cols, int n_eig_vals,
              math_t *eig_vectors, math_t *eig_vals, EigVecMemUsage memUsage,
              cusolverDnHandle_t cusolverH, cudaStream_t stream,
              std::shared_ptr<deviceAllocator> allocator) {
  int lwork;
  int h_meig;

  CUSOLVER_CHECK(cusolverDnsyevdx_bufferSize(
    cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I,
    CUBLAS_FILL_MODE_UPPER, n_rows, in, n_cols, math_t(0.0), math_t(0.0),
    n_cols - n_eig_vals + 1, n_cols, &h_meig, eig_vals, &lwork));

  device_buffer<math_t> d_work(allocator, stream, lwork);
  device_buffer<int> d_dev_info(allocator, stream, 1);
  device_buffer<math_t> d_eig_vectors(allocator, stream, 0);

  if (memUsage == OVERWRITE_INPUT) {
    CUSOLVER_CHECK(cusolverDnsyevdx(
      cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I,
      CUBLAS_FILL_MODE_UPPER, n_rows, in, n_cols, math_t(0.0), math_t(0.0),
      n_cols - n_eig_vals + 1, n_cols, &h_meig, eig_vals, d_work.data(), lwork,
      d_dev_info.data(), stream));
  } else if (memUsage == COPY_INPUT) {
    d_eig_vectors.resize(n_rows * n_cols, stream);
    MLCommon::Matrix::copy(in, d_eig_vectors.data(), n_rows, n_cols, stream);

    CUSOLVER_CHECK(cusolverDnsyevdx(
      cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I,
      CUBLAS_FILL_MODE_UPPER, n_rows, eig_vectors, n_cols, math_t(0.0),
      math_t(0.0), n_cols - n_eig_vals + 1, n_cols, &h_meig, eig_vals,
      d_work.data(), lwork, d_dev_info.data(), stream));
  }

  CUDA_CHECK(cudaGetLastError());

  int dev_info;
  updateHost(&dev_info, d_dev_info.data(), 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ASSERT(dev_info == 0,
         "eig.h: eigensolver couldn't converge to a solution. "
         "This usually occurs when some of the features do not vary enough.");

  if (memUsage == OVERWRITE_INPUT) {
    Matrix::truncZeroOrigin(in, n_rows, eig_vectors, n_rows, n_eig_vals,
                            stream);
  } else if (memUsage == COPY_INPUT) {
    Matrix::truncZeroOrigin(d_eig_vectors.data(), n_rows, eig_vectors, n_rows,
                            n_eig_vals, stream);
  }
}

#endif

/**
 * @defgroup overloaded function for eig decomp with Jacobi method for the
 * column-major symmetric matrices (in parameter)
 * @param n_rows: number of rows of the input
 * @param n_cols: number of cols of the input
 * @param eig_vectors: eigenvectors
 * @param eig_vals: eigen values
 * @param tol: error tolerance for the jacobi method. Algorithm stops when the
 * error is below tol
 * @param sweeps: number of sweeps in the Jacobi algorithm. The more the better
 * accuracy.
 * @param cusolverH cusolver handle
 * @param allocator device allocator for temporary buffers during computation
 * @{
 */
template <typename math_t>
void eigJacobi(const math_t *in, int n_rows, int n_cols, math_t *eig_vectors,
               math_t *eig_vals, cusolverDnHandle_t cusolverH,
               cudaStream_t stream, std::shared_ptr<deviceAllocator> allocator,
               math_t tol = 1.e-7, int sweeps = 15) {
  syevjInfo_t syevj_params = nullptr;
  CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params));
  CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(syevj_params, tol));
  CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(syevj_params, sweeps));

  int lwork;
  CUSOLVER_CHECK(cusolverDnsyevj_bufferSize(
    cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n_rows,
    eig_vectors, n_cols, eig_vals, &lwork, syevj_params));

  device_buffer<math_t> d_work(allocator, stream, lwork);
  device_buffer<int> dev_info(allocator, stream, 1);

  MLCommon::Matrix::copy(in, eig_vectors, n_rows, n_cols, stream);

  CUSOLVER_CHECK(cusolverDnsyevj(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
                                 CUBLAS_FILL_MODE_UPPER, n_rows, eig_vectors,
                                 n_cols, eig_vals, d_work.data(), lwork,
                                 dev_info.data(), syevj_params, stream));

  int executed_sweeps;
  CUSOLVER_CHECK(
    cusolverDnXsyevjGetSweeps(cusolverH, syevj_params, &executed_sweeps));

  CUDA_CHECK(cudaGetLastError());
  CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params));
}

};  // end namespace LinAlg
};  // end namespace MLCommon
