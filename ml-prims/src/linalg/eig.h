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

#include "cuda_utils.h"
#include "cusolver_wrappers.h"
#include "device_allocator.h"
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
 * @param mgr device allocator for temporary buffers during computation
 * @{
 */
template <typename math_t>
void eigDC(const math_t *in, int n_rows, int n_cols, math_t *eig_vectors,
           math_t *eig_vals, cusolverDnHandle_t cusolverH, cudaStream_t stream,
           DeviceAllocator &mgr) {
  int lwork;
  CUSOLVER_CHECK(cusolverDnsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
                                            CUBLAS_FILL_MODE_UPPER, n_rows, in,
                                            n_cols, eig_vals, &lwork));

  math_t *d_work = (math_t *)mgr.alloc(sizeof(math_t) * lwork);
  int *d_dev_info = (int *)mgr.alloc(sizeof(int));

  MLCommon::Matrix::copy(in, eig_vectors, n_rows, n_cols, stream);

  CUSOLVER_CHECK(cusolverDnsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
                                 CUBLAS_FILL_MODE_UPPER, n_rows, eig_vectors,
                                 n_cols, eig_vals, d_work, lwork, d_dev_info,
                                 stream));
  CUDA_CHECK(cudaGetLastError());

  int dev_info;
  updateHost(&dev_info, d_dev_info, 1);
  ASSERT(dev_info == 0,
         "eig.h: eigensolver couldn't converge to a solution. "
         "This usually occurs when some of the features do not vary enough.");

  mgr.free(d_work, stream);
  mgr.free(d_dev_info, stream);
}


/**
 * @defgroup overloaded function for eig decomp with Jacobi method for the
 * column-major symmetric matrices (in parameter)
 * @param n_rows: number of rows of the input
 * @param n_cols: number of cols of the input
 * @param eig_vectors: eigenvectors
 * @param eig_vals: eigen values
 * @{
 */
template <typename math_t>
void eigJacobi(const math_t *in, int n_rows, int n_cols, math_t *eig_vectors,
               math_t *eig_vals, cusolverDnHandle_t cusolverH, cudaStream_t stream) {
  math_t tol = 1.e-7;
  int sweeps = 15;
  eigJacobi(in, eig_vectors, eig_vals, tol, sweeps, n_rows, n_cols, cusolverH, stream);
}

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
 * @param mgr device allocator for temporary buffers during computation
 * @{
 */
template <typename math_t>
void eigJacobi(const math_t *in, int n_rows, int n_cols, math_t *eig_vectors,
               math_t *eig_vals, math_t tol, int sweeps,
               cusolverDnHandle_t cusolverH, cudaStream_t stream, 
               DeviceAllocator &mgr) {
  syevjInfo_t syevj_params = nullptr;
  CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params));
  CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(syevj_params, tol));
  CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(syevj_params, sweeps));

  int lwork;
  CUSOLVER_CHECK(cusolverDnsyevj_bufferSize(
    cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n_rows,
    eig_vectors, n_cols, eig_vals, &lwork, syevj_params));

  math_t *d_work = (math_t *)mgr.alloc(sizeof(math_t) * lwork);
  int *dev_info = (int *)mgr.alloc(sizeof(int));

  MLCommon::Matrix::copy(in, eig_vectors, n_rows, n_cols, stream);

  CUSOLVER_CHECK(cusolverDnsyevj(
    cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n_rows,
    eig_vectors, n_cols, eig_vals, d_work, lwork, dev_info, syevj_params, stream));

  int executed_sweeps;
  CUSOLVER_CHECK(
    cusolverDnXsyevjGetSweeps(cusolverH, syevj_params, &executed_sweeps));

  mgr.free(d_work, stream);
  mgr.free(dev_info, stream);

  CUDA_CHECK(cudaGetLastError());
  CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params));
}

}; // end namespace LinAlg
}; // end namespace MLCommon
