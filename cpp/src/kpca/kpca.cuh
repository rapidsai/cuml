/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <cuml/matrix/kernelparams.h>
#include <cuml/decomposition/params.hpp>
#include <matrix/kernelfactory.cuh>
#include <matrix/kernelmatrices.cuh>
#include <raft/linalg/cublas_wrappers.h>
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>
#include <raft/linalg/eig.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/mr/device/allocator.hpp>
#include <raft/mr/host/allocator.hpp>
#include <tsvd/tsvd.cuh>
#include <common/device_utils.cuh>
#include <rmm/device_vector.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>


/**
 * @brief perform fit operation for kernel PCA. Generates eigenvalues and alphas (unscaled eigenvectors, per sklearn API)
 * @param[in] handle: cuml handle object
 * @param[in] input: data to fit using kernel PCA. Size n_rows x n_cols. The size of the data is indicated in prms.
 * @param[out] alphas: unscaled eigenvectors of the kernel matrix.  scaling occurs in the transform function. Size n_cols * n_components.
 * @param[out] lambdas: lambdas (eigenvalues) of the principal components. Size n_components * 1.
 * @param[in] prms: data structure that includes all the parameters from data size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename value_t, typename enum_solver = ML::solver>
void kpcaFit(const raft::handle_t &handle, value_t *input, value_t *alphas,
             value_t *lambdas, const ML::paramsKPCA &prms, cudaStream_t stream) {
  auto cublas_handle = handle.get_cublas_handle();
  auto allocator = handle.get_device_allocator();
  auto thrust_policy = rmm::exec_policy(stream);

  //  TODO: defer assertions to python layer
  ASSERT(prms.n_cols > 1,
         "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(prms.n_rows > 1,
         "Parameter n_rows: number of rows cannot be less than two");
  ASSERT(
    prms.n_components > 0,
    "Parameter n_components: number of components cannot be less than one");

  int n_components = prms.n_components;
  if (n_components > prms.n_rows) n_components = prms.n_rows;

  MLCommon::Matrix::GramMatrixBase<value_t> *kernel =
    MLCommon::Matrix::KernelFactory<value_t>::create(prms.kernel, cublas_handle);
  
  rmm::device_uvector<value_t> kernel_mat(prms.n_rows * prms.n_rows, stream);
  kernel->evaluate(input, prms.n_rows, prms.n_cols, input, prms.n_rows,
         kernel_mat.data(), false, stream, prms.n_rows, prms.n_rows, prms.n_rows);

  //  create centering matrix (I - 1/nrows)
  rmm::device_uvector<value_t> i_mat(prms.n_rows * prms.n_rows, stream);
  rmm::device_uvector<value_t> diag(prms.n_rows, stream);
  rmm::device_uvector<value_t> centering_mat(prms.n_rows * prms.n_rows, stream);
  thrust::fill(thrust_policy, diag.begin(), diag.end(), 1.0f);
  raft::matrix::initializeDiagonalMatrix(diag.data(), i_mat.data(), prms.n_rows, prms.n_rows, stream);
  value_t inv_n_rows = 1.0/prms.n_rows;
  raft::linalg::subtractScalar(centering_mat.data(), i_mat.data(), inv_n_rows, prms.n_rows * prms.n_rows, stream);
  i_mat.release();
  diag.release();

  //  center the kernel matrix: K' = (I - 1/nrows) K (I - 1/nrows)
  rmm::device_uvector<value_t> temp_mat(prms.n_rows * prms.n_rows, stream);
  raft::linalg::gemm(handle, temp_mat.data(), centering_mat.data(), kernel_mat.data()
                     , prms.n_rows, prms.n_rows, prms.n_rows
                     , true, true, true, stream);
  raft::linalg::gemm(handle, kernel_mat.data(), temp_mat.data(), centering_mat.data()
                     , prms.n_rows, prms.n_rows, prms.n_rows
                     , true, true, true, stream);
  temp_mat.release();
  centering_mat.release();
  
  //  either Jacobi (iterative power method) or DnC eigendecomp
  if (prms.algorithm == enum_solver::COV_EIG_JACOBI) {
    raft::linalg::eigJacobi(handle, kernel_mat.data(), prms.n_rows, prms.n_rows, alphas,
                            lambdas, stream, (value_t)prms.tol,
                            prms.n_iterations);
  } else {
    raft::linalg::eigDC(handle, kernel_mat.data(), prms.n_rows, prms.n_rows, alphas,
                            lambdas, stream);
  }

  raft::matrix::colReverse(alphas, prms.n_rows, prms.n_rows, stream);
  raft::matrix::rowReverse(lambdas, prms.n_rows, 1, stream);
  ML::signFlip(lambdas, prms.n_rows, n_components, alphas, prms.n_cols,
           allocator, stream);
}

/**
 * @brief performs transform operation for the pca. Transforms the data to kernel eigenspace.
 * @param[in] handle: the internal cuml handle object
 * @param[in] input: data to transform. Size n_rows x n_components.
 * @param[in] alphas: principal components of the input data. Size n_rows * n_components.
 * @param[in] lambdas: singular values of the data. Size n_components * 1.
 * @param[out] trans_input:  the transformed data. Size n_cols * n_components.
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename value_t>
void kpcaTransform(const raft::handle_t &handle, value_t *input,
                   value_t *alphas, value_t *lambdas,
                   value_t *trans_input, const ML::paramsKPCA &prms,
                   cudaStream_t stream) {
  //  TODO: defer assertions to python layer
  ASSERT(prms.n_cols > 1,
         "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(prms.n_rows > 0,
         "Parameter n_rows: number of rows cannot be less than one");
  ASSERT(
    prms.n_components > 0,
    "Parameter n_components: number of components cannot be less than one");

  rmm::device_uvector<value_t> sqrt_vals(prms.n_components, stream);
  raft::matrix::seqRoot(lambdas, sqrt_vals.data(), prms.n_components, stream);
  raft::matrix::copy(alphas, trans_input, prms.n_components, prms.n_rows, stream);
  raft::matrix::matrixVectorBinaryMult(trans_input, sqrt_vals.data(), prms.n_rows, prms.n_components, 
                                         false, true, stream);
}
