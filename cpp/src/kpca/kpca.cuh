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
#include <cuml/common/logger.hpp>
#include <raft/distance/kernels.cuh>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/core/handle.hpp>
#include <raft/linalg/eig.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>
#include <tsvd/tsvd.cuh>
#include <common/device_utils.cuh>
#include <rmm/device_vector.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>


namespace ML {

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
  auto thrust_policy = rmm::exec_policy(stream);
  CUML_LOG_INFO("kpcaFit with n_rows = %d, n_cols = %d, n_components = %d, kernel = %d, algorithm = %d",
            prms.n_rows, prms.n_cols, prms.n_components, prms.kernel, prms.algorithm);
  raft::print_device_vector("input", input, prms.n_rows * prms.n_cols, std::cout);
  //  TODO: defer assertions to python layer
  ASSERT(prms.n_cols > 1,
         "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(prms.n_rows > 1,
         "Parameter n_rows: number of rows cannot be less than two");
  ASSERT(
    prms.n_components > 0,
    "Parameter n_components: number of components cannot be less than one");

  size_t n_components = prms.n_components;
  if (n_components > prms.n_rows) n_components = prms.n_rows;
  raft::distance::kernels::GramMatrixBase<value_t> *kernel =
  raft::distance::kernels::KernelFactory<value_t>::create(prms.kernel);
  raft::device_matrix_view<const value_t, int, raft::layout_f_contiguous> X_view =
    raft::make_device_strided_matrix_view<const value_t, int, raft::layout_f_contiguous>(input, prms.n_rows, prms.n_cols, 0);
  raft::device_matrix_view<const value_t, int, raft::layout_f_contiguous> Y_view =
    raft::make_device_strided_matrix_view<const value_t, int, raft::layout_f_contiguous>(input, prms.n_rows, prms.n_cols, 0);
  CUML_LOG_INFO("X_view.stride(0) = %d, X_view.stride(1) = %d", X_view.stride(0), X_view.stride(1));
  CUML_LOG_INFO("X_view.extent(0) = %d, X_view.extent(1) = %d", X_view.extent(0), X_view.extent(1));
  rmm::device_uvector<value_t> kernel_mat(prms.n_rows * prms.n_rows, stream);
  raft::device_matrix_view<value_t, int, raft::layout_f_contiguous> kernel_input =
    raft::make_device_strided_matrix_view<value_t, int, raft::layout_f_contiguous>(kernel_mat.data(), prms.n_rows, prms.n_rows, 0);
  CUML_LOG_INFO("kernel_input.stride(0) = %d, kernel_input.stride(1) = %d", kernel_input.stride(0), kernel_input.stride(1));
  CUML_LOG_INFO("kernel_input.extent(0) = %d, kernel_input.extent(1) = %d", kernel_input.extent(0), kernel_input.extent(1));
  raft::print_device_vector("kernel_mat", kernel_mat.data(), prms.n_rows * prms.n_rows, std::cout);
  // Evaluate kernel matrix
  kernel->evaluate(handle, X_view, Y_view, kernel_input, (value_t*) nullptr, (value_t*) nullptr);
  raft::print_device_vector("kernel_mat2", kernel_mat.data(), prms.n_rows * prms.n_rows, std::cout);


  // raft::print_device_vector("kernel_mat", kernel_mat.data(), prms.n_rows * prms.n_rows, std::cout);
  // kernel->evaluate(handle, X_view, X_view, kernel_input, (value_t*) nullptr, (value_t*) nullptr);
  // raft::print_device_vector("kernel_mat2", kernel_mat.data(), prms.n_rows * prms.n_rows, std::cout);
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
  raft::print_device_vector("centering_mat", centering_mat.data(), prms.n_rows * prms.n_rows, std::cout);

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
  raft::print_device_vector("kernel_mat_centered", kernel_mat.data(), prms.n_rows * prms.n_rows, std::cout);
  //  either Jacobi (iterative power method) or DnC eigendecomp
  if (prms.algorithm == enum_solver::COV_EIG_JACOBI) {
    raft::linalg::eigJacobi(handle, kernel_mat.data(), prms.n_rows, prms.n_rows, alphas,
                            lambdas, stream, (value_t)prms.tol,
                            prms.n_iterations);
  } else {
    raft::linalg::eigDC(handle, kernel_mat.data(), prms.n_rows, prms.n_rows, alphas,
                            lambdas, stream);
  }

  raft::print_device_vector("alphas", alphas, prms.n_rows * prms.n_rows, std::cout);
  raft::print_device_vector("lambdas", lambdas, prms.n_rows, std::cout);
  raft::matrix::colReverse(alphas, prms.n_rows, prms.n_rows, stream);
  raft::matrix::rowReverse(lambdas, prms.n_rows, std::size_t(1), stream);
  ML::signFlip(lambdas, prms.n_rows, prms.n_rows, alphas, prms.n_rows, stream);
  raft::print_device_vector("col reversed alphas", alphas, prms.n_rows * prms.n_rows, std::cout);
  raft::print_device_vector("row reversed lambdas", lambdas, prms.n_components, std::cout);
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
  raft::print_device_vector("input lambdas", lambdas, prms.n_components, std::cout);
  raft::print_device_vector("input alphas", alphas, prms.n_rows * prms.n_rows, std::cout);

  rmm::device_uvector<value_t> sqrt_vals(prms.n_components, stream);
  raft::matrix::seqRoot(lambdas, sqrt_vals.data(), prms.n_components, stream);
  raft::matrix::copy(alphas, trans_input, prms.n_components, prms.n_rows, stream);
  raft::matrix::matrixVectorBinaryMult(trans_input, sqrt_vals.data(), prms.n_rows, prms.n_components, 
                                         false, true, stream);
  raft::print_device_vector("sqrt_vals", sqrt_vals.data(),  prms.n_components, std::cout);
  raft::print_device_vector("final lambdas", lambdas, prms.n_components, std::cout);
  raft::print_device_vector("final alphas", alphas, prms.n_rows * prms.n_rows, std::cout);
  raft::print_device_vector("final trans_input", trans_input,  prms.n_components * prms.n_rows, std::cout);
}
}