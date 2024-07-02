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
#include <raft/core/handle.hpp>
#include <raft/distance/kernels.cuh>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/eig.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/util/cuda_utils.cuh>
#include <tsvd/tsvd.cuh>
#include <common/device_utils.cuh>
#include <rmm/device_vector.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <cuml/common/logger.hpp>


namespace ML {

template <typename value_t>
void evaluateCenteredKernelMatrix(const raft::handle_t &handle, value_t *input1, int n_rows1, value_t *input2, int n_rows2, int n_cols,
                                  const MLCommon::Matrix::KernelParams &kernel_params, rmm::device_uvector<value_t> &kernel_mat, cudaStream_t stream) {
  auto thrust_policy = rmm::exec_policy(stream);
  CUML_LOG_INFO("TOMAS evaluateCenteredKernelMatrix()");
  raft::distance::kernels::GramMatrixBase<value_t> *kernel =
    raft::distance::kernels::KernelFactory<value_t>::create(kernel_params);
  raft::device_matrix_view<const value_t, int, raft::layout_f_contiguous> input1_view =
    raft::make_device_strided_matrix_view<const value_t, int, raft::layout_f_contiguous>(input1, n_rows1, n_cols, 0);
  raft::device_matrix_view<const value_t, int, raft::layout_f_contiguous> input2_view =
    raft::make_device_strided_matrix_view<const value_t, int, raft::layout_f_contiguous>(input2, n_rows2, n_cols, 0);

  raft::device_matrix_view<value_t, int, raft::layout_f_contiguous> kernel_view =
    raft::make_device_strided_matrix_view<value_t, int, raft::layout_f_contiguous>(kernel_mat.data(), n_rows1, n_rows2, 0);

  CUML_LOG_INFO("TOMAS evaluateCenteredKernelMatrix() before eval()");
  // Evaluate kernel matrix
  kernel->evaluate(handle, input1_view, input2_view, kernel_view, (value_t*) nullptr, (value_t*) nullptr);
  CUML_LOG_INFO("TOMAS evaluateCenteredKernelMatrix() before centering()");

  // Create centering matrix (I - 1/nrows)
  int max_n_rows = std::max(n_rows1, n_rows2);
  rmm::device_uvector<value_t> i_mat(max_n_rows * max_n_rows, stream);
  rmm::device_uvector<value_t> diag(max_n_rows, stream);
  rmm::device_uvector<value_t> centering_mat(max_n_rows * max_n_rows, stream);

  thrust::fill(thrust_policy, diag.begin(), diag.end(), 1.0f);
  thrust::fill(thrust_policy, i_mat.begin(), i_mat.end(), 0.0f);

  raft::matrix::initializeDiagonalMatrix(diag.data(), i_mat.data(), max_n_rows, max_n_rows, stream);
  value_t inv_n_rows1 = 1.0 / n_rows1;
  value_t inv_n_rows2 = 1.0 / n_rows2;
  raft::linalg::subtractScalar(centering_mat.data(), i_mat.data(), inv_n_rows1, n_rows1 * n_rows1, stream);
  raft::linalg::subtractScalar(centering_mat.data(), i_mat.data(), inv_n_rows2, n_rows2 * n_rows2, stream);

  i_mat.release();
  diag.release();
  CUML_LOG_INFO("TOMAS evaluateCenteredKernelMatrix() after centering()");

  // Center the kernel matrix: K' = (I - 1/nrows) K (I - 1/nrows)
  rmm::device_uvector<value_t> temp_mat(n_rows1 * n_rows2, stream);
  raft::linalg::gemm(handle, temp_mat.data(), centering_mat.data(), kernel_mat.data(),
                     n_rows1, n_rows2, n_rows2,
                     true, true, true, stream);
  raft::linalg::gemm(handle, kernel_mat.data(), temp_mat.data(), centering_mat.data(),
                     n_rows1, n_rows2, n_rows1,
                     true, true, true, stream);

  temp_mat.release();
  centering_mat.release();
}


/**
 * @brief perform fit operation for kernel PCA. Generates eigenvalues and eigenvectors (unscaled)
 * @param[in] handle: cuml handle object
 * @param[in] input: data to fit using kernel PCA. Size n_rows x n_cols. The size of the data is indicated in prms.
 * @param[out] eigenvectors: unscaled eigenvectors of the kernel matrix. Scaling occurs in the transform function. Size n_rows * n_components.
 * @param[out] eigenvalues: eigenvalues of the principal components. Size n_components * 1.
 * @param[in] prms: data structure that includes all the parameters from data size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename value_t, typename enum_solver = ML::solver>
void kpcaFit(const raft::handle_t &handle, value_t *input, value_t *eigenvectors,
             value_t *eigenvalues, const ML::paramsKPCA &prms, cudaStream_t stream) {
  auto cublas_handle = handle.get_cublas_handle();
  auto thrust_policy = rmm::exec_policy(stream);
  CUML_LOG_INFO("TOMAS kpcaFit()");
  //  TODO: defer assertions to python layer
  ASSERT(prms.n_cols > 1,
         "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(prms.n_rows > 1,
         "Parameter n_rows: number of rows cannot be less than two");
  ASSERT(
    prms.n_components > 0,
    "Parameter n_components: number of components cannot be less than one");

  rmm::device_uvector<value_t> alphas(prms.n_rows * prms.n_rows, stream);
  rmm::device_uvector<value_t> lambdas(prms.n_rows, stream);

  // Evaluate and center the kernel matrix
  rmm::device_uvector<value_t> kernel_mat(prms.n_rows * prms.n_rows, stream);
  evaluateCenteredKernelMatrix(handle, input, prms.n_rows, input, prms.n_rows, prms.n_cols, prms.kernel, kernel_mat, stream);

  //  either Jacobi (iterative power method) or DnC eigendecomp
  if (prms.algorithm == enum_solver::COV_EIG_JACOBI) {
    raft::linalg::eigJacobi(handle, kernel_mat.data(), prms.n_rows, prms.n_rows, alphas.data(),
                            lambdas.data(), stream, (value_t) prms.tol,
                            prms.n_iterations);
  } else {
    raft::linalg::eigDC(handle, kernel_mat.data(), prms.n_rows, prms.n_rows, alphas.data(),
                        lambdas.data(), stream);
  }

  raft::matrix::colReverse(alphas.data(), prms.n_rows, prms.n_rows, stream);
  raft::matrix::rowReverse(lambdas.data(), prms.n_rows, std::size_t(1), stream);
  raft::matrix::copy(alphas.data(), eigenvectors, prms.n_rows, prms.n_components, stream);
  raft::matrix::copy(lambdas.data(), eigenvalues, prms.n_components, std::size_t(1), stream);
  ML::signFlip(eigenvectors, prms.n_rows, prms.n_components, eigenvalues, std::size_t(0), stream);
}

/**
 * @brief performs transform operation for the pca. Transforms the data to kernel eigenspace.
 * @param[in] handle: the internal cuml handle object
 * @param[in] input: data to transform. Size n_rows x n_components.
 * @param[in] eigenvectors: principal components of the input data. Size n_rows * n_components.
 * @param[in] eigenvalues: singular values of the data. Size n_components * 1.
 * @param[out] trans_input:  the transformed data. Size n_rows * n_components.
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename value_t>
void kpcaTransform(const raft::handle_t &handle, value_t *input,
                   value_t *eigenvectors, value_t *eigenvalues,
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
  raft::matrix::seqRoot(eigenvalues, sqrt_vals.data(), prms.n_components, stream);
  raft::matrix::copy(eigenvectors, trans_input, prms.n_components, prms.n_rows, stream);
  raft::matrix::matrixVectorBinaryMult(trans_input, sqrt_vals.data(), prms.n_rows, prms.n_components, 
                                         false, true, stream);
}

template <typename value_t>
__global__ void divideEigenvectors(value_t* eigenvectors, const value_t* sqrt_vals, int n_rows, int n_components) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < n_components && row < n_rows) {
        eigenvectors[col * n_rows + row] /= sqrt_vals[col];
    }
}

/**
 * @brief performs transform operation for the pca. Transforms the data to kernel eigenspace.
 * @param[in] handle: the internal cuml handle object
 * @param[in] fit_input: data to transform. Size fit_n_rows x n_cols.
 * @param[in] input: data to transform. Size n_rows x n_cols.
 * @param[in] eigenvectors: principal components of the input data. Size n_rows * n_components.
 * @param[in] eigenvalues: singular values of the data. Size n_components * 1.
 * @param[out] trans_input:  the transformed data. Size n_rows * n_components.
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename value_t>
void kpcaTransform2(const raft::handle_t &handle, value_t *fit_input, value_t *input,
                   value_t *eigenvectors, value_t *eigenvalues,
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
  ASSERT(
    prms.n_training_samples > 0,
    "Parameter n_components: number of components cannot be less than one");
  rmm::device_uvector<value_t> kernel_mat(prms.n_rows * prms.n_training_samples, stream);
  CUML_LOG_INFO("TOMAS kpcaTransform2() before evaluateCenteredKernelMatrix");
  evaluateCenteredKernelMatrix(handle, fit_input, prms.n_training_samples, input, prms.n_rows, prms.n_cols, prms.kernel, kernel_mat, stream);
  rmm::device_uvector<value_t> sqrt_vals(prms.n_components, stream);
  raft::matrix::seqRoot(eigenvalues, sqrt_vals.data(), prms.n_components, stream);
  // Use raft::linalg::eltwiseDivide to divide each eigenvector by the corresponding sqrt_val
  for (int i = 0; i < prms.n_components; ++i) {
      value_t* col_start = eigenvectors + i * prms.n_rows; // Pointer to the start of the i-th column
      raft::linalg::eltwiseDivide(
          col_start,    // Output pointer
          col_start,    // Input pointer
          sqrt_vals.data() + i, // Scalar value for division
          prms.n_rows,  // Number of elements
          stream);      // CUDA stream
  }
  // CODE HERE on eigenvectors and sqrt_vals.data()
  raft::linalg::gemm(handle, trans_input, kernel_mat.data(), sqrt_vals.data(), prms.n_rows, prms.n_training_samples, prms.n_components, 
                      true, false, false, stream);
}
}