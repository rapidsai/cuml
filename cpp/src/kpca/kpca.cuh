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

#include <common/device_utils.cuh>
#include <cuda_runtime.h>
#include <cuml/matrix/kernelparams.h>
#include <cuml/decomposition/params.hpp>
#include <cuml/common/utils.hpp>
#include <raft/core/handle.hpp>
#include <raft/distance/kernels.cuh>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/eig.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/mean_center.cuh>
#include <raft/util/cuda_utils.cuh>
#include <rmm/device_vector.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <tsvd/tsvd.cuh>
#include <utility>


namespace ML {
template <typename T>
CUML_KERNEL void subtract_mean_kernel(T* mat, const T* row_means, const T* col_means, T overall_mean, int n_rows, int n_cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n_rows && col < n_cols) {
        const int index = col * n_rows + row;
        mat[index] = mat[index] - row_means[row] - col_means[col] + overall_mean;
    }
}

template <typename value_t>
CUML_KERNEL void divideBySqrtKernel(value_t* eigenvectors, const value_t* sqrt_vals, size_t n_training_samples, size_t n_components) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < n_components && row < n_training_samples) {
        size_t idx = col * n_training_samples + row;
        eigenvectors[idx] /= sqrt_vals[col];
    }
}

template <typename value_t>
void evaluateCenteredKernelMatrix(const raft::handle_t &handle, value_t *input1, int n_rows1, value_t *input2, int n_rows2, int n_cols,
                                  const MLCommon::Matrix::KernelParams &kernel_params, rmm::device_uvector<value_t> &kernel_mat, cudaStream_t stream) {
    auto thrust_policy = rmm::exec_policy(stream);
    raft::distance::kernels::GramMatrixBase<value_t> *kernel =
        raft::distance::kernels::KernelFactory<value_t>::create(kernel_params);
    raft::device_matrix_view<const value_t, int, raft::layout_f_contiguous> input1_view =
        raft::make_device_strided_matrix_view<const value_t, int, raft::layout_f_contiguous>(input1, n_rows1, n_cols, 0);
    raft::device_matrix_view<const value_t, int, raft::layout_f_contiguous> input2_view =
        raft::make_device_strided_matrix_view<const value_t, int, raft::layout_f_contiguous>(input2, n_rows2, n_cols, 0);

    raft::device_matrix_view<value_t, int, raft::layout_f_contiguous> kernel_view =
        raft::make_device_strided_matrix_view<value_t, int, raft::layout_f_contiguous>(kernel_mat.data(), n_rows1, n_rows2, 0);

    // Evaluate kernel matrix
    kernel->evaluate(handle, input1_view, input2_view, kernel_view, (value_t*) nullptr, (value_t*) nullptr);
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
  //  TODO: defer assertions to python layer
  ASSERT(prms.n_cols > 1,
         "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(prms.n_rows > 1,
         "Parameter n_rows: number of rows cannot be less than two");
  ASSERT(
    prms.n_components > 0,
    "Parameter n_components: number of components cannot be less than one");
  // raft::print_device_vector("kpcaFit() input", input, prms.n_rows * prms.n_cols, std::cout);
  rmm::device_uvector<value_t> alphas(prms.n_rows * prms.n_rows, stream);
  rmm::device_uvector<value_t> lambdas(prms.n_rows, stream);

  // Evaluate and center the kernel matrix
  rmm::device_uvector<value_t> kernel_mat(prms.n_rows * prms.n_rows, stream);
  evaluateCenteredKernelMatrix(handle, input, prms.n_rows, input, prms.n_rows, prms.n_cols, prms.kernel, kernel_mat, stream);
  // Mean-center the kernel matrix
  rmm::device_uvector<value_t> row_means(prms.n_rows, stream);

  // Step 1: Compute row means
  raft::stats::mean(row_means.data(), kernel_mat.data(), prms.n_rows, prms.n_rows, false, true, stream);

  // Step 3: Compute overall mean
  value_t overall_mean;
  thrust::device_ptr<value_t> d_kernel_mat_ptr(kernel_mat.data());
  value_t sum = thrust::reduce(thrust_policy, d_kernel_mat_ptr, d_kernel_mat_ptr + prms.n_rows * prms.n_rows, static_cast<value_t>(0));
  overall_mean = sum / (prms.n_rows * prms.n_rows);
  // Output the overall mean

  // Step 4: Mean-center the matrix
  dim3 grid((prms.n_rows + 31) / 32, (prms.n_rows + 31) / 32);
  dim3 block(32, 32);
  subtract_mean_kernel<value_t><<<grid, block, 0, stream>>>(kernel_mat.data(), row_means.data(), row_means.data(), overall_mean, prms.n_rows, prms.n_rows);
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
 * @param[out] eigenvectors: principal components of the input data. Size n_rows * n_components.
 * @param[out] eigenvalues: singular values of the data. Size n_components * 1.
 * @param[out] trans_input:  the transformed data. Size n_rows * n_components.
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename value_t>
void kpcaFitTransform(const raft::handle_t &handle, value_t *input,
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
  kpcaFit(handle, input, eigenvectors, eigenvalues, prms, stream);
  rmm::device_uvector<value_t> sqrt_vals(prms.n_components, stream);
  raft::matrix::seqRoot(eigenvalues, sqrt_vals.data(), prms.n_components, stream);
  raft::matrix::copy(eigenvectors, trans_input, prms.n_components, prms.n_rows, stream);
  raft::matrix::matrixVectorBinaryMult(trans_input, sqrt_vals.data(), prms.n_rows, prms.n_components, 
                                         false, true, stream);
}

/**
 * @brief performs transform operation for the pca. Transforms the data to kernel eigenspace.
 * @param[in] handle: the internal cuml handle object
 * @param[in] fit_input: data to transform. Size n_training_samples x n_cols.
 * @param[in] input: data to transform. Size n_rows x n_cols.
 * @param[in] eigenvectors: principal components of the input data. Size n_training_samples * n_components.
 * @param[in] eigenvalues: singular values of the data. Size n_components * 1.
 * @param[out] trans_input:  the transformed data. Size n_rows * n_components.
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename value_t>
void kpcaTransform(const raft::handle_t &handle, value_t *fit_input, value_t *input,
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
    "Parameter n_training_samples: number of training samples cannot be less than one");
  rmm::device_uvector<value_t> kernel_mat(prms.n_rows * prms.n_training_samples, stream);
  rmm::device_uvector<value_t> kernel_mat2(prms.n_training_samples * prms.n_training_samples, stream);
  auto thrust_policy = rmm::exec_policy(stream);
  thrust::fill(thrust_policy, kernel_mat.begin(), kernel_mat.end(), 0.0f);
  evaluateCenteredKernelMatrix(handle, input, prms.n_rows, fit_input, prms.n_training_samples, prms.n_cols, prms.kernel, kernel_mat, stream);
  evaluateCenteredKernelMatrix(handle, fit_input, prms.n_training_samples, fit_input, prms.n_training_samples, prms.n_cols, prms.kernel, kernel_mat2, stream);



  // Mean-center the kernel matrix
  rmm::device_uvector<value_t> row_means(prms.n_training_samples, stream);
  rmm::device_uvector<value_t> col_means(prms.n_rows, stream);

  // Step 1: Compute row means
  raft::stats::mean(row_means.data(), kernel_mat2.data(), prms.n_training_samples, prms.n_training_samples, false, true, stream);

  // Step 2: Compute column means
  raft::stats::mean(col_means.data(), kernel_mat.data(), prms.n_rows, prms.n_training_samples, false, true, stream);
  // Step 3: Compute overall mean
  value_t overall_mean;
  thrust::device_ptr<value_t> d_kernel_mat_ptr(kernel_mat2.data());
  value_t sum = thrust::reduce(thrust_policy, d_kernel_mat_ptr, d_kernel_mat_ptr + prms.n_rows * prms.n_rows, static_cast<value_t>(0));
  overall_mean = sum / (prms.n_rows * prms.n_rows);
  // Output the overall mean

  // Step 4: Mean-center the matrix
  dim3 grid((prms.n_rows + 31) / 32, (prms.n_training_samples + 31) / 32);
  dim3 block(32, 32);
  subtract_mean_kernel<value_t><<<grid, block, 0, stream>>>(kernel_mat.data(), col_means.data(), row_means.data(), overall_mean, prms.n_rows, prms.n_training_samples);




  rmm::device_uvector<value_t> sqrt_vals(prms.n_components, stream);
  raft::matrix::seqRoot(eigenvalues, sqrt_vals.data(), prms.n_components, stream);

  // Launch kernel to divide each element in eigenvectors by the corresponding sqrt_val
  dim3 gridDim((prms.n_components + 31) / 32, (prms.n_training_samples + 31) / 32);
  dim3 blockDim(32, 32);
  divideBySqrtKernel<<<gridDim, blockDim, 0, stream>>>(eigenvectors, sqrt_vals.data(), prms.n_training_samples, prms.n_components);

  raft::linalg::gemm(handle, trans_input, kernel_mat.data(), eigenvectors, prms.n_rows, prms.n_components, prms.n_training_samples, 
                      true, true, true, stream);
}
}