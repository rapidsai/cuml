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

#include <raft/linalg/cublas_wrappers.h>
#include <raft/linalg/transpose.h>
#include <cuml/common/device_buffer.hpp>
#include <cuml/decomposition/params.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>
#include <raft/linalg/eig.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/mean_center.cuh>
#include <stats/cov.cuh>
#include <tsvd/tsvd.cuh>
#include <matrix/kernelfactory.cuh>
#include <cuml/matrix/kernelparams.h>
#include <matrix/kernelmatrices.cuh>

namespace ML {

using namespace MLCommon;




// template <typename math_t, typename enum_solver = solver>
// void truncCompExpVars(const raft::handle_t &handle, math_t *in,
//                       math_t *components, math_t *explained_var,
//                       math_t *explained_var_ratio,
//                       const paramsTSVDTemplate<enum_solver> prms,
//                       cudaStream_t stream) {
//   int len = prms.n_cols * prms.n_cols;
//   auto allocator = handle.get_device_allocator();
//   device_buffer<math_t> components_all(allocator, stream, len);
//   device_buffer<math_t> explained_var_all(allocator, stream, prms.n_cols);
//   device_buffer<math_t> explained_var_ratio_all(allocator, stream, prms.n_cols);

//   calEig<math_t, enum_solver>(handle, in, components_all.data(),
//                               explained_var_all.data(), prms, stream);
//   raft::matrix::truncZeroOrigin(components_all.data(), prms.n_cols, components,
//                                 prms.n_components, prms.n_cols, stream);
//   raft::matrix::ratio(handle, explained_var_all.data(),
//                       explained_var_ratio_all.data(), prms.n_cols, stream);
//   raft::matrix::truncZeroOrigin(explained_var_all.data(), prms.n_cols,
//                                 explained_var, prms.n_components, 1, stream);
//   raft::matrix::truncZeroOrigin(explained_var_ratio_all.data(), prms.n_cols,
//                                 explained_var_ratio, prms.n_components, 1,
//                                 stream);
// }



/**
 * @brief perform fit operation for the pca. Generates eigenvectors, explained vars, singular vals, etc.
 * @param[in] handle: cuml handle object
 * @param[in] input: the data is fitted to PCA. Size n_rows x n_cols. The size of the data is indicated in prms.
 * @param[out] components: the principal components of the input data. Size n_cols * n_components.
 * @param[out] explained_var: explained variances (eigenvalues) of the principal components. Size n_components * 1.
 * @param[out] explained_var_ratio: the ratio of the explained variance and total variance. Size n_components * 1.
 * @param[out] singular_vals: singular values of the data. Size n_components * 1
 * @param[out] mu: mean of all the features (all the columns in the data). Size n_cols * 1.
 * @param[out] noise_vars: variance of the noise. Size 1 * 1 (scalar).
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename math_t>
void kpcaFit(const raft::handle_t &handle, math_t *input, math_t *components,
            math_t *explained_var, math_t *explained_var_ratio,
            math_t *singular_vals, math_t *mu, math_t *noise_vars,
            const paramsPCA &prms, cudaStream_t stream) {
  
  auto cublas_handle = handle.get_cublas_handle();
  ASSERT(prms.n_cols > 1,
         "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(prms.n_rows > 1,
         "Parameter n_rows: number of rows cannot be less than two");
  ASSERT(
    prms.n_components > 0,
    "Parameter n_components: number of components cannot be less than one");
  std::cout << "IN KPCA FIT 0 \n";

  int n_components = prms.n_components;
  if (n_components > prms.n_cols) n_components = prms.n_cols;

  Matrix::KernelParams kparam{Matrix::RBF, 0, 1, 0};
  Matrix::GramMatrixBase<math_t> *kernel =
    Matrix::KernelFactory<math_t>::create(kparam, cublas_handle);





  std::cout << "IN KPCA FIT 3\n";
  
  //  get the eigenvectors and eigenvalues of the gram matrix
  // truncCompExpVars(handle, kernel, components, explained_var, explained_var_ratio, prms, stream);

  std::cout << "END KPCA FIT\n";
}

/**
 * @brief perform fit and transform operations for the pca. Generates transformed data, eigenvectors, explained vars, singular vals, etc.
 * @param[in] handle: cuml handle object
 * @param[in] input: the data is fitted to PCA. Size n_rows x n_cols. The size of the data is indicated in prms.
 * @param[out] trans_input: the transformed data. Size n_rows * n_components.
 * @param[out] components: the principal components of the input data. Size n_cols * n_components.
 * @param[out] explained_var: explained variances (eigenvalues) of the principal components. Size n_components * 1.
 * @param[out] explained_var_ratio: the ratio of the explained variance and total variance. Size n_components * 1.
 * @param[out] singular_vals: singular values of the data. Size n_components * 1
 * @param[out] mu: mean of all the features (all the columns in the data). Size n_cols * 1.
 * @param[out] noise_vars: variance of the noise. Size 1 * 1 (scalar).
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename math_t>
void kpcaFitTransform(const raft::handle_t &handle, math_t *input,
                     math_t *trans_input, math_t *components,
                     math_t *explained_var, math_t *explained_var_ratio,
                     math_t *singular_vals, math_t *mu, math_t *noise_vars,
                     const paramsPCA &prms, cudaStream_t stream) {

  std::cout << "IN KPCA FIT-TRANSFORM \n";
  kpcaFit(handle, input, components, explained_var, explained_var_ratio,
         singular_vals, mu, noise_vars, prms, stream);
  kpcaTransform(handle, input, components, trans_input, singular_vals, mu, prms,
               stream);
  signFlip(trans_input, prms.n_rows, prms.n_components, components, prms.n_cols,
           handle.get_device_allocator(), stream);
  std::cout << "KPCA FIT-TRANSFORM DONE \n";
}

/**
 * @brief performs transform operation for the pca. Transforms the data to eigenspace.
 * @param[in] handle: the internal cuml handle object
 * @param[in] input: the data is transformed. Size n_rows x n_components.
 * @param[in] components: principal components of the input data. Size n_cols * n_components.
 * @param[out] trans_input:  the transformed data. Size n_rows * n_components.
 * @param[in] singular_vals: singular values of the data. Size n_components * 1.
 * @param[in] mu: mean value of the input data
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename math_t>
void kpcaTransform(const raft::handle_t &handle, math_t *input,
                  math_t *components, math_t *trans_input,
                  math_t *singular_vals, math_t *mu, const paramsPCA &prms,
                  cudaStream_t stream) {
  ASSERT(prms.n_cols > 1,
         "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(prms.n_rows > 0,
         "Parameter n_rows: number of rows cannot be less than one");
  ASSERT(
    prms.n_components > 0,
    "Parameter n_components: number of components cannot be less than one");
  
  std::cout << "IN KPCA TRANSFORM 1\n";
  raft::stats::meanCenter(input, input, mu, prms.n_cols, prms.n_rows, false,
                          true, stream);
  
  tsvdTransform(handle, input, components, trans_input, prms, stream);
  raft::stats::meanAdd(input, input, mu, prms.n_cols, prms.n_rows, false, true,
                       stream);
  std::cout << "KPCA TRANSFORM DONE \n";
}

};  // end namespace ML
