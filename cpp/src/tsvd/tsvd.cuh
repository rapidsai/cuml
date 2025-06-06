/*
 * Copyright (c) 2018-2025, NVIDIA CORPORATION.
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

#include <cuml/decomposition/params.hpp>

#include <raft/core/handle.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/eig.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/rsvd.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/stddev.cuh>
#include <raft/stats/sum.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace ML {

template <typename math_t>
void calCompExpVarsSvd(const raft::handle_t& handle,
                       math_t* in,
                       math_t* components,
                       math_t* singular_vals,
                       math_t* explained_vars,
                       math_t* explained_var_ratio,
                       const paramsTSVD& prms,
                       cudaStream_t stream)
{
  auto cusolver_handle = handle.get_cusolver_dn_handle();
  auto cublas_handle   = handle.get_cublas_handle();

  auto diff    = prms.n_cols - prms.n_components;
  math_t ratio = math_t(diff) / math_t(prms.n_cols);
  ASSERT(ratio >= math_t(0.2),
         "Number of components should be less than at least 80 percent of the "
         "number of features");

  std::size_t p = static_cast<std::size_t>(math_t(0.1) * math_t(prms.n_cols));
  // int p = int(math_t(prms.n_cols) / math_t(4));
  ASSERT(p >= 5, "RSVD should be used where the number of columns are at least 50");

  auto total_random_vecs = prms.n_components + p;
  ASSERT(total_random_vecs < prms.n_cols,
         "RSVD should be used where the number of columns are at least 50");

  rmm::device_uvector<math_t> components_temp(prms.n_cols * prms.n_components, stream);
  math_t* left_eigvec = nullptr;
  raft::linalg::rsvdFixedRank(handle,
                              in,
                              prms.n_rows,
                              prms.n_cols,
                              singular_vals,
                              left_eigvec,
                              components_temp.data(),
                              prms.n_components,
                              p,
                              true,
                              false,
                              true,
                              false,
                              (math_t)prms.tol,
                              prms.n_iterations,
                              stream);

  raft::linalg::transpose(
    handle, components_temp.data(), components, prms.n_cols, prms.n_components, stream);
  raft::matrix::power(singular_vals, explained_vars, math_t(1), prms.n_components, stream);
  raft::matrix::ratio(handle, explained_vars, explained_var_ratio, prms.n_components, stream);
}

template <typename math_t, typename enum_solver = solver>
void calEig(const raft::handle_t& handle,
            math_t* in,
            math_t* components,
            math_t* explained_var,
            const paramsTSVDTemplate<enum_solver>& prms,
            cudaStream_t stream)
{
  auto cusolver_handle = handle.get_cusolver_dn_handle();

  if (prms.algorithm == enum_solver::COV_EIG_JACOBI) {
    raft::linalg::eigJacobi(handle,
                            in,
                            prms.n_cols,
                            prms.n_cols,
                            components,
                            explained_var,
                            stream,
                            (math_t)prms.tol,
                            prms.n_iterations);
  } else {
    raft::linalg::eigDC(handle, in, prms.n_cols, prms.n_cols, components, explained_var, stream);
  }

  raft::matrix::colReverse(components, prms.n_cols, prms.n_cols, stream);
  raft::linalg::transpose(components, prms.n_cols, stream);

  raft::matrix::rowReverse(explained_var, prms.n_cols, std::size_t(1), stream);
}

/**
 * @defgroup sign flip for PCA and tSVD. This is used to stabilize the sign of column major eigen
 * vectors
 * @param input: input matrix that will be used to determine the sign.
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param components: components matrix.
 * @param n_cols_comp: number of columns of components matrix
 * @param stream cuda stream
 * @{
 */
template <typename math_t>
void signFlip(math_t* input,
              std::size_t n_rows,
              std::size_t n_cols,
              math_t* components,
              std::size_t n_cols_comp,
              cudaStream_t stream)
{
  auto counting = thrust::make_counting_iterator(0);
  auto m        = n_rows;

  thrust::for_each(
    rmm::exec_policy(stream), counting, counting + n_cols, [=] __device__(std::size_t idx) {
      auto d_i = idx * m;
      auto end = d_i + m;

      math_t max            = 0.0;
      std::size_t max_index = 0;
      for (auto i = d_i; i < end; i++) {
        math_t val = input[i];
        if (val < 0.0) { val = -val; }
        if (val > max) {
          max       = val;
          max_index = i;
        }
      }

      if (input[max_index] < 0.0) {
        for (auto i = d_i; i < end; i++) {
          input[i] = -input[i];
        }

        auto len = n_cols * n_cols_comp;
        for (auto i = idx; i < len; i = i + n_cols) {
          components[i] = -components[i];
        }
      }
    });
}

/**
 * @brief perform fit operation for the tsvd. Generates eigenvectors, explained vars, singular vals,
 * etc.
 * @param[in] handle: the internal cuml handle object
 * @param[in] input: the data is fitted to PCA. Size n_rows x n_cols. The size of the data is
 * indicated in prms.
 * @param[out] components: the principal components of the input data. Size n_cols * n_components.
 * @param[out] singular_vals: singular values of the data. Size n_components * 1
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename math_t>
void tsvdFit(const raft::handle_t& handle,
             math_t* input,
             math_t* components,
             math_t* singular_vals,
             const paramsTSVD& prms,
             cudaStream_t stream)
{
  auto cublas_handle = handle.get_cublas_handle();

  ASSERT(prms.n_cols > 1, "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(prms.n_rows > 1, "Parameter n_rows: number of rows cannot be less than two");
  ASSERT(prms.n_components > 0,
         "Parameter n_components: number of components cannot be less than one");

  auto n_components = prms.n_components;
  if (prms.n_components > prms.n_cols) n_components = prms.n_cols;

  size_t len = prms.n_cols * prms.n_cols;
  rmm::device_uvector<math_t> input_cross_mult(len, stream);

  math_t alpha = math_t(1);
  math_t beta  = math_t(0);
  raft::linalg::gemm(handle,
                     input,
                     prms.n_rows,
                     prms.n_cols,
                     input,
                     input_cross_mult.data(),
                     prms.n_cols,
                     prms.n_cols,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     alpha,
                     beta,
                     stream);

  rmm::device_uvector<math_t> components_all(len, stream);
  rmm::device_uvector<math_t> explained_var_all(prms.n_cols, stream);

  calEig(
    handle, input_cross_mult.data(), components_all.data(), explained_var_all.data(), prms, stream);

  raft::matrix::truncZeroOrigin(
    components_all.data(), prms.n_cols, components, n_components, prms.n_cols, stream);

  math_t scalar = math_t(1);
  raft::matrix::seqRoot(explained_var_all.data(), singular_vals, scalar, n_components, stream);
}

/**
 * @brief performs fit and transform operations for the tsvd. Generates transformed data,
 * eigenvectors, explained vars, singular vals, etc.
 * @param[in] handle: the internal cuml handle object
 * @param[in] input: the data is fitted to PCA. Size n_rows x n_cols. The size of the data is
 * indicated in prms.
 * @param[out] trans_input: the transformed data. Size n_rows * n_components.
 * @param[out] components: the principal components of the input data. Size n_cols * n_components.
 * @param[out] explained_var: explained variances (eigenvalues) of the principal components. Size
 * n_components * 1.
 * @param[out] explained_var_ratio: the ratio of the explained variance and total variance. Size
 * n_components * 1.
 * @param[out] singular_vals: singular values of the data. Size n_components * 1
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename math_t>
void tsvdFitTransform(const raft::handle_t& handle,
                      math_t* input,
                      math_t* trans_input,
                      math_t* components,
                      math_t* explained_var,
                      math_t* explained_var_ratio,
                      math_t* singular_vals,
                      const paramsTSVD& prms,
                      cudaStream_t stream)
{
  tsvdFit(handle, input, components, singular_vals, prms, stream);
  tsvdTransform(handle, input, components, trans_input, prms, stream);

  signFlip(trans_input, prms.n_rows, prms.n_components, components, prms.n_cols, stream);

  rmm::device_uvector<math_t> mu_trans(prms.n_components, stream);
  raft::stats::mean<false>(
    mu_trans.data(), trans_input, prms.n_components, prms.n_rows, false, stream);
  raft::stats::vars<false>(
    explained_var, trans_input, mu_trans.data(), prms.n_components, prms.n_rows, false, stream);

  rmm::device_uvector<math_t> mu(prms.n_cols, stream);
  rmm::device_uvector<math_t> vars(prms.n_cols, stream);

  raft::stats::mean<false>(mu.data(), input, prms.n_cols, prms.n_rows, false, stream);
  raft::stats::vars<false>(vars.data(), input, mu.data(), prms.n_cols, prms.n_rows, false, stream);

  rmm::device_scalar<math_t> total_vars(stream);
  raft::stats::sum<false>(total_vars.data(), vars.data(), std::size_t(1), prms.n_cols, stream);

  math_t total_vars_h;
  raft::update_host(&total_vars_h, total_vars.data(), 1, stream);
  handle.sync_stream(stream);
  math_t scalar = math_t(1) / total_vars_h;

  raft::linalg::scalarMultiply(
    explained_var_ratio, explained_var, scalar, prms.n_components, stream);
}

/**
 * @brief performs transform operation for the tsvd. Transforms the data to eigenspace.
 * @param[in] handle the internal cuml handle object
 * @param[in] input: the data is transformed. Size n_rows x n_components.
 * @param[in] components: principal components of the input data. Size n_cols * n_components.
 * @param[out] trans_input: output that is transformed version of input
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename math_t>
void tsvdTransform(const raft::handle_t& handle,
                   math_t* input,
                   math_t* components,
                   math_t* trans_input,
                   const paramsTSVD& prms,
                   cudaStream_t stream)
{
  ASSERT(prms.n_cols > 1, "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(prms.n_rows > 0, "Parameter n_rows: number of rows cannot be less than one");
  ASSERT(prms.n_components > 0,
         "Parameter n_components: number of components cannot be less than one");

  math_t alpha = math_t(1);
  math_t beta  = math_t(0);
  raft::linalg::gemm(handle,
                     input,
                     prms.n_rows,
                     prms.n_cols,
                     components,
                     trans_input,
                     prms.n_rows,
                     prms.n_components,
                     CUBLAS_OP_N,
                     CUBLAS_OP_T,
                     alpha,
                     beta,
                     stream);
}

/**
 * @brief performs inverse transform operation for the tsvd. Transforms the transformed data back to
 * original data.
 * @param[in] handle the internal cuml handle object
 * @param[in] trans_input: the data is fitted to PCA. Size n_rows x n_components.
 * @param[in] components: transpose of the principal components of the input data. Size n_components
 * * n_cols.
 * @param[out] input: the data is fitted to PCA. Size n_rows x n_cols.
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] stream cuda stream
 */
template <typename math_t>
void tsvdInverseTransform(const raft::handle_t& handle,
                          math_t* trans_input,
                          math_t* components,
                          math_t* input,
                          const paramsTSVD& prms,
                          cudaStream_t stream)
{
  ASSERT(prms.n_cols > 1, "Parameter n_cols: number of columns cannot be less than one");
  ASSERT(prms.n_rows > 0, "Parameter n_rows: number of rows cannot be less than one");
  ASSERT(prms.n_components > 0,
         "Parameter n_components: number of components cannot be less than one");

  math_t alpha = math_t(1);
  math_t beta  = math_t(0);

  raft::linalg::gemm(handle,
                     trans_input,
                     prms.n_rows,
                     prms.n_components,
                     components,
                     input,
                     prms.n_rows,
                     prms.n_cols,
                     CUBLAS_OP_N,
                     CUBLAS_OP_N,
                     alpha,
                     beta,
                     stream);
}

};  // end namespace ML
