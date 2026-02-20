/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/decomposition/pca.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/linalg/pca.cuh>

namespace ML {

/**
 * @brief Convert ML::paramsPCA to raft::linalg::paramsPCA.
 */
raft::linalg::paramsPCA to_raft_params(const paramsPCA& ml_prms)
{
  raft::linalg::paramsPCA prms;
  prms.n_rows       = ml_prms.n_rows;
  prms.n_cols       = ml_prms.n_cols;
  prms.gpu_id       = ml_prms.gpu_id;
  prms.tol          = ml_prms.tol;
  prms.n_iterations = ml_prms.n_iterations;
  prms.n_components = ml_prms.n_components;
  prms.algorithm    = ml_prms.algorithm;
  prms.copy         = ml_prms.copy;
  prms.whiten       = ml_prms.whiten;
  return prms;
}

template <typename math_t>
void pca_fit_impl(const raft::handle_t& handle,
                  math_t* input,
                  math_t* components,
                  math_t* explained_var,
                  math_t* explained_var_ratio,
                  math_t* singular_vals,
                  math_t* mu,
                  math_t* noise_vars,
                  const paramsPCA& prms,
                  bool flip_signs_based_on_U)
{
  auto raft_prms = to_raft_params(prms);
  raft::linalg::pca_fit(
    handle,
    raft_prms,
    raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
      input, prms.n_rows, prms.n_cols),
    raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
      components, prms.n_components, prms.n_cols),
    raft::make_device_vector_view<math_t, std::size_t>(explained_var, prms.n_components),
    raft::make_device_vector_view<math_t, std::size_t>(explained_var_ratio, prms.n_components),
    raft::make_device_vector_view<math_t, std::size_t>(singular_vals, prms.n_components),
    raft::make_device_vector_view<math_t, std::size_t>(mu, prms.n_cols),
    raft::make_device_scalar_view<math_t, std::size_t>(noise_vars),
    flip_signs_based_on_U);
}

template <typename math_t>
void pca_fit_transform_impl(const raft::handle_t& handle,
                            math_t* input,
                            math_t* trans_input,
                            math_t* components,
                            math_t* explained_var,
                            math_t* explained_var_ratio,
                            math_t* singular_vals,
                            math_t* mu,
                            math_t* noise_vars,
                            const paramsPCA& prms,
                            bool flip_signs_based_on_U)
{
  auto raft_prms = to_raft_params(prms);
  raft::linalg::pca_fit_transform(
    handle,
    raft_prms,
    raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
      input, prms.n_rows, prms.n_cols),
    raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
      trans_input, prms.n_rows, prms.n_components),
    raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
      components, prms.n_components, prms.n_cols),
    raft::make_device_vector_view<math_t, std::size_t>(explained_var, prms.n_components),
    raft::make_device_vector_view<math_t, std::size_t>(explained_var_ratio, prms.n_components),
    raft::make_device_vector_view<math_t, std::size_t>(singular_vals, prms.n_components),
    raft::make_device_vector_view<math_t, std::size_t>(mu, prms.n_cols),
    raft::make_device_scalar_view<math_t, std::size_t>(noise_vars),
    flip_signs_based_on_U);
}

template <typename math_t>
void pca_inverse_transform_impl(const raft::handle_t& handle,
                                math_t* trans_input,
                                math_t* components,
                                math_t* singular_vals,
                                math_t* mu,
                                math_t* input,
                                const paramsPCA& prms)
{
  auto raft_prms = to_raft_params(prms);
  raft::linalg::pca_inverse_transform(
    handle,
    raft_prms,
    raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
      trans_input, prms.n_rows, prms.n_components),
    raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
      components, prms.n_components, prms.n_cols),
    raft::make_device_vector_view<math_t, std::size_t>(singular_vals, prms.n_components),
    raft::make_device_vector_view<math_t, std::size_t>(mu, prms.n_cols),
    raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
      input, prms.n_rows, prms.n_cols));
}

template <typename math_t>
void pca_transform_impl(const raft::handle_t& handle,
                        math_t* input,
                        math_t* components,
                        math_t* trans_input,
                        math_t* singular_vals,
                        math_t* mu,
                        const paramsPCA& prms)
{
  auto raft_prms = to_raft_params(prms);
  raft::linalg::pca_transform(
    handle,
    raft_prms,
    raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
      input, prms.n_rows, prms.n_cols),
    raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
      components, prms.n_components, prms.n_cols),
    raft::make_device_vector_view<math_t, std::size_t>(singular_vals, prms.n_components),
    raft::make_device_vector_view<math_t, std::size_t>(mu, prms.n_cols),
    raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
      trans_input, prms.n_rows, prms.n_components));
}

void pcaFit(const raft::handle_t& handle,
            float* input,
            float* components,
            float* explained_var,
            float* explained_var_ratio,
            float* singular_vals,
            float* mu,
            float* noise_vars,
            const paramsPCA& prms,
            bool flip_signs_based_on_U)
{
  pca_fit_impl(handle,
               input,
               components,
               explained_var,
               explained_var_ratio,
               singular_vals,
               mu,
               noise_vars,
               prms,
               flip_signs_based_on_U);
}

void pcaFit(const raft::handle_t& handle,
            double* input,
            double* components,
            double* explained_var,
            double* explained_var_ratio,
            double* singular_vals,
            double* mu,
            double* noise_vars,
            const paramsPCA& prms,
            bool flip_signs_based_on_U)
{
  pca_fit_impl(handle,
               input,
               components,
               explained_var,
               explained_var_ratio,
               singular_vals,
               mu,
               noise_vars,
               prms,
               flip_signs_based_on_U);
}

void pcaFitTransform(const raft::handle_t& handle,
                     float* input,
                     float* trans_input,
                     float* components,
                     float* explained_var,
                     float* explained_var_ratio,
                     float* singular_vals,
                     float* mu,
                     float* noise_vars,
                     const paramsPCA& prms,
                     bool flip_signs_based_on_U)
{
  pca_fit_transform_impl(handle,
                         input,
                         trans_input,
                         components,
                         explained_var,
                         explained_var_ratio,
                         singular_vals,
                         mu,
                         noise_vars,
                         prms,
                         flip_signs_based_on_U);
}

void pcaFitTransform(const raft::handle_t& handle,
                     double* input,
                     double* trans_input,
                     double* components,
                     double* explained_var,
                     double* explained_var_ratio,
                     double* singular_vals,
                     double* mu,
                     double* noise_vars,
                     const paramsPCA& prms,
                     bool flip_signs_based_on_U)
{
  pca_fit_transform_impl(handle,
                         input,
                         trans_input,
                         components,
                         explained_var,
                         explained_var_ratio,
                         singular_vals,
                         mu,
                         noise_vars,
                         prms,
                         flip_signs_based_on_U);
}

void pcaInverseTransform(const raft::handle_t& handle,
                         float* trans_input,
                         float* components,
                         float* singular_vals,
                         float* mu,
                         float* input,
                         const paramsPCA& prms)
{
  pca_inverse_transform_impl(handle, trans_input, components, singular_vals, mu, input, prms);
}

void pcaInverseTransform(const raft::handle_t& handle,
                         double* trans_input,
                         double* components,
                         double* singular_vals,
                         double* mu,
                         double* input,
                         const paramsPCA& prms)
{
  pca_inverse_transform_impl(handle, trans_input, components, singular_vals, mu, input, prms);
}

void pcaTransform(const raft::handle_t& handle,
                  float* input,
                  float* components,
                  float* trans_input,
                  float* singular_vals,
                  float* mu,
                  const paramsPCA& prms)
{
  pca_transform_impl(handle, input, components, trans_input, singular_vals, mu, prms);
}

void pcaTransform(const raft::handle_t& handle,
                  double* input,
                  double* components,
                  double* trans_input,
                  double* singular_vals,
                  double* mu,
                  const paramsPCA& prms)
{
  pca_transform_impl(handle, input, components, trans_input, singular_vals, mu, prms);
}

};  // end namespace ML
