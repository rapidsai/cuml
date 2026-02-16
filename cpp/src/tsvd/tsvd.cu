/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/decomposition/tsvd.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/linalg/tsvd.cuh>

namespace ML {

namespace {

raft::linalg::paramsTSVD to_raft_params(const paramsTSVD& ml_prms)
{
  raft::linalg::paramsTSVD prms;
  prms.n_rows       = ml_prms.n_rows;
  prms.n_cols       = ml_prms.n_cols;
  prms.gpu_id       = ml_prms.gpu_id;
  prms.tol          = ml_prms.tol;
  prms.n_iterations = ml_prms.n_iterations;
  prms.verbose      = ml_prms.verbose;
  prms.n_components = ml_prms.n_components;
  prms.algorithm    = static_cast<raft::linalg::solver>(static_cast<int>(ml_prms.algorithm));
  return prms;
}

template <typename math_t>
void tsvd_fit_impl(raft::handle_t& handle,
                   math_t* input,
                   math_t* components,
                   math_t* singular_vals,
                   const paramsTSVD& prms,
                   bool flip_signs_based_on_U)
{
  auto raft_prms = to_raft_params(prms);
  raft::linalg::tsvd_fit(
    handle,
    raft_prms,
    raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
      input, prms.n_rows, prms.n_cols),
    raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
      components, prms.n_components, prms.n_cols),
    raft::make_device_vector_view<math_t, std::size_t>(singular_vals, prms.n_components),
    flip_signs_based_on_U);
}

template <typename math_t>
void tsvd_fit_transform_impl(raft::handle_t& handle,
                             math_t* input,
                             math_t* trans_input,
                             math_t* components,
                             math_t* explained_var,
                             math_t* explained_var_ratio,
                             math_t* singular_vals,
                             const paramsTSVD& prms,
                             bool flip_signs_based_on_U)
{
  auto raft_prms = to_raft_params(prms);
  raft::linalg::tsvd_fit_transform(
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
    flip_signs_based_on_U);
}

template <typename math_t>
void tsvd_transform_impl(raft::handle_t& handle,
                         math_t* input,
                         math_t* components,
                         math_t* trans_input,
                         const paramsTSVD& prms)
{
  auto raft_prms = to_raft_params(prms);
  raft::linalg::tsvd_transform(handle,
                               raft_prms,
                               raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
                                 input, prms.n_rows, prms.n_cols),
                               raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
                                 components, prms.n_components, prms.n_cols),
                               raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
                                 trans_input, prms.n_rows, prms.n_components));
}

template <typename math_t>
void tsvd_inverse_transform_impl(raft::handle_t& handle,
                                 math_t* trans_input,
                                 math_t* components,
                                 math_t* input,
                                 const paramsTSVD& prms)
{
  auto raft_prms = to_raft_params(prms);
  raft::linalg::tsvd_inverse_transform(
    handle,
    raft_prms,
    raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
      trans_input, prms.n_rows, prms.n_components),
    raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
      components, prms.n_components, prms.n_cols),
    raft::make_device_matrix_view<math_t, std::size_t, raft::col_major>(
      input, prms.n_rows, prms.n_cols));
}

}  // anonymous namespace

void tsvdFit(raft::handle_t& handle,
             float* input,
             float* components,
             float* singular_vals,
             const paramsTSVD& prms,
             bool flip_signs_based_on_U)
{
  tsvd_fit_impl(handle, input, components, singular_vals, prms, flip_signs_based_on_U);
}

void tsvdFit(raft::handle_t& handle,
             double* input,
             double* components,
             double* singular_vals,
             const paramsTSVD& prms,
             bool flip_signs_based_on_U)
{
  tsvd_fit_impl(handle, input, components, singular_vals, prms, flip_signs_based_on_U);
}

void tsvdFitTransform(raft::handle_t& handle,
                      float* input,
                      float* trans_input,
                      float* components,
                      float* explained_var,
                      float* explained_var_ratio,
                      float* singular_vals,
                      const paramsTSVD& prms,
                      bool flip_signs_based_on_U)
{
  tsvd_fit_transform_impl(handle,
                          input,
                          trans_input,
                          components,
                          explained_var,
                          explained_var_ratio,
                          singular_vals,
                          prms,
                          flip_signs_based_on_U);
}

void tsvdFitTransform(raft::handle_t& handle,
                      double* input,
                      double* trans_input,
                      double* components,
                      double* explained_var,
                      double* explained_var_ratio,
                      double* singular_vals,
                      const paramsTSVD& prms,
                      bool flip_signs_based_on_U)
{
  tsvd_fit_transform_impl(handle,
                          input,
                          trans_input,
                          components,
                          explained_var,
                          explained_var_ratio,
                          singular_vals,
                          prms,
                          flip_signs_based_on_U);
}

void tsvdTransform(raft::handle_t& handle,
                   float* input,
                   float* components,
                   float* trans_input,
                   const paramsTSVD& prms)
{
  tsvd_transform_impl(handle, input, components, trans_input, prms);
}

void tsvdTransform(raft::handle_t& handle,
                   double* input,
                   double* components,
                   double* trans_input,
                   const paramsTSVD& prms)
{
  tsvd_transform_impl(handle, input, components, trans_input, prms);
}

void tsvdInverseTransform(raft::handle_t& handle,
                          float* trans_input,
                          float* components,
                          float* input,
                          const paramsTSVD& prms)
{
  tsvd_inverse_transform_impl(handle, trans_input, components, input, prms);
}

void tsvdInverseTransform(raft::handle_t& handle,
                          double* trans_input,
                          double* components,
                          double* input,
                          const paramsTSVD& prms)
{
  tsvd_inverse_transform_impl(handle, trans_input, components, input, prms);
}

};  // end namespace ML
