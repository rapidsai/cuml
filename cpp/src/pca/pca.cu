/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pca.cuh"

#include <cuml/decomposition/pca.hpp>

#include <raft/core/handle.hpp>

namespace ML {

void pcaFit(raft::handle_t& handle,
            float* input,
            float* components,
            float* explained_var,
            float* explained_var_ratio,
            float* singular_vals,
            float* mu,
            float* noise_vars,
            const paramsPCA& prms,
            bool flip_signs_based_on_U = false)
{
  pcaFit(handle,
         input,
         components,
         explained_var,
         explained_var_ratio,
         singular_vals,
         mu,
         noise_vars,
         prms,
         handle.get_stream(),
         flip_signs_based_on_U);
}

void pcaFit(raft::handle_t& handle,
            double* input,
            double* components,
            double* explained_var,
            double* explained_var_ratio,
            double* singular_vals,
            double* mu,
            double* noise_vars,
            const paramsPCA& prms,
            bool flip_signs_based_on_U = false)
{
  pcaFit(handle,
         input,
         components,
         explained_var,
         explained_var_ratio,
         singular_vals,
         mu,
         noise_vars,
         prms,
         handle.get_stream(),
         flip_signs_based_on_U);
}

void pcaFitTransform(raft::handle_t& handle,
                     float* input,
                     float* trans_input,
                     float* components,
                     float* explained_var,
                     float* explained_var_ratio,
                     float* singular_vals,
                     float* mu,
                     float* noise_vars,
                     const paramsPCA& prms,
                     bool flip_signs_based_on_U = false)
{
  pcaFitTransform(handle,
                  input,
                  trans_input,
                  components,
                  explained_var,
                  explained_var_ratio,
                  singular_vals,
                  mu,
                  noise_vars,
                  prms,
                  handle.get_stream(),
                  flip_signs_based_on_U);
}

void pcaFitTransform(raft::handle_t& handle,
                     double* input,
                     double* trans_input,
                     double* components,
                     double* explained_var,
                     double* explained_var_ratio,
                     double* singular_vals,
                     double* mu,
                     double* noise_vars,
                     const paramsPCA& prms,
                     bool flip_signs_based_on_U = false)
{
  pcaFitTransform(handle,
                  input,
                  trans_input,
                  components,
                  explained_var,
                  explained_var_ratio,
                  singular_vals,
                  mu,
                  noise_vars,
                  prms,
                  handle.get_stream(),
                  flip_signs_based_on_U);
}

void pcaInverseTransform(raft::handle_t& handle,
                         float* trans_input,
                         float* components,
                         float* singular_vals,
                         float* mu,
                         float* input,
                         const paramsPCA& prms)
{
  pcaInverseTransform(
    handle, trans_input, components, singular_vals, mu, input, prms, handle.get_stream());
}

void pcaInverseTransform(raft::handle_t& handle,
                         double* trans_input,
                         double* components,
                         double* singular_vals,
                         double* mu,
                         double* input,
                         const paramsPCA& prms)
{
  pcaInverseTransform(
    handle, trans_input, components, singular_vals, mu, input, prms, handle.get_stream());
}

void pcaTransform(raft::handle_t& handle,
                  float* input,
                  float* components,
                  float* trans_input,
                  float* singular_vals,
                  float* mu,
                  const paramsPCA& prms)
{
  pcaTransform(
    handle, input, components, trans_input, singular_vals, mu, prms, handle.get_stream());
}

void pcaTransform(raft::handle_t& handle,
                  double* input,
                  double* components,
                  double* trans_input,
                  double* singular_vals,
                  double* mu,
                  const paramsPCA& prms)
{
  pcaTransform(
    handle, input, components, trans_input, singular_vals, mu, prms, handle.get_stream());
}

};  // end namespace ML
