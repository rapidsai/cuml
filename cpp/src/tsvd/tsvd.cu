/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tsvd.cuh"

#include <cuml/decomposition/tsvd.hpp>

#include <raft/core/handle.hpp>

namespace ML {

void tsvdFit(raft::handle_t& handle,
             float* input,
             float* components,
             float* singular_vals,
             const paramsTSVD& prms,
             bool flip_signs_based_on_U = false)
{
  tsvdFit(
    handle, input, components, singular_vals, prms, handle.get_stream(), flip_signs_based_on_U);
}

void tsvdFit(raft::handle_t& handle,
             double* input,
             double* components,
             double* singular_vals,
             const paramsTSVD& prms,
             bool flip_signs_based_on_U = false)
{
  tsvdFit(
    handle, input, components, singular_vals, prms, handle.get_stream(), flip_signs_based_on_U);
}

void tsvdFitTransform(raft::handle_t& handle,
                      float* input,
                      float* trans_input,
                      float* components,
                      float* explained_var,
                      float* explained_var_ratio,
                      float* singular_vals,
                      const paramsTSVD& prms,
                      bool flip_signs_based_on_U = false)
{
  tsvdFitTransform(handle,
                   input,
                   trans_input,
                   components,
                   explained_var,
                   explained_var_ratio,
                   singular_vals,
                   prms,
                   handle.get_stream(),
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
                      bool flip_signs_based_on_U = false)
{
  tsvdFitTransform(handle,
                   input,
                   trans_input,
                   components,
                   explained_var,
                   explained_var_ratio,
                   singular_vals,
                   prms,
                   handle.get_stream(),
                   flip_signs_based_on_U);
}

void tsvdTransform(raft::handle_t& handle,
                   float* input,
                   float* components,
                   float* trans_input,
                   const paramsTSVD& prms)
{
  tsvdTransform(handle, input, components, trans_input, prms, handle.get_stream());
}

void tsvdTransform(raft::handle_t& handle,
                   double* input,
                   double* components,
                   double* trans_input,
                   const paramsTSVD& prms)
{
  tsvdTransform(handle, input, components, trans_input, prms, handle.get_stream());
}

void tsvdInverseTransform(raft::handle_t& handle,
                          float* trans_input,
                          float* components,
                          float* input,
                          const paramsTSVD& prms)
{
  tsvdInverseTransform(handle, trans_input, components, input, prms, handle.get_stream());
}

void tsvdInverseTransform(raft::handle_t& handle,
                          double* trans_input,
                          double* components,
                          double* input,
                          const paramsTSVD& prms)
{
  tsvdInverseTransform(handle, trans_input, components, input, prms, handle.get_stream());
}

};  // end namespace ML
