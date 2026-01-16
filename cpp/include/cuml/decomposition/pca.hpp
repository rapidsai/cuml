/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "params.hpp"

namespace raft {
class handle_t;
}

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
            bool flip_signs_based_on_U);
void pcaFit(raft::handle_t& handle,
            double* input,
            double* components,
            double* explained_var,
            double* explained_var_ratio,
            double* singular_vals,
            double* mu,
            double* noise_vars,
            const paramsPCA& prms,
            bool flip_signs_based_on_U);
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
                     bool flip_signs_based_on_U);
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
                     bool flip_signs_based_on_U);
void pcaInverseTransform(raft::handle_t& handle,
                         float* trans_input,
                         float* components,
                         float* singular_vals,
                         float* mu,
                         float* input,
                         const paramsPCA& prms);
void pcaInverseTransform(raft::handle_t& handle,
                         double* trans_input,
                         double* components,
                         double* singular_vals,
                         double* mu,
                         double* input,
                         const paramsPCA& prms);
void pcaTransform(raft::handle_t& handle,
                  float* input,
                  float* components,
                  float* trans_input,
                  float* singular_vals,
                  float* mu,
                  const paramsPCA& prms);
void pcaTransform(raft::handle_t& handle,
                  double* input,
                  double* components,
                  double* trans_input,
                  double* singular_vals,
                  double* mu,
                  const paramsPCA& prms);

};  // end namespace ML
